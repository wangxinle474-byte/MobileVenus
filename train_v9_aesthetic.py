"""
Distill v9 — 美学感知微调 (Aesthetic-Aware Fine-tuning)
基于 v7 checkpoint，策略:
  1. param_loss 作为唯一梯度来源 (diff_isp 梯度链不稳定)
  2. diff_isp 渲染 + 图像损失 + MUSIQ 美学分: no_grad 监控
  3. 增强版 diff_isp (tone curve + clarity) 提升渲染质量
"""
import sys, json, time, argparse, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT          = Path('/root/autodl-tmp')
CODE_DIR      = ROOT / 'IntelligenceCamera'
FIVEK_JPEG    = ROOT / 'fivek_jpeg'
DATA_DIR      = ROOT / 'data'
FIVEK_PARAMS  = DATA_DIR / 'fivek_expert_params.json'
FIVEK_CONSENS = DATA_DIR / 'fivek_expert_consensus.json'
STAGE_A_CKPT  = ROOT / 'checkpoints' / 'distill_v6' / 'stage_a' / 'best.pt'
V7_CKPT       = ROOT / 'checkpoints' / 'distill_v7' / 'stage_b' / 'best.pt'
OUTPUT_DIR    = ROOT / 'checkpoints' / 'distill_v9'

sys.path.insert(0, str(CODE_DIR))

def _make_denorm_fn():
    from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
    import torch
    def denorm(params_norm, device):
        result = {}
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_RANGES[name]
            result[name] = (lo + (params_norm[:, i] + 1.0) / 2.0 * (hi - lo)).to(device)
        return result
    return PARAM_NAMES, PARAM_RANGES, denorm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',       type=int,   default=30)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=5e-5)
    parser.add_argument('--image_size',   type=int,   default=224)
    parser.add_argument('--val_ratio',    type=float, default=0.1)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--log_every',    type=int,   default=50)
    parser.add_argument('--aesthetic_every', type=int, default=5)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
    from training.fivek_8param.dataset_expert import build_expert_datasets
    from training.fivek_8param.loss import DistillParamLoss
    from training.semantic_distill.config import DistillConfig
    from models.diff_isp import apply_diff_isp, ssim_loss
    from training.aesthetic_loss import AestheticLoss

    PARAM_NAMES, PARAM_RANGES, denorm = _make_denorm_fn()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    expert_dirs = {}
    for ex in ['a', 'b', 'c', 'd', 'e']:
        p = ROOT / f'fivek_expert_{ex}'
        if p.exists():
            expert_dirs[ex] = str(p)
    logger.info(f'Expert: {list(expert_dirs.keys())}')

    train_ds, val_ds = build_expert_datasets(
        str(FIVEK_PARAMS), str(FIVEK_CONSENS),
        orig_jpeg_dir=str(FIVEK_JPEG), expert_dirs=expert_dirs,
        image_size=args.image_size, val_ratio=args.val_ratio,
        seed=args.seed, val_expert='c',
    )
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f'Data: train={len(train_ds)}, val={len(val_ds)}')

    cfg = DistillConfig()
    stage_a = SemanticDistillModel(image_size=cfg.image_size, visual_dim=cfg.visual_dim, semantic_dim=cfg.semantic_dim, text_dim=cfg.text_dim)
    sa_state = torch.load(str(STAGE_A_CKPT), map_location='cpu', weights_only=False)
    stage_a.load_state_dict(sa_state['model_state_dict'])
    model = DistillParamModel(stage_a, decoder_hidden=cfg.decoder_hidden)

    if V7_CKPT.exists():
        v7_state = torch.load(str(V7_CKPT), map_location='cpu', weights_only=False)
        model.load_state_dict(v7_state['model_state_dict'])
        logger.info(f'v7 loaded (val_loss={v7_state.get("val_loss","?")})')
    else:
        logger.error(f'v7 not found: {V7_CKPT}')
        return

    model.to(device)
    logger.info(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    aesthetic_eval = None
    try:
        aesthetic_eval = AestheticLoss(metric_name='musiq-ava', target_score=5.0, input_size=224, device=str(device))
        if not aesthetic_eval.available:
            aesthetic_eval = None
    except Exception as e:
        logger.warning(f'MUSIQ failed: {e}')

    param_crit = DistillParamLoss(align_weight=0.0, use_consensus=True)
    optimizer = optim.AdamW([
        {'params': model.decoder.parameters(), 'lr': args.lr},
        {'params': model.vision_encoder.parameters(), 'lr': args.lr * 0.1},
        {'params': model.semantic_projector.parameters(), 'lr': args.lr * 0.1},
    ], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(OUTPUT_DIR) / 'stage_b'
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')
    best_aes = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        s_param = s_img = s_aes_score = 0.0
        n = 0

        for i, batch in enumerate(train_loader):
            images       = batch['image'].to(device)
            raw_image    = batch['raw_image'].to(device)
            expert_image = batch['expert_image'].to(device)
            params_gt    = batch['params'].to(device)
            weights      = batch['weights'].to(device)

            out = model(images)
            pred_norm = out['params_norm']
            p_loss, _ = param_crit(pred_norm, params_gt, weights)

            optimizer.zero_grad()
            p_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            s_param += p_loss.item()
            n += 1

            if (i + 1) % args.aesthetic_every == 0:
                with torch.no_grad():
                    pred_phys = denorm(pred_norm, device)
                    rendered = apply_diff_isp(raw_image.float(), pred_phys)
                    rendered = torch.nan_to_num(rendered, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
                    s_img += F.l1_loss(rendered, expert_image.float()).item()
                    if aesthetic_eval is not None:
                        _, aes_score = aesthetic_eval(rendered)
                        s_aes_score += aes_score.mean().item()

            if (i + 1) % args.log_every == 0 and n > 0:
                n_aes = max(1, (i + 1) // args.aesthetic_every)
                logger.info(f'  [{i+1}/{len(train_loader)}] param={s_param/n:.4f} img={s_img/n_aes:.4f} aes={s_aes_score/n_aes:.2f}')

        model.eval()
        v_param = v_img = v_aes = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                raw_image    = batch['raw_image'].to(device)
                expert_image = batch['expert_image'].to(device)
                images       = batch['image'].to(device)
                params_gt    = batch['params'].to(device)
                weights      = batch['weights'].to(device)
                out = model(images)
                vp_loss, _ = param_crit(out['params_norm'], params_gt, weights)
                v_param += vp_loss.item()
                pred_phys = denorm(out['params_norm'], device)
                rendered = apply_diff_isp(raw_image.float(), pred_phys)
                rendered = torch.nan_to_num(rendered, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
                v_img += F.l1_loss(rendered, expert_image.float()).item()
                if aesthetic_eval is not None:
                    _, aes_score = aesthetic_eval(rendered)
                    v_aes += aes_score.mean().item()
                v_n += 1

        val_param = v_param / max(v_n, 1)
        val_img = v_img / max(v_n, 1)
        val_aes = v_aes / max(v_n, 1)
        scheduler.step()
        elapsed = time.time() - t0
        logger.info(f'Ep {epoch}/{args.epochs} | train_param={s_param/max(n,1):.4f} | val_param={val_param:.4f} val_img={val_img:.4f} val_aes={val_aes:.2f} | {elapsed:.1f}s')

        history.append({'epoch': epoch, 'train_param': s_param/max(n,1), 'val_param': val_param, 'val_img': val_img, 'val_aesthetic_score': val_aes})

        if val_param < best_val:
            best_val = val_param
            if val_aes > best_aes:
                best_aes = val_aes
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_param, 'val_img': val_img, 'val_aesthetic': val_aes}, out_dir / 'best.pt')
            logger.info(f'  * best: param={val_param:.4f} img={val_img:.4f} aes={val_aes:.2f}')

    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict()}, out_dir / 'final.pt')
    json.dump(history, open(out_dir / 'history.json', 'w'), indent=2)
    logger.info(f'v9 done! best_param={best_val:.4f} best_aes={best_aes:.2f}')

if __name__ == '__main__':
    main()
