"""
Distill v9 — 美学感知微调 (Aesthetic-Aware Fine-tuning)

基于 v7 checkpoint，新增:
  1. diff_isp 渲染参与梯度回传 (不再 no_grad)
  2. MUSIQ 美学评分作为辅助损失 (冻结 MUSIQ，梯度链: MUSIQ → rendered → params → model)
  3. 图像重建损失 (L1 + SSIM vs Expert C)

损失 = param_weight * param_loss
     + img_weight * image_loss
     + aesthetic_weight * aesthetic_loss

AutoDL 用法:
  python train_v9_aesthetic.py
  python train_v9_aesthetic.py --aesthetic_weight 0.2 --img_weight 1.0

前置条件:
  /root/autodl-tmp/checkpoints/distill_v7/stage_b/best.pt
  /root/autodl-tmp/checkpoints/distill_v6/stage_a/best.pt
  pip install pyiqa
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
    parser.add_argument('--param_weight',      type=float, default=1.0)
    parser.add_argument('--img_weight',        type=float, default=0.5,
                        help='图像重建损失权重 (L1+SSIM vs Expert)')
    parser.add_argument('--aesthetic_weight',  type=float, default=0.1,
                        help='美学损失权重 (MUSIQ reward)')
    parser.add_argument('--target_score',      type=float, default=5.0,
                        help='目标美学分 (hinge loss 阈值)')
    parser.add_argument('--epochs',       type=int,   default=20,
                        help='微调轮数 (基于v7, 不需要太多)')
    parser.add_argument('--batch_size',   type=int,   default=8,
                        help='MUSIQ 占显存, batch 要小')
    parser.add_argument('--lr',           type=float, default=3e-5,
                        help='微调 lr 要小')
    parser.add_argument('--image_size',   type=int,   default=224)
    parser.add_argument('--val_ratio',    type=float, default=0.1)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--log_every',    type=int,   default=50)
    parser.add_argument('--aesthetic_every', type=int, default=2,
                        help='每 N 个 batch 计算一次美学损失 (节省显存)')
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
    logger.info(f'权重: param={args.param_weight}, img={args.img_weight}, '
                f'aesthetic={args.aesthetic_weight}')

    # ── 数据 ──
    expert_dirs = {}
    for ex in ['a', 'b', 'c', 'd', 'e']:
        p = ROOT / f'fivek_expert_{ex}'
        if p.exists():
            expert_dirs[ex] = str(p)
    logger.info(f'可用 Expert: {list(expert_dirs.keys())}')

    train_ds, val_ds = build_expert_datasets(
        str(FIVEK_PARAMS), str(FIVEK_CONSENS),
        orig_jpeg_dir=str(FIVEK_JPEG),
        expert_dirs=expert_dirs,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        val_expert='c',
    )
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    logger.info(f'数据: train={len(train_ds)}, val={len(val_ds)}')

    # ── 模型: 从 v7 加载 ──
    cfg = DistillConfig()
    stage_a = SemanticDistillModel(
        image_size=cfg.image_size, visual_dim=cfg.visual_dim,
        semantic_dim=cfg.semantic_dim, text_dim=cfg.text_dim,
    )
    sa_state = torch.load(str(STAGE_A_CKPT), map_location='cpu', weights_only=False)
    stage_a.load_state_dict(sa_state['model_state_dict'])

    model = DistillParamModel(stage_a, decoder_hidden=cfg.decoder_hidden)

    # 加载 v7 权重
    if V7_CKPT.exists():
        v7_state = torch.load(str(V7_CKPT), map_location='cpu', weights_only=False)
        model.load_state_dict(v7_state['model_state_dict'])
        v7_val = v7_state.get('val_loss', '?')
        logger.info(f'v7 权重加载完成 (val_loss={v7_val})')
    else:
        logger.error(f'v7 checkpoint 不存在: {V7_CKPT}')
        return

    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f'模型参数量: {n_params:.2f}M')

    # ── 美学损失 ──
    aesthetic_loss_fn = None
    if args.aesthetic_weight > 0:
        aesthetic_loss_fn = AestheticLoss(
            metric_name='musiq-ava',
            target_score=args.target_score,
            input_size=224,
            device=str(device),
        )
        if not aesthetic_loss_fn.available:
            logger.warning('MUSIQ 加载失败, 仅用 param + img loss')
            aesthetic_loss_fn = None

    # ── 优化器 (全参数微调, 小学习率) ──
    param_crit = DistillParamLoss(align_weight=0.0, use_consensus=True)

    optimizer = optim.AdamW([
        {'params': model.decoder.parameters(),            'lr': args.lr},
        {'params': model.vision_encoder.parameters(),     'lr': args.lr * 0.1},
        {'params': model.semantic_projector.parameters(), 'lr': args.lr * 0.1},
    ], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(OUTPUT_DIR) / 'stage_b'
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    best_aes = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        s_total = s_param = s_img = s_aes = s_aes_score = 0.0
        n = 0

        for i, batch in enumerate(train_loader):
            images       = batch['image'].to(device)
            raw_image    = batch['raw_image'].to(device)
            expert_image = batch['expert_image'].to(device)
            params_gt    = batch['params'].to(device)
            weights      = batch['weights'].to(device)

            out = model(images)
            pred_norm = out['params_norm']

            # 1. Parameter loss (始终有梯度)
            p_loss, _ = param_crit(pred_norm, params_gt, weights)

            # 2. 图像重建 (diff_isp 参与梯度!)
            pred_phys = denorm(pred_norm, device)
            rendered = apply_diff_isp(raw_image.float(), pred_phys)
            rendered = torch.nan_to_num(rendered, nan=0.0, posinf=1.0, neginf=0.0)
            rendered = rendered.clamp(0, 1)

            img_loss = F.l1_loss(rendered, expert_image.float()) + \
                       ssim_loss(rendered, expert_image.float())

            total = args.param_weight * p_loss + args.img_weight * img_loss

            # 3. 美学损失 (每 N 个 batch 计算一次, 节省显存)
            aes_val = 0.0
            aes_score_val = 0.0
            if aesthetic_loss_fn is not None and (i % args.aesthetic_every == 0):
                aes_loss, aes_score = aesthetic_loss_fn(rendered)
                total = total + args.aesthetic_weight * aes_loss
                aes_val = aes_loss.item()
                aes_score_val = aes_score.mean().item()

            optimizer.zero_grad()
            total.backward()

            # 梯度裁剪 (diff_isp 可能产生大梯度)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                optimizer.zero_grad()
                logger.warning(f'  NaN/Inf grad at batch {i}, skip')
                continue

            optimizer.step()

            s_total     += total.item()
            s_param     += p_loss.item()
            s_img       += img_loss.item()
            s_aes       += aes_val
            s_aes_score += aes_score_val
            n += 1

            if (i + 1) % args.log_every == 0:
                logger.info(
                    f'  [{i+1}/{len(train_loader)}] '
                    f'loss={s_total/n:.4f} p={s_param/n:.4f} '
                    f'img={s_img/n:.4f} aes={s_aes/n:.3f} '
                    f'score={s_aes_score/n:.2f}'
                )

        # ── Val ──
        model.eval()
        v_img_sum = 0.0
        v_aes_sum = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in val_loader:
                raw_image    = batch['raw_image'].to(device)
                expert_image = batch['expert_image'].to(device)
                images       = batch['image'].to(device)
                out = model(images)
                pred_phys = denorm(out['params_norm'], device)
                rendered = apply_diff_isp(raw_image.float(), pred_phys)
                rendered = torch.nan_to_num(rendered, nan=0.0, posinf=1.0, neginf=0.0)
                rendered = rendered.clamp(0, 1)

                v_img_sum += (F.l1_loss(rendered, expert_image.float()) +
                              ssim_loss(rendered, expert_image.float())).item()

                if aesthetic_loss_fn is not None:
                    _, aes_score = aesthetic_loss_fn(rendered)
                    v_aes_sum += aes_score.mean().item()

                v_n += 1

        val_img = v_img_sum / max(v_n, 1)
        val_aes = v_aes_sum / max(v_n, 1)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f'Ep {epoch}/{args.epochs} | '
            f'train={s_total/n:.4f} (p={s_param/n:.4f} img={s_img/n:.4f} '
            f'aes={s_aes/n:.3f} score={s_aes_score/n:.2f}) | '
            f'val_img={val_img:.4f} val_aes={val_aes:.2f} | {elapsed:.1f}s'
        )

        history.append({
            'epoch': epoch,
            'train_loss': s_total / n,
            'train_param': s_param / n,
            'train_img': s_img / n,
            'train_aesthetic': s_aes / n,
            'train_aes_score': s_aes_score / n,
            'val_img': val_img,
            'val_aesthetic_score': val_aes,
        })

        # 保存: 综合考虑 val_img 和美学分
        combined = val_img - 0.1 * val_aes  # 越小越好 (img 低 + 美学高)
        if combined < best_val or val_aes > best_aes + 0.05:
            if combined < best_val:
                best_val = combined
            if val_aes > best_aes:
                best_aes = val_aes
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_img': val_img,
                'val_aesthetic': val_aes,
            }, out_dir / 'best.pt')
            logger.info(f'  ★ best: val_img={val_img:.4f} aes={val_aes:.2f}')

    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict()},
               out_dir / 'final.pt')
    json.dump(history, open(out_dir / 'history.json', 'w'), indent=2)
    logger.info(f'v9 Aesthetic 完成! best_val_img={best_val:.4f} best_aes={best_aes:.2f}')
    logger.info(f'Checkpoint: {out_dir / "best.pt"}')


if __name__ == '__main__':
    main()
