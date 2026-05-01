"""Distill v10 — 端到端美学训练: image_loss通过diff_isp反向传播"""
import sys, json, time, argparse, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path('/root/autodl-tmp')
CODE_DIR = ROOT / 'IntelligenceCamera'
sys.path.insert(0, str(CODE_DIR))

def _make_denorm_fn():
    from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
    import torch
    lo_list = [PARAM_RANGES[p][0] for p in PARAM_NAMES]
    hi_list = [PARAM_RANGES[p][1] for p in PARAM_NAMES]
    def denorm_differentiable(params_norm, device):
        lo = torch.tensor(lo_list, device=device, dtype=params_norm.dtype)
        hi = torch.tensor(hi_list, device=device, dtype=params_norm.dtype)
        params_phys = lo.unsqueeze(0) + (params_norm + 1.0) / 2.0 * (hi - lo).unsqueeze(0)
        result = {}
        for i, name in enumerate(PARAM_NAMES):
            result[name] = params_phys[:, i]
        return result
    return PARAM_NAMES, PARAM_RANGES, denorm_differentiable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--param_weight', type=float, default=1.0)
    parser.add_argument('--img_weight', type=float, default=5.0)
    parser.add_argument('--ssim_weight', type=float, default=1.0)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--aesthetic_every', type=int, default=10)
    args = parser.parse_args()

    import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
    from training.fivek_8param.dataset_expert import build_expert_datasets
    from training.fivek_8param.loss import DistillParamLoss
    from training.semantic_distill.config import DistillConfig
    from models.diff_isp import apply_diff_isp, ssim_loss

    PARAM_NAMES, PARAM_RANGES, denorm = _make_denorm_fn()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    logger.info(f'Config: param_w={args.param_weight} img_w={args.img_weight} ssim_w={args.ssim_weight} grad_clip={args.grad_clip} warmup={args.warmup_epochs}')

    expert_dirs = {}
    for ex in ['a','b','c','d','e']:
        p = ROOT / f'fivek_expert_{ex}'
        if p.exists(): expert_dirs[ex] = str(p)

    train_ds, val_ds = build_expert_datasets(
        str(ROOT/'data'/'fivek_expert_params.json'), str(ROOT/'data'/'fivek_expert_consensus.json'),
        orig_jpeg_dir=str(ROOT/'fivek_jpeg'), expert_dirs=expert_dirs,
        image_size=args.image_size, val_ratio=args.val_ratio, seed=args.seed, val_expert='c',
    )
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f'Data: train={len(train_ds)}, val={len(val_ds)}')

    cfg = DistillConfig()
    stage_a = SemanticDistillModel(image_size=cfg.image_size, visual_dim=cfg.visual_dim, semantic_dim=cfg.semantic_dim, text_dim=cfg.text_dim)
    sa_state = torch.load(str(ROOT/'checkpoints'/'distill_v6'/'stage_a'/'best.pt'), map_location='cpu', weights_only=False)
    stage_a.load_state_dict(sa_state['model_state_dict'])
    model = DistillParamModel(stage_a, decoder_hidden=cfg.decoder_hidden)

    v8_ckpt = ROOT/'checkpoints'/'distill_v8'/'stage_b'/'best.pt'
    if v8_ckpt.exists():
        v8_state = torch.load(str(v8_ckpt), map_location='cpu', weights_only=False)
        model.load_state_dict(v8_state['model_state_dict'])
        logger.info(f'v8 loaded (val_loss={v8_state.get("val_loss","?")})')
    else:
        logger.error(f'v8 not found: {v8_ckpt}'); return

    model.to(device)
    logger.info(f'Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    aesthetic_eval = None
    try:
        from training.aesthetic_loss import AestheticLoss
        aesthetic_eval = AestheticLoss(metric_name='musiq-ava', target_score=5.0, input_size=224, device=str(device))
        if not aesthetic_eval.available: aesthetic_eval = None
    except: pass

    param_crit = DistillParamLoss(align_weight=0.0, use_consensus=True)
    optimizer = optim.AdamW([
        {'params': model.decoder.parameters(), 'lr': args.lr},
        {'params': model.vision_encoder.parameters(), 'lr': args.lr*0.1},
        {'params': model.semantic_projector.parameters(), 'lr': args.lr*0.1},
    ], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(ROOT/'checkpoints'/'distill_v10'/'stage_b')
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf'); best_aes = 0.0; nan_count = 0; history = []

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        use_img_loss = epoch > args.warmup_epochs
        model.train()
        s_param=s_img=s_ssim=s_total=s_aes=0.0; n=n_aes=0

        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            raw_image = batch['raw_image'].to(device)
            expert_image = batch['expert_image'].to(device)
            params_gt = batch['params'].to(device)
            weights = batch['weights'].to(device)

            out = model(images)
            pred_norm = out['params_norm']
            p_loss, _ = param_crit(pred_norm, params_gt, weights)

            img_loss = torch.tensor(0.0, device=device)
            ssim_v = torch.tensor(0.0, device=device)
            if use_img_loss:
                pred_phys = denorm(pred_norm, device)
                rendered = apply_diff_isp(raw_image.float(), pred_phys).clamp(0, 1)
                if torch.isnan(rendered).any():
                    nan_count += 1
                    rendered = torch.nan_to_num(rendered, nan=0.5)
                img_loss = F.l1_loss(rendered, expert_image.float())
                ssim_v = ssim_loss(rendered, expert_image.float())

            total = args.param_weight*p_loss + args.img_weight*img_loss + args.ssim_weight*ssim_v

            optimizer.zero_grad()
            total.backward()

            grad_ok = True
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    grad_ok = False; break
            if grad_ok:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            else:
                nan_count += 1; optimizer.zero_grad()

            s_param += p_loss.item(); s_img += img_loss.item(); s_ssim += ssim_v.item()
            s_total += total.item(); n += 1

            if (i+1) % args.aesthetic_every == 0 and aesthetic_eval is not None:
                with torch.no_grad():
                    if not use_img_loss:
                        pred_phys = denorm(pred_norm.detach(), device)
                        rendered = apply_diff_isp(raw_image.float(), pred_phys).clamp(0,1)
                        rendered = torch.nan_to_num(rendered, nan=0.5)
                    _, aes_s = aesthetic_eval(rendered.detach())
                    s_aes += aes_s.mean().item(); n_aes += 1

            if (i+1) % args.log_every == 0 and n > 0:
                aes_str = f' aes={s_aes/max(n_aes,1):.2f}' if n_aes > 0 else ''
                img_str = f' img={s_img/n:.4f} ssim={s_ssim/n:.4f}' if use_img_loss else ' [warmup]'
                logger.info(f'  [{i+1}/{len(train_loader)}] total={s_total/n:.4f} param={s_param/n:.4f}{img_str}{aes_str}')

        model.eval()
        v_param=v_img=v_aes=0.0; v_n=v_n_aes=0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                raw_image = batch['raw_image'].to(device)
                expert_image = batch['expert_image'].to(device)
                params_gt = batch['params'].to(device)
                weights = batch['weights'].to(device)
                out = model(images)
                vp_loss, _ = param_crit(out['params_norm'], params_gt, weights)
                v_param += vp_loss.item()
                pred_phys = denorm(out['params_norm'], device)
                rendered = apply_diff_isp(raw_image.float(), pred_phys)
                rendered = torch.nan_to_num(rendered, nan=0.5).clamp(0,1)
                v_img += F.l1_loss(rendered, expert_image.float()).item()
                if aesthetic_eval is not None:
                    _, aes_s = aesthetic_eval(rendered)
                    v_aes += aes_s.mean().item(); v_n_aes += 1
                v_n += 1

        val_param = v_param/max(v_n,1); val_img = v_img/max(v_n,1); val_aes = v_aes/max(v_n_aes,1)
        scheduler.step()
        elapsed = time.time() - t0
        phase = 'WARMUP' if not use_img_loss else 'E2E'
        logger.info(f'Ep {epoch}/{args.epochs} [{phase}] | train={s_total/max(n,1):.4f} param={s_param/max(n,1):.4f} img={s_img/max(n,1):.4f} | val_param={val_param:.4f} val_img={val_img:.4f} val_aes={val_aes:.2f} | nan={nan_count} | {elapsed:.1f}s')
        history.append({'epoch':epoch,'phase':phase,'train_total':s_total/max(n,1),'val_param':val_param,'val_img':val_img,'val_aes':val_aes,'nan_count':nan_count})

        if val_img < best_val:
            best_val = val_img
            if val_aes > best_aes: best_aes = val_aes
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'val_loss':val_param,'val_img':val_img,'val_aes':val_aes}, out_dir/'best.pt')
            logger.info(f'  ★ best: img={val_img:.4f} param={val_param:.4f} aes={val_aes:.2f}')
        if epoch % 10 == 0:
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict()}, out_dir/f'ep{epoch}.pt')

    torch.save({'epoch':args.epochs,'model_state_dict':model.state_dict()}, out_dir/'final.pt')
    json.dump(history, open(out_dir/'history.json','w'), indent=2)
    logger.info(f'v10 done! best_img={best_val:.4f} best_aes={best_aes:.2f} nan={nan_count}')

if __name__ == '__main__':
    main()
