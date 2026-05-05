"""
Distill v13 — V5 策略: V3 架构 + 多尺度 MUSIQ + EMA + Warm Restarts

核心改进 (vs v12 V3):
  1. 多尺度 MUSIQ Loss: 512 全图 + 384 随机裁剪, 梯度信号更丰富
  2. 纯 MUSIQ 优化: Phase2 去掉 L1/SSIM/CLIP, 全部梯度集中在 MUSIQ
  3. EMA (Exponential Moving Average): decay=0.999, 验证时用 EMA 模型
  4. CosineAnnealingWarmRestarts: T_0=50, 多次逃出局部最优
  5. 随机水平翻转增强
  6. musiq_weight=8.0 (v3 用 5.0)
  7. 200 epochs (v3 用 120)

V3 best: MUSIQ=4.21 @ Ep111
V5 目标: MUSIQ ≥ 4.30

AutoDL:
  python training/main/train_v13_multiscale.py
  python training/main/train_v13_multiscale.py --epochs 200 --musiq_weight 8.0
"""
import sys, json, time, argparse, logging, copy, random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path('/root/autodl-tmp')
sys.path.insert(0, str(ROOT / 'IntelligenceCamera'))


def _make_denorm_fn():
    from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
    import torch
    lo_list = [PARAM_RANGES[p][0] for p in PARAM_NAMES]
    hi_list = [PARAM_RANGES[p][1] for p in PARAM_NAMES]

    def denorm(params_norm, device):
        lo = torch.tensor(lo_list, device=device, dtype=params_norm.dtype)
        hi = torch.tensor(hi_list, device=device, dtype=params_norm.dtype)
        params_phys = lo.unsqueeze(0) + (params_norm + 1.0) / 2.0 * (hi - lo).unsqueeze(0)
        return {name: params_phys[:, i] for i, name in enumerate(PARAM_NAMES)}

    return PARAM_NAMES, PARAM_RANGES, denorm


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        """Apply EMA weights (for validation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original weights (after validation)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2,
                        help='512×512 + V3 9M, batch=2 on single GPU')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--l1_weight', type=float, default=1.0,
                        help='Phase1 L1 权重')
    parser.add_argument('--ssim_weight', type=float, default=0.5,
                        help='Phase1 SSIM 权重')
    parser.add_argument('--musiq_weight', type=float, default=8.0,
                        help='Phase2 MUSIQ 权重 (v3 用 5.0, 这里更激进)')
    parser.add_argument('--phase1_epochs', type=int, default=3,
                        help='Phase 1 warmup (仅 L1+SSIM), 比 v3 更短')
    parser.add_argument('--phase2_l1_weight', type=float, default=0.0,
                        help='Phase2 L1 权重 (纯 MUSIQ, 设为 0)')
    parser.add_argument('--clipiqa_weight', type=float, default=0.0,
                        help='Phase2 CLIPIQA 权重 (纯 MUSIQ, 设为 0)')
    parser.add_argument('--accum_steps', type=int, default=8,
                        help='Gradient accumulation (effective batch=16)')
    parser.add_argument('--base_ch', type=int, default=48,
                        help='V3 base channels')
    parser.add_argument('--param_version', type=str, default='v8')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的 checkpoint 路径')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay factor')
    parser.add_argument('--crop_size', type=int, default=384,
                        help='Multi-scale MUSIQ 的裁剪尺寸')
    parser.add_argument('--warmrestart_t0', type=int, default=50,
                        help='CosineAnnealingWarmRestarts T_0')
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
    from training.semantic_distill.config import DistillConfig
    from models.diff_isp import apply_diff_isp, ssim_loss
    from models.refinement_net_v3 import RefinementNetV3

    PARAM_NAMES, _, denorm = _make_denorm_fn()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    logger.info(f'=== V5 Strategy: V3 arch + MultiScale MUSIQ + EMA + WarmRestarts ===')
    logger.info(f'Config: base_ch={args.base_ch} musiq_w={args.musiq_weight} '
                f'phase2_l1={args.phase2_l1_weight} clipiqa={args.clipiqa_weight} '
                f'ema={args.ema_decay} crop={args.crop_size} '
                f'accum={args.accum_steps} eff_batch={args.batch_size * args.accum_steps} '
                f'lr={args.lr} epochs={args.epochs} T_0={args.warmrestart_t0}')

    # ── 数据 ──
    from training.fivek_8param.dataset_expert import build_expert_datasets

    class HiResWrapper:
        """Wraps base 224 dataset to provide 512 images + random flip."""
        def __init__(self, base_dataset, augment=False):
            self.base = base_dataset
            self.augment = augment
            from torchvision import transforms as T
            self.resize_512 = T.Resize((512, 512))
            self.to_tensor = T.ToTensor()

        def __len__(self): return len(self.base)

        def __getitem__(self, idx):
            item = self.base[idx]
            import random as _random
            from PIL import Image as _Image

            s = self.base.samples[idx]
            expert_paths = s.get('expert_paths', {})
            if self.base.is_train and expert_paths:
                chosen = _random.choice(list(expert_paths.keys()))
            else:
                chosen = self.base.val_expert
            expert_path = expert_paths.get(chosen) or ''

            try:
                pil_orig = _Image.open(s['orig_path']).convert('RGB')
                pil_expert = _Image.open(expert_path).convert('RGB')
                raw_512 = self.to_tensor(self.resize_512(pil_orig))
                expert_512 = self.to_tensor(self.resize_512(pil_expert))
            except:
                raw_512 = torch.zeros(3, 512, 512)
                expert_512 = torch.zeros(3, 512, 512)

            # Random horizontal flip (augmentation)
            if self.augment and _random.random() > 0.5:
                raw_512 = raw_512.flip(-1)
                expert_512 = expert_512.flip(-1)

            return {
                'image_224': item['image'],
                'raw_512': raw_512, 'expert_512': expert_512,
                'params': item['params'], 'weights': item['weights'],
            }

    expert_dirs = {}
    for ex in ['a', 'b', 'c', 'd', 'e']:
        p = ROOT / f'fivek_expert_{ex}'
        if p.exists():
            expert_dirs[ex] = str(p)

    train_ds_base, val_ds_base = build_expert_datasets(
        str(ROOT / 'data' / 'fivek_expert_params.json'),
        str(ROOT / 'data' / 'fivek_expert_consensus.json'),
        orig_jpeg_dir=str(ROOT / 'fivek_jpeg'), expert_dirs=expert_dirs,
        image_size=224, val_ratio=args.val_ratio, seed=args.seed, val_expert='c',
    )
    train_ds = HiResWrapper(train_ds_base, augment=True)
    val_ds = HiResWrapper(val_ds_base, augment=False)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    logger.info(f'Data: train={len(train_ds)}, val={len(val_ds)}')

    # ── 冻结参数模型 ──
    cfg = DistillConfig()
    stage_a = SemanticDistillModel(
        image_size=cfg.image_size, visual_dim=cfg.visual_dim,
        semantic_dim=cfg.semantic_dim, text_dim=cfg.text_dim,
    )
    sa_state = torch.load(str(ROOT / 'checkpoints' / 'distill_v6' / 'stage_a' / 'best.pt'),
                          map_location='cpu', weights_only=False)
    stage_a.load_state_dict(sa_state['model_state_dict'])
    param_model = DistillParamModel(stage_a, decoder_hidden=cfg.decoder_hidden)
    pm_ckpt = ROOT / 'checkpoints' / f'distill_{args.param_version}' / 'stage_b' / 'best.pt'
    if pm_ckpt.exists():
        pm_state = torch.load(str(pm_ckpt), map_location='cpu', weights_only=False)
        param_model.load_state_dict(pm_state['model_state_dict'])
        logger.info(f'Param model {args.param_version} loaded')
    else:
        logger.error(f'Not found: {pm_ckpt}'); return

    param_model.to(device).eval()
    for p in param_model.parameters():
        p.requires_grad = False

    # ── RefinementNet V3 (proven best architecture) ──
    refine = RefinementNetV3(base_ch=args.base_ch).to(device)
    total_p, train_p = refine.count_params()
    logger.info(f'RefinementNetV3: {total_p/1e6:.2f}M params ({train_p/1e6:.2f}M trainable)')

    # ── EMA ──
    ema = EMA(refine, decay=args.ema_decay)
    logger.info(f'EMA initialized with decay={args.ema_decay}')

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        refine.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        # Re-initialize EMA from resumed model
        ema = EMA(refine, decay=args.ema_decay)
        logger.info(f'Resumed from {args.resume} (epoch {start_epoch-1})')

    # ── MUSIQ (no CLIPIQA in V5 — pure MUSIQ) ──
    musiq_model = None
    clipiqa_model = None
    try:
        import pyiqa
    except Exception as e:
        logger.warning(f'pyiqa import failed: {e}')
    else:
        try:
            musiq_model = pyiqa.create_metric('musiq-ava', device=device)
            for p in musiq_model.parameters():
                p.requires_grad = False
            logger.info('MUSIQ-AVA loaded')
        except Exception as e:
            logger.warning(f'MUSIQ load failed: {e}')

        if args.clipiqa_weight > 0:
            try:
                clipiqa_model = pyiqa.create_metric('clipiqa+', device=device)
                for p in clipiqa_model.parameters():
                    p.requires_grad = False
                logger.info('CLIPIQA+ loaded')
            except Exception as e:
                logger.warning(f'CLIPIQA+ load failed: {e}')

    # ── 优化器 + Warm Restarts ──
    optimizer = optim.AdamW(refine.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.warmrestart_t0, T_mult=2, eta_min=1e-6)
    logger.info(f'Scheduler: CosineAnnealingWarmRestarts T_0={args.warmrestart_t0} T_mult=2')

    out_dir = ROOT / 'checkpoints' / 'refinement_v5'
    out_dir.mkdir(parents=True, exist_ok=True)

    best_musiq = 0.0
    best_l1 = float('inf')
    history = []

    def random_crop_384(x, crop_size=384):
        """Random crop for multi-scale MUSIQ evaluation."""
        _, _, h, w = x.shape
        if h <= crop_size or w <= crop_size:
            return F.interpolate(x, size=(crop_size, crop_size),
                                 mode='bilinear', align_corners=False)
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        return x[:, :, top:top + crop_size, left:left + crop_size]

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        use_aesthetic = epoch > args.phase1_epochs and musiq_model is not None

        # Phase 切换
        if use_aesthetic:
            l1_w = args.phase2_l1_weight    # 0.0
            ssim_w = 0.0
            musiq_w = args.musiq_weight      # 8.0
            clipiqa_w = args.clipiqa_weight   # 0.0
            phase = 'P2-MUSIQ'
        else:
            l1_w = args.l1_weight
            ssim_w = args.ssim_weight
            musiq_w = 0.0
            clipiqa_w = 0.0
            phase = 'P1-L1'

        # ── Train ──
        refine.train()
        s_l1 = s_ssim = s_musiq = s_clip = s_total = s_aes = 0.0
        n = n_aes = 0

        for i, batch in enumerate(train_loader):
            img_224 = batch['image_224'].to(device)
            raw_512 = batch['raw_512'].to(device)
            expert_512 = batch['expert_512'].to(device)

            # 冻结管线: param(224) → diff_isp(512)
            with torch.no_grad():
                out = param_model(img_224)
                pred_phys = denorm(out['params_norm'], device)
                rendered = apply_diff_isp(raw_512.float(), pred_phys).clamp(0, 1)
                rendered = torch.nan_to_num(rendered, nan=0.5)

            # RefinementNet V3 (有梯度)
            enhanced = refine(rendered)

            # L1 + SSIM vs Expert C (Phase 1)
            l1_loss = F.l1_loss(enhanced, expert_512.float())
            ssim_v = ssim_loss(enhanced, expert_512.float())

            # MUSIQ loss (Phase 2)
            musiq_loss = torch.tensor(0.0, device=device)
            clip_loss = torch.tensor(0.0, device=device)

            if use_aesthetic and musiq_model is not None:
                # Scale 1: 512px 全图
                score_512 = musiq_model(enhanced)

                # Scale 2: 384px 随机裁剪 (不同的梯度信号)
                crop_384 = random_crop_384(enhanced, args.crop_size)
                score_384 = musiq_model(crop_384)

                # 多尺度平均: 直接最大化
                musiq_loss = -(score_512.mean() + score_384.mean()) / 2.0

            if use_aesthetic and clipiqa_model is not None and clipiqa_w > 0:
                clip_score = clipiqa_model(enhanced)
                clip_loss = -clip_score.mean()

            total = (l1_w * l1_loss + ssim_w * ssim_v
                     + musiq_w * musiq_loss + clipiqa_w * clip_loss)

            accum = args.accum_steps
            loss_scaled = total / accum
            loss_scaled.backward()

            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                grad_ok = True
                for p in refine.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        grad_ok = False; break
                if grad_ok:
                    nn.utils.clip_grad_norm_(refine.parameters(), 5.0)
                    optimizer.step()
                    ema.update(refine)  # Update EMA after each optimizer step
                else:
                    logger.warning(f'NaN grad at batch {i}')
                optimizer.zero_grad()

            s_l1 += l1_loss.item()
            s_ssim += ssim_v.item()
            s_musiq += musiq_loss.item()
            s_clip += clip_loss.item()
            s_total += total.item()
            n += 1

            # MUSIQ 监控
            if (i + 1) % 3 == 0 and musiq_model is not None:
                with torch.no_grad():
                    aes = musiq_model(enhanced.detach()).mean().item()
                    s_aes += aes; n_aes += 1

            if (i + 1) % args.log_every == 0 and n > 0:
                aes_str = f' MUSIQ={s_aes/max(n_aes,1):.2f}' if n_aes > 0 else ''
                clip_str = f' clip_l={s_clip/n:.4f}' if clipiqa_w > 0 else ''
                lr_now = optimizer.param_groups[0]['lr']
                logger.info(
                    f'  [{i+1}/{len(train_loader)}] '
                    f'total={s_total/n:.4f} l1={s_l1/n:.4f} ssim={s_ssim/n:.4f}'
                    f' musiq_l={s_musiq/n:.4f}{clip_str}{aes_str}'
                    f' lr={lr_now:.2e}'
                )

        scheduler.step()

        # ── Val (use EMA model) ──
        ema.apply_shadow(refine)
        refine.eval()
        v_l1 = v_aes = v_clip = 0.0
        v_n = v_n_aes = 0

        with torch.no_grad():
            for batch in val_loader:
                img_224 = batch['image_224'].to(device)
                raw_512 = batch['raw_512'].to(device)
                expert_512 = batch['expert_512'].to(device)

                out = param_model(img_224)
                pred_phys = denorm(out['params_norm'], device)
                rendered = apply_diff_isp(raw_512.float(), pred_phys).clamp(0, 1)
                rendered = torch.nan_to_num(rendered, nan=0.5)

                enhanced = refine(rendered)
                v_l1 += F.l1_loss(enhanced, expert_512.float()).item()

                if musiq_model is not None:
                    v_aes += musiq_model(enhanced).mean().item()
                    v_n_aes += 1
                if clipiqa_model is not None:
                    v_clip += clipiqa_model(enhanced).mean().item()

                v_n += 1

        ema.restore(refine)  # Restore training weights

        val_l1 = v_l1 / max(v_n, 1)
        val_aes = v_aes / max(v_n_aes, 1)
        val_clip = v_clip / max(v_n, 1)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        logger.info(
            f'Ep {epoch}/{args.epochs} [{phase}] | '
            f'train={s_total/max(n,1):.4f} l1={s_l1/max(n,1):.4f} | '
            f'val_l1={val_l1:.4f} val_MUSIQ={val_aes:.2f} val_CLIP={val_clip:.3f} | '
            f'lr={lr_now:.2e} | {elapsed:.1f}s'
        )

        history.append({
            'epoch': epoch, 'phase': phase, 'lr': lr_now,
            'val_l1': val_l1, 'val_musiq': val_aes, 'val_clipiqa': val_clip,
        })

        # 保存逻辑
        save_it = False
        if use_aesthetic:
            if val_aes > best_musiq:
                best_musiq = val_aes; save_it = True
        else:
            if val_l1 < best_l1:
                best_l1 = val_l1; save_it = True

        if save_it:
            # Save EMA model as best
            ema.apply_shadow(refine)
            torch.save({
                'epoch': epoch,
                'model_state_dict': refine.state_dict(),
                'val_l1': val_l1, 'val_musiq': val_aes, 'val_clipiqa': val_clip,
                'config': {'base_ch': args.base_ch, 'param_version': args.param_version,
                           'strategy': 'v5_multiscale'},
            }, out_dir / 'best.pt')
            ema.restore(refine)
            logger.info(f'  ★ best: l1={val_l1:.4f} MUSIQ={val_aes:.2f} CLIP={val_clip:.3f}')

        if epoch % 10 == 0:
            ema.apply_shadow(refine)
            torch.save({'epoch': epoch, 'model_state_dict': refine.state_dict()},
                       out_dir / f'ep{epoch}.pt')
            ema.restore(refine)

    # Final save
    ema.apply_shadow(refine)
    torch.save({'epoch': args.epochs, 'model_state_dict': refine.state_dict()},
               out_dir / 'final.pt')
    ema.restore(refine)

    json.dump(history, open(out_dir / 'history.json', 'w'), indent=2)
    logger.info(f'v13 done! best_MUSIQ={best_musiq:.2f} best_l1={best_l1:.4f}')
    logger.info(f'Checkpoint dir: {out_dir}')


if __name__ == '__main__':
    main()
