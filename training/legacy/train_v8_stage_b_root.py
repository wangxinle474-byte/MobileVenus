"""
Distill v8 — Stage B: 多数据集多专家联合训练 (FiveK + PPR10K)

基于 v7 架构, 核心改进:
  1. FiveK (Expert A/C/D/E) + PPR10K (Expert A/B/C) 联合训练 → ~50K 训练对
  2. 保留 v7 的退化-修复对比学习
  3. PPR10K 3专家一致性权重自动计算
  4. 验证分别报告 FiveK 和 PPR10K 指标

AutoDL 用法:
  # 1. 先解析 PPR10K XMP 参数
  python tools/data/parse_ppr10k_xmp.py \
    --xmp_source_dir  /root/autodl-tmp/PPR10K/xmp_source \
    --xmp_target_dirs /root/autodl-tmp/PPR10K/xmp_target_a \
                      /root/autodl-tmp/PPR10K/xmp_target_b \
                      /root/autodl-tmp/PPR10K/xmp_target_c \
    --output /root/autodl-tmp/data/ppr10k_params.json

  # 2. 训练
  python training/main/train_v8_stage_b.py
  python training/main/train_v8_stage_b.py --no_ppr10k          # 仅 FiveK 多专家 (对照)
  python training/main/train_v8_stage_b.py --no_contrastive      # 无对比学习 (消融)

前置条件:
  /root/autodl-tmp/checkpoints/distill_v6/stage_a/best.pt
  /root/autodl-tmp/fivek_jpeg/
  /root/autodl-tmp/fivek_expert_c/          (必须)
  /root/autodl-tmp/fivek_expert_a/          (可选)
  /root/autodl-tmp/data/fivek_expert_params.json
  /root/autodl-tmp/data/fivek_expert_consensus.json
  /root/autodl-tmp/data/ppr10k_params.json  (parse_ppr10k_xmp.py 生成)
  /root/autodl-tmp/PPR10K/train_val_images_tif_360p/  (PPR10K 图片)
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
PPR10K_PARAMS = DATA_DIR / 'ppr10k_params.json'
PPR10K_IMAGES = ROOT / 'PPR10K' / 'train_val_images_tif_360p'
V6_STAGE_A    = ROOT / 'checkpoints' / 'distill_v6' / 'stage_a' / 'best.pt'
STAGE_A_CKPT  = V6_STAGE_A
OUTPUT_DIR    = ROOT / 'checkpoints' / 'distill_v8'

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


def batch_degrade(raw_image, device):
    """GPU 批量退化 (与 v7 相同)"""
    import torch
    B = raw_image.shape[0]
    degraded = raw_image.clone()
    delta = {}

    ev_shift = torch.empty(B, 1, 1, 1, device=device).uniform_(-1.5, 1.5)
    degraded = degraded * (2.0 ** ev_shift)
    delta['ev_compensation'] = -ev_shift.view(B)

    wb_shift = torch.empty(B, 1, 1, 1, device=device).uniform_(-1000, 1000)
    wb_factor = wb_shift / 6500.0
    degraded[:, 0:1] = degraded[:, 0:1] * (1.0 + wb_factor * 0.3)
    degraded[:, 2:3] = degraded[:, 2:3] * (1.0 - wb_factor * 0.3)
    delta['white_balance'] = -wb_shift.view(B)

    contrast_reduce = torch.empty(B, 1, 1, 1, device=device).uniform_(0, 50)
    factor = 1.0 - contrast_reduce / 100.0
    degraded = 0.5 + (degraded - 0.5) * factor
    delta['contrast'] = contrast_reduce.view(B)

    shadow_crush = torch.empty(B, 1, 1, 1, device=device).uniform_(0, 30)
    degraded = degraded - (shadow_crush / 100.0 * 0.15)
    delta['shadows'] = shadow_crush.view(B)

    hi_blow = torch.empty(B, 1, 1, 1, device=device).uniform_(0, 30)
    lum = degraded.mean(dim=1, keepdim=True)
    hi_mask = ((lum - 0.5) / 0.5).clamp(0, 1) ** 2
    degraded = degraded + hi_mask * (hi_blow / 100.0 * 0.2)
    delta['highlights'] = hi_blow.view(B)

    sat_reduce = torch.empty(B, 1, 1, 1, device=device).uniform_(0, 40)
    gray = degraded.mean(dim=1, keepdim=True)
    sat_factor = 1.0 - sat_reduce / 100.0
    degraded = gray + (degraded - gray) * sat_factor
    delta['saturation'] = sat_reduce.view(B)

    return degraded.clamp(0, 1), delta


def adjust_gt_params(params_gt_norm, delta, PARAM_NAMES, PARAM_RANGES, device):
    """根据退化补偿量调整归一化GT参数 (与 v7 相同)"""
    import torch
    adjusted = params_gt_norm.clone()
    for i, name in enumerate(PARAM_NAMES):
        if name in delta:
            lo, hi = PARAM_RANGES[name]
            delta_norm = delta[name].to(device) / ((hi - lo) / 2.0)
            adjusted[:, i] = (adjusted[:, i] + delta_norm).clamp(-1, 1)
    return adjusted


def main():
    parser = argparse.ArgumentParser(description='Distill v8: FiveK + PPR10K 联合训练')
    parser.add_argument('--param_weight',       type=float, default=1.0)
    parser.add_argument('--consistency_weight',  type=float, default=0.1)
    parser.add_argument('--no_contrastive',      action='store_true',
                        help='禁用对比学习')
    parser.add_argument('--no_ppr10k',           action='store_true',
                        help='禁用 PPR10K (仅 FiveK 多专家, 用于消融)')
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--image_size',   type=int,   default=224)
    parser.add_argument('--val_ratio',    type=float, default=0.1)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--log_every',    type=int,   default=50)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, ConcatDataset
    from torchvision import transforms as T

    from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
    from training.fivek_8param.dataset_expert import build_expert_datasets
    from training.fivek_8param.loss import DistillParamLoss
    from training.semantic_distill.config import DistillConfig
    from models.diff_isp import apply_diff_isp, ssim_loss

    PARAM_NAMES, PARAM_RANGES, denorm = _make_denorm_fn()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    use_contrastive = not args.no_contrastive
    use_ppr10k = not args.no_ppr10k
    logger.info(f'对比学习: {"✅" if use_contrastive else "❌"}  '
                f'PPR10K: {"✅" if use_ppr10k else "❌"}')

    img_normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # ══════════════════════════════════════════════
    # 数据集
    # ══════════════════════════════════════════════

    # FiveK 多专家
    expert_dirs = {}
    for ex in ['a', 'b', 'c', 'd', 'e']:
        p = ROOT / f'fivek_expert_{ex}'
        if p.exists():
            expert_dirs[ex] = str(p)
    logger.info(f'FiveK 可用 Expert: {list(expert_dirs.keys())}')

    fivek_train, fivek_val = build_expert_datasets(
        str(FIVEK_PARAMS), str(FIVEK_CONSENS),
        orig_jpeg_dir=str(FIVEK_JPEG),
        expert_dirs=expert_dirs,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        val_expert='c',
    )
    logger.info(f'FiveK: train={len(fivek_train)}, val={len(fivek_val)}')

    # PPR10K
    ppr10k_train, ppr10k_val = None, None
    if use_ppr10k:
        if not PPR10K_PARAMS.exists():
            logger.warning(f'PPR10K 参数文件不存在: {PPR10K_PARAMS}')
            logger.warning('请先运行: python tools/data/parse_ppr10k_xmp.py')
            logger.warning('将仅使用 FiveK 数据训练')
            use_ppr10k = False
        elif not PPR10K_IMAGES.exists():
            logger.warning(f'PPR10K 图片目录不存在: {PPR10K_IMAGES}')
            use_ppr10k = False

    if use_ppr10k:
        from training.fivek_8param.dataset_ppr10k import build_ppr10k_datasets
        ppr10k_train, ppr10k_val = build_ppr10k_datasets(
            str(PPR10K_PARAMS), str(PPR10K_IMAGES),
            image_size=args.image_size,
            seed=args.seed,
            val_expert='a',
        )
        logger.info(f'PPR10K: train={len(ppr10k_train)}, val={len(ppr10k_val)}')

    # 合并数据集
    if use_ppr10k and ppr10k_train is not None:
        train_ds = ConcatDataset([fivek_train, ppr10k_train])
        val_ds = fivek_val  # 验证仅用 FiveK Expert C (标准 benchmark)
        logger.info(f'合并 train: {len(train_ds)} '
                    f'(FiveK={len(fivek_train)} + PPR10K={len(ppr10k_train)})')
    else:
        train_ds = fivek_train
        val_ds = fivek_val
        logger.info(f'仅 FiveK: train={len(train_ds)}')

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # PPR10K 单独验证 loader (报告跨域泛化性)
    ppr10k_val_loader = None
    if use_ppr10k and ppr10k_val is not None:
        ppr10k_val_loader = DataLoader(ppr10k_val, args.batch_size, shuffle=False,
                                       num_workers=4, pin_memory=True)

    # ══════════════════════════════════════════════
    # 模型
    # ══════════════════════════════════════════════
    if not STAGE_A_CKPT.exists():
        raise FileNotFoundError(f'找不到 Stage A: {STAGE_A_CKPT}')
    cfg = DistillConfig()
    stage_a = SemanticDistillModel(
        image_size=cfg.image_size, visual_dim=cfg.visual_dim,
        semantic_dim=cfg.semantic_dim, text_dim=cfg.text_dim,
    )
    state = torch.load(str(STAGE_A_CKPT), map_location='cpu', weights_only=False)
    stage_a.load_state_dict(state['model_state_dict'])
    model = DistillParamModel(stage_a, decoder_hidden=cfg.decoder_hidden).to(device)
    model.freeze_backbone()
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Stage A: {STAGE_A_CKPT}  可训练参数: {n_train_params/1e6:.2f}M')

    param_crit = DistillParamLoss(align_weight=0.0, use_consensus=True)
    unfreeze_at = args.epochs // 3
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(OUTPUT_DIR) / 'stage_b'
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    history = []

    # ══════════════════════════════════════════════
    # 训练循环
    # ══════════════════════════════════════════════
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # 解冻 backbone
        if epoch == unfreeze_at + 1:
            model.unfreeze_backbone()
            remaining = args.epochs - epoch + 1
            optimizer = optim.AdamW([
                {'params': model.decoder.parameters(),            'lr': args.lr * 0.5},
                {'params': model.vision_encoder.parameters(),     'lr': args.lr * 0.05},
                {'params': model.semantic_projector.parameters(), 'lr': args.lr * 0.05},
            ], weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=1e-6)
            logger.info(f'★ 解冻 backbone @ epoch {epoch}')

        # ── Train ──
        model.train()
        s_total = s_param = s_param_deg = s_consist = 0.0
        n = 0

        for i, batch in enumerate(train_loader):
            images    = batch['image'].to(device)
            raw_image = batch['raw_image'].to(device)
            params_gt = batch['params'].to(device)
            weights   = batch['weights'].to(device)

            # 原图路径
            out_orig = model(images, return_embedding=use_contrastive)
            pred_norm_orig = out_orig['params_norm']
            p_loss_orig, _ = param_crit(pred_norm_orig, params_gt, weights)

            total_loss = p_loss_orig
            p_loss_deg_val = 0.0
            consist_val = 0.0

            # 退化路径 (对比学习)
            if use_contrastive:
                with torch.no_grad():
                    degraded, delta = batch_degrade(raw_image, device)
                    degraded_normed = img_normalize(degraded)
                    params_gt_deg = adjust_gt_params(
                        params_gt, delta, PARAM_NAMES, PARAM_RANGES, device)

                out_deg = model(degraded_normed, return_embedding=True)
                pred_norm_deg = out_deg['params_norm']
                p_loss_deg, _ = param_crit(pred_norm_deg, params_gt_deg, weights)

                sem_orig = out_orig['semantic_emb']
                sem_deg = out_deg['semantic_emb']
                consistency_loss = F.mse_loss(sem_deg, sem_orig.detach())

                total_loss = p_loss_orig + p_loss_deg + \
                             args.consistency_weight * consistency_loss
                p_loss_deg_val = p_loss_deg.item()
                consist_val = consistency_loss.item()

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            s_total     += total_loss.item()
            s_param     += p_loss_orig.item()
            s_param_deg += p_loss_deg_val
            s_consist   += consist_val
            n += 1

            if (i + 1) % args.log_every == 0:
                log_str = (f'  [{i+1}/{len(train_loader)}] '
                           f'loss={s_total/n:.4f} p={s_param/n:.4f}')
                if use_contrastive:
                    log_str += f' p_deg={s_param_deg/n:.4f} consist={s_consist/n:.4f}'
                logger.info(log_str)

        # ── Validation: FiveK Expert C ──
        model.eval()
        v_param_sum, v_img_sum, v_n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                raw_image    = batch['raw_image'].to(device)
                expert_image = batch['expert_image'].to(device)
                images       = batch['image'].to(device)
                params_gt    = batch['params'].to(device)
                weights      = batch['weights'].to(device)

                out = model(images)
                p_loss, _ = param_crit(out['params_norm'], params_gt, weights)
                v_param_sum += p_loss.item()

                pred_phys = denorm(out['params_norm'], device)
                pred_img  = apply_diff_isp(raw_image.float(), pred_phys)
                pred_img  = torch.nan_to_num(pred_img, nan=0.0, posinf=1.0, neginf=0.0)
                v_img_sum += (F.l1_loss(pred_img, expert_image.float()) +
                              ssim_loss(pred_img, expert_image.float())).item()
                v_n += 1
        val_loss = v_param_sum / max(v_n, 1)
        val_img  = v_img_sum / max(v_n, 1)

        # ── Validation: PPR10K (跨域泛化) ──
        ppr10k_val_loss = 0.0
        if ppr10k_val_loader is not None:
            pv_sum, pv_n = 0.0, 0
            with torch.no_grad():
                for batch in ppr10k_val_loader:
                    images    = batch['image'].to(device)
                    params_gt = batch['params'].to(device)
                    weights   = batch['weights'].to(device)
                    out = model(images)
                    p_loss, _ = param_crit(out['params_norm'], params_gt, weights)
                    pv_sum += p_loss.item()
                    pv_n += 1
            ppr10k_val_loss = pv_sum / max(pv_n, 1)

        scheduler.step()
        elapsed = time.time() - t0

        # 日志
        train_str = f'train={s_total/n:.4f} (p={s_param/n:.4f}'
        if use_contrastive:
            train_str += f' p_deg={s_param_deg/n:.4f} consist={s_consist/n:.4f}'
        train_str += ')'
        val_str = f'val_fivek={val_loss:.4f}(img={val_img:.4f})'
        if ppr10k_val_loader:
            val_str += f' val_ppr10k={ppr10k_val_loss:.4f}'
        logger.info(f'Ep {epoch}/{args.epochs} | {train_str} | {val_str} | {elapsed:.1f}s')

        history.append({
            'epoch': epoch,
            'train_loss': s_total/n,
            'train_param': s_param/n,
            'train_param_deg': s_param_deg/n,
            'train_consistency': s_consist/n,
            'val_fivek_param': val_loss,
            'val_fivek_img': val_img,
            'val_ppr10k_param': ppr10k_val_loss,
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': best_val, 'val_img': val_img,
                        'ppr10k_val': ppr10k_val_loss}, out_dir / 'best.pt')
            logger.info(f'  ★ best={best_val:.4f}')

    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict()},
               out_dir / 'final.pt')
    json.dump(history, open(out_dir / 'history.json', 'w'), indent=2)

    logger.info(f'\n{"="*60}')
    logger.info(f'v8 Stage B 完成!')
    logger.info(f'  FiveK best val: {best_val:.4f}')
    if ppr10k_val_loader:
        logger.info(f'  PPR10K val:     {ppr10k_val_loss:.4f}')
    logger.info(f'  数据规模: FiveK={len(fivek_train)}'
                + (f' + PPR10K={len(ppr10k_train)}' if use_ppr10k and ppr10k_train else ''))
    logger.info(f'  对比学习: {"✅" if use_contrastive else "❌"}')
    logger.info(f'  Checkpoint: {out_dir / "best.pt"}')
    logger.info(f'{"="*60}')


if __name__ == '__main__':
    main()
