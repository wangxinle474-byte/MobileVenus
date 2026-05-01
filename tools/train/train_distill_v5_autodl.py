"""
AutoDL 端 Distill v5 训练脚本
核心改进:
  ① ColorJitter 已去除（消除 input/target 不一致）
  ② Stage B 使用真实 Expert C 图像作为 GT（有监督图像级训练）
     损失: L1(ISP(orig, pred_params), expert_c) + SSIM(...)
  ③ 复用 v4 Stage A 权重（FiveK 域对齐，无需重跑）

前置条件 (AutoDL):
  /root/autodl-tmp/fivek_jpeg/             — FiveK 原始图片
  /root/autodl-tmp/fivek_expert_c/         — Expert C retouched JPEG
                                             (用 download_fivek_expert_c.py 下载)
  /root/autodl-tmp/data/
    fivek_expert_params.json
    fivek_expert_consensus.json
  /root/autodl-tmp/checkpoints/distill_v4/stage_a/best.pt   — v4 Stage A 权重
  /root/autodl-tmp/MobileVenus/            — 训练代码

用法:
  python train_distill_v5_autodl.py                     # 复用 v4 Stage A + Expert C GT
  python train_distill_v5_autodl.py --img_weight 1.0    # 纯图像损失，无参数正则
  python train_distill_v5_autodl.py --param_weight 0.1  # 加参数辅助损失
  python train_distill_v5_autodl.py --retrain_stage_a   # 重新训练 Stage A
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── AutoDL 路径配置 ──────────────────────────────────────────────
ROOT           = Path('/root/autodl-tmp')
FIVEK_JPEG     = ROOT / 'fivek_jpeg'
DATA_DIR       = ROOT / 'data'
CODE_DIR       = ROOT / 'MobileVenus'
EMBED_NPZ      = ROOT / 'fivek_text_embeddings.npz'
FIVEK_JSON     = ROOT / 'fivek_stage_a.json'

FIVEK_PARAMS   = DATA_DIR / 'fivek_expert_params.json'
FIVEK_CONSENS  = DATA_DIR / 'fivek_expert_consensus.json'

V4_STAGE_A     = ROOT / 'checkpoints' / 'distill_v4' / 'stage_a' / 'best.pt'
OUTPUT_DIR     = ROOT / 'checkpoints' / 'distill_v5'


# ── 工具：参数反归一化（batch 版） ──────────────────────────────────

def _make_denorm_fn():
    sys.path.insert(0, str(CODE_DIR))
    from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
    import torch

    def denormalize_params_batch(params_norm, device):
        """params_norm: (B, 8) [-1,1] → dict param_name: (B,) physical"""
        result = {}
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_RANGES[name]
            result[name] = (lo + (params_norm[:, i] + 1.0) / 2.0 * (hi - lo)).to(device)
        return result

    return PARAM_NAMES, denormalize_params_batch


# ── Step 1: Stage A（复用 v4，或重新训练） ─────────────────────────

def step1_stage_a(retrain: bool):
    logger.info('=' * 60)
    if not retrain and V4_STAGE_A.exists():
        logger.info(f'复用 v4 Stage A 权重: {V4_STAGE_A}')
        sys.path.insert(0, str(CODE_DIR))
        from training.semantic_distill.config import DistillConfig
        cfg = DistillConfig(
            text_embed_file      = str(EMBED_NPZ),
            venus_image_root     = str(FIVEK_JPEG),
            fivek_data_file      = str(FIVEK_PARAMS),
            fivek_consensus_file = str(FIVEK_CONSENS),
            fivek_jpeg_dir       = str(FIVEK_JPEG),
            output_dir           = str(OUTPUT_DIR),
        )
        # 把 v4 stage_a 权重复制到 v5 输出目录
        import shutil
        sa_out = Path(OUTPUT_DIR) / 'stage_a'
        sa_out.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(V4_STAGE_A), str(sa_out / 'best.pt'))
        logger.info(f'  已复制到 {sa_out}/best.pt')
        return cfg

    # 重新训练 Stage A
    logger.info('Step 1: Stage A 训练 (FiveK 域视觉-语义对齐)')
    sys.path.insert(0, str(CODE_DIR))
    from training.semantic_distill.config import DistillConfig
    from training.semantic_distill.trainer import train_stage_a

    cfg = DistillConfig(
        text_embed_file      = str(EMBED_NPZ),
        venus_image_root     = str(FIVEK_JPEG),
        fivek_data_file      = str(FIVEK_PARAMS),
        fivek_consensus_file = str(FIVEK_CONSENS),
        fivek_jpeg_dir       = str(FIVEK_JPEG),
        output_dir           = str(OUTPUT_DIR),
        stage_a_epochs       = 25,
        stage_a_lr           = 5e-4,
        stage_a_batch_size   = 32,
        num_workers          = 4,
        fp16                 = True,
    )
    t0 = time.time()
    train_stage_a(cfg)
    logger.info(f'  Stage A 完成, 耗时 {(time.time()-t0)/60:.1f} min')
    return cfg


# ── Step 2: Stage B（Expert C 真实 GT 有监督图像训练） ──────────────

def step2_stage_b_expert_gt(cfg, img_weight: float, param_weight: float):
    """
    Stage B: 以 Expert C retouched 图像为真实 GT
    训练损失 = λ_img * [L1 + SSIM](ISP(orig, pred), expert_c)
             + λ_param * MSE(pred_params_norm, gt_params_norm)  [辅助正则]
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    logger.info('=' * 60)
    logger.info(f'Step 2: Stage B 训练 (Expert C GT, img_w={img_weight} param_w={param_weight})')

    sys.path.insert(0, str(CODE_DIR))
    from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
    from training.fivek_8param.dataset_expert import build_expert_datasets
    from training.fivek_8param.loss import DistillParamLoss
    from models.diff_isp import apply_diff_isp, ssim_loss

    PARAM_NAMES, denormalize_params_batch = _make_denorm_fn()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'  设备: {device}')

    stage_b_epochs     = 50
    stage_b_lr         = 1e-4
    stage_b_batch_size = 16
    unfreeze_at        = stage_b_epochs // 3

    # ── 数据：所有可用 Expert 目录 ──
    expert_dirs = {
        ex: str(ROOT / f'fivek_expert_{ex}')
        for ex in ['a', 'b', 'c', 'd', 'e']
        if (ROOT / f'fivek_expert_{ex}').exists()
    }
    logger.info(f'  可用 Expert 目录: {list(expert_dirs.keys())}')

    train_ds, val_ds = build_expert_datasets(
        str(FIVEK_PARAMS), str(FIVEK_CONSENS),
        orig_jpeg_dir=str(FIVEK_JPEG),
        expert_dirs=expert_dirs,
        image_size=cfg.image_size, val_ratio=cfg.val_ratio, seed=cfg.seed,
        val_expert='c',
    )
    train_loader = DataLoader(train_ds, batch_size=stage_b_batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=stage_b_batch_size,
                              shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    logger.info(f'  数据: train={len(train_ds)}, val={len(val_ds)}')

    # ── 模型 ──
    stage_a_ckpt  = str(Path(cfg.output_dir) / 'stage_a' / 'best.pt')
    stage_a_model = SemanticDistillModel(
        image_size=cfg.image_size, visual_dim=cfg.visual_dim,
        semantic_dim=cfg.semantic_dim, text_dim=cfg.text_dim,
    )
    state = torch.load(stage_a_ckpt, map_location='cpu', weights_only=False)
    stage_a_model.load_state_dict(state['model_state_dict'])
    model = DistillParamModel(stage_a_model, decoder_hidden=cfg.decoder_hidden).to(device)
    model.freeze_backbone()
    logger.info(f'  Stage A 权重: {stage_a_ckpt}')
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'  可训练参数: {n_train/1e6:.2f}M')

    param_criterion = DistillParamLoss(align_weight=0.0, use_consensus=True)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=stage_b_lr, weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stage_b_epochs, eta_min=cfg.min_lr)
    scaler = torch.amp.GradScaler('cuda') if cfg.fp16 and device.type == 'cuda' else None

    out_dir = Path(cfg.output_dir) / 'stage_b'
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_img = float('inf')
    history = []

    for epoch in range(1, stage_b_epochs + 1):
        t0 = time.time()

        # 解冻 backbone
        if epoch == unfreeze_at + 1:
            model.unfreeze_backbone()
            remaining = stage_b_epochs - epoch + 1
            optimizer = optim.AdamW([
                {'params': model.decoder.parameters(),           'lr': stage_b_lr * 0.5},
                {'params': model.vision_encoder.parameters(),    'lr': stage_b_lr * 0.05},
                {'params': model.semantic_projector.parameters(),'lr': stage_b_lr * 0.05},
            ], weight_decay=cfg.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=cfg.min_lr)
            logger.info(f'  ★ 解冻 backbone @ epoch {epoch}')

        # ── 训练 loop ──
        model.train()
        sum_total = sum_img = sum_param = 0.0
        n = 0

        for i, batch in enumerate(train_loader):
            images       = batch['image'].to(device)
            raw_image    = batch['raw_image'].to(device)     # [0, 1]
            expert_image = batch['expert_image'].to(device)  # [0, 1] ← 真实 GT
            params_gt    = batch['params'].to(device)
            weights      = batch['weights'].to(device)

            with torch.amp.autocast('cuda', enabled=scaler is not None):
                out = model(images)
                pred_norm = out['params_norm']  # (B, 8) 归一化

                # ① 图像级损失：ISP(orig, pred) vs Expert C
                pred_phys  = denormalize_params_batch(pred_norm, device)
                pred_img   = apply_diff_isp(raw_image.float(), pred_phys)
                img_l1     = F.l1_loss(pred_img, expert_image.float())
                img_ssim   = ssim_loss(pred_img, expert_image.float())
                i_loss     = img_l1 + img_ssim

                # ② 参数辅助正则（可选，防止 ISP 近似误差主导训练）
                p_loss, _  = param_criterion(pred_norm, params_gt, weights)

                loss = img_weight * i_loss + param_weight * p_loss

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            sum_total += loss.item()
            sum_img   += i_loss.item()
            sum_param += p_loss.item()
            n += 1

            if (i + 1) % cfg.log_every == 0:
                logger.info(
                    f'  [{i+1}/{len(train_loader)}] '
                    f'loss={sum_total/n:.4f}  '
                    f'img(L1+SSIM)={sum_img/n:.4f}  '
                    f'param={sum_param/n:.4f}'
                )

        # ── 验证：计算 val 集图像损失 ──
        model.eval()
        val_img_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                raw_image    = batch['raw_image'].to(device)
                expert_image = batch['expert_image'].to(device)
                images       = batch['image'].to(device)
                out = model(images)
                pred_phys = denormalize_params_batch(out['params_norm'], device)
                pred_img  = apply_diff_isp(raw_image.float(), pred_phys)
                val_img_sum += (F.l1_loss(pred_img, expert_image.float())
                                + ssim_loss(pred_img, expert_image.float())).item()
                val_n += 1
        val_img_loss = val_img_sum / max(val_n, 1)

        scheduler.step()
        elapsed = time.time() - t0
        logger.info(
            f'Epoch {epoch}/{stage_b_epochs} | '
            f'train={sum_total/n:.4f} (img={sum_img/n:.4f} param={sum_param/n:.4f}) | '
            f'val_img={val_img_loss:.4f} | {elapsed:.1f}s'
        )

        history.append({
            'epoch': epoch,
            'train_loss': sum_total / n,
            'train_img_loss': sum_img / n,
            'train_param_loss': sum_param / n,
            'val_img_loss': val_img_loss,
        })

        if val_img_loss < best_val_img:
            best_val_img = val_img_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_img_loss': val_img_loss,
                'img_weight': img_weight,
                'param_weight': param_weight,
            }, out_dir / 'best.pt')
            logger.info(f'  ★ 新最优 val_img_loss={best_val_img:.4f}')

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f'Stage B 完成! 最优 val_img_loss={best_val_img:.4f}')


# ── main ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_weight',   type=float, default=1.0,
                        help='图像损失权重 λ_img (default: 1.0)')
    parser.add_argument('--param_weight', type=float, default=0.05,
                        help='参数辅助正则权重 λ_param (default: 0.05)')
    parser.add_argument('--retrain_stage_a', action='store_true',
                        help='重新训练 Stage A (默认复用 v4 权重)')
    parser.add_argument('--skip_stage_b', action='store_true',
                        help='仅跑 Stage A')
    args = parser.parse_args()

    logger.info('=' * 60)
    logger.info('  Distill v5 — Expert C 真实 GT 有监督图像训练')
    logger.info(f'  img_weight={args.img_weight}  param_weight={args.param_weight}')
    logger.info('=' * 60)

    # 检查前置文件
    for p, desc in [
        (FIVEK_JPEG,   'fivek_jpeg 目录（原始图）'),
        (FIVEK_PARAMS, 'fivek_expert_params.json'),
        (FIVEK_CONSENS,'fivek_expert_consensus.json'),
        (CODE_DIR,     'MobileVenus 代码目录'),
    ]:
        if not p.exists():
            logger.error(f'❌ 缺少: {p}  ({desc})')
            sys.exit(1)
        n = sum(1 for _ in p.glob('*.jpg')) if p.is_dir() else 0
        logger.info(f'  ✅ {p}' + (f'  ({n} 张)' if p.is_dir() else ''))

    # 检查 Expert GT 目录（至少需要 Expert C）
    found_experts = [ex for ex in ['a', 'b', 'c', 'd', 'e']
                     if (ROOT / f'fivek_expert_{ex}').exists()]
    if not found_experts:
        logger.error('❌ 未找到任何 fivek_expert_{a-e}/ 目录')
        logger.error('   请先运行: python MobileVenus/tools/download_fivek_expert_c.py')
        sys.exit(1)
    if 'c' not in found_experts:
        logger.error('❌ 验证需要 Expert C，但 fivek_expert_c/ 不存在')
        logger.error('   请先运行: python MobileVenus/tools/download_fivek_expert_c.py --experts c')
        sys.exit(1)
    for ex in found_experts:
        d = ROOT / f'fivek_expert_{ex}'
        n = sum(1 for _ in d.glob('*.jpg'))
        logger.info(f'  ✅ Expert {ex.upper()}: {n} 张  ({d})')

    if not args.retrain_stage_a and not V4_STAGE_A.exists():
        logger.error(f'❌ 未找到 v4 Stage A 权重: {V4_STAGE_A}')
        logger.error('   请先运行 train_distill_v4_autodl.py，或传入 --retrain_stage_a')
        sys.exit(1)

    cfg = step1_stage_a(retrain=args.retrain_stage_a)

    if not args.skip_stage_b:
        t0 = time.time()
        step2_stage_b_expert_gt(cfg,
                                img_weight=args.img_weight,
                                param_weight=args.param_weight)
        logger.info(f'Stage B 耗时 {(time.time()-t0)/60:.1f} min')

    logger.info('=' * 60)
    logger.info('Distill v5 训练完成!')
    logger.info(f'权重: {OUTPUT_DIR}/stage_a/best.pt')
    logger.info(f'      {OUTPUT_DIR}/stage_b/best.pt')
    logger.info('下一步: 下载 checkpoints/distill_v5/ → 运行 eval_psnr_ssim.py --model distill_v5')


if __name__ == '__main__':
    main()
