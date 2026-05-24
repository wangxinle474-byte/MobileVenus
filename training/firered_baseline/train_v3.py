"""FireRed v3: action-aware param weighting.

v2 (pixel loss) 失败诊断:
  - 7D ISP 非单射, pixel loss 找到 "渲染相似但参数偏离" 的局部解
  - |dP|_phys 反而 +40% 恶化, R 腰斩

v3 修复思路:
  - 不引入 pixel loss
  - 改 param-weighted MSE: 对每个样本, 根据 action 调整 7D 权重向量
  - 主参数 (action primary param) 权重 5.0 — 强信号
  - secondary params 权重 1.0 — 弱信号 (跟主参数协同)
  - 不相关 params 权重 0.2 — 抑制噪声学习
  - 保留 tier-aware sample weighting

预期: wb action 上模型有动力 "敢" 输出极端 wb (4000 或 9000), 而非保守 6000.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from training.firered_baseline.train import (  # noqa: E402
    PARAM_NAMES, ACTIONS, ACTION_TO_IDX, ACTION_PRIMARY_PARAM,
    FireRedPseudoDataset, FireRed7DModel,
    per_param_metrics, per_action_param_metrics,
    build_data, mean_baseline, TIER_WEIGHTS,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================
# Action-aware param weighting
# ============================================================
# 每个 action 对应的 secondary params (主参数以外的相关参数)
ACTION_SECONDARY = {
    'contrast':   ['brightness', 'shadows', 'highlights', 'clarity'],
    'saturation': ['clarity'],
    'shadows':    ['brightness', 'highlights'],
    'highlights': ['brightness', 'shadows'],
    'wb':         ['saturation'],  # wb 变化常伴随饱和度补偿
}


def build_action_param_weights(primary_w: float = 5.0,
                                secondary_w: float = 1.0,
                                other_w: float = 0.2,
                                clarity_w: float = 0.1) -> torch.Tensor:
    """构建 (5, 7) 矩阵, [action_idx, param_idx] = 权重."""
    W = torch.full((len(ACTIONS), len(PARAM_NAMES)), other_w,
                   dtype=torch.float32)
    for ai, a in enumerate(ACTIONS):
        primary = ACTION_PRIMARY_PARAM[a]
        pi = PARAM_NAMES.index(primary)
        W[ai, pi] = primary_w
        for sp in ACTION_SECONDARY.get(a, []):
            spi = PARAM_NAMES.index(sp)
            W[ai, spi] = secondary_w
        # clarity 权重全局压低 (因为 GT 大多 ≈ 0)
        ci = PARAM_NAMES.index('clarity')
        if W[ai, ci] > clarity_w:
            W[ai, ci] = clarity_w
    return W  # (5, 7)


def action_weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                              action_idxs: torch.Tensor,
                              sample_w: torch.Tensor,
                              param_w_matrix: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (B, 7) 归一化空间
    action_idxs:  (B,) ∈ [0, 5)
    sample_w:     (B,) tier 权重
    param_w_matrix: (5, 7) 每个 action 的 param 权重
    """
    pw = param_w_matrix[action_idxs]                # (B, 7)
    err2 = (pred - target) ** 2 * pw                # (B, 7)
    # per-sample loss (mean over params, weighted)
    err_per_sample = err2.sum(dim=1) / (pw.sum(dim=1) + 1e-6)  # (B,)
    err_per_sample = err_per_sample * sample_w
    if sample_w.sum() < 1e-6:
        return err_per_sample.mean()
    return err_per_sample.sum() / (sample_w.sum() + 1e-6)


# ============================================================
# Trainer
# ============================================================
def train_one_epoch(model, loader, opt, param_w_matrix, device, log_every=30):
    model.train()
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        img = batch['image'].to(device, non_blocking=True)
        a_oh = batch['action_onehot'].to(device, non_blocking=True)
        a_idx = batch['action_idx'].to(device, non_blocking=True)
        tgt = batch['params_norm'].to(device, non_blocking=True)
        sw = batch['tier_weight'].to(device, non_blocking=True)
        pred = model(img, a_oh)
        loss = action_weighted_mse_loss(pred, tgt, a_idx, sw, param_w_matrix)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item()
        n += 1
        if (i + 1) % log_every == 0:
            logger.info(f'  [{i+1}/{len(loader)}] loss={tot/n:.4f}')
    return tot / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, param_w_matrix, device):
    model.eval()
    losses, n = 0.0, 0
    all_pred, all_tgt, all_act = [], [], []
    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        a_oh = batch['action_onehot'].to(device, non_blocking=True)
        a_idx = batch['action_idx'].to(device, non_blocking=True)
        tgt = batch['params_norm'].to(device, non_blocking=True)
        sw = batch['tier_weight'].to(device, non_blocking=True)
        pred = model(img, a_oh)
        loss = action_weighted_mse_loss(pred, tgt, a_idx, sw, param_w_matrix)
        losses += loss.item()
        n += 1
        all_pred.append(pred.cpu())
        all_tgt.append(tgt.cpu())
        all_act.append(batch['action_idx'])
    losses /= max(n, 1)
    pred_all = torch.cat(all_pred, dim=0)
    tgt_all = torch.cat(all_tgt, dim=0)
    act_all = torch.cat(all_act, dim=0)
    mae, r = per_param_metrics(pred_all, tgt_all)
    per_act = per_action_param_metrics(pred_all, tgt_all, act_all)
    return losses, mae, r, per_act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='checkpoints/firered_v3')
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_every', type=int, default=30)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--primary_w', type=float, default=5.0,
                    help='action primary param 权重')
    ap.add_argument('--secondary_w', type=float, default=1.0)
    ap.add_argument('--other_w', type=float, default=0.2)
    ap.add_argument('--clarity_w', type=float, default=0.1)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device={device}, lr={args.lr}, batch={args.batch_size}, '
                f'epochs={args.epochs}, dropout={args.dropout}, '
                f'primary_w={args.primary_w}, secondary_w={args.secondary_w}, '
                f'other_w={args.other_w}')

    train_s, val_s = build_data(
        Path(args.jsonl), args.val_ratio,
        ('A excellent', 'B good', 'C acceptable'), args.seed)
    logger.info(f'train action dist: '
                f'{dict(Counter([s["action"] for s in train_s]))}')
    logger.info(f'val   action dist: '
                f'{dict(Counter([s["action"] for s in val_s]))}')

    train_ds = FireRedPseudoDataset(train_s, args.image_size, is_train=True)
    val_ds = FireRedPseudoDataset(val_s, args.image_size, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    base = mean_baseline(train_s, val_s)
    logger.info('\n=== "predict per-action mean" baseline (val MAE) ===')
    for a in ACTIONS:
        if a in base:
            b = base[a]
            logger.info(f'  {a:<11s} primary={b["primary"]:<14s} '
                        f'mean_pred={b["mean_pred"]:+8.2f}  '
                        f'mae={b["mae"]:8.2f}  n={b["n"]}')

    model = FireRed7DModel(image_size=args.image_size,
                           dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'\nmodel: {n_params/1e6:.2f}M params')

    # Action-aware param weight matrix
    param_w_matrix = build_action_param_weights(
        args.primary_w, args.secondary_w, args.other_w, args.clarity_w
    ).to(device)
    logger.info('\n=== action-aware param weight matrix (5 × 7) ===')
    logger.info(f'{"action":<12s}' + ''.join(f'{p[:4]:>7s}' for p in PARAM_NAMES))
    for ai, a in enumerate(ACTIONS):
        row = '  '.join(f'{param_w_matrix[ai, pi].item():5.2f}'
                        for pi in range(7))
        logger.info(f'{a:<12s}  {row}')

    opt = optim.AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float('inf')
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, param_w_matrix,
                                  device, log_every=args.log_every)
        val_loss, val_mae, val_r, val_per_act = evaluate(
            model, val_loader, param_w_matrix, device)
        sched.step()
        elapsed = time.time() - t0

        logger.info(f'Ep {epoch:3d}/{args.epochs}  '
                    f'train={tr_loss:.4f}  val={val_loss:.4f}  '
                    f'elapsed={elapsed:.1f}s  lr={sched.get_last_lr()[0]:.2e}')
        per_param_str = '  '.join(
            f'{p[:4]}=MAE{val_mae[i]:.1f},R{val_r[i]:+.2f}'
            for i, p in enumerate(PARAM_NAMES))
        logger.info(f'  per-param: {per_param_str}')
        per_act_str = '  '.join(
            f'{a[:4]}({v["primary"][:4]})=MAE{v["mae"]:.1f},R{v["r"]:+.2f}'
            for a, v in val_per_act.items())
        logger.info(f'  per-action(primary): {per_act_str}')

        history.append({
            'epoch': epoch,
            'train_loss': tr_loss,
            'val_loss': val_loss,
            'val_mae': val_mae.tolist(),
            'val_pearson_r': val_r.tolist(),
            'val_per_action': val_per_act,
            'lr': sched.get_last_lr()[0],
            'elapsed': elapsed,
        })
        json.dump(history, open(out_dir / 'history.json', 'w'),
                  indent=2, default=float)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae.tolist(),
                'val_pearson_r': val_r.tolist(),
                'val_per_action': val_per_act,
                'baseline_per_action': base,
                'param_w_matrix': param_w_matrix.cpu().tolist(),
                'args': vars(args),
            }, out_dir / 'best.pt')
            logger.info(f'  * best val={val_loss:.4f}  saved')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f'  early stop: no improve for {no_improve} epochs')
                break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_mae': val_mae.tolist(),
        'val_pearson_r': val_r.tolist(),
        'val_per_action': val_per_act,
        'args': vars(args),
    }, out_dir / 'last.pt')

    logger.info(f'\n{"="*78}\n[DONE] best_val_loss={best_val:.4f} @ Ep{best_epoch}')
    logger.info(f'  checkpoint: {out_dir/"best.pt"}')
    logger.info(f'  history:    {out_dir/"history.json"}\n{"="*78}')


if __name__ == '__main__':
    main()
