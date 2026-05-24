"""FireRed v2: 加 pixel reconstruction loss 修复 mis-direction 问题.

v1 诊断结论 (`diagnose_under_prediction.py`):
  - L1(orig, model_render) ≈ L1(orig, inv_render) ≈ 0.05 (改动幅度对了)
  - 但 L1(model, FR) > L1(orig, FR): 模型改动方向不对, 主要 wb action

修复:
  - 主 loss: weighted_mse_loss(P_pred_norm, P_inv_norm)  (跟 v1 一样)
  - 辅 loss: L1( apply_diff_isp(orig, P_pred), apply_diff_isp(orig, P_inv) )
            → 直接强制渲染像素对齐, 让模型对"视觉效果"负责
  - α (pixel loss 权重) ramp-up: 0 → 0.5 (前 5 epoch 只跟 param loss, 之后逐步加)
  - 渲染分辨率 128 (训练时), 减小 GPU 开销

复用 train.py 中的 dataset / model / build_data 等.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.diff_isp import apply_diff_isp  # noqa: E402
from training.firered_baseline.train import (  # noqa: E402
    PARAM_NAMES, ACTIONS, ACTION_TO_IDX, FireRed7DModel,
    PARAM_NORM, normalize_param, denormalize_t,
    weighted_mse_loss, per_param_metrics, per_action_param_metrics,
    build_data, mean_baseline, ACTION_PRIMARY_PARAM, TIER_WEIGHTS,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


class FireRedV2Dataset(Dataset):
    """v2 加返回 orig_render (低分辨率) 用于 pixel loss."""

    def __init__(self, samples: list, image_size: int = 256,
                 render_size: int = 128, is_train: bool = True):
        self.samples = samples
        self.image_size = image_size
        self.render_size = render_size
        self.is_train = is_train
        self.resize_input = T.Resize((image_size, image_size))
        self.resize_render = T.Resize((render_size, render_size))
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.color_jitter = T.ColorJitter(
            brightness=0.05, contrast=0.05, saturation=0.05, hue=0.0)
        self.hflip_p = 0.5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s['orig_path']).convert('RGB')
        except Exception as e:
            logger.warning(f'load fail {s["orig_path"]}: {e}')
            img = Image.new('RGB', (self.image_size, self.image_size))

        # train: 一次决定是否 hflip + color jitter, 两路用同样变换
        do_flip = self.is_train and random.random() < self.hflip_p
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.is_train:
            img = self.color_jitter(img)

        # backbone 输入 (256, normalized)
        img_in = self.resize_input(img)
        x_input = self.norm(self.to_tensor(img_in))

        # 渲染输入 (128, [0,1] 不 normalize, 喂 diff_isp)
        img_render = self.resize_render(img)
        x_render = self.to_tensor(img_render)  # (3, render_size, render_size)

        action_idx = ACTION_TO_IDX[s['action']]
        action_oh = torch.zeros(len(ACTIONS), dtype=torch.float32)
        action_oh[action_idx] = 1.0

        P = s['P_inferred']
        params_norm = torch.tensor(
            [normalize_param(p, P.get(p) or 0.0) for p in PARAM_NAMES],
            dtype=torch.float32)
        # 物理空间 P (用于 pixel loss target render)
        params_phys = torch.tensor(
            [(P.get(p) or 0.0) for p in PARAM_NAMES], dtype=torch.float32)

        tier_w = TIER_WEIGHTS.get(s['quality_tier'], 0.0)

        return {
            'image': x_input,
            'image_render': x_render,
            'action_onehot': action_oh,
            'action_idx': action_idx,
            'params_norm': params_norm,
            'params_phys': params_phys,
            'tier_weight': torch.tensor(tier_w, dtype=torch.float32),
            'source_image': s['source_image'],
            'action': s['action'],
        }


def render_from_pred_norm(orig: torch.Tensor, pred_norm: torch.Tensor):
    """pred_norm: (B, 7) ∈ [-1,1] → 反归一化到物理 → diff_isp 渲染."""
    B = orig.shape[0]
    params = {}
    for i, p in enumerate(PARAM_NAMES):
        params[p] = denormalize_t(p, pred_norm[:, i])
    return apply_diff_isp(orig, params)


def render_from_phys(orig: torch.Tensor, params_phys: torch.Tensor):
    """params_phys: (B, 7) 物理值 → diff_isp 渲染."""
    params = {p: params_phys[:, i] for i, p in enumerate(PARAM_NAMES)}
    return apply_diff_isp(orig, params)


def train_one_epoch(model, loader, opt, param_w, device, epoch, args,
                    log_every=30):
    model.train()
    tot_param, tot_pix, tot, n = 0.0, 0.0, 0.0, 0
    n_nan_skip = 0
    # ramp pixel loss weight (从 0 起)
    if epoch <= args.pixel_warmup_epochs:
        pix_w = args.pixel_loss_weight * (epoch - 1) / max(1, args.pixel_warmup_epochs)
    else:
        pix_w = args.pixel_loss_weight

    for i, batch in enumerate(loader):
        img = batch['image'].to(device, non_blocking=True)
        img_r = batch['image_render'].to(device, non_blocking=True)
        a_oh = batch['action_onehot'].to(device, non_blocking=True)
        tgt_norm = batch['params_norm'].to(device, non_blocking=True)
        tgt_phys = batch['params_phys'].to(device, non_blocking=True)
        sw = batch['tier_weight'].to(device, non_blocking=True)

        pred_norm = model(img, a_oh)
        L_param = weighted_mse_loss(pred_norm, tgt_norm, param_w, sw)

        # pixel loss (只在 pix_w > 0 时算, 节省时间; 还可避免 epoch 1 NaN)
        if pix_w > 1e-6:
            pred_render = render_from_pred_norm(img_r, pred_norm)
            with torch.no_grad():
                tgt_render = render_from_phys(img_r, tgt_phys)
            # NaN clean (diff_isp 在 boundary 偶发 NaN)
            pred_render = torch.nan_to_num(pred_render, nan=0.5, posinf=1.0,
                                           neginf=0.0)
            tgt_render = torch.nan_to_num(tgt_render, nan=0.5, posinf=1.0,
                                           neginf=0.0)
            pix_per_sample = (pred_render - tgt_render).abs().mean(
                dim=[1, 2, 3])    # (B,)
            L_pix = (pix_per_sample * sw).sum() / (sw.sum() + 1e-6)
        else:
            L_pix = torch.tensor(0.0, device=device)

        loss = L_param + pix_w * L_pix

        # NaN guard
        if not torch.isfinite(loss):
            n_nan_skip += 1
            opt.zero_grad()
            continue

        opt.zero_grad()
        loss.backward()
        # 清掉 NaN/Inf 梯度防止权重污染
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0,
                                          neginf=0.0)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tot_param += L_param.item()
        tot_pix += L_pix.item()
        tot += loss.item()
        n += 1
        if (i + 1) % log_every == 0:
            logger.info(f'  [{i+1}/{len(loader)}] '
                        f'L_param={tot_param/n:.4f}  L_pix={tot_pix/n:.4f}  '
                        f'pix_w={pix_w:.3f}  tot={tot/n:.4f}'
                        f'{f"  nan_skip={n_nan_skip}" if n_nan_skip else ""}')
    if n_nan_skip > 0:
        logger.warning(f'  epoch {epoch} skipped {n_nan_skip} NaN batches')
    return tot / max(n, 1), tot_param / max(n, 1), tot_pix / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, param_w, device):
    model.eval()
    tot_param, tot_pix, n = 0.0, 0.0, 0
    all_pred, all_tgt, all_act = [], [], []
    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        img_r = batch['image_render'].to(device, non_blocking=True)
        a_oh = batch['action_onehot'].to(device, non_blocking=True)
        tgt_norm = batch['params_norm'].to(device, non_blocking=True)
        tgt_phys = batch['params_phys'].to(device, non_blocking=True)
        sw = batch['tier_weight'].to(device, non_blocking=True)

        pred_norm = model(img, a_oh)
        L_param = weighted_mse_loss(pred_norm, tgt_norm, param_w, sw)
        pred_render = render_from_pred_norm(img_r, pred_norm)
        tgt_render = render_from_phys(img_r, tgt_phys)
        pred_render = torch.nan_to_num(pred_render, nan=0.5, posinf=1.0,
                                       neginf=0.0)
        tgt_render = torch.nan_to_num(tgt_render, nan=0.5, posinf=1.0,
                                      neginf=0.0)
        pix_per_sample = (pred_render - tgt_render).abs().mean(dim=[1, 2, 3])
        L_pix = (pix_per_sample * sw).sum() / (sw.sum() + 1e-6)

        tot_param += L_param.item()
        tot_pix += L_pix.item()
        n += 1
        all_pred.append(pred_norm.cpu())
        all_tgt.append(tgt_norm.cpu())
        all_act.append(batch['action_idx'])
    tot_param /= max(n, 1)
    tot_pix /= max(n, 1)
    pred_all = torch.cat(all_pred, dim=0)
    tgt_all = torch.cat(all_tgt, dim=0)
    act_all = torch.cat(all_act, dim=0)
    mae, r = per_param_metrics(pred_all, tgt_all)
    per_act = per_action_param_metrics(pred_all, tgt_all, act_all)
    return tot_param, tot_pix, mae, r, per_act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='checkpoints/firered_v2')
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--render_size', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_every', type=int, default=30)
    ap.add_argument('--clarity_weight', type=float, default=0.1)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--pixel_loss_weight', type=float, default=0.5,
                    help='pixel L1 loss 终值 (ramp 后)')
    ap.add_argument('--pixel_warmup_epochs', type=int, default=5,
                    help='pixel loss 从 0 → pixel_loss_weight 的 warmup epoch 数')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device={device}, lr={args.lr}, batch={args.batch_size}, '
                f'epochs={args.epochs}, dropout={args.dropout}, '
                f'pixel_w={args.pixel_loss_weight}, '
                f'pixel_warmup={args.pixel_warmup_epochs}')

    train_s, val_s = build_data(
        Path(args.jsonl), args.val_ratio,
        ('A excellent', 'B good', 'C acceptable'), args.seed)

    from collections import Counter
    logger.info(f'train action dist: '
                f'{dict(Counter([s["action"] for s in train_s]))}')
    logger.info(f'val   action dist: '
                f'{dict(Counter([s["action"] for s in val_s]))}')

    train_ds = FireRedV2Dataset(train_s, args.image_size, args.render_size,
                                is_train=True)
    val_ds = FireRedV2Dataset(val_s, args.image_size, args.render_size,
                              is_train=False)
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

    opt = optim.AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    param_w = torch.ones(7, device=device)
    param_w[6] = args.clarity_weight
    logger.info(f'param weights: {param_w.cpu().tolist()}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float('inf')
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_param, tr_pix = train_one_epoch(
            model, train_loader, opt, param_w, device, epoch, args,
            log_every=args.log_every)
        val_param, val_pix, val_mae, val_r, val_per_act = evaluate(
            model, val_loader, param_w, device)
        # combined val loss for early stop (param + 0.5 pixel, fixed weight at val)
        val_loss = val_param + 0.5 * val_pix
        sched.step()
        elapsed = time.time() - t0

        logger.info(f'Ep {epoch:3d}/{args.epochs}  '
                    f'tr_par={tr_param:.4f}  tr_pix={tr_pix:.4f}  '
                    f'val_par={val_param:.4f}  val_pix={val_pix:.4f}  '
                    f'val={val_loss:.4f}  elapsed={elapsed:.1f}s  '
                    f'lr={sched.get_last_lr()[0]:.2e}')
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
            'tr_param': tr_param, 'tr_pix': tr_pix,
            'val_param': val_param, 'val_pix': val_pix,
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
                'val_param': val_param, 'val_pix': val_pix,
                'val_mae': val_mae.tolist(),
                'val_pearson_r': val_r.tolist(),
                'val_per_action': val_per_act,
                'baseline_per_action': base,
                'args': vars(args),
            }, out_dir / 'best.pt')
            logger.info(f'  * best val={val_loss:.4f}  '
                        f'(par={val_param:.4f}, pix={val_pix:.4f})  saved')
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
