"""FireRed v5: 视觉直冲 — pixel loss vs FireRed_edit (NOT inv_render).

用户目标: "改变原图的参数以达到 FireRed 的效果". 只看视觉, 不管参数.

核心改变 vs v1-v4:
  - 主 loss: L1(diff_isp(orig, P_pred), FR_edit)  ← 直接对 FR 做 pixel match
  - 辅 loss: param MSE × 0.1 (弱正则, 防止 P drift 到 numeric 极值)
  - init from v1 best.pt: 已有合理 7D 起点, 避 cold-start 落入 NaN/坏解
  - 短训 (30 epoch), 直接 push 模型到 7D 上限

vs v2 (失败) 的关键差别:
  - v2 用 inv_render 做 pixel target → 模型只学复刻 inv (= 23.94dB), 不会超过
  - v5 用 FR_edit 做 pixel target → 模型尝试逼近 ceiling, 即使非单射也无所谓
    (用户只看视觉效果, 多组 P 产生同一图像都接受)

预期: PSNR(model, FR) 从 v1 的 21.14 → 接近 23 dB (逼近 7D 上限 23.94)
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


class FireRedV5Dataset(Dataset):
    """v5: 加载 FR_edit 作为 pixel loss 的直接目标."""

    def __init__(self, samples, image_size=256, render_size=128, is_train=True):
        self.samples = samples
        self.image_size = image_size
        self.render_size = render_size
        self.is_train = is_train
        self.resize_input = T.Resize((image_size, image_size))
        self.resize_render = T.Resize((render_size, render_size))
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
            logger.warning(f'load orig fail {s["orig_path"]}: {e}')
            img = Image.new('RGB', (self.image_size, self.image_size))
        try:
            fr_img = Image.open(s['target_path']).convert('RGB')
        except Exception as e:
            logger.warning(f'load FR fail {s["target_path"]}: {e}')
            fr_img = img.copy()

        # train: 同步翻转 orig 和 FR (避免 misalignment)
        do_flip = self.is_train and random.random() < self.hflip_p
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            fr_img = fr_img.transpose(Image.FLIP_LEFT_RIGHT)
        # color jitter 只对 orig (作为输入), FR 不变 (是固定目标)
        if self.is_train:
            img = self.color_jitter(img)

        # backbone 输入
        img_in = self.resize_input(img)
        x_input = self.norm(self.to_tensor(img_in))

        # 渲染输入 (orig at render_size, raw [0,1])
        img_render = self.resize_render(img)
        x_render = self.to_tensor(img_render)

        # FR target at render_size
        fr_render = self.resize_render(fr_img)
        x_fr_render = self.to_tensor(fr_render)

        action_idx = ACTION_TO_IDX[s['action']]
        action_oh = torch.zeros(len(ACTIONS), dtype=torch.float32)
        action_oh[action_idx] = 1.0

        P = s['P_inferred']
        params_norm = torch.tensor(
            [normalize_param(p, P.get(p) or 0.0) for p in PARAM_NAMES],
            dtype=torch.float32)

        tier_w = TIER_WEIGHTS.get(s['quality_tier'], 0.0)

        return {
            'image': x_input,
            'image_render': x_render,
            'fr_render': x_fr_render,    # *** v5 关键: 直接 FR target ***
            'action_onehot': action_oh,
            'action_idx': action_idx,
            'params_norm': params_norm,
            'tier_weight': torch.tensor(tier_w, dtype=torch.float32),
            'source_image': s['source_image'],
            'action': s['action'],
        }


def render_from_pred_norm(orig, pred_norm):
    params = {p: denormalize_t(p, pred_norm[:, i])
              for i, p in enumerate(PARAM_NAMES)}
    return apply_diff_isp(orig, params)


def train_one_epoch(model, loader, opt, param_w, device, epoch, args,
                    log_every=30):
    model.train()
    tot_param, tot_pix, tot, n = 0.0, 0.0, 0.0, 0
    n_nan_skip = 0
    pix_w = args.pixel_loss_weight
    par_w = args.param_loss_weight

    for i, batch in enumerate(loader):
        img = batch['image'].to(device, non_blocking=True)
        img_r = batch['image_render'].to(device, non_blocking=True)
        fr_r = batch['fr_render'].to(device, non_blocking=True)
        a_oh = batch['action_onehot'].to(device, non_blocking=True)
        tgt_norm = batch['params_norm'].to(device, non_blocking=True)
        sw = batch['tier_weight'].to(device, non_blocking=True)

        pred_norm = model(img, a_oh)
        L_param = weighted_mse_loss(pred_norm, tgt_norm, param_w, sw)

        pred_render = render_from_pred_norm(img_r, pred_norm)
        pred_render = torch.nan_to_num(pred_render, nan=0.5, posinf=1.0,
                                       neginf=0.0)
        # *** 关键: pixel target 是 FR_edit, 不是 inv_render ***
        pix_per_sample = (pred_render - fr_r).abs().mean(dim=[1, 2, 3])
        L_pix = (pix_per_sample * sw).sum() / (sw.sum() + 1e-6)

        loss = par_w * L_param + pix_w * L_pix

        if not torch.isfinite(loss):
            n_nan_skip += 1
            opt.zero_grad()
            continue

        opt.zero_grad()
        loss.backward()
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
                        f'tot={tot/n:.4f}'
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
        fr_r = batch['fr_render'].to(device, non_blocking=True)
        a_oh = batch['action_onehot'].to(device, non_blocking=True)
        tgt_norm = batch['params_norm'].to(device, non_blocking=True)
        sw = batch['tier_weight'].to(device, non_blocking=True)

        pred_norm = model(img, a_oh)
        L_param = weighted_mse_loss(pred_norm, tgt_norm, param_w, sw)
        pred_render = render_from_pred_norm(img_r, pred_norm)
        pred_render = torch.nan_to_num(pred_render, nan=0.5, posinf=1.0,
                                       neginf=0.0)
        pix_per_sample = (pred_render - fr_r).abs().mean(dim=[1, 2, 3])
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
    ap.add_argument('--out_dir', default='checkpoints/firered_v5')
    ap.add_argument('--init_ckpt', default='checkpoints/firered_v1/best.pt',
                    help='init from v1 best (避 cold-start)')
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--render_size', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=5e-5,
                    help='lower lr (init 已经合理, 慢慢调)')
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_every', type=int, default=30)
    ap.add_argument('--clarity_weight', type=float, default=0.1)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--param_loss_weight', type=float, default=0.1,
                    help='param MSE 弱正则')
    ap.add_argument('--pixel_loss_weight', type=float, default=1.0,
                    help='pixel L1 (vs FR_edit) 主 loss')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device={device}, lr={args.lr}, batch={args.batch_size}, '
                f'epochs={args.epochs}, '
                f'param_w={args.param_loss_weight}, '
                f'pixel_w={args.pixel_loss_weight}, '
                f'init={args.init_ckpt}')

    train_s, val_s = build_data(
        Path(args.jsonl), args.val_ratio,
        ('A excellent', 'B good', 'C acceptable'), args.seed)
    logger.info(f'train action dist: '
                f'{dict(Counter([s["action"] for s in train_s]))}')
    logger.info(f'val   action dist: '
                f'{dict(Counter([s["action"] for s in val_s]))}')

    train_ds = FireRedV5Dataset(train_s, args.image_size, args.render_size,
                                is_train=True)
    val_ds = FireRedV5Dataset(val_s, args.image_size, args.render_size,
                              is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    base = mean_baseline(train_s, val_s)

    model = FireRed7DModel(image_size=args.image_size,
                           dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'model: {n_params/1e6:.2f}M params')

    # *** init from v1 best ***
    if args.init_ckpt and Path(args.init_ckpt).exists():
        ck = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck['model_state_dict'])
        logger.info(f'loaded init from {args.init_ckpt} '
                    f'(was Ep{ck.get("epoch", -1)} val={ck.get("val_loss", 0):.4f})')

    opt = optim.AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    param_w = torch.ones(7, device=device)
    param_w[6] = args.clarity_weight

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float('inf')
    best_epoch = -1
    no_improve = 0

    # 评估初始状态 (v1 init) 作为 baseline
    val_param0, val_pix0, val_mae0, val_r0, val_per_act0 = evaluate(
        model, val_loader, param_w, device)
    logger.info(f'\n[init from v1] val_param={val_param0:.4f}  '
                f'val_pix(vs FR)={val_pix0:.4f}')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_param, tr_pix = train_one_epoch(
            model, train_loader, opt, param_w, device, epoch, args,
            log_every=args.log_every)
        val_param, val_pix, val_mae, val_r, val_per_act = evaluate(
            model, val_loader, param_w, device)
        # *** v5 用 val_pix (vs FR) 作 early stop 指标, 不再用 param ***
        val_loss = val_pix  # 主指标 = pixel L1 vs FR
        sched.step()
        elapsed = time.time() - t0

        logger.info(f'Ep {epoch:3d}/{args.epochs}  '
                    f'tr_par={tr_param:.4f}  tr_pix={tr_pix:.4f}  '
                    f'val_par={val_param:.4f}  val_pix={val_pix:.4f}  '
                    f'elapsed={elapsed:.1f}s  '
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
            'epoch': epoch, 'tr_param': tr_param, 'tr_pix': tr_pix,
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
                'val_loss': val_loss, 'val_param': val_param,
                'val_pix': val_pix, 'val_mae': val_mae.tolist(),
                'val_pearson_r': val_r.tolist(),
                'val_per_action': val_per_act,
                'baseline_per_action': base, 'args': vars(args),
            }, out_dir / 'best.pt')
            logger.info(f'  * best val_pix={val_pix:.4f}  saved')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f'  early stop: no improve for {no_improve} epochs')
                break

    torch.save({
        'epoch': epoch, 'model_state_dict': model.state_dict(),
        'val_loss': val_loss, 'val_mae': val_mae.tolist(),
        'val_pearson_r': val_r.tolist(),
        'val_per_action': val_per_act, 'args': vars(args),
    }, out_dir / 'last.pt')

    logger.info(f'\n{"="*78}\n[DONE] best val_pix={best_val:.4f} @ Ep{best_epoch}')
    logger.info(f'  init val_pix was: {val_pix0:.4f}')
    logger.info(f'  improvement: {val_pix0 - best_val:+.4f} '
                f'({(val_pix0 - best_val) / val_pix0 * 100:+.1f}%)')
    logger.info(f'  checkpoint: {out_dir/"best.pt"}\n{"="*78}')


if __name__ == '__main__':
    main()
