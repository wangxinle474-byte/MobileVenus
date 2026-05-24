"""Path Z: train pure image-domain ResUNet on FireRed pseudo-labels.

Drops the 7D ISP parameterization. No Bezier, no LUT, no NamedCurves.
Direct image-to-image regression: orig + action -> refined.

Compares against:
  - Path A (v11a, 7D ISP + NamedCurves):     24.46 dB ± 0.20
  - Path X (v11a backbone + residual head):  25.28 dB
  - Path Z (this, pure image-domain ResUNet): TBD

Usage:
    python -m training.firered_baseline.train_resunet_imgdomain \\
        --jsonl outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl \\
        --out_dir checkpoints/lut_pathZ_resunet \\
        --batch_size 4 --epochs 60 --lr 3e-4 \\
        --base_ch 48 --image_size 256
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.image_domain_resunet import ImageDomainResUNet  # noqa: E402
from models.diff_isp import ssim_loss  # noqa: E402
from training.firered_baseline.train_lut import (  # noqa: E402
    ACTIONS, LUTDataset, build_data, compute_psnr,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('pathZ')


def train_one_epoch(model, loader, opt, device, args):
    model.train()
    tot_l1, tot_ssim, tot_psnr, n = 0., 0., 0., 0
    for i, batch in enumerate(loader):
        orig = batch['orig'].to(device)
        target = batch['target'].to(device)
        a_oh = batch['action_onehot'].to(device)
        sw = batch['tier_weight'].to(device)

        refined = model(orig, a_oh)

        l1_per = (refined - target).abs().mean(dim=[1, 2, 3])
        L_l1 = (l1_per * sw).sum() / (sw.sum() + 1e-6)
        L_ssim = ssim_loss(refined, target)
        loss = args.l1_weight * L_l1 + args.ssim_weight * L_ssim

        if not torch.isfinite(loss):
            opt.zero_grad()
            continue

        opt.zero_grad()
        loss.backward()
        # Sanitize NaN grads
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad = torch.nan_to_num(p.grad, nan=0.0,
                                          posinf=0.0, neginf=0.0)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            psnr = compute_psnr(refined, target)
        tot_l1 += L_l1.item()
        tot_ssim += L_ssim.item()
        tot_psnr += psnr
        n += 1

        if (i + 1) % args.log_every == 0:
            logger.info(
                f'  [{i+1}/{len(loader)}] '
                f'L1={tot_l1/n:.4f} SSIM={tot_ssim/n:.4f} '
                f'PSNR={tot_psnr/n:.2f}dB')

    return {
        'l1': tot_l1 / max(n, 1),
        'ssim': tot_ssim / max(n, 1),
        'psnr': tot_psnr / max(n, 1),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_l1, tot_ssim, tot_psnr, n = 0., 0., 0., 0
    per_action_psnr = {a: [] for a in ACTIONS}
    for batch in loader:
        orig = batch['orig'].to(device)
        target = batch['target'].to(device)
        a_oh = batch['action_onehot'].to(device)

        refined = model(orig, a_oh)
        l1 = (refined - target).abs().mean()
        ssim = ssim_loss(refined, target)
        # Per-sample PSNR for per-action breakdown
        mse_per = ((refined - target) ** 2).mean(dim=[1, 2, 3]).clamp(min=1e-10)
        psnr_per = (-10.0 * torch.log10(mse_per)).cpu().tolist()
        for action, psnr in zip(batch['action'], psnr_per):
            per_action_psnr[action].append(float(psnr))

        psnr_mean = sum(psnr_per) / len(psnr_per)
        tot_l1 += l1.item()
        tot_ssim += ssim.item()
        tot_psnr += psnr_mean
        n += 1

    return {
        'l1': tot_l1 / max(n, 1),
        'ssim': tot_ssim / max(n, 1),
        'psnr': tot_psnr / max(n, 1),
        'per_action': {a: (sum(v)/len(v) if v else 0.0)
                       for a, v in per_action_psnr.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--base_ch', type=int, default=48)
    ap.add_argument('--delta_scale', type=float, default=1.0)
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split_seed', type=int, default=None)
    ap.add_argument('--l1_weight', type=float, default=1.0)
    ap.add_argument('--ssim_weight', type=float, default=0.5)
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--num_workers', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    split_seed = args.split_seed or args.seed
    train_s, val_s = build_data(
        Path(args.jsonl), args.val_ratio,
        ('A excellent', 'B good', 'C acceptable'), split_seed)
    train_ds = LUTDataset(train_s, args.image_size, is_train=True)
    val_ds = LUTDataset(val_s, args.image_size, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    logger.info(f'train={len(train_ds)}, val={len(val_ds)}')

    # Model
    model = ImageDomainResUNet(
        in_ch=3, base_ch=args.base_ch,
        n_actions=len(ACTIONS), delta_scale=args.delta_scale,
    ).to(device)
    n_params = model.param_count()
    logger.info(f'ImageDomainResUNet base_ch={args.base_ch} '
                f'params={n_params/1e6:.2f}M')

    opt = optim.AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_psnr = 0.0
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, opt, device, args)
        val = evaluate(model, val_loader, device)
        sched.step()
        dt = time.time() - t0
        per_action_str = ' '.join(
            f'{a[:3]}={p:.2f}' for a, p in val['per_action'].items())
        logger.info(
            f'Ep {epoch:3d}/{args.epochs}  '
            f'train: L1={tr["l1"]:.4f} PSNR={tr["psnr"]:.2f}dB  '
            f'val: L1={val["l1"]:.4f} PSNR={val["psnr"]:.2f}dB  '
            f'[{per_action_str}]  {dt:.1f}s')
        history.append({
            'epoch': epoch, 'train': tr, 'val': val, 'time_sec': dt,
        })

        if val['psnr'] > best_psnr:
            best_psnr = val['psnr']
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_psnr': val['psnr'],
                'val_per_action': val['per_action'],
                'args': vars(args),
            }, out_dir / 'best.pt')
            logger.info(f'  * best val_psnr={best_psnr:.2f}dB  saved')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f'  early stop: no improve for '
                            f'{args.patience} epochs')
                break

    # Save history + final summary
    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info('\n' + '=' * 70)
    logger.info(f'[DONE] best_val_psnr={best_psnr:.2f}dB @ Ep{best_epoch}')
    logger.info(f'  checkpoint: {out_dir / "best.pt"}')
    logger.info(f'  v11a baseline:    24.46 dB (Path A)')
    logger.info(f'  Path X (residual): 25.28 dB')
    logger.info(f'  Path Z (this):     {best_psnr:.2f} dB')
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
