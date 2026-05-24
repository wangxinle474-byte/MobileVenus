"""v13: image-domain residual refinement on top of v11-d 7-action base.

Pipeline:
  base = v11d(orig, action)                       # frozen
  refined, delta = refiner(orig, base, action)    # learned
  loss = L1(refined, target) + ssim_w * SSIM
       + lab_w * L1_in_Lab(refined, target)
       + grad_w * L1(grad(refined), grad(target))
       + delta_l1_w * |delta|

Why:
  v11-d 7-action best is 23.99dB but visually still conservative vs FireRed
  (esp. wb 21.67, saturation 22.36, brightness 23.02). A small image-domain
  residual on top of the parametric base captures the FireRed-specific
  color/texture style that 7D ISP + Bezier curves cannot represent.

Usage:
  python -m training.firered_baseline.train_v13_firered_refine \\
    --base_ckpt checkpoints/lut_v11d_firered_7actions_6537/best.pt \\
    --jsonl outputs/firered_v11_existing_7actions/pseudo_labels.jsonl \\
    --out_dir checkpoints/v13a_firered_refine_7actions \\
    --batch_size 4 --epochs 40 --base_ch 32 --image_size 256
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

from models.firered_residual_refiner import FireRedResidualRefiner  # noqa: E402
from models.diff_isp import ssim_loss  # noqa: E402
from training.firered_baseline.train_lut import (  # noqa: E402
    ACTIONS, LUTDataset, NamedCurvesPredictor, build_data, compute_psnr,
    set_actions,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('v13')


# ============================================================
# Auxiliary losses
# ============================================================
def _rgb_to_lab_approx(x: torch.Tensor) -> torch.Tensor:
    """Cheap differentiable RGB -> Lab approximation (sRGB assumed in [0,1]).

    Uses linearized sRGB + standard XYZ matrix + Lab cube root.
    Outputs roughly in (L:[0,1], a:[-1,1], b:[-1,1]).
    """
    eps = 1e-6
    # sRGB -> linear (smooth approx)
    x = x.clamp(0.0, 1.0)
    lin = torch.where(x > 0.04045,
                      ((x + 0.055) / 1.055).clamp(min=eps).pow(2.4),
                      x / 12.92)
    r, g, b = lin[:, 0:1], lin[:, 1:2], lin[:, 2:3]
    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b
    # Reference white D65
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    fx = (X / Xn).clamp(min=eps).pow(1.0 / 3.0)
    fy = (Y / Yn).clamp(min=eps).pow(1.0 / 3.0)
    fz = (Z / Zn).clamp(min=eps).pow(1.0 / 3.0)
    L = (1.16 * fy - 0.16)              # ~[0, 1]
    a = (5.0 * (fx - fy)) * 0.2          # scale to ~[-1, 1]
    b_ = (2.0 * (fy - fz)) * 0.2
    return torch.cat([L, a, b_], dim=1)


def lab_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (_rgb_to_lab_approx(pred) - _rgb_to_lab_approx(target)).abs().mean()


def gradient_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sobel-style edge L1 to encourage matching local structure."""
    def _grad(x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy
    pgx, pgy = _grad(pred)
    tgx, tgy = _grad(target)
    return (pgx - tgx).abs().mean() + (pgy - tgy).abs().mean()


# ============================================================
# v11-d base loader
# ============================================================
def load_v11d_base(ckpt_path: Path, device: torch.device) -> tuple:
    """Load frozen v11-d NamedCurvesPredictor from checkpoint.

    Also calls set_actions() in-place so the global ACTIONS list matches the
    base checkpoint. Returns (model, ckpt_args).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ca = ckpt['args']
    ckpt_actions = ca.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)

    image_size = ca.get('image_size', 256)
    if not ca.get('named_curves', False):
        raise RuntimeError(
            f'base_ckpt {ckpt_path} is not a NamedCurves (v9/v11) model')

    model = NamedCurvesPredictor(
        n_colors=ca.get('nc_n_colors', 6),
        n_control_points=ca.get('nc_n_control_points', 11),
        use_attention=ca.get('nc_use_attention', False),
        per_action_curves=ca.get('nc_per_action_curves', False),
        use_7d_anchor=ca.get('nc_use_7d_anchor', False),
        use_context=ca.get('nc_use_context', False),
        action_gated_context=ca.get('nc_action_gated_context', False),
        use_action_context=ca.get('nc_use_action_context', False),
        use_region_basis=ca.get('nc_use_region_basis', False),
        use_region_param_delta=ca.get('nc_use_region_param_delta', False),
        use_learned_cn=ca.get('nc_use_learned_cn', False),
        use_wb_head=ca.get('nc_use_wb_head', False),
        use_nilut_residual=ca.get('nc_use_nilut_residual', False),
        use_vera_renderer=ca.get('nc_use_vera_renderer', False),
        nilut_hidden=ca.get('nilut_hidden', 32),
        nilut_n_layers=ca.get('nilut_n_layers', 3),
        nilut_n_freq=ca.get('nilut_n_freq', 4),
        nilut_gate_init=ca.get('nilut_gate_init', 1.0),
        use_implicit_head=ca.get('use_implicit_head', False),
        implicit_head_base_ch=ca.get('implicit_head_base_ch', 32),
        implicit_head_gate_init=ca.get('implicit_head_gate_init', 0.0),
        image_size=image_size,
        n_actions=len(ACTIONS),
        dropout=ca.get('dropout', 0.5),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(f'Loaded v11d base: {ckpt_path} '
                f'val_psnr={ckpt.get("val_psnr", 0):.2f}dB '
                f'@ Ep{ckpt.get("epoch", "?")} actions={ACTIONS}')
    return model, ca


# ============================================================
# Train / eval loops
# ============================================================
def _forward_base(base_model: nn.Module, batch: dict,
                  device: torch.device) -> torch.Tensor:
    """Run frozen v11-d to produce base image (no grad)."""
    enc_in = batch['enc_input'].to(device)
    orig = batch['orig'].to(device)
    a_oh = batch['action_onehot'].to(device)
    with torch.no_grad():
        refined, _, _, _ = base_model(enc_in, orig, a_oh)
    return refined


def train_one_epoch(refiner, base_model, loader, opt, device, args):
    refiner.train()
    base_model.eval()
    tot = {'l1': 0., 'ssim': 0., 'lab': 0., 'grad': 0., 'd': 0., 'psnr': 0.}
    n = 0
    for i, batch in enumerate(loader):
        orig = batch['orig'].to(device)
        target = batch['target'].to(device)
        a_oh = batch['action_onehot'].to(device)
        sw = batch['tier_weight'].to(device)

        base = _forward_base(base_model, batch, device)
        refined, delta = refiner(orig, base, a_oh)

        l1_per = (refined - target).abs().mean(dim=[1, 2, 3])
        L_l1 = (l1_per * sw).sum() / (sw.sum() + 1e-6)
        L_ssim = ssim_loss(refined, target)
        L_lab = lab_l1_loss(refined, target)
        L_grad = gradient_l1_loss(refined, target)
        L_delta = delta.abs().mean()
        loss = (args.l1_weight * L_l1
                + args.ssim_weight * L_ssim
                + args.lab_weight * L_lab
                + args.grad_weight * L_grad
                + args.delta_l1_weight * L_delta)

        if not torch.isfinite(loss):
            opt.zero_grad()
            continue

        opt.zero_grad()
        loss.backward()
        for p in refiner.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad = torch.nan_to_num(p.grad, nan=0.0,
                                          posinf=0.0, neginf=0.0)
        nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            psnr = compute_psnr(refined, target)
        tot['l1'] += L_l1.item()
        tot['ssim'] += L_ssim.item()
        tot['lab'] += L_lab.item()
        tot['grad'] += L_grad.item()
        tot['d'] += L_delta.item()
        tot['psnr'] += psnr
        n += 1

        if (i + 1) % args.log_every == 0:
            logger.info(
                f'  [{i+1}/{len(loader)}] '
                f'L1={tot["l1"]/n:.4f} SSIM={tot["ssim"]/n:.4f} '
                f'Lab={tot["lab"]/n:.4f} grad={tot["grad"]/n:.4f} '
                f'|d|={tot["d"]/n:.4f} PSNR={tot["psnr"]/n:.2f}dB')

    return {k: v / max(n, 1) for k, v in tot.items()}


@torch.no_grad()
def evaluate(refiner, base_model, loader, device):
    refiner.eval()
    base_model.eval()
    tot_l1, tot_psnr_refined, tot_psnr_base, n = 0., 0., 0., 0
    per_action_psnr = {a: [] for a in ACTIONS}
    per_action_base = {a: [] for a in ACTIONS}
    for batch in loader:
        orig = batch['orig'].to(device)
        target = batch['target'].to(device)
        a_oh = batch['action_onehot'].to(device)

        base = _forward_base(base_model, batch, device)
        refined, _ = refiner(orig, base, a_oh)
        l1 = (refined - target).abs().mean()

        mse_r = ((refined - target) ** 2).mean(dim=[1, 2, 3]).clamp(min=1e-10)
        psnr_r = (-10.0 * torch.log10(mse_r)).cpu().tolist()
        mse_b = ((base - target) ** 2).mean(dim=[1, 2, 3]).clamp(min=1e-10)
        psnr_b = (-10.0 * torch.log10(mse_b)).cpu().tolist()
        for action, pr, pb in zip(batch['action'], psnr_r, psnr_b):
            per_action_psnr[action].append(float(pr))
            per_action_base[action].append(float(pb))

        tot_l1 += l1.item()
        tot_psnr_refined += sum(psnr_r) / len(psnr_r)
        tot_psnr_base += sum(psnr_b) / len(psnr_b)
        n += 1

    return {
        'l1': tot_l1 / max(n, 1),
        'psnr': tot_psnr_refined / max(n, 1),
        'psnr_base': tot_psnr_base / max(n, 1),
        'per_action': {a: (sum(v)/len(v) if v else 0.0)
                       for a, v in per_action_psnr.items()},
        'per_action_base': {a: (sum(v)/len(v) if v else 0.0)
                            for a, v in per_action_base.items()},
    }


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True,
                    help='Frozen v11-d NamedCurves checkpoint')
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--base_ch', type=int, default=32)
    ap.add_argument('--delta_scale', type=float, default=0.5)
    ap.add_argument('--image_size', type=int, default=None,
                    help='Override base ckpt image_size if given')
    ap.add_argument('--val_ratio', type=float, default=None,
                    help='Override base ckpt val_ratio if given')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split_seed', type=int, default=None)
    ap.add_argument('--l1_weight', type=float, default=1.0)
    ap.add_argument('--ssim_weight', type=float, default=0.5)
    ap.add_argument('--lab_weight', type=float, default=0.5)
    ap.add_argument('--grad_weight', type=float, default=0.1)
    ap.add_argument('--delta_l1_weight', type=float, default=0.02,
                    help='Sparse residual prior; small but non-zero')
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--num_workers', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device: {device}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load v11-d base (also sets ACTIONS via set_actions) ──
    base_model, base_args = load_v11d_base(Path(args.base_ckpt), device)

    image_size = args.image_size or base_args.get('image_size', 256)
    val_ratio = (args.val_ratio if args.val_ratio is not None
                 else base_args.get('val_ratio', 0.2))
    split_seed = (args.split_seed if args.split_seed is not None
                  else base_args.get('split_seed',
                                     base_args.get('seed', args.seed)))
    logger.info(f'image_size={image_size} val_ratio={val_ratio} '
                f'split_seed={split_seed} actions={ACTIONS}')

    # ── Data (uses same split as base ckpt) ──
    train_s, val_s = build_data(
        Path(args.jsonl), val_ratio,
        ('A excellent', 'B good', 'C acceptable'),
        split_seed, action_filter=ACTIONS)
    train_ds = LUTDataset(train_s, image_size, is_train=True)
    val_ds = LUTDataset(val_s, image_size, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    logger.info(f'train={len(train_ds)} val={len(val_ds)}')

    # ── Refiner ──
    refiner = FireRedResidualRefiner(
        base_ch=args.base_ch, n_actions=len(ACTIONS),
        delta_scale=args.delta_scale,
    ).to(device)
    n_params = refiner.param_count()
    logger.info(f'FireRedResidualRefiner base_ch={args.base_ch} '
                f'delta_scale={args.delta_scale} '
                f'params={n_params/1e6:.2f}M')

    opt = optim.AdamW(refiner.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Persist resolved settings for reproducibility (incl. actions list).
    args_dict = vars(args).copy()
    args_dict.update({
        'actions': list(ACTIONS),
        'image_size': image_size,
        'val_ratio': val_ratio,
        'split_seed': split_seed,
        'base_args': base_args,
        'v13_refine': True,
    })

    best_psnr = 0.0
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(refiner, base_model, train_loader, opt,
                             device, args)
        val = evaluate(refiner, base_model, val_loader, device)
        sched.step()
        dt = time.time() - t0
        per_action_str = ' '.join(
            f'{a[:3]}={p:.2f}' for a, p in val['per_action'].items())
        logger.info(
            f'Ep {epoch:3d}/{args.epochs}  '
            f'train: L1={tr["l1"]:.4f} PSNR={tr["psnr"]:.2f}dB  '
            f'val: L1={val["l1"]:.4f} PSNR={val["psnr"]:.2f}dB '
            f'(base={val["psnr_base"]:.2f}dB)  '
            f'[{per_action_str}]  {dt:.1f}s')
        history.append({
            'epoch': epoch, 'train': tr, 'val': val, 'time_sec': dt,
        })

        if val['psnr'] > best_psnr:
            best_psnr = val['psnr']
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'model_state_dict': refiner.state_dict(),
                'epoch': epoch,
                'val_psnr': val['psnr'],
                'val_psnr_base': val['psnr_base'],
                'val_per_action': val['per_action'],
                'val_per_action_base': val['per_action_base'],
                'args': args_dict,
            }, out_dir / 'best.pt')
            logger.info(f'  * best val_psnr={best_psnr:.2f}dB  saved')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f'  early stop: no improve for '
                            f'{args.patience} epochs')
                break

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info('\n' + '=' * 70)
    logger.info(f'[DONE] best_val_psnr={best_psnr:.2f}dB @ Ep{best_epoch}')
    logger.info(f'  checkpoint: {out_dir / "best.pt"}')
    logger.info(f'  base v11-d: {args.base_ckpt}')
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
