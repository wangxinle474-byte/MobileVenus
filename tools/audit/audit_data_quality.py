"""Audit pseudo-label quality: how clean are wb / saturation / etc. edits?

For each val sample, compute:
  1. Channel-mean gain ratio (target/orig per channel)
  2. Residual PSNR = apply linear gain to orig, compare to target.
     If high (~30 dB+): target is approximately a clean linear RGB scaling
     If low (<20 dB): target has substantial non-linear / spatial changes
                       beyond what a 3-gain WB can express. = noisy label
                       for wb action.
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from training.firered_baseline.train_lut import (
    LUTDataset, build_data, ACTIONS,
)


def main():
    _, val_s = build_data(
        Path('outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl'),
        0.2, ('A excellent', 'B good', 'C acceptable'), 42)
    ds = LUTDataset(val_s, 256, is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    records = {a: [] for a in ACTIONS}
    src_to_tier = {s['source_image']: s.get('quality_tier', '?') for s in val_s}

    for b in loader:
        orig = b['orig'][0]
        target = b['target'][0]
        a = b['action'][0]
        src = b['source_image'][0]

        om = orig.mean(dim=[1, 2]).clamp(min=0.01)
        tm = target.mean(dim=[1, 2]).clamp(min=0.01)
        gains = (tm / om)                                    # (3,)

        # Apply linear gain, see residual to target
        corrected = (orig * gains.view(3, 1, 1)).clamp(0, 1)
        mse = ((corrected - target) ** 2).mean().clamp(min=1e-10)
        residual_psnr = float((-10 * torch.log10(mse)).item())

        records[a].append({
            'src': src,
            'gains': gains.tolist(),
            'residual_psnr': residual_psnr,
            'tier': src_to_tier.get(src, '?'),
        })

    print('==== Pseudo-label cleanness audit ====')
    print('residual_psnr = how close target is to a pure linear-gain edit')
    print('(higher = cleaner gain shift; lower = mixed/non-linear edit)\n')
    for a in ACTIONS:
        rs = [r['residual_psnr'] for r in records[a]]
        if not rs:
            continue
        mean_r = sum(rs) / len(rs)
        min_r = min(rs)
        max_r = max(rs)
        print(f'  {a:12s} n={len(rs):2d}  mean={mean_r:.2f}  '
              f'min={min_r:.2f}  max={max_r:.2f}')

    print()
    print('==== wb samples (sorted: noisiest first) ====')
    for r in sorted(records['wb'], key=lambda x: x['residual_psnr']):
        g = r['gains']
        print(f'  {r["src"]:36s} '
              f'gains=({g[0]:.2f},{g[1]:.2f},{g[2]:.2f}) '
              f'residual={r["residual_psnr"]:.1f}dB '
              f'tier={r["tier"]}')

    print()
    print('==== saturation samples (sorted: noisiest first, top-5) ====')
    sat_sorted = sorted(records['saturation'], key=lambda x: x['residual_psnr'])
    for r in sat_sorted[:5]:
        g = r['gains']
        print(f'  {r["src"]:36s} '
              f'gains=({g[0]:.2f},{g[1]:.2f},{g[2]:.2f}) '
              f'residual={r["residual_psnr"]:.1f}dB '
              f'tier={r["tier"]}')


if __name__ == '__main__':
    main()
