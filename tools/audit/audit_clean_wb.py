"""Audit cleanness of new FireRed wb samples (warm + cool).

Compute residual PSNR after pure linear RGB gain correction:
  - High residual = clean WB-like edit (linear gain explains target)
  - Low residual  = mixed edit (non-linear changes beyond WB)

Compare new clean prompts (warm + cool, 49 samples) vs original
'Apply warmer white balance' prompts (7 wb val samples).
"""
from __future__ import annotations
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import torch


def load_resize(p: Path, size=256) -> torch.Tensor:
    img = Image.open(p).convert('RGB').resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def residual_psnr(orig: torch.Tensor, target: torch.Tensor) -> tuple:
    """Apply linear RGB gain to orig, measure residual PSNR vs target.
    Returns (residual_psnr_db, gains_rgb tuple).
    """
    om = orig.mean(dim=[1, 2]).clamp(min=0.01)
    tm = target.mean(dim=[1, 2]).clamp(min=0.01)
    gains = (tm / om)
    corrected = (orig * gains.view(3, 1, 1)).clamp(0, 1)
    mse = ((corrected - target) ** 2).mean().clamp(min=1e-10)
    psnr = float((-10 * torch.log10(mse)).item())
    return psnr, tuple(gains.tolist())


def main():
    cfg_warm = json.load(open('data/teacher_edits_fivek_wb_clean_warm.json',
                                encoding='utf-8'))
    cfg_cool = json.load(open('data/teacher_edits_fivek_wb_clean_cool.json',
                                encoding='utf-8'))

    results = {'warm': [], 'cool': []}
    for label, cfg, out_dir in [
        ('warm', cfg_warm, Path('outputs/teacher_edits/fivek_wb_clean_warm')),
        ('cool', cfg_cool, Path('outputs/teacher_edits/fivek_wb_clean_cool')),
    ]:
        for s in cfg['samples']:
            orig_p = Path(s['orig_path'])
            tgt_p = out_dir / f'{s["idx"]:04d}.png'
            if not orig_p.exists() or not tgt_p.exists():
                continue
            orig = load_resize(orig_p)
            target = load_resize(tgt_p)
            r_psnr, gains = residual_psnr(orig, target)
            results[label].append({
                'idx': s['idx'],
                'src': s['source_image'],
                'gains': gains,
                'residual_psnr': r_psnr,
            })

    print('==== Clean wb samples audit (residual PSNR) ====')
    print('Higher residual_psnr = target closer to pure linear gain (cleaner WB)\n')
    for label in ('warm', 'cool'):
        rs = [r['residual_psnr'] for r in results[label]]
        if not rs:
            continue
        n = len(rs)
        mean = sum(rs) / n
        rs_sorted = sorted(rs)
        med = rs_sorted[n // 2]
        print(f'  {label}  n={n:2d}  mean={mean:.2f}dB  '
              f'median={med:.2f}dB  min={min(rs):.2f}dB  max={max(rs):.2f}dB')

    # Compare to old dirty wb data (residual mean was 23.20 in audit)
    all_rs = (
        [r['residual_psnr'] for r in results['warm']] +
        [r['residual_psnr'] for r in results['cool']]
    )
    print(f'\n  combined n={len(all_rs)}  mean={sum(all_rs)/len(all_rs):.2f}dB')
    print('  Reference: original "wb" val samples residual mean = 23.20 dB')
    delta = sum(all_rs) / len(all_rs) - 23.20
    print(f'  Δ vs original: {delta:+.2f} dB '
          f'(positive = cleaner, ideal: >5 dB improvement)')

    # Sample-by-sample for spot check
    print('\n==== warm samples ====')
    for r in sorted(results['warm'], key=lambda x: x['residual_psnr']):
        g = r['gains']
        print(f'  {r["src"]:42s} gains=({g[0]:.2f},{g[1]:.2f},{g[2]:.2f}) '
              f'residual={r["residual_psnr"]:.2f}dB')
    print('\n==== cool samples ====')
    for r in sorted(results['cool'], key=lambda x: x['residual_psnr']):
        g = r['gains']
        print(f'  {r["src"]:42s} gains=({g[0]:.2f},{g[1]:.2f},{g[2]:.2f}) '
              f'residual={r["residual_psnr"]:.2f}dB')


if __name__ == '__main__':
    main()
