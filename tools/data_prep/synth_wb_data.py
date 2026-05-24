"""Generate synthetic white-balance training samples from FiveK images.

For each source image, apply known R/G/B gains derived from a color
temperature (Planckian approximation) to produce a synthetic target.
The resulting (orig, synth_target) pairs have *perfect* WB labels and
massively expand the wb action training data.

Usage:
    python tools/synth_wb_data.py --n_images 50 --temps 3500,5000,7500,10000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.diff_isp import color_temp_to_rgb_gains  # noqa: E402


def gains_for_temp(temp_k: float) -> torch.Tensor:
    """Color temperature (Kelvin) -> RGB gains (3,) normalized so G=1."""
    t = torch.tensor([temp_k], dtype=torch.float32)
    gains = color_temp_to_rgb_gains(t)
    return gains[0]


def apply_gain_to_image(img_pil: Image.Image, gains_rgb: torch.Tensor
                         ) -> Image.Image:
    """Multiply per-channel RGB gain on a PIL image. Output clipped to [0,1]."""
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    arr = arr * gains_rgb.numpy().reshape(1, 1, 3)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source_jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--output_dir', default='outputs/synth_wb')
    ap.add_argument('--n_images', type=int, default=50,
                    help='Number of distinct FiveK images to use as source')
    ap.add_argument('--temps', type=str,
                    default='3500,4500,5500,7500,10000',
                    help='Comma-separated color temperatures (Kelvin)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / 'imgs'
    img_dir.mkdir(exist_ok=True)

    temps = [float(t.strip()) for t in args.temps.split(',') if t.strip()]
    print(f'Color temperatures: {temps}')

    # Print gain for each temp for sanity
    print('Reference gains (G=1.0):')
    for T in temps:
        g = gains_for_temp(T)
        print(f'  {int(T):5d}K  R/G/B = ({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f})')

    # Load source pool
    lines = Path(args.source_jsonl).read_text(encoding='utf-8').strip().split('\n')
    all_samples = [json.loads(l) for l in lines]
    # Filter to good tiers and non-wb (avoid reusing real wb sources)
    pool = [s for s in all_samples
            if s.get('quality_tier', '') in ('A excellent', 'B good', 'C acceptable')
            and s.get('action', s.get('tone_target', '')) != 'wb']

    # Dedupe by source_image
    seen = set()
    unique_pool = []
    for s in pool:
        if s['source_image'] not in seen:
            seen.add(s['source_image'])
            unique_pool.append(s)
    random.shuffle(unique_pool)

    selected = unique_pool[:args.n_images]
    print(f'\nSelected {len(selected)} unique FiveK images from pool of '
          f'{len(unique_pool)} non-wb samples')

    new_entries = []
    skipped = 0
    for src_s in selected:
        orig_path = Path(src_s['orig_path'])
        if not orig_path.is_absolute():
            orig_path = PROJECT_ROOT / orig_path
        if not orig_path.exists():
            skipped += 1
            continue

        img = Image.open(orig_path).convert('RGB')

        for temp_k in temps:
            gains = gains_for_temp(temp_k)
            synth_target = apply_gain_to_image(img, gains)
            stem = orig_path.stem
            target_fname = f'{stem}_wb{int(temp_k)}K.png'
            target_path_full = img_dir / target_fname
            synth_target.save(target_path_full, quality=95)

            # Build P_inferred: all 0 except white_balance = temp_k
            P_inferred = {
                'white_balance': float(temp_k),
                'brightness': 0.0,
                'contrast': 0.0,
                'shadows': 0.0,
                'highlights': 0.0,
                'saturation': 0.0,
                'clarity': 0.0,
            }

            entry = {
                'rank': -1,
                'idx': -1,
                'source_image': f'synth-{src_s["source_image"]}-{int(temp_k)}K',
                'orig_path': str(orig_path).replace('\\', '/'),
                'target_path': str(target_path_full).replace('\\', '/'),
                'caption': f'Adjust white balance to {int(temp_k)} Kelvin.',
                'tone_target': 'wb',
                'P_inferred': P_inferred,
                'P_init_heuristic': P_inferred,
                'pixel_l1': 0.0, 'pixel_l2': 0.0, 'final_loss': 0.0,
                'delta_target_orig': 0.0,
                'fit_size': list(img.size),
                'runtime_sec': 0.0,
                'verdict': 'SYNTH',
                'action': 'wb',
                '_source': 'outputs/synth_wb',
                'quality_tier': 'A excellent',
                '_synth_gains_rgb': gains.tolist(),
                '_synth_temp_k': float(temp_k),
            }
            new_entries.append(entry)

    out_jsonl = out_dir / 'labels.jsonl'
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    n_src = len(selected) - skipped
    print(f'\nGenerated {len(new_entries)} synth wb samples from '
          f'{n_src} source images (skipped {skipped} missing)')
    print(f'  JSONL: {out_jsonl}')
    print(f'  PNGs:  {img_dir}')


if __name__ == '__main__':
    main()
