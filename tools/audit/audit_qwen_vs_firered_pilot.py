"""Compare Qwen-Image-Edit (via DashScope) vs FireRed (ModelScope Studio) output
quality on the same 3 wb captions.

Metrics (lower = cleaner wb):
  1. brightness_shift (target_mean - orig_mean): how much luminance changed
  2. saturation_shift: how much saturation changed
  3. residual_psnr: how close (orig + inferred WB gains) is to target
  4. inferred_gains: the 3 RGB gains after WB inversion

A "clean WB" edit would have:
  - brightness_shift ~ 0 (no brightness change)
  - saturation_shift ~ 0 (no sat change)
  - residual_psnr > 30 dB (WB gains fully explain the edit)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image


ORIG_DIR = Path('outputs/inverse_fit_pilot/fivek_500_master')
FIRERED_DIR = Path('outputs/teacher_edits/fivek_wb_clean_warm')
FIRERED_COOL_DIR = Path('outputs/teacher_edits/fivek_wb_clean_cool')
QWEN_DIR = Path('outputs/teacher_edits/fivek_wb_clean_qwen_pilot')

# Pilot samples (2 warm + 1 cool)
SAMPLES = [
    {'idx': 850, 'dir': FIRERED_DIR, 'label': 'warm'},
    {'idx': 487, 'dir': FIRERED_DIR, 'label': 'warm'},
    {'idx': 1263, 'dir': FIRERED_COOL_DIR, 'label': 'cool'},
]


def rgb_to_lum(img: np.ndarray) -> float:
    """Mean luminance (Rec.709)."""
    return float(0.2126 * img[..., 0].mean()
                 + 0.7152 * img[..., 1].mean()
                 + 0.0722 * img[..., 2].mean())


def rgb_to_sat(img: np.ndarray) -> float:
    """Mean saturation = (max-min)/(max+eps), in [0,1]."""
    mx = img.max(axis=-1)
    mn = img.min(axis=-1)
    return float(((mx - mn) / (mx + 1e-6)).mean())


def fit_wb_gains(orig: np.ndarray, target: np.ndarray) -> tuple:
    """Find per-channel gain g = target/orig (median per channel)."""
    eps = 1e-4
    gains = []
    for c in range(3):
        ratio = target[..., c] / (orig[..., c] + eps)
        gains.append(float(np.median(ratio)))
    return tuple(gains)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a - b) ** 2).mean())
    if mse < 1e-10:
        return 99.0
    return float(-10 * np.log10(mse))


def load(path: Path, size: int = 512) -> np.ndarray:
    img = Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def analyze(idx: int, orig_path: Path, target_path: Path, label: str):
    orig = load(orig_path)
    target = load(target_path)
    gains = fit_wb_gains(orig, target)
    rendered = np.clip(orig * np.array(gains), 0, 1)
    res_psnr = psnr(rendered, target)
    bri_shift = rgb_to_lum(target) - rgb_to_lum(orig)
    sat_shift = rgb_to_sat(target) - rgb_to_sat(orig)
    return dict(idx=idx, label=label,
                gains=gains, res_psnr=res_psnr,
                bri_shift=bri_shift, sat_shift=sat_shift)


def main():
    print(f"{'idx':<6} {'label':<6} {'teacher':<8} "
          f"{'gains (R,G,B)':<28} {'bri_Δ':<8} {'sat_Δ':<8} "
          f"{'res_PSNR':<8}")
    print('-' * 80)

    # Master captions for orig paths
    with open('outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl',
              encoding='utf-8') as f:
        master = {}
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            master[r['idx']] = r['orig_path']

    rows = []
    for s in SAMPLES:
        orig_path = Path(master[s['idx']])
        firered_path = s['dir'] / f"{s['idx']:04d}.png"
        qwen_path = QWEN_DIR / f"{s['idx']:04d}.png"

        if firered_path.exists():
            r = analyze(s['idx'], orig_path, firered_path, s['label'])
            r['teacher'] = 'FireRed'
            rows.append(r)
            print(f"{r['idx']:<6} {r['label']:<6} {r['teacher']:<8} "
                  f"({r['gains'][0]:5.2f},{r['gains'][1]:5.2f},{r['gains'][2]:5.2f})      "
                  f"{r['bri_shift']:+6.3f}  {r['sat_shift']:+6.3f}  "
                  f"{r['res_psnr']:5.1f} dB")

        if qwen_path.exists():
            r = analyze(s['idx'], orig_path, qwen_path, s['label'])
            r['teacher'] = 'Qwen'
            rows.append(r)
            print(f"{r['idx']:<6} {r['label']:<6} {r['teacher']:<8} "
                  f"({r['gains'][0]:5.2f},{r['gains'][1]:5.2f},{r['gains'][2]:5.2f})      "
                  f"{r['bri_shift']:+6.3f}  {r['sat_shift']:+6.3f}  "
                  f"{r['res_psnr']:5.1f} dB")
        print()

    # Summary
    fr = [r for r in rows if r['teacher'] == 'FireRed']
    qw = [r for r in rows if r['teacher'] == 'Qwen']

    def avg(xs, k):
        return sum(x[k] for x in xs) / max(len(xs), 1)

    print('=' * 80)
    print(f"SUMMARY (n={len(fr)} each)")
    print(f"  FireRed: bri_Δ={avg(fr,'bri_shift'):+.3f}  "
          f"sat_Δ={avg(fr,'sat_shift'):+.3f}  "
          f"res_PSNR={avg(fr,'res_psnr'):.1f} dB")
    print(f"  Qwen   : bri_Δ={avg(qw,'bri_shift'):+.3f}  "
          f"sat_Δ={avg(qw,'sat_shift'):+.3f}  "
          f"res_PSNR={avg(qw,'res_psnr'):.1f} dB")
    print()
    if abs(avg(qw, 'bri_shift')) < abs(avg(fr, 'bri_shift')) and \
       avg(qw, 'res_psnr') > avg(fr, 'res_psnr'):
        print('  [WIN] Qwen outputs are cleaner wb edits (less brightness '
              'shift + higher residual PSNR).')
    elif abs(avg(qw, 'bri_shift')) > abs(avg(fr, 'bri_shift')):
        print('  [LOSS] Qwen still shifts brightness more than FireRed.')
    else:
        print('  [TIE] Mixed signal, need more samples to decide.')


if __name__ == '__main__':
    main()
