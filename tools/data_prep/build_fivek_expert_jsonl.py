"""Build a clean per-action pseudo_labels.jsonl from FiveK Expert (A/B/C/D/E) parameters.

For each FiveK source image and each tone action a in
{wb, brightness, contrast, shadows, highlights, saturation}, render a target
by applying ONLY that action's expert-chosen value via apply_diff_isp,
keeping all other params at identity.

This produces (orig, target) pairs that are:
  * By definition perfectly representable by our 7D ISP (no fitting error)
  * Labeled with REAL Lightroom expert parameters (not LLM-edited pseudo)
  * Per-action structured (matches existing v11 training format)

Avoid v9g_aug failure: targets here come from real human expert (Expert C
by default), not from synthetic Planckian. wb labels are real LR Temperature
values, not gain triplets from a single 1D color-temp axis.

Usage example:
    python tools/build_fivek_expert_jsonl.py --expert C --image_size 512 \
        --output_dir outputs/fivek_expert_c_master --limit 50  # dry-run

    python tools/build_fivek_expert_jsonl.py --expert C --image_size 512 \
        --output_dir outputs/fivek_expert_c_master  # full run

Output:
    {output_dir}/imgs/{stem}__{action}.jpg                # rendered targets
    {output_dir}/pseudo_labels.jsonl                      # one record per (img, action)
    {output_dir}/summary.json                             # action counts + param stats
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.diff_isp import apply_diff_isp  # noqa: E402


# ─────────────────────── Action -> param key + identity ──────────────────
# Identity value for each param when not the active action.
PARAM_IDENTITY = {
    'white_balance': 5500.0,    # Sun / Planckian neutral
    'brightness': 0.0,
    'contrast': 0.0,
    'shadows': 0.0,
    'highlights': 0.0,
    'saturation': 0.0,
    'clarity': 0.0,
}

# Map action name -> the parameter key in apply_diff_isp / Expert params dict.
ACTION_TO_PARAM = {
    'wb':         'white_balance',
    'brightness': 'brightness',
    'contrast':   'contrast',
    'shadows':    'shadows',
    'highlights': 'highlights',
    'saturation': 'saturation',
}

# Minimum delta from identity for a record to be considered "significant".
# Avoids generating identity-equivalent samples (action did nothing).
SIG_DELTA = {
    'wb':         200.0,   # |wb - 5500| >= 200K
    'brightness': 3.0,
    'contrast':   3.0,
    'shadows':    3.0,
    'highlights': 3.0,
    'saturation': 3.0,
}


def build_action_params(expert_params: dict, action: str,
                        device: torch.device) -> dict:
    """Return a (B=1) params dict where only `action`'s param is from expert,
    everything else at identity.
    """
    pkey = ACTION_TO_PARAM[action]
    out = {}
    for k, ident in PARAM_IDENTITY.items():
        if k == pkey:
            v = float(expert_params.get(k))
        else:
            v = float(ident)
        out[k] = torch.tensor([v], dtype=torch.float32, device=device)
    return out


def serialize_p_inferred(expert_params: dict, action: str) -> dict:
    """JSON-serializable P_inferred mirroring the active-action params dict.

    Other keys are at identity (0 or 5500K for wb).
    Includes 'clarity' for compatibility with existing schema.
    """
    pkey = ACTION_TO_PARAM[action]
    out = {}
    for k, ident in PARAM_IDENTITY.items():
        if k == pkey:
            out[k] = float(expert_params.get(k))
        else:
            out[k] = float(ident)
    return out


def load_pil_resized(jpg_path: Path, long_side: int) -> Image.Image:
    img = Image.open(jpg_path).convert('RGB')
    w, h = img.size
    if max(w, h) > long_side:
        if w >= h:
            new_w = long_side
            new_h = int(round(h * long_side / w))
        else:
            new_h = long_side
            new_w = int(round(w * long_side / h))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0  # (H, W, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
    return t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((t * 255 + 0.5).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params_json',
                    default='data/fivek_expert_abcde_params.json',
                    help='Output of parse_fivek_lrcat.py with 5 experts')
    ap.add_argument('--expert', default='C', choices=['A', 'B', 'C', 'D', 'E'],
                    help='Which FiveK expert to use as target (default C)')
    ap.add_argument('--image_dir',
                    default=r'E:/Data/dataset/fivek_jpeg',
                    help='Directory containing FiveK source JPEGs')
    ap.add_argument('--output_dir',
                    default='outputs/fivek_expert_c_master',
                    help='Where to write rendered targets + jsonl')
    ap.add_argument('--image_size', type=int, default=512,
                    help='Long-side resolution for rendered targets')
    ap.add_argument('--actions', nargs='+',
                    default=['wb', 'contrast', 'shadows',
                             'highlights', 'saturation'],
                    help='Which actions to render per source image. '
                         'NOTE: brightness is intentionally excluded by default '
                         'because train_lut.py ACTION_TO_IDX only supports 5 '
                         'actions: wb/contrast/shadows/highlights/saturation. '
                         'If you train with a 6-action variant, add brightness '
                         'back via --actions wb brightness contrast shadows '
                         'highlights saturation.')
    ap.add_argument('--wb_min', type=float, default=2000.0)
    ap.add_argument('--wb_max', type=float, default=12000.0,
                    help='Filter out wb outliers (default cap 12000K, '
                         'avoids the LR-max 50000K uncalibrated entries)')
    ap.add_argument('--limit', type=int, default=0,
                    help='If >0, only render this many source images (dry-run)')
    ap.add_argument('--device', default='cuda',
                    choices=['cuda', 'cpu'])
    ap.add_argument('--jpeg_quality', type=int, default=92)
    ap.add_argument('--skip_existing', action='store_true', default=True,
                    help='Skip rendering if output PNG already exists')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or
                          args.device == 'cpu' else 'cpu')
    print(f'device={device}, expert={args.expert}, image_size={args.image_size}')
    print(f'actions: {args.actions}')

    # ── 1. load expert params ──────────────────────────────────────────
    print(f'Loading {args.params_json}...')
    raw = json.load(open(args.params_json, encoding='utf-8'))
    all_samples = raw['samples']
    expert_samples = [s for s in all_samples if s['expert'] == args.expert]
    print(f'  {len(expert_samples)} records for expert {args.expert}')

    # Index expert params by image name (e.g. "a0002-dgw_005.dng" -> params dict)
    by_image = {}
    skipped_wb = 0
    for s in expert_samples:
        p = s['params']
        wb = p.get('white_balance')
        if wb is None:
            skipped_wb += 1
            continue
        if wb < args.wb_min or wb > args.wb_max:
            skipped_wb += 1
            continue
        by_image[s['image_name']] = p
    print(f'  {len(by_image)} usable images after wb filter '
          f'[{args.wb_min:.0f}, {args.wb_max:.0f}]K  '
          f'(skipped {skipped_wb})')

    # ── 2. locate source jpegs ─────────────────────────────────────────
    image_dir = Path(args.image_dir)
    jpeg_files = {p.name: p for p in image_dir.glob('*.jpg')}
    print(f'  {len(jpeg_files)} source JPEGs in {image_dir}')

    pairs = []
    unmatched = 0
    for dng_name, params in by_image.items():
        jpg_name = dng_name.replace('.dng', '.jpg').replace('.DNG', '.jpg')
        jpg_path = jpeg_files.get(jpg_name)
        if jpg_path is None:
            unmatched += 1
            continue
        pairs.append((jpg_path, params))
    print(f'  matched {len(pairs)} (orig_jpg, expert_params) pairs '
          f'(unmatched {unmatched})')

    if args.limit > 0:
        pairs = pairs[:args.limit]
        print(f'  LIMIT applied -> {len(pairs)} pairs (dry-run)')

    # ── 3. render per-action targets ────────────────────────────────────
    out_dir = Path(args.output_dir)
    img_out_dir = out_dir / 'imgs'
    img_out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / 'pseudo_labels.jsonl'
    summary_path = out_dir / 'summary.json'

    n_total = 0
    n_skip_existing = 0
    n_skip_identity = 0
    action_counts = Counter()
    param_vals = defaultdict(list)
    t_start = time.time()

    with open(jsonl_path, 'w', encoding='utf-8') as fout:
        for i, (jpg_path, params) in enumerate(pairs):
            stem = jpg_path.stem
            # Load once, reuse for all actions
            try:
                img_pil = load_pil_resized(jpg_path, args.image_size)
                img_t = pil_to_tensor(img_pil, device)
            except Exception as e:
                print(f'  [skip] {jpg_path.name}: load error {e}')
                continue

            for action in args.actions:
                pkey = ACTION_TO_PARAM[action]
                val = float(params.get(pkey, PARAM_IDENTITY[pkey]))
                ident = PARAM_IDENTITY[pkey]
                delta = abs(val - ident)
                if delta < SIG_DELTA[action]:
                    n_skip_identity += 1
                    continue

                target_fname = f'{stem}__{action}.jpg'
                target_path = img_out_dir / target_fname

                # skip if existing
                if args.skip_existing and target_path.exists():
                    n_skip_existing += 1
                else:
                    a_params = build_action_params(params, action, device)
                    with torch.no_grad():
                        rendered = apply_diff_isp(img_t, a_params)
                    out_pil = tensor_to_pil(rendered)
                    out_pil.save(target_path, quality=args.jpeg_quality)

                # write jsonl record (same schema as fivek_500_master pseudo_labels)
                P_inf = serialize_p_inferred(params, action)
                entry = {
                    'rank': -1,
                    'idx': i,
                    'source_image': f'fivek-{args.expert.lower()}-{stem}',
                    'orig_path': str(jpg_path).replace('\\', '/'),
                    'target_path': str(target_path).replace('\\', '/'),
                    'caption': f'Adjust {action} per FiveK Expert {args.expert}.',
                    'tone_target': action,
                    'P_inferred': P_inf,
                    'P_init_heuristic': P_inf,
                    'pixel_l1': 0.0, 'pixel_l2': 0.0, 'final_loss': 0.0,
                    'delta_target_orig': 0.0,
                    'fit_size': list(img_pil.size),
                    'runtime_sec': 0.0,
                    'verdict': 'EXPERT_C_RENDER',
                    'action': action,
                    '_source': f'outputs/{out_dir.name}',
                    'quality_tier': 'A excellent',
                    '_expert': args.expert,
                    '_expert_param_value': val,
                }
                fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
                n_total += 1
                action_counts[action] += 1
                param_vals[action].append(val)

            if (i + 1) % 200 == 0:
                el = time.time() - t_start
                rate = (i + 1) / max(el, 1e-3)
                eta = (len(pairs) - (i + 1)) / max(rate, 1e-3)
                print(f'  [{i+1:5d}/{len(pairs)}]  records={n_total}  '
                      f'skip_id={n_skip_identity} skip_exist={n_skip_existing}  '
                      f'{rate:.1f} img/s  eta {eta/60:.1f} min')

    elapsed = time.time() - t_start
    print(f'\nDONE in {elapsed/60:.1f} min')
    print(f'  records written: {n_total}')
    print(f'  per-action:       {dict(action_counts)}')
    print(f'  skipped identity: {n_skip_identity}')
    print(f'  skipped existing: {n_skip_existing}')

    # ── 4. summary.json ────────────────────────────────────────────────
    summary = {
        'expert': args.expert,
        'image_size': args.image_size,
        'wb_filter': [args.wb_min, args.wb_max],
        'n_source_images': len(pairs),
        'n_records': n_total,
        'per_action': {},
    }
    for a, vals in param_vals.items():
        summary['per_action'][a] = {
            'n': len(vals),
            'param_min': float(min(vals)),
            'param_max': float(max(vals)),
            'param_mean': float(sum(vals) / len(vals)),
            'param_median': float(sorted(vals)[len(vals) // 2]),
        }
    summary['elapsed_min'] = elapsed / 60
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'  summary: {summary_path}')
    print(f'  jsonl:   {jsonl_path}')


if __name__ == '__main__':
    main()
