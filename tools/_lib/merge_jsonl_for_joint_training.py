"""Merge multiple pseudo_labels.jsonl files for joint training.

Different jsonl files use different `source_image` formats:
  - clean target (fivek_expert_c_master): "fivek-c-a0001-jmac_DSC1459"
  - inverse_fit pseudo (fivek_500_master): "a0939-IMG_0262.jpg"
  - synthetic WB aug: "synth-warm-X" (preserved)

For joint training to work correctly, the train/val split (by source_image
in train_lut.py:build_data) must group ALL variants of the same FiveK
image together. This tool extracts the canonical FiveK stem
(e.g. "a0001-jmac_DSC1459") and uses it as the unified source_image,
preserving the original via `_orig_source_image`.

Usage:
  python tools/merge_jsonl_for_joint_training.py \
    --inputs clean:outputs/fivek_expert_c_master/pseudo_labels.jsonl \
             pseudo:outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl \
    --upsample clean=1,pseudo=40 \
    --out_jsonl outputs/joint_clean_pseudo/pseudo_labels.jsonl

  # Then train as usual:
  python -u training/firered_baseline/train_lut.py \
    --jsonl outputs/joint_clean_pseudo/pseudo_labels.jsonl \
    --named_curves --nc_n_colors 3 --nc_n_control_points 7 \
    --nc_use_7d_anchor --nc_use_context --nc_action_gated_context \
    --dropout 0.5 --param_weight 0.05 \
    --seed 42 --split_seed 42 \
    --out_dir checkpoints/lut_v11a_joint_seed42 \
    --epochs 50 --patience 10
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def extract_fivek_stem(record: dict) -> str:
    """Extract canonical FiveK image stem (e.g. 'a0001-jmac_DSC1459').

    Handled formats (output is always the trailing 'aXXXX-...' identifier):
      'fivek-c-a0001-jmac_DSC1459'   -> 'a0001-jmac_DSC1459'
      'fivek-a-a3500-IMG_0001'       -> 'a3500-IMG_0001'
      'a0939-IMG_0262.jpg'           -> 'a0939-IMG_0262'
      'a0001-jmac_DSC1459'           -> 'a0001-jmac_DSC1459' (already)
      'synth-warm-1234'              -> 'synth-warm-1234' (preserved as-is)

    Falls back to the original source_image if no aXXXX pattern matches.
    """
    s = record.get('source_image', '')
    # Preserve synthetic samples (they go to train-only via build_data)
    if s.startswith('synth-'):
        return s
    s_no_ext = re.sub(r'\.(jpg|jpeg|png|tif|tiff|webp)$', '', s,
                       flags=re.IGNORECASE)
    # Strip leading 'fivek-X-' prefix (X is a-e for the 5 FiveK experts)
    s_clean = re.sub(r'^fivek-[a-e]-', '', s_no_ext, flags=re.IGNORECASE)
    # Match aXXXX-... pattern (FiveK canonical naming)
    m = re.search(r'(a\d{4,5}-[\w_\-]+)', s_clean)
    if m:
        return m.group(1)
    # Fallback: also try orig_path basename
    op = record.get('orig_path', '')
    if op:
        base = Path(op).stem
        m = re.search(r'(a\d{4,5}-[\w_\-]+)', base)
        if m:
            return m.group(1)
    return s_no_ext


def parse_kv_pairs(s: str) -> dict:
    """Parse 'key1=val1,key2=val2' string."""
    if not s:
        return {}
    out = {}
    for kv in s.split(','):
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True,
                    help='List of "tag:path" pairs, e.g. '
                         'clean:path/to/a.jsonl pseudo:path/to/b.jsonl')
    ap.add_argument('--out_jsonl', required=True)
    ap.add_argument('--upsample', type=str, default='',
                    help='Per-tag upsampling factor, e.g. '
                         '"clean=1,pseudo=40" (default: 1 for all)')
    ap.add_argument('--tier_filter', type=str,
                    default='A excellent,B good,C acceptable',
                    help='Comma-separated quality_tier values to keep')
    ap.add_argument('--cap_per_tag', type=str, default='',
                    help='Per-tag max records, e.g. "clean=20000"')
    ap.add_argument('--no_normalize_source_image', action='store_true',
                    help='Skip source_image normalization (debug only)')
    args = ap.parse_args()

    upsample = {k: int(v) for k, v in parse_kv_pairs(args.upsample).items()}
    cap = {k: int(v) for k, v in parse_kv_pairs(args.cap_per_tag).items()}
    tier_set = set(t.strip() for t in args.tier_filter.split(','))

    out_records = []
    stats = {}
    per_tag_action = defaultdict(Counter)
    per_tag_tier = defaultdict(Counter)
    stem_to_tags = defaultdict(set)

    for inp in args.inputs:
        if ':' not in inp:
            print(f'[skip] bad input format (missing tag): {inp}')
            continue
        tag, path_str = inp.split(':', 1)
        path = Path(path_str)
        if not path.exists():
            print(f'[skip] file not found: {path}')
            continue
        u = upsample.get(tag, 1)
        c = cap.get(tag, None)

        n_in = 0
        n_filtered = 0
        n_kept = 0
        records_kept = []
        with path.open(encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_in += 1
                tier = rec.get('quality_tier', '')
                if tier not in tier_set:
                    n_filtered += 1
                    continue
                action = rec.get('tone_target') or rec.get('action', 'unknown')
                rec['_dataset_tag'] = tag
                rec['_orig_source_image'] = rec.get('source_image', '')
                if not args.no_normalize_source_image:
                    rec['source_image'] = extract_fivek_stem(rec)
                stem_to_tags[rec['source_image']].add(tag)
                per_tag_action[tag][action] += 1
                per_tag_tier[tag][tier] += 1
                records_kept.append(rec)
                n_kept += 1
                if c is not None and n_kept >= c:
                    break

        n_emitted = 0
        for rec in records_kept:
            for _ in range(u):
                out_records.append(rec)
                n_emitted += 1

        stats[tag] = {
            'path': str(path),
            'n_input': n_in,
            'n_filtered_by_tier': n_filtered,
            'n_kept': n_kept,
            'upsample_factor': u,
            'n_emitted': n_emitted,
        }
        print(f'[{tag}] {path.name}: '
              f'{n_in} input → {n_kept} kept (after tier filter, '
              f'{n_filtered} filtered) → {n_emitted} emitted (×{u})')

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f'\n[ok] wrote {len(out_records)} records to {out_path}')

    # Composition summary
    print('\nPer-tag composition:')
    for tag, s in stats.items():
        print(f'  {tag:>10} : {s["n_emitted"]:>6} records '
              f'(actions: ' +
              ', '.join(f'{a}={n}' for a, n in
                         sorted(per_tag_action[tag].items())) + ')')

    # Image overlap analysis
    n_unique_stems = len(stem_to_tags)
    n_overlap = sum(1 for tags in stem_to_tags.values() if len(tags) > 1)
    print(f'\nUnique FiveK images: {n_unique_stems}')
    print(f'  in 1 dataset only: {n_unique_stems - n_overlap}')
    print(f'  in >1 dataset (joint variants): {n_overlap}')

    # Save sidecar summary
    summary = {
        'out_jsonl': str(out_path),
        'n_total_records': len(out_records),
        'n_unique_stems': n_unique_stems,
        'n_stems_in_multiple_datasets': n_overlap,
        'per_tag': stats,
        'per_tag_action_counts': {tag: dict(c)
                                   for tag, c in per_tag_action.items()},
        'per_tag_tier_counts': {tag: dict(c)
                                 for tag, c in per_tag_tier.items()},
    }
    summary_path = out_path.parent / 'merge_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                             encoding='utf-8')
    print(f'[ok] composition summary: {summary_path}')


if __name__ == '__main__':
    main()
