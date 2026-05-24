"""Merge new wb_clean (warm+cool) pseudo_labels into a single JSONL
ready for train_lut.py --extra_jsonl flag.

Adds the missing 'action' and 'quality_tier' fields and filters by
pixel_l1 (drop FAIL samples with L1 > 0.15 = label too noisy to learn).
"""
from __future__ import annotations
import json
from pathlib import Path


L1_THRESHOLD = 0.15   # drop samples with pixel_l1 above this (= label too noisy)


def patch_sample(s: dict) -> dict:
    """Add missing 'action' and 'quality_tier'."""
    s = dict(s)
    # tone_target is 'wb' from caption file
    s['action'] = s.get('tone_target', 'wb')
    # Assign quality tier based on pixel_l1
    l1 = s.get('pixel_l1', 1.0)
    if l1 < 0.05:
        s['quality_tier'] = 'B good'
    elif l1 < 0.10:
        s['quality_tier'] = 'C acceptable'
    else:
        s['quality_tier'] = 'D dirty'   # will be filtered out (not in tier_filter)
    s['verdict'] = ('PASS' if l1 < 0.04
                     else ('WARN' if l1 < 0.10 else 'FAIL'))
    return s


def main():
    in_paths = [
        Path('outputs/inverse_fit_pilot/wb_clean_warm/pseudo_labels.jsonl'),
        Path('outputs/inverse_fit_pilot/wb_clean_cool/pseudo_labels.jsonl'),
    ]
    out_path = Path('outputs/inverse_fit_pilot/wb_clean_merged/pseudo_labels.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for p in in_paths:
        if not p.exists():
            print(f'[WARN] not found: {p}')
            continue
        with open(p, encoding='utf-8') as f:
            samples = [json.loads(l) for l in f if l.strip()]
        all_samples.extend(samples)
        print(f'  loaded {len(samples)} from {p.name}')

    patched = [patch_sample(s) for s in all_samples]

    # Filter by L1 threshold (drop label-noisy samples)
    kept = [s for s in patched if s.get('pixel_l1', 1.0) <= L1_THRESHOLD]
    dropped = [s for s in patched if s.get('pixel_l1', 1.0) > L1_THRESHOLD]

    print(f'\n[FILTER] L1 threshold = {L1_THRESHOLD}')
    print(f'  kept   : {len(kept):2d} / {len(patched):2d}')
    print(f'  dropped: {len(dropped):2d} / {len(patched):2d}')

    # Tier breakdown of kept
    from collections import Counter
    tier_counts = Counter(s['quality_tier'] for s in kept)
    print(f'  kept tiers: {dict(tier_counts)}')

    # Write
    with open(out_path, 'w', encoding='utf-8') as f:
        for s in kept:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'\n[SAVED] {out_path}  ({len(kept)} samples)')


if __name__ == '__main__':
    main()
