"""Split Path Y master captions JSON into per-action files for AutoDL batch.

Input:  data/teacher_edits_fivek_pathY_1500.json
Output: data/pathY_captions/pathY_contrast.json
        data/pathY_captions/pathY_saturation.json
        data/pathY_captions/pathY_shadows.json
        data/pathY_captions/pathY_highlights.json
        data/pathY_captions/pathY_wb.json

Usage:
    python tools/data/data_prep/split_pathY_per_action.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    master = PROJECT_ROOT / 'data/teacher_edits_fivek_pathY_1500.json'
    with open(master, encoding='utf-8') as f:
        doc = json.load(f)

    samples = doc['samples']
    meta = doc['metadata']
    print(f'[INFO] loaded {len(samples)} samples from {master.name}')

    out_dir = PROJECT_ROOT / 'data' / 'pathY_captions'
    out_dir.mkdir(parents=True, exist_ok=True)

    by_action = {}
    for s in samples:
        action = s['tone_target']
        by_action.setdefault(action, []).append(s)

    for action, action_samples in sorted(by_action.items()):
        out_doc = {
            'metadata': {
                **meta,
                'action': action,
                'n_samples': len(action_samples),
                'split_from': str(master),
            },
            'samples': action_samples,
        }
        out_path = out_dir / f'pathY_{action}.json'
        out_path.write_text(
            json.dumps(out_doc, ensure_ascii=False, indent=2),
            encoding='utf-8')
        print(f'  {action:12s} {len(action_samples):4d} samples -> {out_path.name}')

    print(f'\n[OK] {len(by_action)} per-action files in {out_dir}')


if __name__ == '__main__':
    main()
