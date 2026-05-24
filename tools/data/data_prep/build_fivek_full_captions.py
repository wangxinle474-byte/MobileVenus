"""Build captions JSON for ALL 5000 FiveK images × 5 actions = 25,000 samples.

Output: data/teacher_edits_fivek_full_25k.json
        (split into per-action JSONs for parallel runs)

Usage:
    python tools/data/data_prep/build_fivek_full_captions.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIVEK_DIR = Path('E:/Data/dataset/fivek_jpeg')

ACTIONS = [
    ('contrast',   'Increase contrast. Keep the original composition and subject unchanged.'),
    ('saturation', 'Enhance saturation. Keep the original composition and subject unchanged.'),
    ('shadows',    'Lift shadows. Keep the original composition and subject unchanged.'),
    ('highlights', 'Recover highlights. Keep the original composition and subject unchanged.'),
    ('wb',         'Apply warmer white balance. Keep the original composition and subject unchanged.'),
]

FIVEK_RE = re.compile(r'^a(\d{4})-.*\.jpg$')


def main():
    # List all valid FiveK images
    all_jpegs = sorted(
        p.name for p in FIVEK_DIR.glob('*.jpg')
        if FIVEK_RE.match(p.name))
    print(f'[INFO] {len(all_jpegs)} FiveK images found')

    # Build combined JSON + per-action JSONs
    all_samples = []
    per_action_samples = {a: [] for a, _ in ACTIONS}

    for action_name, caption in ACTIONS:
        for img_name in all_jpegs:
            idx = int(FIVEK_RE.match(img_name).group(1))
            sample = {
                'rank': len(per_action_samples[action_name]) + 1,
                'idx': idx,
                'source_image': img_name,
                'orig_path': f'E:/Data/dataset/fivek_jpeg/{img_name}',
                'new_caption': caption,
                'tone_target': action_name,
                'score': 5,
            }
            all_samples.append(sample)
            per_action_samples[action_name].append(sample)

    print(f'[INFO] {len(all_samples)} total samples '
          f'({len(ACTIONS)} actions × {len(all_jpegs)} images)')

    # Write combined JSON
    out_dir = PROJECT_ROOT / 'data'
    out_dir.mkdir(exist_ok=True)

    combined_doc = {
        'metadata': {
            'purpose': 'FiveK full: 5000 images × 5 actions = 25,000 FireRed captions',
            'n_samples': len(all_samples),
            'n_images': len(all_jpegs),
            'actions': [a for a, _ in ACTIONS],
            'source_pool': str(FIVEK_DIR),
            'firered_settings': 'FireRed 1.1 API, Lightning LoRA, cfg=1.0, seed=42',
        },
        'samples': all_samples,
    }
    combined_path = out_dir / 'teacher_edits_fivek_full_25k.json'
    combined_path.write_text(
        json.dumps(combined_doc, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(f'[OK] {combined_path} ({combined_path.stat().st_size / 1024 / 1024:.1f} MB)')

    # Write per-action JSONs (for parallel runs)
    for action_name, _ in ACTIONS:
        action_doc = {
            'metadata': {
                'purpose': f'FiveK full: {action_name} (5000 images)',
                'n_samples': len(per_action_samples[action_name]),
                'action': action_name,
            },
            'samples': per_action_samples[action_name],
        }
        action_path = out_dir / f'teacher_edits_fivek_full_{action_name}.json'
        action_path.write_text(
            json.dumps(action_doc, ensure_ascii=False, indent=2),
            encoding='utf-8')
        print(f'  [OK] {action_path.name} ({len(per_action_samples[action_name])} samples)')

    print(f'\n建议按 action 分批跑:')
    for action_name, _ in ACTIONS:
        print(f'  python tools/data/editor_models/run_firered_online.py '
              f'--captions data/teacher_edits_fivek_full_{action_name}.json '
              f'--input_dir E:/Data/dataset/fivek_jpeg '
              f'--out_dir outputs/teacher_edits/fivek_full/{action_name} '
              f'--resume')


if __name__ == '__main__':
    main()
