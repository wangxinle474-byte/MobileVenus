"""Path Y: build +1500 FireRed captions JSON for AutoDL batch generation.

Picks 300 NEW FiveK images (not in existing 499 used in fivek_500_master),
generates 5 actions × 300 = 1500 caption samples for FireRed inference.

Output: data/teacher_edits_fivek_pathY_1500.json
        + data/teacher_edits_fivek_pathY_300_sources.txt (image-list for sync)

Usage:
    python tools/data/data_prep/build_pathY_1500_captions.py
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIVEK_DIR = Path('E:/Data/dataset/fivek_jpeg')
EXISTING_JSONL = PROJECT_ROOT / 'outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl'

# Match existing per-action template (select_fivek_per_action.py)
ACTIONS = [
    ('contrast',   'Increase contrast. Keep the original composition and subject unchanged.'),
    ('saturation', 'Enhance saturation. Keep the original composition and subject unchanged.'),
    ('shadows',    'Lift shadows. Keep the original composition and subject unchanged.'),
    ('highlights', 'Recover highlights. Keep the original composition and subject unchanged.'),
    ('wb',         'Apply warmer white balance. Keep the original composition and subject unchanged.'),
]

N_NEW_IMAGES = 300
SEED = 42


def main():
    # 1. Load existing 499 used source images
    used = set()
    for line in EXISTING_JSONL.read_text(encoding='utf-8').strip().split('\n'):
        rec = json.loads(line)
        used.add(rec['source_image'])
    print(f'[INFO] {len(used)} source images already used in 500-master')

    # 2. List FiveK candidates not yet used (only standard a####-XXX.jpg format)
    FIVEK_RE = re.compile(r'^a(\d{4})-.*\.jpg$')
    all_jpegs = sorted(
        p.name for p in FIVEK_DIR.glob('*.jpg')
        if FIVEK_RE.match(p.name))
    candidates = [n for n in all_jpegs if n not in used]
    print(f'[INFO] {len(candidates)} candidates available '
          f'({len(all_jpegs)} valid a####-*.jpg - {len(used)} used)')
    assert len(candidates) >= N_NEW_IMAGES, \
        f'not enough candidates: {len(candidates)} < {N_NEW_IMAGES}'

    # 3. Sample 300 deterministically
    rng = random.Random(SEED)
    picked = sorted(rng.sample(candidates, N_NEW_IMAGES))
    print(f'[INFO] picked {len(picked)} new source images (seed={SEED})')

    # 4. Build samples: 5 actions × 300 images = 1500
    samples = []
    for action_name, caption in ACTIONS:
        for i, img in enumerate(picked):
            idx = int(FIVEK_RE.match(img).group(1))  # 'a0123-...' → 123
            samples.append({
                'rank': i + 1,
                'idx': idx,
                'source_image': img,
                'orig_path': f'E:/Data/dataset/fivek_jpeg/{img}',
                'venus_suggestion': '',  # not needed for FireRed
                'new_caption': caption,
                'tone_target': action_name,
                'score': 5,
            })
    print(f'[INFO] built {len(samples)} samples '
          f'({len(ACTIONS)} actions × {N_NEW_IMAGES} images)')

    # 5. Write outputs
    out_json = PROJECT_ROOT / 'data/teacher_edits_fivek_pathY_1500.json'
    out_doc = {
        'metadata': {
            'purpose': 'Path Y: +1500 FireRed pseudo-labels (300 new images × 5 actions)',
            'n_samples': len(samples),
            'n_unique_sources': N_NEW_IMAGES,
            'actions': [a for a, _ in ACTIONS],
            'seed': SEED,
            'source_pool': str(FIVEK_DIR),
            'excluded_existing': str(EXISTING_JSONL),
            'caption_style': 'instruction-style (matches fivek_per_action master)',
            'firered_settings': 'steps=8, cfg=1.0, seed=42 (matches 500-master)',
        },
        'samples': samples,
    }
    out_json.write_text(
        json.dumps(out_doc, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(f'[OK] wrote {out_json} ({out_json.stat().st_size/1024:.1f} KB)')

    # 6. Image-list for AutoDL sync (rsync/scp)
    out_list = PROJECT_ROOT / 'data/teacher_edits_fivek_pathY_300_sources.txt'
    out_list.write_text('\n'.join(picked) + '\n', encoding='utf-8')
    print(f'[OK] wrote {out_list} ({len(picked)} filenames)')

    # Stats summary
    print('\n=== Path Y batch summary ===')
    print(f'  new source images : {N_NEW_IMAGES}')
    print(f'  actions           : {len(ACTIONS)} ({[a for a, _ in ACTIONS]})')
    print(f'  total samples     : {len(samples)}')
    print(f'  estimated AutoDL runtime @ ~12s/img : '
          f'{len(samples) * 12 / 3600:.1f}h '
          f'({len(samples) * 12 / 60:.0f} min)')


if __name__ == '__main__':
    main()
