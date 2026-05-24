"""Generate cleaner FireRed wb caption JSON files for warm + cool directions.

Compared to the original `Apply warmer white balance` prompt, the new prompts:
  1. Cover BOTH warm and cool directions
  2. Explicitly forbid brightness/contrast/saturation changes
  3. Specify the exact effect (warm = orange/yellow tint, cool = blue tint)

Output:
  data/teacher_edits_fivek_wb_clean_warm.json   (25 samples, warm prompt)
  data/teacher_edits_fivek_wb_clean_cool.json   (25 samples, cool prompt)
"""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
from pathlib import Path

# Use the same scoring logic as select_fivek_per_action.py
sys.path.insert(0, str(Path(__file__).parent))
from select_fivek_teacher_candidates import (  # noqa: E402
    score_caption, has_negative, has_suggestion, is_too_positive,
)


# ============================================================
# Cleaner prompts (constrained to WB only)
# ============================================================
WARM_PROMPT = (
    "Apply a warm white balance shift to make the photo look like it was "
    "taken under tungsten or sunset light (more orange/yellow tone). "
    "ONLY adjust the color temperature (white balance) — do NOT change "
    "the brightness, contrast, saturation, sharpness, or any other aspect "
    "of the image. Keep the composition, subject, and exposure identical."
)

COOL_PROMPT = (
    "Apply a cool white balance shift to make the photo look like it was "
    "taken under shade or overcast daylight (more blue tone). "
    "ONLY adjust the color temperature (white balance) — do NOT change "
    "the brightness, contrast, saturation, sharpness, or any other aspect "
    "of the image. Keep the composition, subject, and exposure identical."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='data/fivek_venus_labels.json',
                    help='FiveK Venus labels JSON (with `labels` list)')
    ap.add_argument('--n_warm', type=int, default=25)
    ap.add_argument('--n_cool', type=int, default=25)
    ap.add_argument('--min_score', type=int, default=4)
    ap.add_argument('--seed', type=int, default=44)
    ap.add_argument('--exclude_from_teacher_jsons', nargs='*',
                    default=['data/teacher_edits_fivek_wb_20.json',
                             'data/teacher_edits_fivek_wb_extend80.json'],
                    help='Exclude images already used in existing wb sets')
    ap.add_argument('--out_dir', default='data')
    args = ap.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)

    with open(args.source, encoding='utf-8') as f:
        d = json.load(f)
    labels = d['labels']
    print(f'[INFO] loaded {len(labels)} fivek venus labels')

    # Collect excluded images from existing teacher JSONs to avoid val leakage
    excluded = set()
    for jpath in args.exclude_from_teacher_jsons:
        jp = Path(jpath)
        if not jp.exists():
            print(f'[WARN] teacher JSON not found: {jp}')
            continue
        with open(jp, encoding='utf-8') as f:
            jd = json.load(f)
        for s in jd.get('samples', []):
            img = s.get('source_image')
            if img:
                excluded.add(img)
        print(f'[INFO] excluded +{len(jd.get("samples", []))} from {jp.name}')
    print(f'[INFO] total excluded images: {len(excluded)}')

    # Score and filter candidates (same logic as select_fivek_per_action.py)
    candidates = []
    for lbl in labels:
        suggestion = lbl.get('raw_response', '')
        if not suggestion or len(suggestion) < 50:
            continue
        if is_too_positive(suggestion):
            continue
        if not has_suggestion(suggestion):
            continue
        if has_negative(suggestion):
            continue
        if lbl['image'] in excluded:
            continue
        score, hits = score_caption(suggestion)
        if score < args.min_score:
            continue
        candidates.append({
            'image': lbl['image'],
            'raw_response': suggestion,
            'score': score,
        })

    print(f'[FILTER] {len(candidates)} candidates with score≥{args.min_score}')
    n_need = args.n_warm + args.n_cool
    if len(candidates) < n_need:
        raise RuntimeError(f'Need {n_need} candidates, got {len(candidates)}')

    # Sort by score, then deterministic shuffle
    candidates.sort(key=lambda x: (-x['score'], x['image']))
    top = candidates[:n_need]
    rng = random.Random(args.seed)
    rng.shuffle(top)

    warm_pool = top[:args.n_warm]
    cool_pool = top[args.n_warm:args.n_warm + args.n_cool]

    def _build_samples(pool, caption, action='wb'):
        out = []
        for rank, c in enumerate(pool, start=1):
            img = c['image']
            m = re.match(r'a0*(\d+)-', img)
            idx = int(m.group(1)) if m else -1
            entry = {
                'rank': rank,
                'idx': idx,
                'source_image': img,
                'orig_path': f'E:/Data/dataset/fivek_jpeg/{img}',
                'venus_suggestion': c['raw_response'][:300],
                'new_caption': caption,
                'tone_target': action,
                'score': c['score'],
            }
            out.append(entry)
        return out

    warm_samples = _build_samples(warm_pool, WARM_PROMPT)
    cool_samples = _build_samples(cool_pool, COOL_PROMPT)

    warm_out = {
        'metadata': {
            'purpose': 'FiveK teacher edit single-action: wb (CLEAN warm prompt)',
            'action': 'wb_clean_warm',
            'caption': WARM_PROMPT,
            'n_samples': len(warm_samples),
            'source': args.source,
            'seed': args.seed,
            'group_index': 4,
            'note': 'v2 cleaner prompt with explicit "ONLY WB" constraint',
        },
        'samples': warm_samples,
    }
    cool_out = {
        'metadata': {
            'purpose': 'FiveK teacher edit single-action: wb (CLEAN cool prompt)',
            'action': 'wb_clean_cool',
            'caption': COOL_PROMPT,
            'n_samples': len(cool_samples),
            'source': args.source,
            'seed': args.seed,
            'group_index': 4,
            'note': 'v2 cleaner prompt with explicit "ONLY WB" constraint',
        },
        'samples': cool_samples,
    }

    warm_path = out_dir / 'teacher_edits_fivek_wb_clean_warm.json'
    cool_path = out_dir / 'teacher_edits_fivek_wb_clean_cool.json'
    warm_path.write_text(json.dumps(warm_out, ensure_ascii=False, indent=2),
                          encoding='utf-8')
    cool_path.write_text(json.dumps(cool_out, ensure_ascii=False, indent=2),
                          encoding='utf-8')

    print(f'\nWarm prompt:\n  "{WARM_PROMPT}"')
    print(f'Cool prompt:\n  "{COOL_PROMPT}"')
    print(f'\nWritten:')
    print(f'  {warm_path}  ({len(warm_samples)} samples)')
    print(f'  {cool_path}  ({len(cool_samples)} samples)')


if __name__ == '__main__':
    main()
