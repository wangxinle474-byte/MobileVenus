from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb', 'brightness', 'clarity']


PARAM_IDENTITY = {
    'ev_compensation': 0.0,
    'white_balance': 6000.0,
    'contrast': 0.0,
    'shadows': 0.0,
    'highlights': 0.0,
    'saturation': 0.0,
}


ACTION_PARAM_HINT = {
    'contrast': {'contrast': 35.0},
    'saturation': {'saturation': 30.0},
    'shadows': {'shadows': 35.0},
    'highlights': {'highlights': -35.0},
    'wb': {'white_balance': 7000.0},
    'brightness': {'ev_compensation': 0.7},
    'clarity': {'contrast': 18.0, 'shadows': -8.0, 'highlights': 8.0},
}


def _load_samples(data_dir: Path, action: str) -> dict[int, dict]:
    path = data_dir / f'teacher_edits_fivek_full_{action}.json'
    with path.open(encoding='utf-8') as f:
        cfg = json.load(f)
    return {int(s['idx']): s for s in cfg['samples']}


def _resolve_orig(sample: dict, input_dir: Path) -> Path | None:
    candidates = []
    if sample.get('orig_path'):
        candidates.append(Path(sample['orig_path']))
    if sample.get('source_image'):
        candidates.append(input_dir / sample['source_image'])
    idx = int(sample['idx'])
    candidates.extend([input_dir / f'{idx:04d}.png', input_dir / f'{idx:04d}_orig.png'])
    for p in candidates:
        if p.exists():
            return p
    return None


def _params_for_action(action: str) -> dict:
    params = dict(PARAM_IDENTITY)
    params.update(ACTION_PARAM_HINT.get(action, {}))
    return params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--input_dir', default='E:/Data/dataset/fivek_jpeg')
    ap.add_argument('--teacher_dir', default='outputs/teacher_edits/fivek_full')
    ap.add_argument('--out_dir', default='outputs/firered_v12a_existing')
    ap.add_argument('--actions', nargs='+', default=ACTIONS)
    ap.add_argument('--target_abs', action='store_true')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    input_dir = Path(args.input_dir)
    teacher_dir = Path(args.teacher_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    missing_caption = 0
    missing_orig = 0
    counts = Counter()

    for action in args.actions:
        by_idx = _load_samples(data_dir, action)
        action_dir = teacher_dir / action
        for target in sorted(action_dir.glob('*.png')):
            try:
                idx = int(target.stem)
            except ValueError:
                continue
            sample = by_idx.get(idx)
            if sample is None:
                missing_caption += 1
                continue
            orig = _resolve_orig(sample, input_dir)
            if orig is None:
                missing_orig += 1
                continue
            target_path = target.resolve() if args.target_abs else target.as_posix()
            record = {
                'idx': idx,
                'source_image': sample.get('source_image', orig.name),
                'orig_path': str(orig).replace('\\', '/'),
                'target_path': str(target_path).replace('\\', '/'),
                'caption': sample.get('new_caption', ''),
                'tone_target': action,
                'action': action,
                'quality_tier': 'B good',
                'mean_params': _params_for_action(action),
            }
            records.append(record)
            counts[action] += 1

    out_jsonl = out_dir / 'pseudo_labels.jsonl'
    with out_jsonl.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    summary = {
        'n_records': len(records),
        'counts': dict(counts),
        'missing_caption': missing_caption,
        'missing_orig': missing_orig,
        'actions': args.actions,
        'teacher_dir': str(teacher_dir),
        'input_dir': str(input_dir),
        'jsonl': str(out_jsonl),
    }
    with (out_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
