from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

DEFAULT_ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']
ALL_ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb', 'brightness', 'clarity']
PARAM_KEYS = [
    'white_balance',
    'brightness',
    'contrast',
    'shadows',
    'highlights',
    'saturation',
    'clarity',
]


def convert_record(record: dict, actions: set[str]) -> dict | None:
    action = record.get('action') or record.get('tone_target')
    if action not in actions:
        return None
    mean_params = record.get('mean_params') or {}
    inferred = {key: 0.0 for key in PARAM_KEYS}
    inferred['white_balance'] = float(mean_params.get('white_balance', 6000.0))
    inferred['brightness'] = float(mean_params.get('brightness', mean_params.get('ev_compensation', 0.0)))
    inferred['contrast'] = float(mean_params.get('contrast', 0.0))
    inferred['shadows'] = float(mean_params.get('shadows', 0.0))
    inferred['highlights'] = float(mean_params.get('highlights', 0.0))
    inferred['saturation'] = float(mean_params.get('saturation', 0.0))
    inferred['clarity'] = float(mean_params.get('clarity', 0.0))
    out = dict(record)
    out['action'] = action
    out['tone_target'] = action
    out['quality_tier'] = out.get('quality_tier') or 'B good'
    out['P_inferred'] = inferred
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='outputs/firered_v12a_existing/pseudo_labels.jsonl')
    parser.add_argument('--out_dir', default='outputs/firered_v11a_existing_5actions')
    parser.add_argument('--actions', nargs='+', default=DEFAULT_ACTIONS,
                        choices=ALL_ACTIONS)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'pseudo_labels.jsonl'
    actions = set(args.actions)
    records = []
    for line in in_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        converted = convert_record(json.loads(line), actions)
        if converted is not None:
            records.append(converted)
    with out_path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    summary = {
        'input': str(in_path),
        'output': str(out_path),
        'records': len(records),
        'selected_actions': args.actions,
        'actions': dict(sorted(Counter(record['action'] for record in records).items())),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
