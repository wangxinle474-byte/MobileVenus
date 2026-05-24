"""检查 per-action FireRed batch 是否完整, 把缺失的写到 retry JSON."""
import argparse
import json
from pathlib import Path

ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='20',
                    help='caption JSON 后缀, 例如 20 或 extend80')
    ap.add_argument('--retry_suffix', default=None,
                    help='retry JSON 后缀 (默认 = {suffix}_retry)')
    ap.add_argument('--out_root', default='outputs/teacher_edits/fivek_per_action')
    args = ap.parse_args()

    retry_suffix = args.retry_suffix or f'{args.suffix}_retry'
    total_missing = []
    for a in ACTIONS:
        cap_path = f'data/teacher_edits_fivek_{a}_{args.suffix}.json'
        out_dir = Path(f'{args.out_root}/{a}')
        if not Path(cap_path).exists():
            print(f'{a:<10s} : SKIP (caption {cap_path} missing)')
            continue
        with open(cap_path, encoding='utf-8') as f:
            cap = json.load(f)
        expected = {s['idx']: s for s in cap['samples']}
        actual = set()
        for p in out_dir.glob('*.png'):
            try:
                actual.add(int(p.stem))
            except ValueError:
                pass
        missing = [s for idx, s in expected.items() if idx not in actual]
        missing_ids = [m['idx'] for m in missing]
        print(f'{a:<10s} : {len(actual)}/{len(expected)}  missing: {missing_ids}')
        for m in missing:
            total_missing.append({'action': a, 'sample': m})

    if not total_missing:
        print('\nAll complete!')
        return

    print(f'\nTotal missing: {len(total_missing)}')
    by_action = {}
    for m in total_missing:
        by_action.setdefault(m['action'], []).append(m['sample'])
    for a, samples in by_action.items():
        out = {
            'metadata': {
                'purpose': f'retry missing for {a}',
                'n_samples': len(samples),
                'action': a,
            },
            'samples': samples,
        }
        retry_path = f'data/teacher_edits_fivek_{a}_{retry_suffix}.json'
        with open(retry_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f'  retry json: {retry_path} ({len(samples)} samples)')


if __name__ == '__main__':
    main()
