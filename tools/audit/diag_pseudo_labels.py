"""快速检查 pseudo_labels.jsonl 中 P_inferred 的完整性 (NaN / 缺失 / 范围)."""
import json
import math
import sys
from pathlib import Path
from collections import Counter

JSONL = Path('outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl')
P_KEYS = ['white_balance', 'brightness', 'contrast',
          'shadows', 'highlights', 'saturation', 'clarity']


def main():
    samples = [json.loads(l) for l in JSONL.read_text(encoding='utf-8').splitlines()]
    print(f'total samples: {len(samples)}')

    no_p = 0
    missing_keys = Counter()
    nan_count = Counter()
    none_count = Counter()
    ranges = {k: [float('inf'), float('-inf')] for k in P_KEYS}

    for i, s in enumerate(samples):
        P = s.get('P_inferred')
        if P is None or not isinstance(P, dict):
            no_p += 1
            continue
        for k in P_KEYS:
            if k not in P:
                missing_keys[k] += 1
                continue
            v = P[k]
            if v is None:
                none_count[k] += 1
                continue
            if isinstance(v, float) and math.isnan(v):
                nan_count[k] += 1
                continue
            try:
                vf = float(v)
                ranges[k][0] = min(ranges[k][0], vf)
                ranges[k][1] = max(ranges[k][1], vf)
            except (TypeError, ValueError):
                print(f'[bad] idx={i} key={k} value={v!r}')

    print(f'\nno P_inferred at all: {no_p}')
    print(f'\nmissing keys (per-key count):')
    for k, c in missing_keys.most_common():
        print(f'  {k}: {c}')
    print(f'\nNone values (per-key count):')
    for k, c in none_count.most_common():
        print(f'  {k}: {c}')
    print(f'\nNaN values (per-key count):')
    for k, c in nan_count.most_common():
        print(f'  {k}: {c}')
    print(f'\nValue ranges:')
    for k, (lo, hi) in ranges.items():
        print(f'  {k}: [{lo:.2f}, {hi:.2f}]')

    print(f'\nfirst 3 P_inferred:')
    for s in samples[:3]:
        print(f'  action={s.get("action")} tier={s.get("quality_tier")}: {s.get("P_inferred")}')


if __name__ == '__main__':
    main()
