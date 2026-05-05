"""\u8bca\u65ad fivek_expert_settings \u4e2d\u7684\u53c2\u6570\u5b8c\u5907\u5ea6."""
import json
from collections import Counter

PATH = r'E:\Data\dataset\fivek_expert\fivek_expert_settings.json'
SENTINEL = -999999

d = json.load(open(PATH, encoding='utf-8'))
samples = d['samples']

import sys
EXPERT_FILTER = sys.argv[1] if len(sys.argv) > 1 else 'real'  # 'real' or 'default'

if EXPERT_FILTER == 'default':
    real_experts = [s for s in samples if s['expert'] == 'default']
    print(f'DEFAULT samples: {len(real_experts)}')
else:
    real_experts = [s for s in samples if s['expert'] != 'default']
    print(f'real expert samples: {len(real_experts)}')
print()

# \u68c0\u67e5\u6bcf\u4e2a\u5173\u952e\u53c2\u6570\u7684\u6709\u6548\u7387 (\u4e0d\u662f sentinel)
keys_to_check = ['exposure', 'temperature', 'tint', 'contrast', 'brightness',
                 'shadows', 'highlights', 'saturation', 'vibrance', 'clarity']

print(f'{"key":15s}  {"valid":>6s}  {"sentinel":>8s}  {"unique":>6s}  '
      f'{"min":>8s}  {"max":>8s}  {"mean":>8s}')
for k in keys_to_check:
    vals = [s['raw_settings'].get(k, 0) for s in real_experts]
    valid = [v for v in vals if v != SENTINEL]
    senti = sum(1 for v in vals if v == SENTINEL)
    uniq = len(set(valid))
    if valid:
        print(f'{k:15s}  {len(valid):>6d}  {senti:>8d}  {uniq:>6d}  '
              f'{min(valid):>+8.1f}  {max(valid):>+8.1f}  '
              f'{sum(valid) / len(valid):>+8.2f}')
    else:
        print(f'{k:15s}  no valid values')

# \u4e0d\u540c expert \u7684\u8868\u73b0
print()
print('=== expert \u5206\u522b\u770b ===')
for exp in set(s['expert'] for s in real_experts):
    sub = [s for s in real_experts if s['expert'] == exp]
    print(f'\n{exp[:20]}... ({len(sub)} records)')
    for k in ['contrast', 'shadows', 'highlights', 'saturation', 'vibrance']:
        vals = [s['raw_settings'].get(k, 0) for s in sub if s['raw_settings'].get(k, 0) != SENTINEL]
        if vals:
            print(f'  {k:12s}  n={len(vals):4d}  '
                  f'mean={sum(vals) / len(vals):+7.2f}  '
                  f'range=[{min(vals):+5.0f}, {max(vals):+5.0f}]')
