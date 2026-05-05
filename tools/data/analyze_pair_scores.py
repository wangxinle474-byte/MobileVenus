"""看 pair_scores.json 的统计和具体样本 (AutoDL 上跑)."""
import json
import sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/outputs/pair_scores.json'
d = json.load(open(PATH))
results = d['results']

print('=' * 60)
print(f'Total: {len(results)}')
print(f'Improved (delta>=0.5): {d.get("num_improved", "?")}')

# 统计分数分布
scores_o = [r['score_orig'] for r in results if r['score_orig'] is not None]
scores_e = [r['score_edit'] for r in results if r['score_edit'] is not None]
deltas = [r.get('delta') for r in results if r.get('delta') is not None]

if scores_o:
    print(f'\norig scores: n={len(scores_o)}  mean={sum(scores_o)/len(scores_o):.2f}  '
          f'min={min(scores_o):.1f}  max={max(scores_o):.1f}')
    print(f'  distr: {dict(Counter(round(s) for s in scores_o))}')
if scores_e:
    print(f'edit scores: n={len(scores_e)}  mean={sum(scores_e)/len(scores_e):.2f}  '
          f'min={min(scores_e):.1f}  max={max(scores_e):.1f}')
    print(f'  distr: {dict(Counter(round(s) for s in scores_e))}')
if deltas:
    n_up = sum(1 for x in deltas if x > 0)
    n_zero = sum(1 for x in deltas if x == 0)
    n_down = sum(1 for x in deltas if x < 0)
    print(f'deltas:  n={len(deltas)}  mean={sum(deltas)/len(deltas):+.2f}  '
          f'up={n_up}  zero={n_zero}  down={n_down}')

print()
print('=' * 60)
print('前 5 个样本:')
for r in results[:5]:
    print(f'  [{r["idx"]}] {r["image"]}')
    print(f'    orig={r["score_orig"]}  raw={r["score_raw_orig"]!r}')
    print(f'    edit={r["score_edit"]}  raw={r["score_raw_edit"]!r}')
    print(f'    prompt={r["edit_prompt"][:80]!r}')
    print()
