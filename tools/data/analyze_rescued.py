"""\u5206\u6790\u201c\u5dee\u56fe\u6539\u597d\u201d\u7684\u5206\u6570\u5206\u5e03\u3002

\u5206 3 \u4e2a\u89d2\u5ea6:
  1. delta>=0.5 \u7684 194 \u5bf9 (\u539f\u59cb\u8fc7\u6ee4\u5668\u4fdd\u7559\u7684)
  2. edit>=6 \u7684 633 \u5bf9 (\u65b0\u8fc7\u6ee4\u5668\u4fdd\u7559\u7684)
  3. \u4e24\u4e2a\u4ea4\u96c6, \u8868\u660e\u201c\u771f\u6b63\u5dee\u56fe\u6539\u5230\u597d\u201d
"""
import json
import numpy as np
from collections import Counter


def bucket_of(s):
    if s is None:
        return None
    for lo, hi, lab in [(0, 2, '0-2'), (2, 4, '2-4'), (4, 6, '4-6'),
                         (6, 8, '6-8'), (8, 10.01, '8-10')]:
        if lo <= s < hi:
            return lab


d = json.load(open('outputs/aug_ip2p_full_v1_reparsed.json', encoding='utf-8'))
results = d['results']
pairs = [r for r in results
         if r.get('score_orig_v2') is not None and r.get('score_edit_v2') is not None]


def hist(arr, label, bins=None):
    if bins is None:
        bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.01]
    print(f'\n[{label}]  n={len(arr)}, mean={np.mean(arr):.2f}, std={np.std(arr):.2f}')
    h, _ = np.histogram(arr, bins=bins)
    max_h = max(h) if max(h) > 0 else 1
    for i, n in enumerate(h):
        lo, hi = bins[i], bins[i + 1]
        bar = '=' * int(n / max_h * 50)
        print(f'  [{lo:>4.1f}, {hi:>4.1f})  {n:>4d}  {n/len(arr)*100:5.1f}%  {bar}')


def bucket_dist(arr, label):
    print(f'\n[{label}]  n={len(arr)}')
    bc = Counter(bucket_of(s) for s in arr)
    for b in ['0-2', '2-4', '4-6', '6-8', '8-10']:
        n = bc.get(b, 0)
        bar = '=' * int(n / max(len(arr), 1) * 50)
        print(f'  {b:6s}  {n:4d}  {n/len(arr)*100:5.1f}%  {bar}')


# ============================================================================
print('=' * 75)
print('# \u89d2\u5ea6 1: delta>=0.5 \u7684 194 \u5bf9 (\u539f\u59cb\u8fc7\u6ee4\u5668)')
print('=' * 75)
print('\u201c\u201d\u9636\u6539\u552e\u5f00\u201d \u53d6\u51fa\u4e2d\u7531\u539f\u56fe\u4e0a\u52a0\u63d0\u5236 0.5+ \u7684\u90a3\u90e8\u5206')
improved = [r for r in pairs
            if (r['score_edit_v2'] - r['score_orig_v2']) >= 0.5]
print(f'\n\u603b\u5bf9\u6570: {len(improved)}')

orig_imp = [r['score_orig_v2'] for r in improved]
edit_imp = [r['score_edit_v2'] for r in improved]
delta_imp = [e - o for o, e in zip(orig_imp, edit_imp)]

bucket_dist(orig_imp, '\u539f\u56fe\u8bc4\u5206\u5206\u5e03 (\u88ab\u9009\u4e2d\u7684 194 \u4e2a)')
bucket_dist(edit_imp, '\u7f16\u8f91\u540e\u8bc4\u5206\u5206\u5e03')

# delta \u5206\u5e03
print(f'\n[delta = edit - orig \u5206\u5e03]')
delta_buckets = [
    ('+0.5~+1', sum(1 for d in delta_imp if 0.5 <= d < 1)),
    ('+1~+2',   sum(1 for d in delta_imp if 1 <= d < 2)),
    ('+2~+3',   sum(1 for d in delta_imp if 2 <= d < 3)),
    ('+3~+4',   sum(1 for d in delta_imp if 3 <= d < 4)),
    ('+4~+5',   sum(1 for d in delta_imp if 4 <= d < 5)),
    ('+5+',     sum(1 for d in delta_imp if d >= 5)),
]
for lab, n in delta_buckets:
    bar = '=' * int(n / max(len(improved), 1) * 50)
    print(f'  delta {lab:8s}  {n:4d}  {n/len(improved)*100:5.1f}%  {bar}')


# ============================================================================
print('\n\n' + '=' * 75)
print('# \u89d2\u5ea6 2: edit>=6 \u7684 633 \u5bf9 (\u65b0\u8fc7\u6ee4\u5668 = \u6700\u7ec8\u8bad\u7ec3\u96c6)')
print('=' * 75)
final = [r for r in pairs if r['score_edit_v2'] >= 6.0]
print(f'\n\u603b\u5bf9\u6570: {len(final)}')

orig_f = [r['score_orig_v2'] for r in final]
edit_f = [r['score_edit_v2'] for r in final]
delta_f = [e - o for o, e in zip(orig_f, edit_f)]

bucket_dist(orig_f, '\u539f\u56fe\u8bc4\u5206\u5206\u5e03 (633 \u96c6)')
bucket_dist(edit_f, '\u7f16\u8f91\u540e\u8bc4\u5206\u5206\u5e03')

print(f'\n[delta \u5206\u5e03]')
n_up_strong = sum(1 for d in delta_f if d >= 2)
n_up_mild = sum(1 for d in delta_f if 0.5 <= d < 2)
n_same = sum(1 for d in delta_f if abs(d) < 0.5)
n_down = sum(1 for d in delta_f if d < -0.5)
total = len(final)
for lab, n in [('\u5927\u5e45\u4e0a\u5347 (+2 \u4ee5\u4e0a) "\u5dee\u56fe\u6539\u597d"', n_up_strong),
               ('\u8f7b\u5ea6\u4e0a\u5347 (+0.5~+2)        ',     n_up_mild),
               ('\u539f\u672c\u5c31\u597d \u6301\u5e73 (\u00b10.5)        ',  n_same),
               ('\u8f7b\u5ea6\u4e0b\u964d (-0.5~)         ',     n_down)]:
    bar = '=' * int(n / total * 50)
    print(f'  {lab:35s}  {n:4d}  {n/total*100:5.1f}%  {bar}')


# ============================================================================
print('\n\n' + '=' * 75)
print('# \u89d2\u5ea6 3: \u4ea4\u96c6 = \u8d77\u70b9\u5dee + \u7ec8\u70b9\u9ad8 (\u771f\u6b63\u7684\u201c\u5dee\u6539\u597d\u201d)')
print('=' * 75)

# \u201c\u5dee\u56fe\u6539\u597d\u201d \u7684\u4e25\u683c\u5b9a\u4e49: orig < 6 (\u8fbe\u4e0d\u5230\u597d) \u4e14 edit >= 6 (\u8fbe\u5230\u597d)
rescued = [r for r in pairs
           if r['score_orig_v2'] < 6.0 and r['score_edit_v2'] >= 6.0]
already_good = [r for r in pairs
                if r['score_orig_v2'] >= 6.0 and r['score_edit_v2'] >= 6.0]

print(f'\n\u4e25\u683c\u5b9a\u4e49 \u201c\u5dee\u56fe\u6539\u597d\u201d (orig<6 \u4e14 edit>=6) : {len(rescued)}')
print(f'\u539f\u672c\u5c31\u597d (orig>=6 \u4e14 edit>=6)            : {len(already_good)}')
print(f'\u4e24\u8005\u4e4b\u548c (\u5373 633)                       : {len(rescued) + len(already_good)}')
print(f'\u201c\u5dee\u56fe\u6539\u597d\u201d \u5360\u8bad\u7ec3\u96c6\u6bd4\u91cd          : {len(rescued)/633*100:.1f}%')

# rescued \u7684\u8be6\u7ec6\u5206\u5e03
if rescued:
    orig_r = [r['score_orig_v2'] for r in rescued]
    edit_r = [r['score_edit_v2'] for r in rescued]
    delta_r = [e - o for o, e in zip(orig_r, edit_r)]

    bucket_dist(orig_r, 'rescued \u96c6 \u539f\u56fe\u8bc4\u5206\u5206\u5e03')
    bucket_dist(edit_r, 'rescued \u96c6 \u7f16\u8f91\u540e\u8bc4\u5206\u5206\u5e03')

    print(f'\n[rescued \u96c6 delta \u7edf\u8ba1]')
    print(f'  mean = {np.mean(delta_r):+.2f}')
    print(f'  std  = {np.std(delta_r):.2f}')
    print(f'  min  = {min(delta_r):+.1f}')
    print(f'  max  = {max(delta_r):+.1f}')

# \u540c\u4e00\u539f\u56fe\u8d77\u70b9 \u7684\u63d0\u5347\u8868\u73b0
print(f'\n[\u6309 \u539f\u56fe\u8d77\u70b9\u6863 \u770b "\u5dee\u6539\u597d" \u8868\u73b0]')
print(f'{"\u539f\u56fe\u6863":>10s}  {"\u8be5\u6863\u603b\u6570":>8s}  {"\u88ab rescued":>10s}  {"\u5360\u6bd4":>6s}  '
      f'{"\u5e73\u5747 delta":>9s}  {"\u5e73\u5747 edit":>9s}')
for ob in ['0-2', '2-4', '4-6']:
    in_b = [r for r in pairs if bucket_of(r['score_orig_v2']) == ob]
    in_resc = [r for r in rescued if bucket_of(r['score_orig_v2']) == ob]
    if not in_b:
        continue
    print(f'{ob:>10s}  {len(in_b):>8d}  {len(in_resc):>10d}  '
          f'{len(in_resc)/len(in_b)*100:>5.1f}%  '
          f'{np.mean([r["score_edit_v2"] - r["score_orig_v2"] for r in in_resc]) if in_resc else 0:>+9.2f}  '
          f'{np.mean([r["score_edit_v2"] for r in in_resc]) if in_resc else 0:>9.2f}')


# ============================================================================
print('\n\n' + '=' * 75)
print('# \u89d2\u5ea6 4: \u4e0d\u540c orig \u5206\u6863\u4e0b \u7684\u63d0\u5347\u80fd\u529b')
print('=' * 75)

print(f'\n\u4ec5\u770b "delta\u3001\u8fdb\u6b65" \u8868\u73b0:')
print(f'{"orig\u6863":>10s}  {"\u603b\u6570":>5s}  {"\u5747\u503c":>5s}  {"\u4e2d\u4f4d\u6570":>5s}  '
      f'{"\u6700\u9ad8\u63d0\u5347":>9s}  {"\u6700\u4f4e (\u4e0b\u964d)":>10s}  {"P75 delta":>9s}')
for ob in ['0-2', '2-4', '4-6', '6-8', '8-10']:
    in_b = [r for r in pairs if bucket_of(r['score_orig_v2']) == ob]
    if not in_b:
        continue
    deltas = np.array([r['score_edit_v2'] - r['score_orig_v2'] for r in in_b])
    print(f'{ob:>10s}  {len(in_b):>5d}  '
          f'{deltas.mean():>+5.2f}  '
          f'{np.median(deltas):>+5.1f}  '
          f'{deltas.max():>+9.1f}  '
          f'{deltas.min():>+10.1f}  '
          f'{np.percentile(deltas, 75):>+9.2f}')


# ============================================================================
print('\n\n' + '=' * 75)
print('# \u89d2\u5ea6 5: rescued (\u5dee\u6539\u597d) \u7684\u5377\u8d70 \u6837\u4f8b')
print('=' * 75)

rescued_sorted = sorted(rescued, key=lambda r: -(r['score_edit_v2'] - r['score_orig_v2']))
print(f'\nTop 8 rescued (\u63d0\u5347\u5e45\u5ea6\u6700\u5927):\n')
print(f'{"idx":>4s}  {"image":<12s}  {"orig":>4s} -> {"edit":>4s}  delta  prompt')
for r in rescued_sorted[:8]:
    p = (r.get('edit_prompt') or '').replace('\n', ' ')[:55]
    delta_r = r['score_edit_v2'] - r['score_orig_v2']
    print(f'  {r["idx"]:>4d}  {r["image"]:<14s}  '
          f'{r["score_orig_v2"]:>4.1f} -> {r["score_edit_v2"]:>4.1f}  '
          f'{delta_r:+5.1f}  "{p}"')
