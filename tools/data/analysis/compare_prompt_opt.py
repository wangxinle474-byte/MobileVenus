"""\u5bf9\u6bd4 "\u7ecf\u8fc7 prompt \u4f18\u5316" vs "\u539f\u56fe" \u7684 AesExpert \u8bc4\u5206\u3002

\u201cprompt \u4f18\u5316\u201d \u6307 pipeline:
    \u539f\u56fe + Venus \u751f\u6210\u7684 edit_prompt  \u2192  IP2P \u7f16\u8f91  \u2192  \u8f93\u51fa\u56fe

\u4e24\u8005\u90fd\u7528 AesExpert \u6253\u5206, \u6709 996 \u5bf9 \u53ef\u76f4\u63a5\u5bf9\u9f50.
"""
import json
import numpy as np
from pathlib import Path
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

# \u4ec5\u4fdd\u7559\u4e24\u8fb9\u90fd\u6709 parseable \u8bc4\u5206\u7684
pairs = [r for r in results
         if r.get('score_orig_v2') is not None and r.get('score_edit_v2') is not None]

orig = np.array([r['score_orig_v2'] for r in pairs])
edit = np.array([r['score_edit_v2'] for r in pairs])
delta = edit - orig

print('=' * 70)
print(f'# prompt \u4f18\u5316\u524d\u540e \u00b7 AesExpert \u8bc4\u5206\u5bf9\u6bd4  ({len(pairs)} \u5bf9)')
print('=' * 70)

# --- \u90e8\u5206 1: \u6574\u4f53\u5206\u5e03 ---
print(f'\n[\u6574\u4f53\u7edf\u8ba1]')
print(f'{"":20s}  {"\u6837\u672c":>4s}  {"\u5747\u503c":>5s}  {"\u6807\u5dee":>5s}  {"\u6700\u4f4e":>5s}  {"\u6700\u9ad8":>5s}  '
      f'{"p25":>4s}  {"p50":>4s}  {"p75":>4s}  {"p90":>4s}')
for lab, a in [('\u539f\u56fe (original)', orig), ('\u7f16\u8f91\u540e (prompt\u4f18\u5316\u540e)', edit)]:
    print(f'{lab:20s}  {len(a):>4d}  {a.mean():>5.2f}  {a.std():>5.2f}  '
          f'{a.min():>5.1f}  {a.max():>5.1f}  '
          f'{np.percentile(a,25):>4.1f}  {np.percentile(a,50):>4.1f}  '
          f'{np.percentile(a,75):>4.1f}  {np.percentile(a,90):>4.1f}')

# delta \u7edf\u8ba1
print(f'\n[\u5206\u503c\u53d8\u5316 delta = edit - orig]')
print(f'  mean delta : {delta.mean():+.2f}  (\u8d1f\u503c=\u5e73\u5747\u53d8\u5dee\u4e86)')
print(f'  std delta  :  {delta.std():.2f}')
print(f'  min/max    : {delta.min():+.1f} / {delta.max():+.1f}')

# --- \u90e8\u5206 2: \u6539\u5584 / \u6301\u5e73 / \u53d8\u5dee \u5206\u5e03 ---
print(f'\n[\u7f16\u8f91\u6548\u679c\u7c7b\u578b]')
big_better = int((delta >= 2).sum())
mild_better = int(((delta >= 0.5) & (delta < 2)).sum())
same = int((abs(delta) < 0.5).sum())
mild_worse = int(((delta <= -0.5) & (delta > -2)).sum())
big_worse = int((delta <= -2).sum())
total = len(delta)
for lab, n in [('\u5927\u5e45\u53d8\u597d (+2 \u4ee5\u4e0a)', big_better),
               ('\u8f7b\u5ea6\u53d8\u597d (+0.5~+2)', mild_better),
               ('\u6301\u5e73 (\u00b10.5)',           same),
               ('\u8f7b\u5ea6\u53d8\u5dee (-0.5~-2)', mild_worse),
               ('\u5927\u5e45\u53d8\u5dee (-2 \u4ee5\u4e0b)', big_worse)]:
    bar = '=' * int(n / total * 50)
    print(f'  {lab:25s}  {n:4d}  {n/total*100:5.1f}%  {bar}')

# --- \u90e8\u5206 3: \u6863\u4f4d\u8f6c\u79fb\u77e9\u9635 ---
print(f'\n[\u6863\u4f4d\u8f6c\u79fb\u77e9\u9635]  (\u884c=\u539f\u56fe\u6863, \u5217=\u7f16\u8f91\u540e\u6863)')
buckets = ['0-2', '2-4', '4-6', '6-8', '8-10']
transitions = Counter()
for r in pairs:
    ob = bucket_of(r['score_orig_v2'])
    eb = bucket_of(r['score_edit_v2'])
    transitions[(ob, eb)] += 1

print(f'{"orig \\ edit":12s}', end='')
for b in buckets:
    print(f'{b:>7s}', end='')
print(f'{"row\u6c47":>7s}')
for ob in buckets:
    row_sum = 0
    print(f'{ob:12s}', end='')
    for eb in buckets:
        v = transitions.get((ob, eb), 0)
        print(f'{v:>7d}', end='')
        row_sum += v
    print(f'{row_sum:>7d}')

# \u5bf9\u89d2\u7ebf = \u6263\u91cf, \u53f3\u4e0a\u4e09\u89d2 = \u63d0\u5347, \u5de6\u4e0b\u4e09\u89d2 = \u4e0b\u964d
diag_keep = sum(transitions.get((b, b), 0) for b in buckets)
up_improve = sum(transitions.get((buckets[i], buckets[j]), 0)
                 for i in range(len(buckets)) for j in range(i + 1, len(buckets)))
down_worse = sum(transitions.get((buckets[i], buckets[j]), 0)
                 for i in range(len(buckets)) for j in range(0, i))
print(f'\n  \u6863\u4f4d\u672a\u53d8 (\u5bf9\u89d2\u7ebf): {diag_keep} ({diag_keep/total*100:.1f}%)')
print(f'  \u6863\u4f4d\u4e0a\u79fb (\u53f3\u4e0a): {up_improve} ({up_improve/total*100:.1f}%)')
print(f'  \u6863\u4f4d\u4e0b\u79fb (\u5de6\u4e0b): {down_worse} ({down_worse/total*100:.1f}%)')

# --- \u90e8\u5206 4: \u6848\u4f8b ---
print(f'\n[\u663e\u8457\u6539\u5584 Top 5]')
sorted_pairs = sorted(pairs, key=lambda r: -(r['score_edit_v2'] - r['score_orig_v2']))
print(f'{"idx":>4s}  {"image":>12s}  {"orig":>5s} -> {"edit":>5s}  delta  {"prompt":.50s}')
for r in sorted_pairs[:5]:
    delta_r = r['score_edit_v2'] - r['score_orig_v2']
    p = (r.get('edit_prompt') or '').replace('\n', ' ')[:50]
    print(f'  {r["idx"]:>4d}  {r["image"]:>20s}  '
          f'{r["score_orig_v2"]:>5.1f} -> {r["score_edit_v2"]:>5.1f}  '
          f'{delta_r:+5.1f}  "{p}"')

print(f'\n[\u663e\u8457\u53d8\u5dee Top 5]')
for r in sorted_pairs[-5:][::-1]:
    delta_r = r['score_edit_v2'] - r['score_orig_v2']
    p = (r.get('edit_prompt') or '').replace('\n', ' ')[:50]
    print(f'  {r["idx"]:>4d}  {r["image"]:>20s}  '
          f'{r["score_orig_v2"]:>5.1f} -> {r["score_edit_v2"]:>5.1f}  '
          f'{delta_r:+5.1f}  "{p}"')

# --- \u90e8\u5206 5: \u539f\u56fe\u5206\u6863\u4e0b\u7684\u6539\u5584\u7387 ---
print(f'\n[\u6309\u539f\u56fe\u6863\u4f4d \u770b\u6539\u5584\u7387]')
print(f'{"orig\u6863":>10s}  {"\u603b\u6570":>5s}  {"\u6539\u5584":>5s}  {"\u6539\u5584%":>6s}  '
      f'{"\u6301\u5e73":>5s}  {"\u53d8\u5dee":>5s}  {"\u5e73\u5747 delta":>9s}')
for ob in buckets:
    bpairs = [r for r in pairs if bucket_of(r['score_orig_v2']) == ob]
    if not bpairs:
        continue
    bdelta = [r['score_edit_v2'] - r['score_orig_v2'] for r in bpairs]
    n_up = sum(1 for d in bdelta if d >= 0.5)
    n_same = sum(1 for d in bdelta if abs(d) < 0.5)
    n_down = sum(1 for d in bdelta if d <= -0.5)
    print(f'{ob:>10s}  {len(bpairs):>5d}  {n_up:>5d}  {n_up/len(bpairs)*100:>5.1f}%  '
          f'{n_same:>5d}  {n_down:>5d}  {np.mean(bdelta):>+9.2f}')
