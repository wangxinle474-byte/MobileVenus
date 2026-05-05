"""\u5bf9\u6bd4\u9879\u76ee\u4e2d\u6240\u6709\u7684\u8bc4\u5206\u6765\u6e90\u3002

\u8bc4\u5206\u6e90:
  FiveK \u56fe:
    - Venus (50 \u5f20 eval)
    - AADB AesExpert (5120 \u5f20, 5 \u7ef4 + overall, scale 0-10)
    - Expert consensus (5000 \u5f20 .dng, scale 0-1)
    - New AesExpert (5120 \u5f20 jpg, scale 0-10)
  BA_AesGuide \u56fe:
    - IP2P pair \u8bc4\u5206 (996 \u5bf9, orig + edit)
  AADB (\u4e0d\u5c5e\u672c\u9879\u76ee):
    - Venus pseudo-labels (5000 \u5f20)
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict


def load(p):
    return json.load(open(p, encoding='utf-8'))


def stats_of(arr, name):
    if not arr:
        return f'{name}: empty'
    a = np.array(arr, dtype=float)
    return (f'{name}  n={len(a):5d}  mean={a.mean():5.2f}  std={a.std():4.2f}  '
            f'min={a.min():4.2f}  max={a.max():5.2f}  '
            f'p50={np.percentile(a,50):4.2f}  p90={np.percentile(a,90):4.2f}')


print('=' * 75)
print('# \u90e8\u5206 1: \u6bcf\u4e2a\u8bc4\u5206\u6e90\u7684\u6574\u4f53\u7edf\u8ba1')
print('=' * 75)

# --- FiveK Venus (50 \u5f20, \u591a\u6a21\u578b) ---
venus = load('data/venus_eval_results_all.json')
print(f'\n[Venus FiveK eval]  {venus["num_images"]} \u5f20 FiveK')
for model_key, col in [('original', 'original'), ('baseline', 'baseline'),
                        ('distill_v2', 'distill_v2'), ('distill_v4', 'distill_v4'),
                        ('distill_v5', 'distill_v5')]:
    vals = [r[f'{col}_scores'].get('overall') for r in venus['results']
            if f'{col}_scores' in r and r[f'{col}_scores'] is not None]
    vals = [v for v in vals if v is not None]
    print('  ' + stats_of(vals, f'{model_key:15s} overall'))

# --- FiveK AADB AesExpert (5120, 0-10) ---
fk_aadb = load('data/fivek_aesthetic_scores.json')
fk_aadb_overall = [v['overall'] for v in fk_aadb['scores'].values() if v.get('overall') is not None]
print(f'\n[FiveK AADB AesExpert]  {len(fk_aadb["scores"])} \u5f20 FiveK')
print('  ' + stats_of(fk_aadb_overall, 'overall        '))
# \u5e26\u4e0a 5 \u7ef4\u7ec6\u5206
for dim in ['composition', 'lighting', 'color', 'clarity', 'subject']:
    dvals = [v['dimensions'].get(dim) for v in fk_aadb['scores'].values()
             if 'dimensions' in v and v['dimensions'].get(dim) is not None]
    print('  ' + stats_of(dvals, f'{dim:15s}'))

# --- FiveK Expert consensus (5000 .dng, 0-1) ---
consensus = load('data/fivek_expert_consensus.json')
cons_vals = [v['consensus_score'] for v in consensus['scores'].values()
             if v.get('consensus_score') is not None]
print(f'\n[FiveK Expert consensus]  {len(consensus["scores"])} \u5f20 FiveK .dng')
print('  \u542b\u4e49: expert \u53c2\u6570 std \u8d8a\u5c0f = \u5171\u8bc6\u5ea6\u8d8a\u9ad8, \u8303\u56f4 [0,1]')
print('  ' + stats_of(cons_vals, 'consensus      '))

# --- FiveK New AesExpert (5120, 0-10) ---
new_aes = load('outputs/fivek_aesexpert_scores.json')
new_aes_vals = [v['score'] for v in new_aes['scores'].values() if v.get('score') is not None]
print(f'\n[New AesExpert (\u672c\u4f1a\u8bdd)]  {len(new_aes["scores"])} \u5f20 FiveK JPEG')
print('  ' + stats_of(new_aes_vals, 'score          '))
# parse \u65b9\u6cd5\u5206\u5e03
parse_methods = Counter(v.get('parse_method', 'none') for v in new_aes['scores'].values())
print(f'  parse_method: {dict(parse_methods.most_common())}')

# --- BA_AesGuide IP2P \u8bc4\u5206 ---
ip2p = load('outputs/aug_ip2p_full_v1_reparsed.json')
orig_vals = [r['score_orig_v2'] for r in ip2p['results']
             if r.get('score_orig_v2') is not None]
edit_vals = [r['score_edit_v2'] for r in ip2p['results']
             if r.get('score_edit_v2') is not None]
print(f'\n[IP2P BA_AesGuide]  {len(ip2p["results"])} \u5bf9')
print('  ' + stats_of(orig_vals, 'orig score     '))
print('  ' + stats_of(edit_vals, 'edit score     '))

# --- AADB Venus pseudo-labels (5000, \u4e0d\u662f FiveK) ---
venus_aadb = load('data/venus_pseudo_labels.json')
va_overall = [r['overall'] for r in venus_aadb['labels'] if r.get('overall') is not None]
print(f'\n[Venus AADB pseudo-labels]  {len(venus_aadb["labels"])} \u5f20 AADB (\u4e0d\u662f FiveK)')
print('  ' + stats_of(va_overall, 'overall        '))


# --- \u90e8\u5206 2: \u540c\u4e00\u5f20 FiveK \u56fe\u7684\u591a\u8bc4\u5206\u5bf9\u9f50 ---
print('\n\n')
print('=' * 75)
print('# \u90e8\u5206 2: \u540c\u4e00\u5f20 FiveK \u56fe\u7684\u591a\u8bc4\u5206\u5bf9\u9f50 \u6837\u4f8b')
print('=' * 75)

# \u7edf\u4e00\u7528 .jpg \u4f5c\u4e3a key
# new_aes / fk_aadb \u90fd\u662f .jpg key
# consensus \u662f .dng key  \u2192 \u8f6c\u4e3a .jpg
# venus \u4e2d image = "a0041-IMG_4972" (\u65e0\u540e\u7f00)

cons_jpg = {k.replace('.dng', '.jpg').replace('.DNG', '.jpg'): v
            for k, v in consensus['scores'].items()}
new_aes_jpg = {k: v for k, v in new_aes['scores'].items()}
fk_aadb_jpg = {k: v for k, v in fk_aadb['scores'].items()}

# venus FiveK \u7684 image \u4e0d\u5e26 .jpg, \u9700\u52a0\u4e0a; \u7528\u524d\u7f00\u5339\u914d
venus_fk = {r['image']: r for r in venus['results']}

# \u627e\u540c\u65f6\u5728 new_aes, fk_aadb, consensus \u91cc\u7684\u56fe
common = set(new_aes_jpg.keys()) & set(fk_aadb_jpg.keys()) & set(cons_jpg.keys())
print(f'\n3 \u6e90\u5171\u540c\u56fe\u6570: {len(common)}')

# \u9009 5 \u5f20\u5c55\u793a
samples_to_show = sorted(common)[:5]

print(f'\n{"image":35s}  {"AesExpertNew":>13s}  {"AesExpertAADB":>14s}  {"Consensus":>10s}  {"Venus(50)":>10s}')
print('-' * 95)
for img in samples_to_show:
    new_s = new_aes_jpg[img].get('score', '?')
    aadb_s = fk_aadb_jpg[img].get('overall', '?')
    cons_s = cons_jpg[img].get('consensus_score', '?')
    # \u8f6c\u6210\u201c\u65e0 .jpg\u201d\u7684 stem \u770b Venus
    stem = img.rsplit('.', 1)[0]
    v_entry = venus_fk.get(stem, {})
    v_s = v_entry.get('original_scores', {}).get('overall', '-') if v_entry else '-'
    def f4(v):
        return f'{v:.2f}' if isinstance(v, (int, float)) else str(v)
    print(f'{img:35s}  {f4(new_s):>13s}  {f4(aadb_s):>14s}  {f4(cons_s):>10s}  {f4(v_s):>10s}')


# --- \u90e8\u5206 3: \u8bc4\u5206\u4e4b\u95f4\u7684\u76f8\u5173\u6027 ---
print('\n\n')
print('=' * 75)
print('# \u90e8\u5206 3: \u8bc4\u5206\u4e92\u76f8\u76f8\u5173\u6027 (Pearson)')
print('=' * 75)

pairs = []
for img in common:
    new_s = new_aes_jpg[img].get('score')
    aadb_s = fk_aadb_jpg[img].get('overall')
    cons_s = cons_jpg[img].get('consensus_score')
    if new_s is None or aadb_s is None or cons_s is None:
        continue
    pairs.append((new_s, aadb_s, cons_s))

if pairs:
    A = np.array(pairs, dtype=float)
    n = len(A)
    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    print(f'\n\u57fa\u4e8e {n} \u5f20\u5171\u540c\u56fe:')
    print(f'  AesExpertNew  vs  AesExpertAADB  :  r = {corr(A[:,0], A[:,1]):+.3f}')
    print(f'  AesExpertNew  vs  Consensus       :  r = {corr(A[:,0], A[:,2]):+.3f}')
    print(f'  AesExpertAADB vs  Consensus       :  r = {corr(A[:,1], A[:,2]):+.3f}')
    print('  (+1=\u6b63\u76f8\u5173, 0=\u65e0\u5173, -1=\u8d1f\u76f8\u5173)')


# --- \u90e8\u5206 4: New AesExpert \u7684\u6863\u4f4d\u5206\u5e03 (\u7ec6) ---
print('\n\n')
print('=' * 75)
print('# \u90e8\u5206 4: New AesExpert (\u4e3b\u529b\u8fc7\u6ee4\u5668) \u6863\u4f4d\u5206\u5e03\u7ec6\u770b')
print('=' * 75)

def bucket_of(s):
    for lo, hi, lab in [(0, 2, '0-2'), (2, 4, '2-4'), (4, 6, '4-6'),
                         (6, 8, '6-8'), (8, 10.01, '8-10')]:
        if lo <= s < hi:
            return lab

# FiveK
print(f'\nFiveK ({len(new_aes_vals)} \u5f20):')
bc = Counter(bucket_of(s) for s in new_aes_vals)
for b in ['0-2', '2-4', '4-6', '6-8', '8-10']:
    n = bc.get(b, 0)
    bar = '=' * (n // 100)
    print(f'  {b:6s}  {n:5d}  {n/len(new_aes_vals)*100:5.1f}%  {bar}')

# IP2P orig + edit
print(f'\nIP2P orig ({len(orig_vals)} \u5bf9):')
bc = Counter(bucket_of(s) for s in orig_vals)
for b in ['0-2', '2-4', '4-6', '6-8', '8-10']:
    n = bc.get(b, 0)
    print(f'  {b:6s}  {n:5d}  {n/len(orig_vals)*100:5.1f}%  {"=" * (n // 20)}')

print(f'\nIP2P edit ({len(edit_vals)} \u5bf9):')
bc = Counter(bucket_of(s) for s in edit_vals)
for b in ['0-2', '2-4', '4-6', '6-8', '8-10']:
    n = bc.get(b, 0)
    print(f'  {b:6s}  {n:5d}  {n/len(edit_vals)*100:5.1f}%  {"=" * (n // 20)}')


# --- \u90e8\u5206 5: \u540c\u4e00 image \u4e0a \u6240\u6709\u8bc4\u5206\u7684\u5df6\u5f02 ---
print('\n\n')
print('=' * 75)
print('# \u90e8\u5206 5: \u540c\u4e00\u5f20\u56fe\u4e0d\u540c\u8bc4\u5206\u6e90\u7684\u6700\u5927\u5df6\u5f02')
print('=' * 75)

# \u5f52\u4e00\u5316: AADB overall \u5df2 0-10; new_aes 0-10; consensus 0-1 \u2192 \u00d7 10
diffs = []
for img in common:
    new_s = new_aes_jpg[img].get('score')
    aadb_s = fk_aadb_jpg[img].get('overall')
    cons_s = cons_jpg[img].get('consensus_score')
    if None in (new_s, aadb_s, cons_s):
        continue
    cons_10 = cons_s * 10
    diffs.append((img, new_s, aadb_s, cons_10,
                  max(abs(new_s - aadb_s), abs(new_s - cons_10), abs(aadb_s - cons_10))))

diffs.sort(key=lambda x: -x[-1])
print(f'\n\u5206\u5dee\u6700\u5927\u7684 5 \u5f20 (\u5f52\u4e00\u5316\u5230 0-10):')
print(f'{"image":35s}  {"new":>5s}  {"AADB":>5s}  {"cons*10":>7s}  {"max_diff":>8s}')
for img, n, a, c, d in diffs[:5]:
    print(f'{img:35s}  {n:5.2f}  {a:5.2f}  {c:7.2f}  {d:8.2f}')

print(f'\n\u5206\u5dee\u6700\u5c0f\u7684 5 \u5f20:')
for img, n, a, c, d in diffs[-5:]:
    print(f'{img:35s}  {n:5.2f}  {a:5.2f}  {c:7.2f}  {d:8.2f}')
