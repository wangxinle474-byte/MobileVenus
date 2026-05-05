"""\u5206\u6790\u5f53\u524d\u7684 edit_prompts \u8d28\u91cf\u95ee\u9898, \u5173\u8054 IP2P \u8bc4\u5206\u7ed3\u679c."""
import json
import re
from collections import Counter

prompts = json.load(open('data/venus_edit_prompts.json', encoding='utf-8'))['prompts']
scores = json.load(open('outputs/aug_ip2p_full_v1_reparsed.json', encoding='utf-8'))['results']

# \u5efa img \u2192 score \u6620\u5c04
img2score = {r['image']: r for r in scores}

# \u4e3a prompt \u52a0\u4e0a\u8bc4\u5206
for p in prompts:
    r = img2score.get(p['image'])
    if r:
        p['delta'] = r.get('delta_v2')
        p['score_orig'] = r.get('score_orig_v2')
        p['score_edit'] = r.get('score_edit_v2')

print('=' * 75)
print('# \u5f53\u524d prompt \u8d28\u91cf\u81ea\u8bca')
print('=' * 75)

# --- \u7279\u5f81 1: \u662f\u5426\u7948\u4f7f\u53e5\uff08IP2P \u6700\u4f73\u683c\u5f0f\uff09---
imperative_starters = ('make', 'change', 'add', 'remove', 'turn', 'convert',
                       'increase', 'decrease', 'brighten', 'darken', 'enhance',
                       'reduce', 'moderate', 'adjust', 'soften', 'sharpen')
n_imperative = sum(1 for p in prompts
                   if p['edit_prompt'].lower().lstrip().split()[0] in imperative_starters)
print(f'\n[\u7948\u4f7f\u53e5\u5f00\u5934 (IP2P \u63a8\u8350\u683c\u5f0f)]')
print(f'  \u7948\u4f7f\u53e5\u5f00\u5934: {n_imperative}/{len(prompts)} ({n_imperative/len(prompts)*100:.1f}%)')

# \u7edf\u8ba1\u5f00\u5934\u5355\u8bcd\u5206\u5e03
first_words = Counter(p['edit_prompt'].lower().lstrip().split()[0]
                      if p['edit_prompt'].strip() else '(empty)'
                      for p in prompts)
print(f'\n[\u524d 15 \u79cd\u5f00\u5934]:')
for w, n in first_words.most_common(15):
    mark = ' ->\u7948\u4f7f' if w in imperative_starters else ''
    print(f'  {w:20s}  {n:4d}  ({n/len(prompts)*100:5.1f}%){mark}')

# --- \u7279\u5f81 2: prompt \u957f\u5ea6 ---
lengths = [len(p['edit_prompt'].split()) for p in prompts]
print(f'\n[prompt \u957f\u5ea6\u5206\u5e03 (\u8bcd\u6570)]')
print(f'  mean={sum(lengths)/len(lengths):.1f}  '
      f'min={min(lengths)}  max={max(lengths)}')
print(f'  IP2P \u6700\u4f73 <= 15 \u8bcd:  {sum(1 for l in lengths if l <= 15)} ({sum(1 for l in lengths if l <= 15)/len(lengths)*100:.1f}%)')
print(f'  \u8fc7\u957f > 25 \u8bcd:        {sum(1 for l in lengths if l > 25)} ({sum(1 for l in lengths if l > 25)/len(lengths)*100:.1f}%)')

# --- \u7279\u5f81 3: \u5305\u542b\u7591\u4f3c\u8bed ---
patterns = {
    "\u5305\u542b 'However'":       r'\bhowever\b',
    "\u5305\u542b 'could'":         r'\bcould\b',
    "\u5305\u542b 'possibly'":      r'\bpossibly\b',
    "\u5305\u542b 'benefit from'":  r'\bbenefit from\b',
    "\u5305\u542b 'consider'":      r'\bconsider\b',
    "\u5305\u542b 'suggesting'":    r'\bsuggest',
    "\u4ee5\u9884\u7f6e\u8bcd\u7ed3\u5c3e (for/of/to)": r'\b(for|of|to)\s*[.!?]?\s*$',
    "\u542b\u5b8c\u7ed3\u53e5\u53f7 < \u4e2d\u95f4": r'\. [a-z]',
}
print(f'\n[\u98ce\u9669\u8bcd\u6837 / IP2P \u4e0d\u80fd\u7406\u89e3\u7684\u6a21\u5f0f]')
for lab, pat in patterns.items():
    n = sum(1 for p in prompts if re.search(pat, p['edit_prompt'], flags=re.I))
    print(f'  {lab:35s}  {n:4d} ({n/len(prompts)*100:5.1f}%)')

# --- \u7279\u5f81 4: \u8bc4\u5206\u6548\u679c\u4e0e prompt \u7279\u5f81\u5173\u8054 ---
print(f'\n[\u8bc4\u5206 vs prompt \u7279\u5f81 \u5173\u8054]')
print(f'{"\u5b50\u96c6":40s}  {"\u6837\u672c":>5s}  {"\u5747 delta":>8s}  {"\u6539\u5584\u7387 %":>10s}')
def subset_stat(subset, lab):
    if not subset:
        print(f'  {lab:40s}  (empty)')
        return
    deltas = [s['delta'] for s in subset if s.get('delta') is not None]
    improved = sum(1 for d in deltas if d >= 0.5)
    n = len(deltas)
    print(f'  {lab:40s}  {n:>5d}  {sum(deltas)/n if n else 0:>+8.2f}  '
          f'{improved/n*100 if n else 0:>9.1f}%')

subset_stat(prompts, '\u5168\u90e8')
subset_stat([p for p in prompts if p['edit_prompt'].lower().lstrip().split()
             and p['edit_prompt'].lower().lstrip().split()[0] in imperative_starters],
            '\u7948\u4f7f\u53e5\u5f00\u5934')
subset_stat([p for p in prompts if re.search(r'\bhowever\b', p['edit_prompt'], flags=re.I)],
            "\u542b 'However'")
subset_stat([p for p in prompts if len(p['edit_prompt'].split()) <= 15],
            '<=15 \u8bcd')
subset_stat([p for p in prompts if len(p['edit_prompt'].split()) > 25],
            '>25 \u8bcd')
subset_stat([p for p in prompts if re.search(r'\b(for|of|to)\s*[.!?]?\s*$',
                                              p['edit_prompt'], flags=re.I)],
            '\u88ab\u622a\u65ad\u7684 (dangling prep)')

# --- \u7279\u5f81 5: \u5c55\u793a\u4e09\u7c7b\u6837\u672c ---
print(f'\n[\u6837\u4f8b 1: \u5f53\u524d prompt (\u6700\u5dee\u5f62\u5f0f)]')
for p in sorted(prompts, key=lambda p: p.get('delta', 99))[:5]:
    print(f'  delta={p.get("delta"):+.1f}  "{p["edit_prompt"][:90]}"')

print(f'\n[\u6837\u4f8b 2: \u5f53\u524d prompt (\u6700\u597d\u7ed3\u679c)]')
for p in sorted(prompts, key=lambda p: -(p.get('delta') or -99))[:5]:
    print(f'  delta={p.get("delta"):+.1f}  "{p["edit_prompt"][:90]}"')

print(f'\n[\u6837\u4f8b 3: \u968f\u673a\u4e2d\u70b9]')
mid = sorted(prompts, key=lambda p: abs(p.get('delta', 0) or 0))
for p in mid[10:15]:
    print(f'  delta={p.get("delta"):+.1f}  "{p["edit_prompt"][:90]}"')
