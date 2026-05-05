"""\u4e34\u65f6 \u67e5\u770b fivek_expert_params.json \u7ed3\u6784 (\u4ee5\u786e\u5b9a expert \u6807\u7b7e\u5982\u4f55\u5bf9\u5e94 a/b/c/d/e)\u3002"""
import json
from collections import Counter, defaultdict

d = json.load(open('data/fivek_expert_params.json', encoding='utf-8'))
s = d['samples']
print(f'Total records: {len(s)}')

# per image
per_img = defaultdict(list)
for r in s:
    per_img[r['image_name']].append(r)
print(f'Unique images: {len(per_img)}')

# one image
one = list(per_img.keys())[0]
print(f'\nImage: {one}')
recs = per_img[one]
print(f'Record count: {len(recs)}')
for r in recs:
    ex = r.get('expert', '?')
    ex_s = ex[:12] + '..' if len(ex) > 12 else ex
    print(f"  expert={ex_s:<15s}  ev={r.get('ev_compensation',0):+.2f}  "
          f"wb={r.get('white_balance',0):.0f}  contrast={r.get('contrast',0):+.0f}  "
          f"sat={r.get('saturation',0):+.0f}")

# expert distribution across all records
print('\nExpert distribution (top 10):')
for ex, cnt in Counter(r['expert'] for r in s).most_common(10):
    print(f'  {ex}: {cnt}')
