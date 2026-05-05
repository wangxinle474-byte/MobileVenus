"""\u68c0\u67e5 fivek_expert_settings.json \u7ed3\u6784."""
import json
from collections import Counter

PATH = r'E:\Data\dataset\fivek_expert\fivek_expert_settings.json'
d = json.load(open(PATH, encoding='utf-8'))
samples = d['samples']
print(f'total samples: {len(samples)}')
print()

experts = Counter(s['expert'] for s in samples)
print('expert distribution:')
for k, v in experts.most_common():
    print(f'  {k:20s}  {v}')
print()

unique_imgs = set(s['image_name'] for s in samples)
print(f'unique images: {len(unique_imgs)}')
print()

print('targets keys:    ', list(samples[0]['targets'].keys()))
print('raw_settings keys:', list(samples[0]['raw_settings'].keys()))
print()

# 1 image, 5 experts
first_img = samples[0]['image_name']
matching = [s for s in samples if s['image_name'] == first_img][:8]
print(f'image: {first_img}')
for s in matching:
    t = s['targets']
    rs = s['raw_settings']
    print(f"  id={s['id']:30s}  expert={s['expert']:12s}  "
          f"ev={t.get('ev_compensation'):+.2f}  wb={t.get('white_balance')}  "
          f"contrast={rs.get('contrast'):+}  shadows={rs.get('shadows'):+}  "
          f"highlights={rs.get('highlights'):+}  sat={rs.get('saturation'):+}")
