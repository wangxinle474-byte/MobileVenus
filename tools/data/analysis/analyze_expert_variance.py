"""分析 FiveK 专家参数的专家间分歧程度"""
import json
import numpy as np
from collections import defaultdict

with open('data/fivek_expert_params.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

params = ['ev_compensation', 'white_balance', 'contrast', 'brightness',
          'shadows', 'highlights', 'saturation', 'vibrance', 'clarity']

ranges = {
    'ev_compensation': 6, 'white_balance': 8000, 'contrast': 100,
    'brightness': 100, 'shadows': 100, 'highlights': 100,
    'saturation': 200, 'vibrance': 200, 'clarity': 100
}

# 按图片分组
img_groups = defaultdict(lambda: defaultdict(list))
for s in data['samples']:
    img_groups[s['image_name']][s.get('expert', 'unknown')] = True
    for p in params:
        img_groups[s['image_name']][p].append(s[p])

print(f"总记录: {len(data['samples'])}, 图片: {len(img_groups)}")
print()
print(f"{'参数':<20} {'均值':>8} {'专家间STD':>10} {'STD/范围':>10}  难度")
print("-" * 65)

for p in params:
    stds = []
    vals = []
    for img in img_groups:
        v = img_groups[img][p]
        if isinstance(v, list) and len(v) > 1:
            stds.append(np.std(v))
            vals.extend(v)
    mean_std = np.mean(stds)
    ratio = mean_std / ranges[p] * 100
    difficulty = "★" if ratio < 5 else ("★★" if ratio < 10 else "★★★")
    print(f"{p:<20} {np.mean(vals):>8.1f} {mean_std:>10.1f} {ratio:>9.1f}%  {difficulty}")
