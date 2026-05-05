"""\u6269\u5145\u524d vs \u6269\u5145\u540e \u6570\u636e\u96c6\u5bf9\u6bd4\u3002

\u6269\u5145\u524d (\u672c\u4f1a\u8bdd\u4e4b\u524d\u5df2\u6709):
  - data/instruction_data.json  (80777 \u6837\u672c, 64.6 MB)
  - data/fivek_expert_params.json (46167)
  - data/ppr10k_params.json (11161)
  - ...

\u6269\u5145\u540e (\u672c\u4f1a\u8bdd\u65b0\u751f\u6210):
  - data/aug_ip2p_full.json (633)
  - data/aug_fivek_params_filtered.json (26145)
  - data/aug_final.json (26778)
"""
import json
from pathlib import Path
from collections import Counter


def load_samples(path, key=None):
    if not Path(path).exists():
        return None, []
    d = json.load(open(path, encoding='utf-8'))
    if isinstance(d, list):
        return d, d
    for k in (key, 'samples', 'results', 'labels', 'prompts', 'data'):
        if k and k in d and isinstance(d[k], list):
            return d, d[k]
    return d, []


def stats(samples, name, fields=None):
    print(f'\n=== {name} ===')
    print(f'  total: {len(samples)}')
    if not samples:
        return
    keys = list(samples[0].keys())
    print(f'  keys: {keys[:12]}')

    # source \u5206\u5e03 (\u5982\u679c\u6709)
    if 'source' in keys:
        c = Counter(s.get('source') for s in samples)
        print(f'  by source: {dict(c.most_common(8))}')
    if 'expert' in keys:
        c = Counter(s.get('expert') for s in samples)
        print(f'  by expert: {dict(c.most_common(5))}')

    # \u552f\u4e00\u56fe
    for img_key in ['image_name', 'image', 'source_image']:
        if img_key in keys:
            uniq = len({s.get(img_key) for s in samples})
            print(f'  unique {img_key}: {uniq}')
            break


print('=' * 60)
print('# \u6269\u5145\u524d')
print('=' * 60)

# 1. instruction_data \u662f\u4e3b\u4f53
_, inst = load_samples('data/instruction_data.json')
stats(inst, 'instruction_data.json (\u4e3b\u4f53)')

# 2. fivek params (\u65e9\u671f\u63d0\u53d6, \u53ef\u80fd\u5df2\u88ab\u7528)
_, fk_old = load_samples('data/fivek_expert_params.json')
stats(fk_old, 'fivek_expert_params.json (\u65e9\u671f\u63d0\u53d6)')

# 3. ppr10k
_, ppr = load_samples('data/ppr10k_params.json')
stats(ppr, 'ppr10k_params.json')

# \u672c\u4f1a\u8bdd\u4e4b\u524d \"\u539f\u59cb\" \u8bad\u7ec3\u6837\u672c \u603b\u8ba1
total_before = len(inst)  # \u4e3b\u4f53\u90fd\u5728 instruction_data \u91cc\u4e86
print('\n' + '-' * 40)
print(f'\u4e3b\u8bad\u7ec3\u6587\u4ef6 instruction_data.json: {len(inst)}')
print(f'\u539f\u6709 fivek_expert_params (\u88ab \u7eb3\u5165 instruction_data \u4e2d): {len(fk_old)}')
print(f'\u539f\u6709 ppr10k_params: {len(ppr)}')
print('-' * 40)


print('\n')
print('=' * 60)
print('# \u6269\u5145\u540e (\u672c\u4f1a\u8bdd\u65b0\u589e)')
print('=' * 60)

_, aug_ip2p = load_samples('data/aug_ip2p_full.json')
stats(aug_ip2p, 'aug_ip2p_full.json (Venus->IP2P->AesExpert)')

_, aug_fk_pre = load_samples('data/aug_fivek_params.json')
stats(aug_fk_pre, 'aug_fivek_params.json (FiveK \u65e0\u8fc7\u6ee4)')

_, aug_fk = load_samples('data/aug_fivek_params_filtered.json')
stats(aug_fk, 'aug_fivek_params_filtered.json (FiveK + AesExpert\u8fc7\u6ee4)')

_, aug_final = load_samples('data/aug_final.json')
stats(aug_final, 'aug_final.json (\u7edf\u4e00 schema)')


# \u6df1\u5ea6\u5bf9\u6bd4: instruction_data \u4e2d\u662f\u5426\u5305\u62ec FiveK
print('\n')
print('=' * 60)
print('# instruction_data \u4e0e\u672c\u4f1a\u8bdd FiveK \u91cd\u53e0\u5206\u6790')
print('=' * 60)

if inst and aug_fk:
    inst_imgs = {s.get('image_name') for s in inst if 'image_name' in s}
    fk_imgs = {s.get('image_name') for s in aug_fk}
    overlap = inst_imgs & fk_imgs
    print(f'instruction_data \u4e2d\u552f\u4e00 image_name: {len(inst_imgs)}')
    print(f'aug_fivek_filtered \u4e2d\u552f\u4e00 image_name: {len(fk_imgs)}')
    print(f'\u91cd\u53e0: {len(overlap)}')
    fk_only = fk_imgs - inst_imgs
    inst_only = inst_imgs - fk_imgs
    print(f'\u4ec5 fivek_filtered: {len(fk_only)}')
    print(f'\u4ec5 instruction_data: {len(inst_only)}')


# \u53c2\u6570\u53ef\u7528\u6027 \u5bf9\u6bd4
print('\n')
print('=' * 60)
print('# \u6269\u5145\u524d/\u540e \u53c2\u6570\u8fdb\u4e00\u6b65\u5bf9\u6bd4')
print('=' * 60)

PARAM_KEYS = ['ev_compensation', 'white_balance', 'contrast',
              'shadows', 'highlights', 'saturation']


def param_stats(samples, label, params_path=None):
    print(f'\n[{label}]')
    if not samples:
        print('  empty')
        return
    # \u63a2\u63a2\u53c2\u6570\u5b58\u653e\u4f4d\u7f6e
    if params_path is None:
        s0 = samples[0]
        if 'params' in s0 and isinstance(s0['params'], dict):
            params_path = 'params'
        elif 'target_params' in s0 and isinstance(s0['target_params'], dict):
            params_path = 'target_params'
        elif 'mean_params' in s0 and isinstance(s0['mean_params'], dict):
            params_path = 'mean_params'
        elif 'ev_compensation' in s0:
            params_path = ''   # flat
    print(f'  params_path: {params_path or "(flat)"}')

    for k in PARAM_KEYS:
        vals = []
        for s in samples:
            p = s if not params_path else s.get(params_path, {})
            v = p.get(k) if isinstance(p, dict) else None
            if v is not None:
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    pass
        if vals:
            print(f'  {k:18s} n={len(vals):6d}  mean={sum(vals)/len(vals):+8.2f}  '
                  f'min={min(vals):+7.1f}  max={max(vals):+7.1f}')


param_stats(inst, 'instruction_data \u4e3b\u4f53')
param_stats(aug_final, 'aug_final \u672c\u4f1a\u8bdd\u4ea7\u51fa')
