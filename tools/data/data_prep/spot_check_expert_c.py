"""\u62bd\u67e5 Expert C \u89e3\u6790\u8d28\u91cf: \u968f\u673a 5 \u5f20, \u5c55\u793a raw LR + 7D \u53c2\u6570 + N=499 \u4f2a\u6807\u7b7e\u5bf9\u6bd4."""
import json
import random
from pathlib import Path

# \u52a0\u8f7d Expert ABCDE GT
gt = json.load(open('data/fivek_expert_abcde_params.json', encoding='utf-8'))
gt_c = [s for s in gt['samples'] if s['expert'] == 'C']
print(f'Total C samples: {len(gt_c)}')

# \u968f\u673a\u62bd 5 \u5f20\u770b raw
random.seed(0)
sample = random.sample(gt_c, 5)
print('\n=== Random 5 Expert C samples (raw LR + 7D mapped) ===')
for s in sample:
    print(f'\nimage: {s["image_name"]}  copy={s["lr_copy_name"]}')
    print(f'  7D: {json.dumps(s["params"], indent=8)}')
    raw = s['raw_lr']
    keep = {k: raw.get(k) for k in
            ['Temperature', 'Tint', 'Exposure', 'Brightness', 'Contrast',
             'Shadows', 'Highlights', 'HighlightRecovery', 'FillLight',
             'Saturation', 'Vibrance', 'Clarity', 'ProcessVersion', 'Version']
            if k in raw}
    print(f'  LR_raw: {keep}')

# \u52a0\u8f7d N=499 \u4f2a\u6807\u7b7e
pseudo = []
master_jsonl = Path('outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl')
with open(master_jsonl, encoding='utf-8') as f:
    for line in f:
        pseudo.append(json.loads(line))
print(f'\nTotal pseudo records (N=499): {len(pseudo)}')

# \u67e5 N=499 \u4e2d image_name \u4e0e C GT \u4ea4\u96c6
pseudo_imgs = set()
for p in pseudo:
    sn = p.get('source_image', '')
    if sn:
        # \u8def\u5f84\u53ef\u80fd\u5305\u542b a0001-XXX.jpg, \u8f6c\u5316\u4e3a a0001-XXX.dng
        stem = Path(sn).stem  # a0001-XXX
        pseudo_imgs.add(f'{stem}.dng')

c_imgs = set(s['image_name'] for s in gt_c)
overlap = pseudo_imgs & c_imgs
print(f'\nN=499 unique images: {len(pseudo_imgs)}')
print(f'overlap with Expert C 5000: {len(overlap)}/{len(pseudo_imgs)}')

# \u793a\u4f8b\u91cd\u53e0 5 \u5f20: \u4f2a vs \u771f \u53c2\u6570\u5bf9\u6bd4
print('\n=== Pseudo vs Expert C GT comparison (5 overlap samples) ===')
gt_c_by_name = {s['image_name']: s for s in gt_c}
shown = 0
for p in pseudo:
    sn = p.get('source_image', '')
    if not sn:
        continue
    dng_name = f'{Path(sn).stem}.dng'
    if dng_name not in gt_c_by_name:
        continue
    gt_s = gt_c_by_name[dng_name]
    print(f'\n[{p.get("action","?")}] {dng_name}  '
          f'(L1={p.get("pixel_l1", 0):.4f}, tier={p.get("quality_tier","?")})')
    p_inf = p.get('P_inferred', {})
    p_gt = gt_s['params']
    print(f'    {"param":<12s}{"pseudo":>10s}{"gt_C":>10s}{"diff":>10s}')
    for k in ['white_balance', 'brightness', 'contrast', 'shadows',
              'highlights', 'saturation', 'clarity']:
        ps = p_inf.get(k, 0)
        gt_v = p_gt.get(k, 0)
        if gt_v is None:
            gt_v = 0
        diff = ps - gt_v
        print(f'    {k:<12s}{ps:>10.2f}{gt_v:>10.2f}{diff:>10.2f}')
    shown += 1
    if shown >= 5:
        break
