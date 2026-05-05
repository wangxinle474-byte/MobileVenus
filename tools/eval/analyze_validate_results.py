"""\u5206\u6790 validate_diff_isp_vs_lr.py \u7684 per_sample.json:
- worst / best N \u6837\u672c
- PSNR \u4e0e\u5404\u53c2\u6570\u5927\u5c0f \u7684\u76f8\u5173
- diff_isp vs baseline \u7684 delta \u5206\u5e03 (\u65e0\u6548 / \u6709\u6548 / \u8d1f\u5411)
"""
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IN = PROJECT_ROOT / 'outputs/validate_diff_isp_vs_lr/per_sample.json'

data = json.load(open(IN, encoding='utf-8'))
print(f'Total samples: {len(data)}\n')

# ----- \u5404\u6837\u672c \u4ece best \u5230 worst (\u6309 PSNR gain) -----
for s in data:
    s['psnr_gain'] = s['psnr_fake_vs_expert'] - s['psnr_orig_vs_expert']
    s['dE_gain']   = s['deltaE_orig_vs_expert'] - s['deltaE_fake_vs_expert']  # \u6b63 = \u6539\u5584

data.sort(key=lambda s: s['psnr_gain'])

print('='*90)
print(f'{"BEST 5 (PSNR gain vs do-nothing)":^90}')
print('='*90)
print(f'{"image":35s}  {"PSNR gain":>10s}  {"fake":>6s}  {"orig":>6s}  {"\u0394E gain":>8s}  '
      f'{"ev":>5s}  {"wb":>5s}  {"C":>5s}  {"B":>5s}  {"S":>5s}  {"H":>5s}  {"sat":>5s}')
for s in data[-5:][::-1]:
    p = s['params_used']
    print(f'{s["image"][:35]:35s}  {s["psnr_gain"]:+10.2f}  '
          f'{s["psnr_fake_vs_expert"]:6.2f}  {s["psnr_orig_vs_expert"]:6.2f}  '
          f'{s["dE_gain"]:+8.2f}  '
          f'{p["ev_compensation"]:+5.2f}  {p["white_balance"]:5.0f}  '
          f'{p["contrast"]:+5.0f}  {p["brightness"]:+5.0f}  {p["shadows"]:+5.0f}  '
          f'{p["highlights"]:+5.0f}  {p["saturation"]:+5.0f}')

print('\n' + '='*90)
print(f'{"WORST 5 (PSNR gain vs do-nothing)":^90}')
print('='*90)
print(f'{"image":35s}  {"PSNR gain":>10s}  {"fake":>6s}  {"orig":>6s}  {"\u0394E gain":>8s}  '
      f'{"ev":>5s}  {"wb":>5s}  {"C":>5s}  {"B":>5s}  {"S":>5s}  {"H":>5s}  {"sat":>5s}')
for s in data[:5]:
    p = s['params_used']
    print(f'{s["image"][:35]:35s}  {s["psnr_gain"]:+10.2f}  '
          f'{s["psnr_fake_vs_expert"]:6.2f}  {s["psnr_orig_vs_expert"]:6.2f}  '
          f'{s["dE_gain"]:+8.2f}  '
          f'{p["ev_compensation"]:+5.2f}  {p["white_balance"]:5.0f}  '
          f'{p["contrast"]:+5.0f}  {p["brightness"]:+5.0f}  {p["shadows"]:+5.0f}  '
          f'{p["highlights"]:+5.0f}  {p["saturation"]:+5.0f}')

# ----- \u6709\u6548\u7387 -----
print('\n' + '='*90)
print(f'{"diff_isp Helpfulness Breakdown":^90}')
print('='*90)
gains = [s['psnr_gain'] for s in data]
n_positive = sum(1 for g in gains if g > 0.5)
n_neutral = sum(1 for g in gains if -0.5 <= g <= 0.5)
n_negative = sum(1 for g in gains if g < -0.5)
print(f'  Positive (>+0.5 dB gain, diff_isp \u6539\u5584):  {n_positive:3d} / {len(data)} ({n_positive/len(data)*100:.0f}%)')
print(f'  Neutral  (\u00b10.5 dB, \u6ca1\u663e\u8457\u5dee\u522b):          {n_neutral:3d} / {len(data)} ({n_neutral/len(data)*100:.0f}%)')
print(f'  NEGATIVE (<-0.5 dB, diff_isp \u53cd\u800c\u66f4\u5dee):  {n_negative:3d} / {len(data)} ({n_negative/len(data)*100:.0f}%)')
print(f'  Mean PSNR gain: {np.mean(gains):+.2f} \u00b1 {np.std(gains):.2f}')
print(f'  Median PSNR gain: {np.median(gains):+.2f}')

# ----- \u53c2\u6570\u7edf\u8ba1 -----
print('\n' + '='*90)
print(f'{"Used Parameter Statistics (mean-aggregated across 10 XMP records per image)":^90}')
print('='*90)
keys = ['ev_compensation', 'white_balance', 'contrast', 'brightness',
        'shadows', 'highlights', 'saturation', 'vibrance', 'clarity']
print(f'{"param":20s}  {"mean":>10s}  {"std":>10s}  {"min":>10s}  {"max":>10s}')
for k in keys:
    vals = [s['params_used'][k] for s in data]
    print(f'{k:20s}  {np.mean(vals):10.2f}  {np.std(vals):10.2f}  {np.min(vals):10.2f}  {np.max(vals):10.2f}')

# ----- \u5355\u53c2\u6570 correlation to PSNR gain -----
print('\n' + '='*90)
print(f'{"Correlation: |param| vs PSNR gain (\u8d1f\u503c\u8868\u793a\u5f53\u53c2\u6570\u8d8a\u5927 diff_isp \u8d8a\u5dee)":^90}')
print('='*90)
print(f'{"param":20s}  {"corr(|val|, gain)":>20s}')
for k in keys:
    abs_vals = np.array([abs(s['params_used'][k]) for s in data])
    gains_arr = np.array(gains)
    if abs_vals.std() < 1e-6:
        continue
    corr = float(np.corrcoef(abs_vals, gains_arr)[0, 1])
    marker = '  \u26a0  very negative \u2192 op \u4e0d\u5339\u914d' if corr < -0.3 else ''
    print(f'{k:20s}  {corr:+20.3f}{marker}')

# ----- save Markdown -----
out_md = PROJECT_ROOT / 'docs' / 'diff_isp_validation' / 'ANALYSIS.md'
out_md.parent.mkdir(parents=True, exist_ok=True)

lines = []
lines.append('# diff_isp vs Real Lightroom \u2014 \u9a8c\u8bc1\u62a5\u544a\n')
lines.append(f'\u6837\u672c: {len(data)} \u5f20 FiveK (\u968f\u673a\u5747\u5300\u62bd\u6837, seed=42)\n')
lines.append(f'\u56fe\u50cf\u5c3a\u5bf8: 512\u00d7512\n')
lines.append(f'\u53c2\u6570\u805a\u5408: mean (\u540c\u4e00\u56fe\u7684 ~10 \u6761 XMP \u8bb0\u5f55\u6c42\u5747)\n')

lines.append('\n## 1. \u603b\u4f53\u6307\u6807\n')
lines.append('| Metric | diff_isp vs Expert | Baseline (orig=GT) | Delta |')
lines.append('|---|---|---|---|')
psnr_ours = np.mean([s['psnr_fake_vs_expert'] for s in data])
psnr_base = np.mean([s['psnr_orig_vs_expert'] for s in data])
ssim_ours = np.mean([s['ssim_fake_vs_expert'] for s in data])
ssim_base = np.mean([s['ssim_orig_vs_expert'] for s in data])
de_ours = np.mean([s['deltaE_fake_vs_expert'] for s in data])
de_base = np.mean([s['deltaE_orig_vs_expert'] for s in data])
lines.append(f'| PSNR | **{psnr_ours:.2f} dB** | {psnr_base:.2f} dB | **{psnr_ours-psnr_base:+.2f}** |')
lines.append(f'| SSIM | {ssim_ours:.3f} | {ssim_base:.3f} | {ssim_ours-ssim_base:+.3f} |')
lines.append(f'| \u0394E  | {de_ours:.2f} | {de_base:.2f} | {de_ours-de_base:+.2f} |')

lines.append('\n## 2. \u6709\u6548\u7387\n')
lines.append(f'- diff_isp \u6539\u5584 (+0.5 dB+): {n_positive}/{len(data)} ({n_positive/len(data)*100:.0f}%)')
lines.append(f'- \u65e0\u5dee\u522b (\u00b10.5 dB): {n_neutral}/{len(data)} ({n_neutral/len(data)*100:.0f}%)')
lines.append(f'- **\u53cd\u800c\u66f4\u5dee** (-0.5 dB-): **{n_negative}/{len(data)} ({n_negative/len(data)*100:.0f}%)**')

lines.append('\n## 3. \u7ed3\u8bba\n')
if n_positive > n_negative * 2:
    lines.append('\u2705 diff_isp \u5728\u5927\u591a\u6570\u6837\u672c\u4e0a\u8d77\u6b63\u5411\u4f5c\u7528\uff0c\u4f46 PSNR \u7edd\u5bf9\u503c\u8fd8\u5f88\u4f4e (~19 dB)\u3002')
else:
    lines.append('\u26a0\ufe0f diff_isp \u6d82\u6539\u58f0\u6709\u9650 \u2014 \u5927\u90e8\u5206\u6837\u672c\u63d0\u5347\u4e0d\u663e\u8457\u6216\u53cd\u4f8b\u3002\u8003\u8651\u91cd\u65b0\u6821\u51c6\u6216\u66ff\u6362\u3002')

out_md.write_text('\n'.join(lines), encoding='utf-8')
print(f'\n[DONE] \u5206\u6790\u62a5\u544a\u4fdd\u5b58\u5230: {out_md}')
