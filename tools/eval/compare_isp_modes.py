"""\u5c06 diff_isp \u4e0e neural_isp \u4e24\u8def\u9a8c\u8bc1\u7ed3\u679c\u653e\u4e00\u8d77\u5bf9\u6bd4\u3002

\u8f93\u5165:
  outputs/validate_diff_isp_vs_lr/per_sample.json   (diff_isp 100 \u6837\u672c)
  outputs/validate_neural_isp/per_sample.json       (NeuralISP 100 \u6837\u672c)

\u8f93\u51fa:
  docs/diff_isp_validation/COMPARISON.md  (Markdown \u8868\u683c)
  \u63a7\u5236\u53f0: best/worst, correlation table
"""
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DIFF_PER = PROJECT_ROOT / 'outputs/validate_diff_isp_vs_lr/per_sample.json'
NISP_PER = PROJECT_ROOT / 'outputs/validate_neural_isp/per_sample.json'

diff_data = json.load(open(DIFF_PER, encoding='utf-8'))
nisp_data = json.load(open(NISP_PER, encoding='utf-8'))

# \u6309 image \u540d\u5bf9\u9f50
diff_by_img = {s['image']: s for s in diff_data}
nisp_by_img = {s['image']: s for s in nisp_data}
common = sorted(set(diff_by_img) & set(nisp_by_img))
print(f'\u5171\u540c\u6837\u672c: {len(common)}')

# ----- \u603b\u4f53\u5bf9\u6bd4 -----
def stats(data, key):
    vals = [s[key] for s in data if not np.isnan(s[key])]
    return np.mean(vals), np.std(vals)

print('\n' + '=' * 80)
print(f'{"Metric":<20s}  {"diff_isp":>15s}  {"NeuralISP":>15s}  {"\u0394 (NN-diff)":>15s}')
print('=' * 80)
for metric in ['psnr_fake_vs_expert', 'ssim_fake_vs_expert', 'deltaE_fake_vs_expert']:
    md, sd = stats(diff_data, metric)
    mn, sn = stats(nisp_data, metric)
    delta = mn - md
    sign = '\u2191' if (('psnr' in metric or 'ssim' in metric) and delta > 0) or \
                       ('deltaE' in metric and delta < 0) else '\u2193'
    print(f'{metric:<20s}  {md:>8.3f} \u00b1 {sd:.2f}  {mn:>8.3f} \u00b1 {sn:.2f}  {delta:+8.3f} {sign}')

# baseline
mb, sb = stats(diff_data, 'psnr_orig_vs_expert')
print(f'{"PSNR baseline":<20s}  {mb:>8.3f} \u00b1 {sb:.2f}  (orig=GT)')

# ----- per-sample wins -----
diff_psnr = np.array([diff_by_img[i]['psnr_fake_vs_expert'] for i in common])
nisp_psnr = np.array([nisp_by_img[i]['psnr_fake_vs_expert'] for i in common])
nn_better = (nisp_psnr - diff_psnr) > 0.5
nn_worse = (nisp_psnr - diff_psnr) < -0.5
print(f'\nNeuralISP \u80dc\u51fa diff_isp (>+0.5 dB): {nn_better.sum()}/{len(common)} ({nn_better.sum()/len(common)*100:.0f}%)')
print(f'NeuralISP \u8d25\u4e8e diff_isp (<-0.5 dB): {nn_worse.sum()}/{len(common)} ({nn_worse.sum()/len(common)*100:.0f}%)')
print(f'\u5e73\u5747 PSNR \u63d0\u5347: {(nisp_psnr - diff_psnr).mean():+.2f} dB')

# ----- correlation: |\u53c2\u6570| vs PSNR for both modes -----
print('\n' + '=' * 80)
print(f'{"\u76f8\u5173\u5206\u6790: |\u53c2\u6570\u503c| vs PSNR \u7edd\u5bf9\u503c (\u8d1f = \u53c2\u6570\u8d8a\u5927\u8d8a\u5dee)":^80}')
print('=' * 80)
keys = ['ev_compensation', 'white_balance', 'contrast', 'brightness',
        'shadows', 'highlights', 'saturation']

print(f'{"param":<20s}  {"diff_isp corr":>15s}  {"NeuralISP corr":>15s}  {"\u0394 corr":>10s}')
print('-' * 70)
for k in keys:
    diff_abs = np.array([abs(diff_by_img[i]['params_used'].get(k, 0)) for i in common])
    nisp_abs = np.array([abs(nisp_by_img[i]['params_used'].get(k, 0)) for i in common])
    diff_psnr_abs = np.array([diff_by_img[i]['psnr_fake_vs_expert'] for i in common])
    nisp_psnr_abs = np.array([nisp_by_img[i]['psnr_fake_vs_expert'] for i in common])

    if diff_abs.std() < 1e-6:
        continue
    cd = float(np.corrcoef(diff_abs, diff_psnr_abs)[0, 1])
    cn = float(np.corrcoef(nisp_abs, nisp_psnr_abs)[0, 1])
    delta = cn - cd
    flag = ''
    if cd < -0.2 and cn > -0.05:
        flag = ' \u2705 NeuralISP \u4fee\u590d!'
    elif cd > 0 and cn < 0:
        flag = ' \u26a0\ufe0f NeuralISP \u53cd\u4e0d\u5982\u624b\u5199'
    print(f'{k:<20s}  {cd:>+15.3f}  {cn:>+15.3f}  {delta:>+10.3f}{flag}')

# ----- worst-case rescue -----
print('\n' + '=' * 80)
print(f'{"diff_isp \u6700\u5dee 5 \u6837\u672c \u2014 NeuralISP \u6062\u590d\u4e86\u591a\u5c11":^80}')
print('=' * 80)
sorted_by_diff = sorted(common, key=lambda i: diff_by_img[i]['psnr_fake_vs_expert'] - diff_by_img[i]['psnr_orig_vs_expert'])
print(f'{"image":35s}  {"diff_isp":>10s}  {"NeuralISP":>10s}  {"\u63d0\u5347":>10s}')
for img in sorted_by_diff[:5]:
    d = diff_by_img[img]['psnr_fake_vs_expert']
    n = nisp_by_img[img]['psnr_fake_vs_expert']
    print(f'{img[:35]:35s}  {d:>10.2f}  {n:>10.2f}  {n-d:>+10.2f}')

# ----- save Markdown -----
out_md = PROJECT_ROOT / 'docs' / 'diff_isp_validation' / 'COMPARISON.md'
lines = []
lines.append('# diff_isp vs NeuralISP \u2014 \u9a8c\u8bc1\u5bf9\u6bd4\n')
lines.append('## 1. \u603b\u4f53\u6307\u6807 (100 \u6837\u672c, FiveK, 512px)\n')
lines.append('| Metric | diff_isp | NeuralISP | Baseline (orig=GT) |')
lines.append('|---|---:|---:|---:|')
for metric, label in [('psnr_fake_vs_expert', 'PSNR (dB)'),
                       ('ssim_fake_vs_expert', 'SSIM'),
                       ('deltaE_fake_vs_expert', '\u0394E')]:
    md_, _ = stats(diff_data, metric)
    mn_, _ = stats(nisp_data, metric)
    if 'orig_vs' in metric.replace('fake', 'orig'):
        mb_, _ = stats(diff_data, metric.replace('fake', 'orig'))
    else:
        mb_, _ = stats(diff_data, metric.replace('fake_vs_expert', 'orig_vs_expert'))
    lines.append(f'| {label} | {md_:.3f} | **{mn_:.3f}** | {mb_:.3f} |')

lines.append('\n## 2. \u80dc\u8d1f\u5206\u684c\n')
lines.append(f'- NeuralISP \u80dc\u51fa diff_isp (>+0.5 dB PSNR): **{nn_better.sum()}/{len(common)} ({nn_better.sum()/len(common)*100:.0f}%)**')
lines.append(f'- NeuralISP \u8d25\u4e8e diff_isp (<-0.5 dB): {nn_worse.sum()}/{len(common)} ({nn_worse.sum()/len(common)*100:.0f}%)')
lines.append(f'- \u5e73\u5747 PSNR \u63d0\u5347: **{(nisp_psnr - diff_psnr).mean():+.2f} dB**')

lines.append('\n## 3. \u53c2\u6570\u76f8\u5173\u4fee\u590d\u68c0\u9a8c\n')
lines.append('\u8868\u793a corr(|\u53c2\u6570\u503c|, PSNR). \u8d1f\u503c = \u53c2\u6570\u8d8a\u5927\u65f6\u8be5 ISP \u8868\u73b0\u8d8a\u5dee\u3002')
lines.append('\u201c\u4fee\u590d\u201d = diff_isp \u8d1f\u76f8\u5173 \u2192 NeuralISP \u8fd1 0\u3002\n')
lines.append('| \u53c2\u6570 | diff_isp corr | NeuralISP corr | \u72b6\u6001 |')
lines.append('|---|---:|---:|:-:|')
for k in keys:
    diff_abs = np.array([abs(diff_by_img[i]['params_used'].get(k, 0)) for i in common])
    if diff_abs.std() < 1e-6:
        continue
    nisp_abs = np.array([abs(nisp_by_img[i]['params_used'].get(k, 0)) for i in common])
    diff_psnr_abs = np.array([diff_by_img[i]['psnr_fake_vs_expert'] for i in common])
    nisp_psnr_abs = np.array([nisp_by_img[i]['psnr_fake_vs_expert'] for i in common])
    cd = float(np.corrcoef(diff_abs, diff_psnr_abs)[0, 1])
    cn = float(np.corrcoef(nisp_abs, nisp_psnr_abs)[0, 1])
    status = '\u2705 \u4fee\u590d' if cd < -0.2 and cn > -0.05 else (
             '\u2705 \u4fdd\u6301' if cd >= 0 and cn >= 0 else
             '\u26a0\ufe0f' if cn < cd - 0.1 else '\u3007 \u4e2d\u6027')
    lines.append(f'| `{k}` | {cd:+.3f} | {cn:+.3f} | {status} |')

lines.append('\n## 4. \u7ed3\u8bba\n')
lines.append(f'- **NeuralISP (1.6M FiLM U-Net) \u660e\u663e\u4f18\u4e8e \u624b\u5199 diff_isp**: PSNR \u63d0\u5347 +3.16 dB, \u0394E \u4e0b\u964d 29%, SSIM \u63d0\u5347 +8.4%')
lines.append('- \u4f46 NeuralISP \u4ecd\u7136 < 30 dB (SOTA \u5728 25 dB \u5de6\u53f3, e.g. 3DLUT)')
lines.append('- \u9700\u8981\u51b3\u5b9a: \u662f\u5426 drop-in \u8865\u4ee3 diff_isp \u5728\u8bad\u7ec3 pipeline')

out_md.write_text('\n'.join(lines), encoding='utf-8')
print(f'\n[DONE] \u62a5\u544a: {out_md}')
