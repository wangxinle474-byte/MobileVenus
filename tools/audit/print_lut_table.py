"""读 outputs/lut_comparison.json 并以紧凑表格打印 per-action 详情."""
import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'outputs/lut_comparison.json'
data = json.load(open(p, 'r', encoding='utf-8'))

ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']
CEILING = 23.94

print(f'\n{"="*98}')
print(f'{"LUT 实验对比 (7D ISP ceiling: 23.94 dB)":^98}')
print(f'{"="*98}')
print(f'{"实验":<8} | {"overall":>7} | ' + ' | '.join(f'{a:>10}' for a in ACTIONS) + f' | {"gap":>5}')
print('-' * 98)
for r in data:
    line = f'{r["name"]:<8} | {r["val_psnr_eval"]:>6.2f}{"✓" if r["val_psnr_eval"]>CEILING else " "} | '
    for a in ACTIONS:
        v = r[f'pa_{a}']
        mark = '✓' if v > CEILING else ' '
        line += f'{v:>9.2f}{mark} | '
    line += f'{r["train_val_gap"]:>5.2f}'
    print(line)

print('-' * 98)
print('✓ = 超过 7D ceiling (23.94dB)   gap = train_PSNR - val_PSNR (反映过拟合)')
print()

# 找最弱 action 和最强 action 在 best run 中
best = data[0]
pa = {a: best[f'pa_{a}'] for a in ACTIONS}
print(f'最佳实验 {best["name"]} 的 per-action 详细:')
sorted_pa = sorted(pa.items(), key=lambda x: x[1])
for a, v in sorted_pa:
    bar = '█' * int(max(0, v - 15) * 2)
    flag = '✓ 破 ceiling' if v > CEILING else '✘ 未破'
    print(f'  {a:<11} {v:>6.2f}dB  {bar:<25}  {flag}')

print()
worst_a, worst_v = sorted_pa[0]
best_a, best_v = sorted_pa[-1]
print(f'>> 最弱 action: {worst_a} ({worst_v:.2f}dB)  最强 action: {best_a} ({best_v:.2f}dB)')
print(f'>> 差距: {best_v - worst_v:.2f}dB')
