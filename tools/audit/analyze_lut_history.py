"""分析 train_lut.py 训练历史: 找出 train/val PSNR 走势和过拟合点."""
import json
import sys
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/lut_v1/history.json'
h = json.load(open(path, encoding='utf-8'))

print(f'Loaded {len(h)} epochs from {path}\n')
print(f'{"Ep":>4} | {"Train PSNR":>10} | {"Val PSNR":>9} | {"Train L1":>9} | {"Val L1":>8} | gap')
print('-' * 70)
best_val = 0.0
best_ep = -1
for r in h:
    tp = r['train']['psnr']
    vp = r['val']['psnr']
    tl = r['train']['l1']
    vl = r['val']['l1']
    gap = tp - vp
    marker = ''
    if vp > best_val:
        best_val = vp
        best_ep = r['epoch']
        marker = ' *'
    print(f'{r["epoch"]:>4} | {tp:>10.2f} | {vp:>9.2f} | {tl:>9.4f} | {vl:>8.4f} | {gap:+5.2f}{marker}')

print('\n' + '=' * 70)
print(f'BEST: Ep{best_ep}  val_PSNR={best_val:.2f} dB')
print(f'7D ISP ceiling reference: ~23.94 dB')
diff = best_val - 23.94
print(f'Diff vs 7D: {diff:+.2f} dB ({"突破" if diff > 0 else "未突破"})')

# weight distribution at best epoch
best_rec = next(r for r in h if r['epoch'] == best_ep)
if 'weight_mean' in best_rec['val']:
    print(f'\nLUT fusion weights at best epoch: {best_rec["val"]["weight_mean"]}')
