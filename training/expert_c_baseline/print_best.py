"""打印 best.pt 的关键指标 + vs mean baseline 对比."""
import sys
from pathlib import Path
import torch

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/expert_c_v1/best.pt'
c = torch.load(ckpt_path, map_location='cpu', weights_only=False)

PARAM_NAMES = [
    'white_balance', 'brightness', 'contrast',
    'shadows', 'highlights', 'saturation', 'clarity',
]
# mean baseline (从训练日志硬编码, Ep 0 之前打印的)
MEAN_BASELINE = {
    'white_balance': 783.73,
    'brightness': 13.41,
    'contrast': 14.45,
    'shadows': 6.58,
    'highlights': 15.92,
    'saturation': 9.03,
    'clarity': 0.01,
}

print(f'ckpt: {ckpt_path}')
print(f'epoch: {c["epoch"]}, val_loss: {c["val_loss"]:.4f}')
print()
print(f'{"param":<15s}{"base_MAE":>10s}{"best_MAE":>10s}{"dMAE":>8s}  {"R":>6s}  {"verdict":<15s}')
for i, p in enumerate(PARAM_NAMES):
    base = MEAN_BASELINE[p]
    mae = c['val_mae'][i]
    r = c['val_pearson_r'][i]
    d = mae - base
    dpct = 100 * d / base if base > 0 else 0
    if r >= 0.5:
        v = 'STRONG signal'
    elif r >= 0.25:
        v = 'MILD signal'
    elif r >= 0.1:
        v = 'WEAK signal'
    else:
        v = 'NO signal'
    print(f'{p:<15s}{base:>10.2f}{mae:>10.2f}{d:>+8.2f}  {r:>+6.2f}  {v:<15s}')
