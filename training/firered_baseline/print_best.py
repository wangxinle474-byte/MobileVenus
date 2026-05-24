"""打印 FireRed baseline best.pt 的关键指标 + 与 mean baseline / Expert C baseline 对比."""
import sys
from pathlib import Path
import torch

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/firered_v1/best.pt'
c = torch.load(ckpt_path, map_location='cpu', weights_only=False)

PARAM_NAMES = [
    'white_balance', 'brightness', 'contrast',
    'shadows', 'highlights', 'saturation', 'clarity',
]
ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']

print(f'ckpt: {ckpt_path}')
print(f'epoch: {c["epoch"]}, val_loss: {c["val_loss"]:.4f}')
print()
print('=== 7D per-param (整体, 跨所有 action) ===')
print(f'{"param":<15s}{"val_MAE":>10s}{"R":>7s}{"verdict":<15s}')
for i, p in enumerate(PARAM_NAMES):
    mae = c['val_mae'][i]
    r = c['val_pearson_r'][i]
    if r >= 0.5:
        v = 'STRONG signal'
    elif r >= 0.25:
        v = 'MILD signal'
    elif r >= 0.1:
        v = 'WEAK signal'
    else:
        v = 'NO signal'
    print(f'{p:<15s}{mae:>10.2f}{r:>+7.2f}  {v:<15s}')

print()
print('=== per-action primary param (核心: 模型在每个 action 主导参数上的精度) ===')
per_act = c.get('val_per_action', {})
base_per = c.get('baseline_per_action', {})
print(f'{"action":<11s}{"primary":<14s}{"base_MAE":>10s}{"model_MAE":>11s}{"dMAE":>8s}{"R":>7s}{"verdict":<15s}')
for a in ACTIONS:
    if a not in per_act:
        continue
    info = per_act[a]
    base_info = base_per.get(a, {})
    base_mae = base_info.get('mae', float('nan'))
    mae = info['mae']
    r = info['r']
    d = mae - base_mae
    dpct = (100 * d / base_mae) if base_mae > 0 else 0
    if r >= 0.5:
        v = 'STRONG signal'
    elif r >= 0.25:
        v = 'MILD signal'
    elif r >= 0.1:
        v = 'WEAK signal'
    else:
        v = 'NO signal'
    print(f'{a:<11s}{info["primary"]:<14s}{base_mae:>10.2f}{mae:>11.2f}{d:>+8.2f}{r:>+7.2f}  {v:<15s}')
