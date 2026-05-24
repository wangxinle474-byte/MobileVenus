"""v7 单独的 per-action PSNR 诊断 (避免重跑所有实验)."""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.firered_baseline.train_lut import (  # noqa: E402
    HybridLUTPredictor, PerActionLUTPredictor,
    LUTDataset, build_data, ACTIONS,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
JSONL = ROOT / 'outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl'


@torch.no_grad()
def eval_ckpt(ckpt_path, val_loader):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt['args']
    if a.get('hybrid_wb'):
        m = HybridLUTPredictor(
            n_luts=a['n_luts'], lut_dim=a['lut_dim'],
            image_size=a['image_size'], dropout=a['dropout'])
        name = 'v7 (hybrid)'
    else:
        m = PerActionLUTPredictor(
            n_luts=a['n_luts'], lut_dim=a['lut_dim'],
            image_size=a['image_size'], dropout=a['dropout'])
        name = 'v6 (per-action)'
    m.load_state_dict(ckpt['model_state_dict'])
    m.to(device).eval()

    pa = {a: [] for a in ACTIONS}
    for batch in val_loader:
        ei = batch['enc_input'].to(device)
        og = batch['orig'].to(device)
        tg = batch['target'].to(device)
        oh = batch['action_onehot'].to(device)
        out, _ = m(ei, og, oh)
        for b in range(out.shape[0]):
            mse = ((out[b] - tg[b]) ** 2).mean()
            psnr = (-10 * torch.log10(mse.clamp(min=1e-10))).item()
            pa[batch['action'][b]].append(psnr)
    pa = {k: float(np.mean(v)) for k, v in pa.items()}
    overall = float(np.mean(list(pa.values())))
    return name, pa, overall


# 构建 val (与训练同 seed)
train_s, val_s = build_data(
    JSONL, val_ratio=0.2,
    tier_filter=('A excellent', 'B good', 'C acceptable'), seed=42)
val_ds = LUTDataset(val_s, image_size=256, is_train=False)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

print(f'\n{"="*82}')
print(f'{"v6 vs v7 per-action PSNR 对比":^82}')
print(f'{"="*82}')

results = {}
for run in ['lut_v6', 'lut_v7']:
    ckpt = ROOT / f'checkpoints/{run}/best.pt'
    if not ckpt.exists():
        print(f'[SKIP] {run}')
        continue
    name, pa, overall = eval_ckpt(ckpt, val_loader)
    results[run] = (name, pa, overall)

# 表头
print(f'{"action":<12} | ' + ' | '.join(
    f'{r:^15}' for r in results) + ' | delta')
print('-' * 82)

CEIL = 23.94
for a in ACTIONS:
    line = f'{a:<12} | '
    vs = []
    for run in results:
        v = results[run][1][a]
        vs.append(v)
        flag = '✓' if v > CEIL else ' '
        line += f'{v:>10.2f}{flag}     | '
    delta = vs[1] - vs[0] if len(vs) == 2 else 0
    sign = '+' if delta >= 0 else ''
    line += f'{sign}{delta:.2f}'
    print(line)

print('-' * 82)
overall_line = f'{"OVERALL":<12} | '
ovs = []
for run in results:
    o = results[run][2]
    ovs.append(o)
    overall_line += f'{o:>10.2f}       | '
if len(ovs) == 2:
    od = ovs[1] - ovs[0]
    overall_line += f'{"+" if od>=0 else ""}{od:.2f}'
print(overall_line)
print(f'\n7D ISP ceiling: 23.94 dB')

if 'lut_v6' in results and 'lut_v7' in results:
    wb_v6 = results['lut_v6'][1]['wb']
    wb_v7 = results['lut_v7'][1]['wb']
    print(f'\n>> wb 变化: v6 {wb_v6:.2f}dB → v7 {wb_v7:.2f}dB '
          f'({"+" if wb_v7-wb_v6>=0 else ""}{wb_v7-wb_v6:.2f})')
    if wb_v7 > wb_v6:
        print('   3-gain 提升了 wb! 但其他 action 退步 → 共享 encoder 受干扰')
    else:
        print('   3-gain 反而拖累 wb → FiveK 的 wb 不是纯乘性变换')
