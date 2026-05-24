"""Per-action eval for v10a (v9a retrained on 9D-fit Qwen wb pseudo_labels).

v10a uses Off-Planckian inverse_fit (PARAM_SPEC extended with wb_u, wb_v).
Model architecture is identical to v9a (predicts 7D); only the pseudo-label
quality changes.
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from training.firered_baseline.train_lut import (
    NamedCurvesPredictor, LUTDataset, build_data, ACTIONS,
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = 'checkpoints/lut_v10a_offplanckian/best.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ca = ckpt['args']
    model = NamedCurvesPredictor(
        n_colors=ca['nc_n_colors'], n_control_points=ca['nc_n_control_points'],
        use_attention=ca['nc_use_attention'],
        per_action_curves=ca['nc_per_action_curves'],
        use_7d_anchor=ca['nc_use_7d_anchor'],
        use_context=ca.get('nc_use_context', False),
        use_learned_cn=ca.get('nc_use_learned_cn', False),
        use_wb_head=ca.get('nc_use_wb_head', False),
        use_nilut_residual=ca.get('nc_use_nilut_residual', False),
        image_size=ca['image_size'], dropout=ca['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _, val_s = build_data(
        Path('outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl'),
        0.2, ('A excellent', 'B good', 'C acceptable'), 42)
    ds = LUTDataset(val_s, 256, is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    per_action = {a: [] for a in ACTIONS}
    with torch.no_grad():
        for b in loader:
            r, _, _, _ = model(b['enc_input'].to(device),
                                b['orig'].to(device),
                                b['action_onehot'].to(device))
            mse = ((r - b['target'].to(device)) ** 2).mean().clamp(min=1e-10)
            psnr = float((-10 * torch.log10(mse)).item())
            per_action[b['action'][0]].append(psnr)

    print(f'\nv10a_offplanckian best_val_psnr={ckpt.get("val_psnr",0):.2f}dB '
          f'@ Ep{ckpt.get("epoch","?")}')
    print(f'\nv10a_offplanckian Per-action PSNR:')
    for a, lst in sorted(per_action.items(),
                         key=lambda x: sum(x[1])/max(len(x[1]),1)):
        mean = sum(lst)/max(len(lst),1)
        p10 = sorted(lst)[max(0, len(lst)//10)]
        p90 = sorted(lst)[min(len(lst)-1, 9*len(lst)//10)]
        print(f'  {a:12s} n={len(lst):2d}  mean={mean:.2f}  '
              f'p10={p10:.2f}  p90={p90:.2f}')

    print(f'\nReference (from docs/experiments_v1_to_v9.md):')
    print(f'  v9a baseline   : wb=22.35 dB, total=24.52 dB')
    print(f'  v9a_cleanwb    : wb=20.24 dB, total=23.99 dB')
    print(f'  v9a_qwen       : wb=20.79 dB, total=24.03 dB')
    print(f'  v9g_aug        : wb=16.14 dB, total=23.47 dB')


if __name__ == '__main__':
    main()
