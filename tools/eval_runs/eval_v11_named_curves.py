"""Generic per-action eval for v11 NamedCurves region/context variants."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from training.firered_baseline.train_lut import (
    ACTIONS,
    LUTDataset,
    NamedCurvesPredictor,
    build_data,
    set_actions,
)


def _get(ca: dict, key: str, default):
    return ca[key] if key in ca else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--jsonl', type=str,
                    default='outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl')
    ap.add_argument('--image_size', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--split_seed', type=int, default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = ckpt['args']
    ckpt_actions = ca.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)
    image_size = args.image_size or _get(ca, 'image_size', 256)

    model = NamedCurvesPredictor(
        n_colors=_get(ca, 'nc_n_colors', 3),
        n_control_points=_get(ca, 'nc_n_control_points', 7),
        use_attention=_get(ca, 'nc_use_attention', False),
        per_action_curves=_get(ca, 'nc_per_action_curves', False),
        use_7d_anchor=_get(ca, 'nc_use_7d_anchor', True),
        use_context=_get(ca, 'nc_use_context', False),
        action_gated_context=_get(ca, 'nc_action_gated_context', False),
        use_action_context=_get(ca, 'nc_use_action_context', False),
        use_region_basis=_get(ca, 'nc_use_region_basis', False),
        use_region_param_delta=_get(ca, 'nc_use_region_param_delta', False),
        use_learned_cn=_get(ca, 'nc_use_learned_cn', False),
        use_wb_head=_get(ca, 'nc_use_wb_head', False),
        use_nilut_residual=_get(ca, 'nc_use_nilut_residual', False),
        use_vera_renderer=_get(ca, 'nc_use_vera_renderer', False),
        use_implicit_head=_get(ca, 'use_implicit_head', False),
        implicit_head_base_ch=_get(ca, 'implicit_head_base_ch', 32),
        implicit_head_gate_init=_get(ca, 'implicit_head_gate_init', 0.0),
        nilut_hidden=_get(ca, 'nilut_hidden', 32),
        nilut_n_layers=_get(ca, 'nilut_n_layers', 3),
        nilut_n_freq=_get(ca, 'nilut_n_freq', 4),
        nilut_gate_init=_get(ca, 'nilut_gate_init', 1.0),
        image_size=image_size,
        n_actions=len(ACTIONS),
        dropout=_get(ca, 'dropout', 0.5),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    split_seed = args.split_seed
    if split_seed is None:
        split_seed = _get(ca, 'split_seed', _get(ca, 'seed', 42))
    val_ratio = _get(ca, 'val_ratio', 0.2)
    _, val_s = build_data(
        Path(args.jsonl), val_ratio, ('A excellent', 'B good', 'C acceptable'),
        split_seed, action_filter=ACTIONS)
    ds = LUTDataset(val_s, image_size, is_train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    per_action = {a: [] for a in ACTIONS}
    with torch.no_grad():
        for batch in loader:
            refined, _, _, _ = model(
                batch['enc_input'].to(device),
                batch['orig'].to(device),
                batch['action_onehot'].to(device),
            )
            target = batch['target'].to(device)
            mse_per = ((refined - target) ** 2).mean(dim=[1, 2, 3]).clamp(min=1e-10)
            psnr_per = (-10.0 * torch.log10(mse_per)).detach().cpu().tolist()
            for action, psnr in zip(batch['action'], psnr_per):
                per_action[action].append(float(psnr))

    all_vals = [v for lst in per_action.values() for v in lst]
    overall = sum(all_vals) / max(len(all_vals), 1)

    print(f'\ncheckpoint: {args.ckpt}')
    print(f'best_val_psnr={ckpt.get("val_psnr", 0):.2f}dB @ Ep{ckpt.get("epoch", "?")}')
    print(f'eval_overall={overall:.2f}dB')
    print('flags: '
          f'context={_get(ca, "nc_use_context", False)}, '
          f'action_gated_context={_get(ca, "nc_action_gated_context", False)}, '
          f'action_context={_get(ca, "nc_use_action_context", False)}, '
          f'region_basis={_get(ca, "nc_use_region_basis", False)}, '
          f'region_param_delta={_get(ca, "nc_use_region_param_delta", False)}')

    print('\nPer-action PSNR:')
    for action, values in sorted(per_action.items(),
                                 key=lambda item: sum(item[1]) / max(len(item[1]), 1)):
        if not values:
            continue
        vals = sorted(values)
        mean = sum(vals) / len(vals)
        p10 = vals[max(0, len(vals) // 10)]
        p90 = vals[min(len(vals) - 1, 9 * len(vals) // 10)]
        print(f'  {action:12s} n={len(vals):2d}  mean={mean:.2f}  '
              f'p10={p10:.2f}  p90={p90:.2f}')


if __name__ == '__main__':
    main()
