"""Evaluate model's 7D parameter prediction quality vs GT (inverse_fit pseudo-label).

Reports:
  - Per-action MAE on the *active* parameter (e.g. for action=wb,
    we look at white_balance pred vs gt).
  - Per-parameter MAE across all val samples (regardless of action).
  - Distributional stats (p10/p50/p90) and pred-vs-gt scatter dump.

Usage:
  python tools/eval_param_prediction.py \
    --ckpt checkpoints/lut_v11a_seed42/best.pt \
    --jsonl outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl \
    --split_seed 42

Output:
  - stdout: human-readable tables
  - {ckpt_dir}/param_pred_eval.json: full per-sample dump for plotting
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from training.firered_baseline.train_lut import (  # noqa: E402
    ACTIONS,
    ACTION_TO_PARAM_7D_INDICES,
    LUTDataset,
    NamedCurvesPredictor,
    PARAM_NAMES_7D,
    PARAM_NORM_7D,
    build_data,
    set_actions,
)


def _get(ca: dict, key: str, default):
    return ca[key] if key in ca else default


def denormalize(name: str, v_norm: float) -> float:
    cfg = PARAM_NORM_7D[name]
    return float(v_norm) * cfg['scale'] + cfg['center']


def percentile(values, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return float(s[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--jsonl', type=str,
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--image_size', type=int, default=None)
    ap.add_argument('--split_seed', type=int, default=None)
    ap.add_argument('--out_json', type=str, default=None,
                    help='Output JSON path (default: alongside ckpt)')
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
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Collect: per-sample (action, action_idx, gt_p7_phys, pred_p7_phys)
    samples = []
    with torch.no_grad():
        for batch in loader:
            enc_in = batch['enc_input'].to(device)
            orig = batch['orig'].to(device)
            a_oh = batch['action_onehot'].to(device)
            gt_p7 = batch['params_norm_7d'].to(device)  # (1, 7)

            _, _, _, pred_p7 = model(enc_in, orig, a_oh)  # (1, 7)

            action = batch['action'][0]
            action_idx = ACTIONS.index(action)
            active_param_idx = ACTION_TO_PARAM_7D_INDICES[action_idx]

            gt_norm = gt_p7[0].cpu().tolist()
            pred_norm = pred_p7[0].cpu().tolist()
            gt_phys = [denormalize(n, v) for n, v in zip(PARAM_NAMES_7D, gt_norm)]
            pred_phys = [denormalize(n, v) for n, v in zip(PARAM_NAMES_7D, pred_norm)]

            samples.append({
                'idx': len(samples),
                'source_image': batch['source_image'][0],
                'action': action,
                'action_idx': action_idx,
                'active_param': PARAM_NAMES_7D[active_param_idx],
                'active_param_idx': active_param_idx,
                'gt_norm': gt_norm,
                'pred_norm': pred_norm,
                'gt_phys': gt_phys,
                'pred_phys': pred_phys,
            })

    n_total = len(samples)

    # ------------------------------------------------------------------
    # Aggregation 1: per-action, ACTIVE parameter only
    # ------------------------------------------------------------------
    print('\n' + '=' * 78)
    print(f'checkpoint: {args.ckpt}')
    print(f'best_val_psnr={ckpt.get("val_psnr", 0):.2f}dB '
          f'@ Ep{ckpt.get("epoch", "?")}')
    print(f'jsonl: {args.jsonl}')
    print(f'val samples: {n_total}')
    print('=' * 78)

    print('\n[1] Per-action MAE on the ACTIVE parameter (physical units):')
    print(f'{"action":<12} {"param":<14} {"n":>4} '
          f'{"gt_mean":>10} {"pred_mean":>10} '
          f'{"MAE":>8} {"p50_err":>8} {"p90_err":>8} '
          f'{"unit":<6}')
    print('-' * 96)

    UNITS = {
        'white_balance': 'K',
        'brightness': '',
        'contrast': '',
        'shadows': '',
        'highlights': '',
        'saturation': '',
        'clarity': '',
    }

    per_action_active = {}
    for action in ACTIONS:
        sub = [s for s in samples if s['action'] == action]
        if not sub:
            continue
        param_idx = sub[0]['active_param_idx']
        param_name = sub[0]['active_param']
        gt_vals = [s['gt_phys'][param_idx] for s in sub]
        pred_vals = [s['pred_phys'][param_idx] for s in sub]
        errs = [abs(p - g) for p, g in zip(pred_vals, gt_vals)]
        mae = sum(errs) / len(errs)
        gt_mean = sum(gt_vals) / len(gt_vals)
        pred_mean = sum(pred_vals) / len(pred_vals)
        p50 = percentile(errs, 0.5)
        p90 = percentile(errs, 0.9)
        per_action_active[action] = {
            'n': len(sub),
            'param': param_name,
            'gt_mean': gt_mean,
            'pred_mean': pred_mean,
            'mae': mae,
            'p50_err': p50,
            'p90_err': p90,
        }
        print(f'{action:<12} {param_name:<14} {len(sub):>4} '
              f'{gt_mean:>10.2f} {pred_mean:>10.2f} '
              f'{mae:>8.2f} {p50:>8.2f} {p90:>8.2f} '
              f'{UNITS[param_name]:<6}')

    # ------------------------------------------------------------------
    # Aggregation 2: per-parameter MAE across ALL samples
    # ------------------------------------------------------------------
    print('\n[2] Per-parameter MAE across ALL val samples (physical units):')
    print('   (most non-active params have gt=center; this shows whether')
    print('    model correctly outputs ~center for non-active params)')
    print(f'{"param":<14} {"n":>4} '
          f'{"gt_mean":>10} {"pred_mean":>10} '
          f'{"MAE":>8} {"p50_err":>8} {"p90_err":>8}')
    print('-' * 80)
    per_param = {}
    for i, name in enumerate(PARAM_NAMES_7D):
        gt_vals = [s['gt_phys'][i] for s in samples]
        pred_vals = [s['pred_phys'][i] for s in samples]
        errs = [abs(p - g) for p, g in zip(pred_vals, gt_vals)]
        mae = sum(errs) / len(errs)
        gt_mean = sum(gt_vals) / len(gt_vals)
        pred_mean = sum(pred_vals) / len(pred_vals)
        per_param[name] = {
            'n': len(samples),
            'gt_mean': gt_mean,
            'pred_mean': pred_mean,
            'mae': mae,
            'p50_err': percentile(errs, 0.5),
            'p90_err': percentile(errs, 0.9),
        }
        print(f'{name:<14} {len(samples):>4} '
              f'{gt_mean:>10.2f} {pred_mean:>10.2f} '
              f'{mae:>8.2f} {percentile(errs, 0.5):>8.2f} '
              f'{percentile(errs, 0.9):>8.2f}')

    # ------------------------------------------------------------------
    # Aggregation 3: signed bias (pred − gt) for active param
    # ------------------------------------------------------------------
    print('\n[3] Signed bias (pred − gt) for ACTIVE parameter:')
    print('    (indicates systematic over/under-prediction)')
    print(f'{"action":<12} {"param":<14} {"bias_mean":>10} '
          f'{"bias_std":>10} {"% pred>gt":>10}')
    print('-' * 64)
    per_action_bias = {}
    for action in ACTIONS:
        sub = [s for s in samples if s['action'] == action]
        if not sub:
            continue
        param_idx = sub[0]['active_param_idx']
        param_name = sub[0]['active_param']
        diffs = [s['pred_phys'][param_idx] - s['gt_phys'][param_idx]
                 for s in sub]
        mean_d = sum(diffs) / len(diffs)
        var_d = sum((d - mean_d) ** 2 for d in diffs) / max(len(diffs), 1)
        std_d = var_d ** 0.5
        pct_pos = sum(1 for d in diffs if d > 0) / len(diffs) * 100.0
        per_action_bias[action] = {
            'param': param_name,
            'bias_mean': mean_d,
            'bias_std': std_d,
            'pct_pred_gt_gt': pct_pos,
        }
        print(f'{action:<12} {param_name:<14} '
              f'{mean_d:>+10.2f} {std_d:>10.2f} {pct_pos:>9.1f}%')

    # ------------------------------------------------------------------
    # Dump full samples to JSON for plotting
    # ------------------------------------------------------------------
    out_json_path = (Path(args.out_json) if args.out_json else
                     Path(args.ckpt).parent / 'param_pred_eval.json')
    out = {
        'ckpt': str(args.ckpt),
        'best_val_psnr': float(ckpt.get('val_psnr', 0)),
        'best_epoch': int(ckpt.get('epoch', -1)),
        'jsonl': str(args.jsonl),
        'actions': list(ACTIONS),
        'val_ratio': val_ratio,
        'split_seed': split_seed,
        'n_val': n_total,
        'per_action_active': per_action_active,
        'per_param': per_param,
        'per_action_bias': per_action_bias,
        'samples': samples,
    }
    out_json_path.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'\n[dump] full per-sample data → {out_json_path}')


if __name__ == '__main__':
    main()
