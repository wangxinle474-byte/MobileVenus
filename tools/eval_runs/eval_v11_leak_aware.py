"""Leak-aware per-action eval for v11 NamedCurves checkpoints.

When a model is trained on a merged jsonl (e.g. joint_all_pseudo) whose
source_image keys are normalized stems (e.g. 'a0939-IMG_0262'), evaluating
that model on the un-normalized fivek_500_master jsonl with the standard
eval tool can produce inflated numbers because the val=74 split there may
contain images the model actually trained on (just under different
source_image strings).

This tool:
  1. Reproduces the model's training-time train/val split by applying the
     same build_data() to the training jsonl, collecting `train_stems`.
  2. For each record in the eval jsonl, computes its canonical FiveK
     stem (via the same normalization used at training-time merge).
  3. Splits eval samples into:
       - "leaked"     : stem ∈ train_stems   (model has seen this image)
       - "leak-free"  : stem ∉ train_stems   (model has NEVER seen this)
  4. Reports per-action PSNR on BOTH subsets so we can compare fairly
     against baselines that used a different jsonl/split.

Usage:
  python tools/eval_v11_leak_aware.py \
    --ckpt checkpoints/lut_v11a_multiteacher_seed42/best.pt \
    --training_jsonl outputs/joint_all_pseudo/pseudo_labels.jsonl \
    --eval_jsonl outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl \
    --split_seed 42

Output: prints per-action PSNR on full / leaked / leak-free subsets;
optionally writes JSON summary via --out_json.
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
    LUTDataset,
    NamedCurvesPredictor,
    build_data,
    set_actions,
)
# Reuse the same stem normalizer the merge tool used at training-time
from tools._lib.merge_jsonl_for_joint_training import extract_fivek_stem  # noqa: E402


def _get(ca: dict, key: str, default):
    return ca[key] if key in ca else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--training_jsonl', type=str, required=True,
                    help='The jsonl that was used to TRAIN the checkpoint. '
                         'Used to reproduce the train/val split and gather '
                         'train_stems for leak detection.')
    ap.add_argument('--eval_jsonl', type=str, required=True,
                    help='The jsonl with the records we want PSNR on. '
                         'Each record\'s source_image is normalized via the '
                         'same extract_fivek_stem used at training-time merge '
                         'and checked for membership in train_stems.')
    ap.add_argument('--split_seed', type=int, default=None,
                    help='Same split_seed used at training time')
    ap.add_argument('--val_ratio', type=float, default=None)
    ap.add_argument('--image_size', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--out_json', type=str, default='',
                    help='Optional path to save full JSON summary')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = ckpt['args']
    ckpt_actions = ca.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)
    image_size = args.image_size or _get(ca, 'image_size', 256)
    split_seed = args.split_seed if args.split_seed is not None \
        else _get(ca, 'split_seed', _get(ca, 'seed', 42))
    val_ratio = args.val_ratio if args.val_ratio is not None \
        else _get(ca, 'val_ratio', 0.2)

    # ── Build model from saved args ──────────────────────────────────────
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

    # ── Step 1: reproduce training-time train stems ──────────────────────
    train_s, val_s = build_data(
        Path(args.training_jsonl), val_ratio,
        ('A excellent', 'B good', 'C acceptable'),
        split_seed, action_filter=ACTIONS)
    train_stems = set(s['source_image'] for s in train_s)
    train_val_stems = train_stems | set(s['source_image'] for s in val_s)
    print(f'[training_jsonl] {args.training_jsonl}')
    print(f'  train: {len(train_s)} records / {len(train_stems)} unique stems')
    print(f'  val:   {len(val_s)} records / '
          f'{len(set(s["source_image"] for s in val_s))} unique stems')

    # ── Step 2: load eval jsonl + apply same stem normalization ─────────
    lines = Path(args.eval_jsonl).read_text(encoding='utf-8').strip().split('\n')
    raw_eval = [json.loads(l) for l in lines]
    tier_set = ('A excellent', 'B good', 'C acceptable')
    raw_eval = [s for s in raw_eval if s.get('quality_tier', '') in tier_set
                and s.get('action') in ACTIONS]
    # validate target file exists (same as build_data)
    valid_eval = []
    for s in raw_eval:
        tp = s['target_path']
        if not Path(tp).is_absolute():
            tp = PROJECT_ROOT / tp
        if Path(tp).exists():
            valid_eval.append(s)
    print(f'\n[eval_jsonl] {args.eval_jsonl}')
    print(f'  valid records (tier+target): {len(valid_eval)} / {len(raw_eval)}')

    # Normalize stems (same as merge tool used at training-time)
    for s in valid_eval:
        s['_canonical_stem'] = extract_fivek_stem(s)

    # ── Step 3: split into leaked / leak-free ────────────────────────────
    leaked = [s for s in valid_eval if s['_canonical_stem'] in train_stems]
    leak_free = [s for s in valid_eval if s['_canonical_stem'] not in train_stems]
    in_val = [s for s in valid_eval if s['_canonical_stem'] in
              (train_val_stems - train_stems)]
    print(f'  leaked (stem in TRAIN of training_jsonl):    '
          f'{len(leaked)} ({100*len(leaked)/max(1,len(valid_eval)):.1f}%)')
    print(f'  in TRAINING_JSONL VAL (also seen, not train): '
          f'{len(in_val)} ({100*len(in_val)/max(1,len(valid_eval)):.1f}%)')
    print(f'  leak-free (never seen in training_jsonl):    '
          f'{len(leak_free)} ({100*len(leak_free)/max(1,len(valid_eval)):.1f}%)')

    # ── Step 4: compute PSNR on each subset ──────────────────────────────
    def _eval_subset(samples, label):
        if not samples:
            print(f'\n[{label}] (empty, skip)')
            return {'overall': None, 'per_action': {}, 'n': 0}
        ds = LUTDataset(samples, image_size, is_train=False)
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
                mse_per = ((refined - target) ** 2).mean(dim=[1, 2, 3]) \
                    .clamp(min=1e-10)
                psnr_per = (-10.0 * torch.log10(mse_per)).detach().cpu().tolist()
                for action, psnr in zip(batch['action'], psnr_per):
                    per_action[action].append(float(psnr))
        all_vals = [v for lst in per_action.values() for v in lst]
        overall = sum(all_vals) / max(len(all_vals), 1)
        print(f'\n[{label}] n={len(samples)}, overall PSNR = {overall:.2f} dB')
        print(f'  Per-action:')
        pa_summary = {}
        for a in ACTIONS:
            vals = per_action[a]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            vals_sorted = sorted(vals)
            p10 = vals_sorted[max(0, int(0.1 * len(vals_sorted)) - 1)]
            p90 = vals_sorted[min(len(vals_sorted) - 1,
                                  int(0.9 * len(vals_sorted)))]
            print(f'    {a:>11}  n={len(vals):>3}  mean={mean:.2f}  '
                  f'p10={p10:.2f}  p90={p90:.2f}')
            pa_summary[a] = {
                'n': len(vals), 'mean': mean, 'p10': p10, 'p90': p90,
            }
        return {
            'overall': overall, 'n': len(samples),
            'per_action': pa_summary,
        }

    print(f'\ncheckpoint: {args.ckpt}')
    print(f'best_val_psnr={ckpt.get("val_psnr", 0):.2f}dB '
          f'@ Ep{ckpt.get("epoch", "?")}')

    full = _eval_subset(valid_eval, 'FULL eval set (raw, may be leaked)')
    leaked_res = _eval_subset(leaked, 'LEAKED subset (model trained on these)')
    in_val_res = _eval_subset(in_val, 'IN-VAL subset '
                                       '(model saw at training-time val)')
    leak_free_res = _eval_subset(leak_free, 'LEAK-FREE subset '
                                             '(unseen by model)')

    # ── Output JSON summary ──────────────────────────────────────────────
    summary = {
        'ckpt': args.ckpt,
        'training_jsonl': args.training_jsonl,
        'eval_jsonl': args.eval_jsonl,
        'split_seed': split_seed,
        'val_ratio': val_ratio,
        'actions': list(ACTIONS),
        'best_val_psnr_at_ckpt': float(ckpt.get('val_psnr', 0)),
        'epoch_at_ckpt': ckpt.get('epoch', None),
        'training_train_records': len(train_s),
        'training_val_records': len(val_s),
        'training_unique_train_stems': len(train_stems),
        'eval_total_records': len(valid_eval),
        'eval_leaked': len(leaked),
        'eval_in_val_of_training': len(in_val),
        'eval_leak_free': len(leak_free),
        'eval_leak_fraction': len(leaked) / max(1, len(valid_eval)),
        'subsets': {
            'full': full,
            'leaked': leaked_res,
            'in_val': in_val_res,
            'leak_free': leak_free_res,
        },
    }
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding='utf-8')
        print(f'\n[ok] summary written to {args.out_json}')


if __name__ == '__main__':
    main()
