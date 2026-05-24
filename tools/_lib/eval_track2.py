"""Track 2 unified eval — runs leak-aware evaluation on three val sets.

Given a Track 2 checkpoint (Path A or Path B), evaluates against:
  1. Clean Expert C (in-domain for Track 2 checkpoints; 15,739 records)
  2. FireRed pseudo (Track 1 canonical baseline; 372 records, val=74 subset)
  3. MMArt-PPR10k 250 (real LR XMP; PPR10K-disjoint → 100% leak-free)

The training_jsonl is auto-detected from ckpt['args']['jsonl'] so leak-aware
partition is computed correctly even when the eval set differs from training.

Outputs a single combined JSON with per-set per-action breakdown plus a
Track-2-specific comparison table that mirrors the four-cell ablation in
docs/track2_plan.md §1.

Usage:
  python tools/eval_track2.py \
    --ckpt checkpoints/lut_track2B_nilut_clean_seed42/best.pt \
    --out_dir outputs/eval_track2/B
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
from tqdm import tqdm

from training.firered_baseline.train_lut import (  # noqa: E402
    ACTIONS,
    LUTDataset,
    NamedCurvesPredictor,
    build_data,
    set_actions,
)
from tools._lib.merge_jsonl_for_joint_training import extract_fivek_stem  # noqa: E402


# =====================================================================
# Track 2 standard eval suite — fixed across all checkpoints
# =====================================================================
TRACK2_EVAL_SETS = {
    'clean_expertC': {
        'jsonl': 'outputs/fivek_expert_c_master/pseudo_labels.jsonl',
        'desc': 'Clean Expert C (15,739 noise-free targets, in-domain '
                'for Track 2)',
    },
    'firered_pseudo': {
        'jsonl': 'outputs/inverse_fit_pilot/fivek_500_master/'
                 'pseudo_labels.jsonl',
        'desc': 'FireRed pseudo (372 records, Track 1 baseline; '
                'cross-distribution for Track 2)',
    },
    'mmart_real_lr': {
        'jsonl': 'outputs/mmart_pseudo_labels/v1_250/pseudo_labels.jsonl',
        'desc': 'MMArt-PPR10k 250 (real LR XMP; PPR10K-disjoint, '
                'fully leak-free against FiveK by construction)',
    },
}

# Tier filter must match training-time build_data
TIER_FILTER = ('A excellent', 'B good', 'C acceptable')


def _get(ca: dict, key: str, default):
    return ca[key] if key in ca else default


def build_model_from_ckpt(ckpt: dict, image_size: int,
                          device: torch.device) -> NamedCurvesPredictor:
    """Reconstruct a NamedCurvesPredictor from saved ckpt['args'].

    Reads both v9-v11 flags AND the Track 2 additions (force_nilut_gate,
    vera_latent_dim, vera_hidden, vera_n_layers, vera_gate_init).
    Falls back to v11a defaults for missing keys so old checkpoints
    still load correctly.
    """
    ca = ckpt['args']
    ckpt_actions = ca.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)
    model = NamedCurvesPredictor(
        # v9 base
        n_colors=_get(ca, 'nc_n_colors', 3),
        n_control_points=_get(ca, 'nc_n_control_points', 7),
        use_attention=_get(ca, 'nc_use_attention', False),
        per_action_curves=_get(ca, 'nc_per_action_curves', False),
        use_7d_anchor=_get(ca, 'nc_use_7d_anchor', True),
        # v9d/v11
        use_context=_get(ca, 'nc_use_context', False),
        action_gated_context=_get(ca, 'nc_action_gated_context', False),
        use_action_context=_get(ca, 'nc_use_action_context', False),
        use_region_basis=_get(ca, 'nc_use_region_basis', False),
        use_region_param_delta=_get(ca, 'nc_use_region_param_delta', False),
        use_learned_cn=_get(ca, 'nc_use_learned_cn', False),
        use_wb_head=_get(ca, 'nc_use_wb_head', False),
        # v10b NILUT
        use_nilut_residual=_get(ca, 'nc_use_nilut_residual', False),
        nilut_hidden=_get(ca, 'nilut_hidden', 32),
        nilut_n_layers=_get(ca, 'nilut_n_layers', 3),
        nilut_n_freq=_get(ca, 'nilut_n_freq', 4),
        nilut_gate_init=_get(ca, 'nilut_gate_init', 1.0),
        # Track 2 additions
        force_nilut_gate=_get(ca, 'nc_force_nilut_gate', False),
        # v10c VeraRenderer
        use_vera_renderer=_get(ca, 'nc_use_vera_renderer', False),
        vera_latent_dim=_get(ca, 'vera_latent_dim', 32),
        vera_hidden=_get(ca, 'vera_hidden', 64),
        vera_n_layers=_get(ca, 'vera_n_layers', 4),
        vera_gate_init=_get(ca, 'vera_gate_init', 0.0),
        # Path X: implicit residual head (post-v11)
        use_implicit_head=_get(ca, 'use_implicit_head', False),
        implicit_head_base_ch=_get(ca, 'implicit_head_base_ch', 32),
        implicit_head_gate_init=_get(ca, 'implicit_head_gate_init', 0.0),
        image_size=image_size,
        n_actions=len(ACTIONS),
        dropout=_get(ca, 'dropout', 0.5),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model


def eval_subset(model: NamedCurvesPredictor, samples: list, image_size: int,
                batch_size: int, device: torch.device,
                label: str) -> dict:
    """Run model on a list of samples; return overall + per-action PSNR."""
    if not samples:
        print(f'  [{label}] (empty, skip)', flush=True)
        return {'overall': None, 'per_action': {}, 'n': 0}
    ds = LUTDataset(samples, image_size, is_train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    per_action = {a: [] for a in ACTIONS}
    with torch.no_grad():
        pbar = tqdm(loader, desc=f'  [{label}]', total=len(loader),
                    mininterval=1.0, dynamic_ncols=True,
                    file=sys.stderr, leave=True)
        for batch in pbar:
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
        pa_summary[a] = {'n': len(vals), 'mean': mean,
                          'p10': p10, 'p90': p90}
    print(f'  [{label}] n={len(samples)}  overall PSNR = {overall:.2f} dB',
          flush=True)
    for a in ACTIONS:
        if a in pa_summary:
            s = pa_summary[a]
            print(f'      {a:>11}  n={s["n"]:>4}  mean={s["mean"]:.2f}  '
                  f'p10={s["p10"]:.2f}  p90={s["p90"]:.2f}', flush=True)
    return {'overall': overall, 'n': len(samples),
            'per_action': pa_summary}


def run_leak_aware_eval(model: NamedCurvesPredictor,
                         training_jsonl: Path, eval_jsonl: Path,
                         split_seed: int, val_ratio: float,
                         image_size: int, batch_size: int,
                         device: torch.device,
                         eval_label: str) -> dict:
    """One leak-aware evaluation pass: full / leaked / leak-free subsets."""
    print(f'\n=== {eval_label} ===')
    print(f'  training_jsonl: {training_jsonl}')
    print(f'  eval_jsonl    : {eval_jsonl}')

    # 1) reproduce training-time train/val stems
    # Both sides MUST use extract_fivek_stem(...) for canonical comparison;
    # the raw `source_image` field has dataset-specific prefixes (e.g.
    # 'fivek-c-a0001-...') that won't match the eval-side canonical stems.
    train_s, val_s = build_data(training_jsonl, val_ratio, TIER_FILTER,
                                  split_seed, action_filter=ACTIONS)
    train_stems = set(extract_fivek_stem(s) for s in train_s)
    val_stems = set(extract_fivek_stem(s) for s in val_s)
    train_val_stems = train_stems | val_stems

    # 2) load eval jsonl
    lines = eval_jsonl.read_text(encoding='utf-8').strip().split('\n')
    raw = [json.loads(l) for l in lines if l.strip()]
    raw = [s for s in raw if s.get('quality_tier', '') in TIER_FILTER
           and s.get('action') in ACTIONS]
    valid = []
    for s in raw:
        tp = s['target_path']
        if not Path(tp).is_absolute():
            tp = PROJECT_ROOT / tp
        if Path(tp).exists():
            valid.append(s)

    # 3) normalize stems and split
    for s in valid:
        s['_canonical_stem'] = extract_fivek_stem(s)
    leaked = [s for s in valid if s['_canonical_stem'] in train_stems]
    leak_free = [s for s in valid if
                  s['_canonical_stem'] not in train_val_stems]
    in_val = [s for s in valid if
               s['_canonical_stem'] in (val_stems - train_stems)]

    print(f'  records: total={len(valid)}, leaked={len(leaked)} '
          f'({100*len(leaked)/max(1,len(valid)):.1f}%), '
          f'in_val={len(in_val)}, leak_free={len(leak_free)}')

    # 4) eval each subset
    full_res = eval_subset(model, valid, image_size, batch_size, device,
                           label='FULL    ')
    leaked_res = eval_subset(model, leaked, image_size, batch_size, device,
                              label='LEAKED  ')
    in_val_res = eval_subset(model, in_val, image_size, batch_size, device,
                              label='IN-VAL  ')
    leak_free_res = eval_subset(model, leak_free, image_size, batch_size,
                                 device, label='LEAK-FREE')

    return {
        'eval_jsonl': str(eval_jsonl),
        'training_jsonl': str(training_jsonl),
        'records_total': len(valid),
        'records_leaked': len(leaked),
        'records_in_val': len(in_val),
        'records_leak_free': len(leak_free),
        'leak_fraction': len(leaked) / max(1, len(valid)),
        'subsets': {
            'full': full_res,
            'leaked': leaked_res,
            'in_val': in_val_res,
            'leak_free': leak_free_res,
        },
    }


def print_summary_table(results: dict, ckpt_path: str):
    """Compact Track 2 comparison table — leak-free overall PSNR per set."""
    print('\n' + '=' * 70)
    print(f'TRACK 2 SUMMARY  ckpt: {ckpt_path}')
    print('=' * 70)
    print(f'{"eval set":<24} {"n":>5} {"leak-free":>10} {"full":>8}')
    print('-' * 70)
    for name, res in results.items():
        if not isinstance(res, dict) or 'subsets' not in res:
            continue
        lf = res['subsets']['leak_free']
        full = res['subsets']['full']
        lf_psnr = f'{lf["overall"]:.2f}' if lf['overall'] else 'n/a'
        full_psnr = f'{full["overall"]:.2f}' if full['overall'] else 'n/a'
        print(f'{name:<24} {res["records_leak_free"]:>5} '
              f'{lf_psnr:>10} {full_psnr:>8}')
    print('=' * 70)
    print('Reference (Track 1, v11a baseline on FireRed pseudo val=74):')
    print('  baseline overall = 24.46 dB; clean-target transfer = 21.36 dB;')
    print('  joint clean+pseudo = 22.80 dB; MMArt head-to-head = 20.79 dB.')
    print('=' * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True,
                    help='Track 2 checkpoint (Path A or Path B)')
    ap.add_argument('--out_dir', type=str, required=True,
                    help='Directory to write track2_eval.json')
    ap.add_argument('--training_jsonl', type=str, default='',
                    help='Override training jsonl (default: read from '
                         'ckpt[\'args\'][\'jsonl\'])')
    ap.add_argument('--split_seed', type=int, default=None)
    ap.add_argument('--val_ratio', type=float, default=None)
    ap.add_argument('--image_size', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--skip', type=str, nargs='+', default=[],
                    help='Skip specified eval sets (e.g. --skip mmart_real_lr)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = ckpt['args']
    image_size = args.image_size or _get(ca, 'image_size', 256)
    split_seed = args.split_seed if args.split_seed is not None \
        else _get(ca, 'split_seed', _get(ca, 'seed', 42))
    val_ratio = args.val_ratio if args.val_ratio is not None \
        else _get(ca, 'val_ratio', 0.2)

    # Auto-detect training_jsonl
    if args.training_jsonl:
        training_jsonl = Path(args.training_jsonl)
    else:
        training_jsonl = Path(_get(ca, 'jsonl',
                                    str(PROJECT_ROOT /
                                        TRACK2_EVAL_SETS['clean_expertC']
                                            ['jsonl'])))
    if not training_jsonl.is_absolute():
        training_jsonl = PROJECT_ROOT / training_jsonl

    print(f'Loading checkpoint: {args.ckpt}')
    print(f'  best_val_psnr={ckpt.get("val_psnr", 0):.2f} dB '
          f'@ Ep{ckpt.get("epoch", "?")}')
    print(f'  training_jsonl (auto): {training_jsonl}')
    print(f'  Track 2 flags: '
          f'force_nilut_gate={_get(ca, "nc_force_nilut_gate", False)}, '
          f'use_nilut_residual={_get(ca, "nc_use_nilut_residual", False)}, '
          f'nilut_gate_init={_get(ca, "nilut_gate_init", 1.0)}, '
          f'use_vera_renderer={_get(ca, "nc_use_vera_renderer", False)}, '
          f'vera_gate_init={_get(ca, "vera_gate_init", 0.0)}')

    model = build_model_from_ckpt(ckpt, image_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  model params: {n_params:,}')

    # Run leak-aware eval on each Track 2 standard set
    results = {}
    for name, info in TRACK2_EVAL_SETS.items():
        if name in args.skip:
            print(f'\n[skip] {name}')
            continue
        eval_path = PROJECT_ROOT / info['jsonl']
        if not eval_path.exists():
            print(f'\n[warn] eval set not found: {eval_path} — skipping')
            results[name] = {'error': 'jsonl not found',
                              'jsonl': str(eval_path)}
            continue
        results[name] = run_leak_aware_eval(
            model, training_jsonl, eval_path,
            split_seed, val_ratio, image_size,
            args.batch_size, device, eval_label=info['desc'])

    print_summary_table(results, args.ckpt)

    # Save combined JSON
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'ckpt': args.ckpt,
        'best_val_psnr_at_ckpt': float(ckpt.get('val_psnr', 0)),
        'epoch_at_ckpt': ckpt.get('epoch', None),
        'training_jsonl': str(training_jsonl),
        'split_seed': split_seed,
        'val_ratio': val_ratio,
        'image_size': image_size,
        'actions': list(ACTIONS),
        'model_n_params': n_params,
        'flags': {
            'use_nilut_residual': bool(_get(ca, 'nc_use_nilut_residual',
                                             False)),
            'force_nilut_gate': bool(_get(ca, 'nc_force_nilut_gate', False)),
            'nilut_hidden': _get(ca, 'nilut_hidden', 32),
            'nilut_n_layers': _get(ca, 'nilut_n_layers', 3),
            'nilut_gate_init': _get(ca, 'nilut_gate_init', 1.0),
            'use_vera_renderer': bool(_get(ca, 'nc_use_vera_renderer',
                                             False)),
            'vera_latent_dim': _get(ca, 'vera_latent_dim', 32),
            'vera_hidden': _get(ca, 'vera_hidden', 64),
            'vera_n_layers': _get(ca, 'vera_n_layers', 4),
            'vera_gate_init': _get(ca, 'vera_gate_init', 0.0),
        },
        'eval_sets': results,
    }
    out_json = out_dir / 'track2_eval.json'
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding='utf-8')
    print(f'\n[ok] summary written to {out_json}')


if __name__ == '__main__':
    main()
