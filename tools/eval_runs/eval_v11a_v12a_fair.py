from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.firered_baseline.train_lut import (
    ACTIONS as V11_ACTIONS,
    LUTDataset,
    NamedCurvesPredictor,
    set_actions,
)
from training.main.train_v12a_firered import (
    ACTIONS as V12_ACTIONS,
    ActionConditionedRefinementNetV4,
    FireRedV12ADataset,
    load_jsonl,
    load_param_model,
    make_denorm_fn,
    split_by_source,
)
from models.diff_isp import apply_diff_isp


def _get(cfg: dict, key: str, default):
    return cfg[key] if key in cfg else default


@torch.no_grad()
def psnr_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred.float() - target.float()).pow(2).flatten(1).mean(dim=1).clamp_min(1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def summarize(values_by_action: dict[str, list[float]]) -> dict:
    all_values = [v for values in values_by_action.values() for v in values]
    per_action = {}
    for action, values in sorted(values_by_action.items()):
        if not values:
            continue
        vals = torch.tensor(values, dtype=torch.float32)
        per_action[action] = {
            'n': int(vals.numel()),
            'psnr': float(vals.mean().item()),
            'p10': float(vals.kthvalue(max(1, int(vals.numel() * 0.1))).values.item()),
            'p90': float(vals.kthvalue(min(vals.numel(), max(1, int(vals.numel() * 0.9)))).values.item()),
        }
    overall = float(torch.tensor(all_values, dtype=torch.float32).mean().item()) if all_values else 0.0
    return {'overall_psnr': overall, 'n': len(all_values), 'per_action': per_action}


def load_v11_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt['args']
    ckpt_actions = cfg.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)
    model = NamedCurvesPredictor(
        n_colors=_get(cfg, 'nc_n_colors', 3),
        n_control_points=_get(cfg, 'nc_n_control_points', 7),
        use_attention=_get(cfg, 'nc_use_attention', False),
        per_action_curves=_get(cfg, 'nc_per_action_curves', False),
        use_7d_anchor=_get(cfg, 'nc_use_7d_anchor', True),
        use_context=_get(cfg, 'nc_use_context', False),
        action_gated_context=_get(cfg, 'nc_action_gated_context', False),
        use_action_context=_get(cfg, 'nc_use_action_context', False),
        use_region_basis=_get(cfg, 'nc_use_region_basis', False),
        use_region_param_delta=_get(cfg, 'nc_use_region_param_delta', False),
        use_learned_cn=_get(cfg, 'nc_use_learned_cn', False),
        use_wb_head=_get(cfg, 'nc_use_wb_head', False),
        use_nilut_residual=_get(cfg, 'nc_use_nilut_residual', False),
        use_vera_renderer=_get(cfg, 'nc_use_vera_renderer', False),
        use_implicit_head=_get(cfg, 'use_implicit_head', False),
        implicit_head_base_ch=_get(cfg, 'implicit_head_base_ch', 32),
        implicit_head_gate_init=_get(cfg, 'implicit_head_gate_init', 0.0),
        nilut_hidden=_get(cfg, 'nilut_hidden', 32),
        nilut_n_layers=_get(cfg, 'nilut_n_layers', 3),
        nilut_n_freq=_get(cfg, 'nilut_n_freq', 4),
        nilut_gate_init=_get(cfg, 'nilut_gate_init', 1.0),
        image_size=_get(cfg, 'image_size', 256),
        n_actions=len(V11_ACTIONS),
        dropout=_get(cfg, 'dropout', 0.5),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model, ckpt


@torch.no_grad()
def eval_v11(model, samples: list[dict], image_size: int, batch_size: int, device: torch.device):
    ds = LUTDataset(samples, image_size=image_size, is_train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    values_by_action = defaultdict(list)
    for batch in loader:
        output, _, _, _ = model(
            batch['enc_input'].to(device),
            batch['orig'].to(device),
            batch['action_onehot'].to(device),
        )
        target = batch['target'].to(device)
        psnr = psnr_per_sample(output, target).detach().cpu().tolist()
        for action, value in zip(batch['action'], psnr):
            values_by_action[str(action)].append(float(value))
    return summarize(values_by_action)


@torch.no_grad()
def eval_v12(
    ckpt_path: Path,
    samples: list[dict],
    asset_root: Path,
    orig_dir: Path | None,
    base_ch: int,
    batch_size: int,
    device: torch.device,
):
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    model = ActionConditionedRefinementNetV4(
        base_ch=base_ch,
        n_actions=len(V12_ACTIONS),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    param_version = cfg.get('param_version', 'v8') if isinstance(cfg, dict) else 'v8'
    param_model = load_param_model(asset_root, param_version, device)
    denorm = make_denorm_fn()

    ds = FireRedV12ADataset(samples, PROJECT_ROOT, orig_dir, is_train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    values_512 = defaultdict(list)
    values_256 = defaultdict(list)
    for batch_idx, batch in enumerate(loader):
        img_224 = batch['image_224'].to(device)
        raw_512 = batch['raw_512'].to(device)
        target_512 = batch['target_512'].to(device)
        action_oh = batch['action_onehot'].to(device)
        out = param_model(img_224)
        params_key = 'norm_params' if 'norm_params' in out else 'params_norm'
        pred_phys = denorm(out[params_key], device)
        rendered = apply_diff_isp(raw_512.float(), pred_phys).clamp(0, 1)
        rendered = torch.nan_to_num(rendered, nan=0.5)
        enhanced = model(rendered, action_oh).clamp(0, 1)

        psnr512 = psnr_per_sample(enhanced, target_512).detach().cpu().tolist()
        enhanced_256 = F.interpolate(enhanced, size=(256, 256), mode='bilinear', align_corners=False)
        target_256 = F.interpolate(target_512, size=(256, 256), mode='bilinear', align_corners=False)
        psnr256 = psnr_per_sample(enhanced_256, target_256).detach().cpu().tolist()
        batch_samples = samples[batch_idx * batch_size: batch_idx * batch_size + len(psnr512)]
        for sample, v512, v256 in zip(batch_samples, psnr512, psnr256):
            action = str(sample['action'])
            values_512[action].append(float(v512))
            values_256[action].append(float(v256))
    result = summarize(values_256)
    result['overall_psnr_512'] = summarize(values_512)['overall_psnr']
    result['per_action_512'] = summarize(values_512)['per_action']
    return result, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', default='outputs/firered_v12a_existing/pseudo_labels.jsonl')
    ap.add_argument('--v11_ckpt', default='checkpoints/lut_v11a_action_gated_context/best.pt')
    ap.add_argument('--v12_ckpt', default='checkpoints/refinement_v12a_firered_6537/best.pt')
    ap.add_argument('--asset_root', default='.')
    ap.add_argument('--orig_dir', default='E:/Data/dataset/fivek_jpeg')
    ap.add_argument('--out', default='outputs/fair_eval_v11a_v12a/results.json')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--batch_size_v11', type=int, default=4)
    ap.add_argument('--batch_size_v12', type=int, default=1)
    ap.add_argument('--base_ch', type=int, default=48)
    ap.add_argument('--max_samples', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_absolute():
        jsonl_path = PROJECT_ROOT / jsonl_path
    samples = load_jsonl(jsonl_path)
    _, val_samples = split_by_source(samples, args.val_ratio, args.seed)

    v11_model, v11_ckpt = load_v11_model(PROJECT_ROOT / args.v11_ckpt, device)
    v11_actions = list(V11_ACTIONS)
    val_samples = [s for s in val_samples if s.get('action') in set(v11_actions)]
    if args.max_samples > 0:
        val_samples = val_samples[:args.max_samples]

    print(f'device={device}')
    print(f'val_samples={len(val_samples)} actions={v11_actions}')

    v11_image_size = int(v11_ckpt['args'].get('image_size', 256))
    v11_result = eval_v11(v11_model, val_samples, v11_image_size, args.batch_size_v11, device)
    print(f'v11a fair 256px PSNR={v11_result["overall_psnr"]:.2f} n={v11_result["n"]}')

    v12_result, v12_ckpt = eval_v12(
        PROJECT_ROOT / args.v12_ckpt,
        val_samples,
        PROJECT_ROOT / args.asset_root if not Path(args.asset_root).is_absolute() else Path(args.asset_root),
        Path(args.orig_dir) if args.orig_dir else None,
        args.base_ch,
        args.batch_size_v12,
        device,
    )
    print(f'v12a fair 256px PSNR={v12_result["overall_psnr"]:.2f} n={v12_result["n"]}')
    print(f'v12a fair 512px PSNR={v12_result["overall_psnr_512"]:.2f} n={v12_result["n"]}')

    result = {
        'jsonl': str(jsonl_path),
        'seed': args.seed,
        'val_ratio': args.val_ratio,
        'filtered_actions': v11_actions,
        'n_val': len(val_samples),
        'v11': {
            'ckpt': args.v11_ckpt,
            'checkpoint_epoch': v11_ckpt.get('epoch'),
            'checkpoint_val_psnr': v11_ckpt.get('val_psnr'),
            'fair_eval': v11_result,
        },
        'v12': {
            'ckpt': args.v12_ckpt,
            'checkpoint_epoch': v12_ckpt.get('epoch'),
            'checkpoint_val_psnr': v12_ckpt.get('val_psnr'),
            'base_ch': args.base_ch,
            'fair_eval': v12_result,
        },
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
