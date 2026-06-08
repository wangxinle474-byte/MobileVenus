"""v13a 自主决策 action: 跑全部 7 个 action, 用 L1 delta 选最大的作为模型选择.

每个 action 的 v13a 输出与原图的 L1 距离 = 模型认为该维度需要调整的强度.
选 L1 最大的 action 作为模型的自主决策结果.

Panel: Original | v13a (model auto-pick) | Expert C GT
另外: 在底部显示 7 个 action 的 L1 delta 排序.

用法:
  D:\\anaconda\\envs\\Venus\\python.exe tools/data/data_prep/preview_v13a_auto_action.py --n 12
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from models.diff_isp import apply_diff_isp  # noqa: E402
from models.firered_residual_refiner import FireRedResidualRefiner  # noqa: E402
from training.firered_baseline.train_lut import (  # noqa: E402
    ACTIONS, ACTION_TO_IDX, PARAM_NAMES_7D, PARAM_NORM_7D,
    NamedCurvesPredictor, set_actions,
)

try:
    import torchvision.transforms as T
except ImportError:
    raise ImportError('需要 torchvision。用 Venus conda: '
                      'D:\\anaconda\\envs\\Venus\\python.exe')

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

ACTION_TO_PARAM = {
    'wb': 'white_balance', 'brightness': 'brightness',
    'contrast': 'contrast', 'shadows': 'shadows',
    'highlights': 'highlights', 'saturation': 'saturation',
    'clarity': 'clarity',
}

ACTION_CN = {
    'contrast': '对比度', 'saturation': '饱和度', 'shadows': '暗部',
    'highlights': '高光', 'wb': '白平衡', 'brightness': '亮度',
    'clarity': '清晰度',
}


def load_v11d_base(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    ca = ckpt['args']
    if ca.get('actions') is not None:
        set_actions(ca['actions'])
    model = NamedCurvesPredictor(
        n_colors=ca.get('nc_n_colors', 3),
        n_control_points=ca.get('nc_n_control_points', 7),
        use_attention=ca.get('nc_use_attention', False),
        per_action_curves=ca.get('nc_per_action_curves', False),
        use_7d_anchor=ca.get('nc_use_7d_anchor', True),
        use_context=ca.get('nc_use_context', False),
        action_gated_context=ca.get('nc_action_gated_context', False),
        use_action_context=ca.get('nc_use_action_context', False),
        use_region_basis=ca.get('nc_use_region_basis', False),
        use_region_param_delta=ca.get('nc_use_region_param_delta', False),
        use_learned_cn=ca.get('nc_use_learned_cn', False),
        use_wb_head=ca.get('nc_use_wb_head', False),
        use_nilut_residual=ca.get('nc_use_nilut_residual', False),
        use_vera_renderer=ca.get('nc_use_vera_renderer', False),
        nilut_hidden=ca.get('nilut_hidden', 32),
        nilut_n_layers=ca.get('nilut_n_layers', 3),
        nilut_n_freq=ca.get('nilut_n_freq', 4),
        nilut_gate_init=ca.get('nilut_gate_init', 1.0),
        use_implicit_head=ca.get('use_implicit_head', False),
        implicit_head_base_ch=ca.get('implicit_head_base_ch', 32),
        implicit_head_gate_init=ca.get('implicit_head_gate_init', 0.0),
        image_size=ca.get('image_size', 256),
        n_actions=len(ACTIONS),
        dropout=ca.get('dropout', 0.5),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ca


def load_v13a_refiner(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    ca = ckpt['args']
    refiner = FireRedResidualRefiner(
        base_ch=ca.get('base_ch', 32),
        n_actions=len(ACTIONS),
        delta_scale=ca.get('delta_scale', 0.5),
    ).to(device)
    refiner.load_state_dict(ckpt['model_state_dict'], strict=True)
    refiner.eval()
    info = {'val_psnr': ckpt.get('val_psnr', 0),
            'epoch': ckpt.get('epoch', '?')}
    return refiner, info


def preprocess(img: Image.Image, image_size: int, device: torch.device):
    resize = T.Resize((image_size, image_size))
    to_tensor = T.ToTensor()
    norm = T.Normalize(NORM_MEAN, NORM_STD)
    img_r = resize(img)
    orig_t = to_tensor(img_r).unsqueeze(0).to(device)
    enc_t = norm(orig_t.squeeze(0)).unsqueeze(0).to(device)
    return enc_t, orig_t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255
           ).astype(np.uint8)
    return Image.fromarray(arr)


def add_label(img: Image.Image, text: str, bg=(30, 41, 59),
              fontsize=14) -> Image.Image:
    W, H = img.size
    bar_h = 28
    new = Image.new('RGB', (W, H + bar_h), bg)
    draw = ImageDraw.Draw(new)
    try:
        font = ImageFont.truetype("msyh.ttc", fontsize)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", fontsize - 2)
        except Exception:
            font = ImageFont.load_default()
    draw.text((4, 4), text, fill=(255, 255, 255), font=font)
    new.paste(img, (0, bar_h))
    return new


def normalize_expert_param(name: str, val: float) -> float:
    cfg = PARAM_NORM_7D[name]
    return abs((val - cfg['center']) / cfg['scale'])


def expert_dominant(gt_params: dict, available_actions) -> str:
    best_a, best_s = available_actions[0], -1.0
    for a in available_actions:
        s = normalize_expert_param(ACTION_TO_PARAM[a], gt_params.get(
            ACTION_TO_PARAM[a], 0))
        if s > best_s:
            best_s, best_a = s, a
    return best_a, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt',
                    default='checkpoints/lut_v11d_firered_7actions_6537/best.pt')
    ap.add_argument('--refiner_ckpt',
                    default='checkpoints/v13a_firered_refine_7actions/best.pt')
    ap.add_argument('--gt_json', default='data/fivek_expert_abcde_params.json')
    ap.add_argument('--jpeg_dir', default=r'E:\Data\dataset\fivek_jpeg')
    ap.add_argument('--expert', default='C')
    ap.add_argument('--out_dir', default='outputs/preview_v13a_auto_action')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--n_calib', type=int, default=100,
                    help='Samples for per-action L1 baseline calibration')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f'[load] v11d base + v13a refiner')
    base_model, base_args = load_v11d_base(Path(args.base_ckpt), device)
    image_size = base_args.get('image_size', 256)
    refiner, refiner_info = load_v13a_refiner(
        Path(args.refiner_ckpt), device)
    print(f'[load] val_psnr={refiner_info["val_psnr"]:.2f}dB '
          f'actions={list(ACTIONS)}')

    # Expert C GT
    with open(args.gt_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    expert_records = {s['image_name']: s
                      for s in data['samples']
                      if s['expert'] == args.expert}
    jpeg_dir = Path(args.jpeg_dir)
    available = []
    for name, rec in expert_records.items():
        stem = name.rsplit('.', 1)[0]
        jpg = jpeg_dir / f'{stem}.jpg'
        if jpg.exists():
            available.append((name, str(jpg), rec))
    print(f'[data] matches: {len(available)}')

    rng = random.Random(args.seed)
    samples = rng.sample(available, min(args.n, len(available)))

    # ── Calibration: compute per-action mean L1 delta on N random samples ──
    n_calib = min(args.n_calib, len(available))
    print(f'\n[calib] Running calibration on {n_calib} samples to '
          f'normalize per-action L1 baseline...')
    calib_samples = rng.sample(available, n_calib)
    per_action_deltas = {a: [] for a in ACTIONS}
    for j, (_, jpg_path, _) in enumerate(calib_samples):
        if (j + 1) % 20 == 0:
            print(f'  calib [{j+1}/{n_calib}]')
        orig_pil = Image.open(jpg_path).convert('RGB')
        enc_input, orig_t = preprocess(orig_pil, image_size, device)
        with torch.no_grad():
            for action in ACTIONS:
                a_idx = ACTION_TO_IDX[action]
                a_oh = torch.zeros(1, len(ACTIONS), dtype=torch.float32,
                                   device=device)
                a_oh[0, a_idx] = 1.0
                base_out, _, _, _ = base_model(enc_input, orig_t, a_oh)
                refined, _ = refiner(orig_t, base_out, a_oh)
                per_action_deltas[action].append(
                    (refined - orig_t).abs().mean().item())

    # Compute mean & std per action
    action_stats = {}
    for action in ACTIONS:
        arr = np.array(per_action_deltas[action])
        action_stats[action] = {
            'mean': float(arr.mean()),
            'std': float(arr.std() + 1e-6),
        }
    print('[calib] Per-action L1 baseline:')
    for action in ACTIONS:
        s = action_stats[action]
        print(f'  {action:12s}  mean={s["mean"]:.4f}  std={s["std"]:.4f}')

    previews = []
    log = []
    n_match_expert = 0

    for i, (dng_name, jpg_path, gt_rec) in enumerate(samples):
        stem = dng_name.rsplit('.', 1)[0]
        gt_p = gt_rec['params']
        gt_params = {n: float(gt_p.get(n, 0) or 0) for n in PARAM_NAMES_7D}
        expert_a, expert_score = expert_dominant(gt_params, list(ACTIONS))

        orig_pil = Image.open(jpg_path).convert('RGB')
        enc_input, orig_t = preprocess(orig_pil, image_size, device)

        # Run all 7 actions through v13a, compute L1 delta + z-score
        action_results = {}
        with torch.no_grad():
            for action in ACTIONS:
                a_idx = ACTION_TO_IDX[action]
                a_oh = torch.zeros(1, len(ACTIONS), dtype=torch.float32,
                                   device=device)
                a_oh[0, a_idx] = 1.0
                base_out, _, _, _ = base_model(enc_input, orig_t, a_oh)
                refined, _ = refiner(orig_t, base_out, a_oh)
                l1_delta = (refined - orig_t).abs().mean().item()
                stats = action_stats[action]
                z = (l1_delta - stats['mean']) / stats['std']
                action_results[action] = {
                    'refined': refined.cpu(),
                    'l1_delta': l1_delta,
                    'z_score': z,
                }

        # Pick action with max z-score = "this image needs this action more
        # than typical" (normalized across per-action baselines)
        sorted_actions = sorted(action_results.items(),
                                key=lambda kv: kv[1]['z_score'],
                                reverse=True)
        model_action = sorted_actions[0][0]
        model_l1 = sorted_actions[0][1]['l1_delta']
        model_z = sorted_actions[0][1]['z_score']

        match = (model_action == expert_a)
        if match:
            n_match_expert += 1
        match_str = '✓' if match else '✗'
        print(f'[{i+1}/{len(samples)}] {stem}  '
              f'model={model_action} (z={model_z:+.2f}, L1={model_l1:.4f})  '
              f'expert={expert_a}  {match_str}')

        # Top-3 ranking text (by z-score)
        top3 = sorted_actions[:3]
        rank_txt = ' > '.join(
            f'{ACTION_CN[a]}(z={r["z_score"]:+.1f})' for a, r in top3)

        # Expert C GT render
        gt_params_t = {n: torch.tensor([gt_params[n]], dtype=torch.float32,
                                        device=device)
                       for n in PARAM_NAMES_7D}
        with torch.no_grad():
            gt_render = apply_diff_isp(orig_t, gt_params_t).clamp(0, 1)

        # 3-panel
        model_refined = action_results[model_action]['refined'].to(device)
        gt_txt = ' '.join(
            f'{n[:3]}={gt_params[n]:.0f}' if n == 'white_balance'
            else f'{n[:3]}={gt_params[n]:+.0f}'
            for n in PARAM_NAMES_7D)

        panels = [
            add_label(tensor_to_pil(orig_t), f'原图 {stem}'),
            add_label(tensor_to_pil(model_refined),
                      f'v13a 自动→ {ACTION_CN[model_action]} '
                      f'({model_action})  | TOP3: {rank_txt}', fontsize=12),
            add_label(tensor_to_pil(gt_render),
                      f'Expert {args.expert} GT (dom={expert_a}): {gt_txt}',
                      fontsize=12),
        ]

        pw, ph = panels[0].size
        gap = 3
        strip = Image.new('RGB', (pw * 3 + gap * 2, ph), (15, 23, 42))
        for j, p in enumerate(panels):
            strip.paste(p, (j * (pw + gap), 0))

        strip_path = out_dir / f'{i+1:02d}_{stem}.jpg'
        strip.save(strip_path, quality=92)
        previews.append(strip_path.name)
        log.append({
            'stem': stem,
            'model_action': model_action,
            'model_l1_delta': model_l1,
            'model_z_score': model_z,
            'expert_dominant': expert_a,
            'expert_score': expert_score,
            'match': match,
            'all_l1_deltas': {a: r['l1_delta']
                              for a, r in action_results.items()},
            'all_z_scores': {a: r['z_score']
                             for a, r in action_results.items()},
            'expert_c_gt': gt_params,
        })

    # Contact sheet
    if previews:
        strips = [Image.open(out_dir / p) for p in previews]
        sw, sh = strips[0].size
        vgap = 4
        contact = Image.new('RGB',
                            (sw, sh * len(strips) + vgap * (len(strips) - 1)),
                            (30, 30, 30))
        for idx, s in enumerate(strips):
            contact.paste(s, (0, idx * (sh + vgap)))
        contact_path = out_dir / '_contact.jpg'
        contact.save(contact_path, quality=85)
        print(f'\n[done] contact: {contact_path}')

    # Summary
    print(f'\n=== Summary ===')
    print(f'Total: {len(samples)}')
    print(f'Model-Expert agreement: {n_match_expert}/{len(samples)} '
          f'({100*n_match_expert/len(samples):.0f}%)')

    # Action distribution
    from collections import Counter
    model_dist = Counter(d['model_action'] for d in log)
    expert_dist = Counter(d['expert_dominant'] for d in log)
    print(f'Model picks:  {dict(model_dist.most_common())}')
    print(f'Expert picks: {dict(expert_dist.most_common())}')

    log_path = out_dir / 'auto_action_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': len(samples),
                'model_expert_agreement': n_match_expert,
                'agreement_rate': n_match_expert / len(samples),
                'model_distribution': dict(model_dist),
                'expert_distribution': dict(expert_dist),
            },
            'samples': log,
        }, f, indent=2, ensure_ascii=False, default=float)
    print(f'[done] log: {log_path}')
    print(f'[done] output: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
