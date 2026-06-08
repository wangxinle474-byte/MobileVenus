"""v13a SOTA 渲染 vs Expert C GT 对比 (3-panel).

每张图选择 Expert C 最大归一化变化的维度作为 dominant action,
用 v13a (v11d base + refiner, 25.22dB) 渲染该 action.

Panel: Original | v13a (dominant action) | Expert C GT (apply_diff_isp)

用法:
  D:\\anaconda\\envs\\Venus\\python.exe tools/data/data_prep/preview_v13a_render.py --n 12
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

# action → param name (for dominant action selection)
ACTION_TO_PARAM = {
    'wb': 'white_balance',
    'brightness': 'brightness',
    'contrast': 'contrast',
    'shadows': 'shadows',
    'highlights': 'highlights',
    'saturation': 'saturation',
    'clarity': 'clarity',
}

ACTION_CN = {
    'contrast': '对比度', 'saturation': '饱和度', 'shadows': '暗部',
    'highlights': '高光', 'wb': '白平衡', 'brightness': '亮度',
    'clarity': '清晰度',
}


def normalize_expert_param(name: str, val: float) -> float:
    """Returns |normalized value| in roughly [0, 1] range."""
    cfg = PARAM_NORM_7D[name]
    return abs((val - cfg['center']) / cfg['scale'])


def pick_dominant_action(gt_params: dict, available_actions) -> str:
    """Pick the action whose param has the largest |normalized| value."""
    best_action = available_actions[0]
    best_score = -1.0
    for action in available_actions:
        param_name = ACTION_TO_PARAM[action]
        score = normalize_expert_param(param_name, gt_params.get(param_name, 0))
        if score > best_score:
            best_score = score
            best_action = action
    return best_action, best_score


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
    info = {
        'val_psnr': ckpt.get('val_psnr', 0),
        'epoch': ckpt.get('epoch', '?'),
    }
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


def add_label(img: Image.Image, text: str, bg=(30, 41, 59)) -> Image.Image:
    W, H = img.size
    bar_h = 28
    new = Image.new('RGB', (W, H + bar_h), bg)
    draw = ImageDraw.Draw(new)
    try:
        font = ImageFont.truetype("msyh.ttc", 14)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
    draw.text((4, 4), text, fill=(255, 255, 255), font=font)
    new.paste(img, (0, bar_h))
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt',
                    default='checkpoints/lut_v11d_firered_7actions_6537/best.pt')
    ap.add_argument('--refiner_ckpt',
                    default='checkpoints/v13a_firered_refine_7actions/best.pt')
    ap.add_argument('--gt_json', default='data/fivek_expert_abcde_params.json')
    ap.add_argument('--jpeg_dir', default=r'E:\Data\dataset\fivek_jpeg')
    ap.add_argument('--expert', default='C')
    ap.add_argument('--out_dir', default='outputs/preview_v13a_render')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f'[load] v11d base: {args.base_ckpt}')
    base_model, base_args = load_v11d_base(Path(args.base_ckpt), device)
    image_size = base_args.get('image_size', 256)
    print(f'[load] v13a refiner: {args.refiner_ckpt}')
    refiner, refiner_info = load_v13a_refiner(
        Path(args.refiner_ckpt), device)
    print(f'[load] refiner val_psnr={refiner_info["val_psnr"]:.2f}dB '
          f'@Ep{refiner_info["epoch"]} actions={list(ACTIONS)}')

    # Load Expert C GT
    with open(args.gt_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    expert_records = {s['image_name']: s
                      for s in data['samples']
                      if s['expert'] == args.expert}

    # Match JPEGs
    jpeg_dir = Path(args.jpeg_dir)
    available = []
    for name, rec in expert_records.items():
        stem = name.rsplit('.', 1)[0]
        jpg = jpeg_dir / f'{stem}.jpg'
        if jpg.exists():
            available.append((name, str(jpg), rec))
    print(f'[data] Expert {args.expert} matches: {len(available)}')

    rng = random.Random(args.seed)
    samples = rng.sample(available, min(args.n, len(available)))

    # Inference
    previews = []
    log = []
    for i, (dng_name, jpg_path, gt_rec) in enumerate(samples):
        stem = dng_name.rsplit('.', 1)[0]

        # Expert C GT params (full 7D)
        gt_p = gt_rec['params']
        gt_params = {n: float(gt_p.get(n, 0) or 0) for n in PARAM_NAMES_7D}

        # Pick dominant action based on Expert C edit
        action, score = pick_dominant_action(gt_params, list(ACTIONS))
        print(f'[{i+1}/{len(samples)}] {stem}  dominant={action} '
              f'(score={score:.2f})')

        orig_pil = Image.open(jpg_path).convert('RGB')

        # Run v13a (v11d base → refiner) for dominant action
        enc_input, orig_t = preprocess(orig_pil, image_size, device)
        a_idx = ACTION_TO_IDX[action]
        a_oh = torch.zeros(1, len(ACTIONS), dtype=torch.float32, device=device)
        a_oh[0, a_idx] = 1.0
        with torch.no_grad():
            base_out, _, _, _ = base_model(enc_input, orig_t, a_oh)
            refined, _ = refiner(orig_t, base_out, a_oh)

        # Expert C GT through apply_diff_isp (same input size as model)
        gt_params_t = {n: torch.tensor([gt_params[n]], dtype=torch.float32,
                                        device=device)
                       for n in PARAM_NAMES_7D}
        with torch.no_grad():
            gt_render = apply_diff_isp(orig_t, gt_params_t).clamp(0, 1)

        # 3-panel
        gt_txt = ' '.join(
            f'{n[:3]}={gt_params[n]:.0f}' if n == 'white_balance'
            else f'{n[:3]}={gt_params[n]:+.0f}'
            for n in PARAM_NAMES_7D)

        panels = [
            add_label(tensor_to_pil(orig_t), f'原图 {stem}'),
            add_label(tensor_to_pil(refined),
                      f'v13a → {ACTION_CN[action]} ({action})'),
            add_label(tensor_to_pil(gt_render),
                      f'Expert {args.expert} GT: {gt_txt}'),
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
            'stem': stem, 'dominant_action': action,
            'score': score, 'expert_c_gt': gt_params,
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

    log_path = out_dir / 'dominant_actions.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False, default=float)
    print(f'[done] log: {log_path}')
    print(f'[done] output: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
