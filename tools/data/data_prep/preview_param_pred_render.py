"""SOTA 模型 7D 参数预测 → apply_diff_isp 渲染 vs Expert C GT 同引擎对比.

策略:
  v11d base 对每个 action 预测 pred_p7 (7D, [-1,1])
  每个 action 只取对应参数维度 (如 wb action → white_balance)
  合并 7 个 action 的预测 → 完整 7D 参数集
  apply_diff_isp(orig, model_params) vs apply_diff_isp(orig, expert_c_params)

用法:
  D:\\anaconda\\envs\\Venus\\python.exe tools/data/data_prep/preview_param_pred_render.py --n 12
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

# action name → index in PARAM_NAMES_7D
ACTION_TO_PARAM_IDX = {
    'wb': 0,           # white_balance
    'brightness': 1,
    'contrast': 2,
    'shadows': 3,
    'highlights': 4,
    'saturation': 5,
    'clarity': 6,
}

ACTION_CN = {
    'contrast': '对比度', 'saturation': '饱和度', 'shadows': '暗部',
    'highlights': '高光', 'wb': '白平衡', 'brightness': '亮度',
    'clarity': '清晰度',
}


def denorm_param(name: str, v: float) -> float:
    cfg = PARAM_NORM_7D[name]
    return v * cfg['scale'] + cfg['center']


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ca = ckpt['args']
    ckpt_actions = ca.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)

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
    info = {
        'val_psnr': ckpt.get('val_psnr', 0),
        'epoch': ckpt.get('epoch', '?'),
        'image_size': ca.get('image_size', 256),
        'use_7d_anchor': ca.get('nc_use_7d_anchor', True),
    }
    return model, info


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
        font = ImageFont.truetype("msyh.ttc", 13)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
    draw.text((4, 4), text, fill=(255, 255, 255), font=font)
    new.paste(img, (0, bar_h))
    return new


def predict_composite_params(model, enc_input, orig_t, device):
    """Run all 7 actions, extract each action's dominant param, compose 7D."""
    composite_norm = torch.zeros(len(PARAM_NAMES_7D))
    per_action_all = {}  # action → full 7D denormalized dict

    with torch.no_grad():
        for action in ACTIONS:
            a_idx = ACTION_TO_IDX[action]
            a_oh = torch.zeros(1, len(ACTIONS), dtype=torch.float32,
                               device=device)
            a_oh[0, a_idx] = 1.0
            _, _, _, pred_p7 = model(enc_input, orig_t, a_oh)
            p7 = pred_p7[0].cpu()  # (7,)

            # Store full denormalized params for this action
            denormed = {}
            for i, name in enumerate(PARAM_NAMES_7D):
                denormed[name] = denorm_param(name, float(p7[i]))
            per_action_all[action] = denormed

            # Take dominant param index for this action
            param_idx = ACTION_TO_PARAM_IDX[action]
            composite_norm[param_idx] = p7[param_idx]

    # Denormalize composite
    composite = {}
    for i, name in enumerate(PARAM_NAMES_7D):
        composite[name] = denorm_param(name, float(composite_norm[i]))

    return composite, per_action_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',
                    default='checkpoints/lut_v11d_firered_7actions_6537/best.pt')
    ap.add_argument('--gt_json', default='data/fivek_expert_abcde_params.json')
    ap.add_argument('--jpeg_dir', default=r'E:\Data\dataset\fivek_jpeg')
    ap.add_argument('--expert', default='C')
    ap.add_argument('--out_dir', default='outputs/preview_param_pred_render')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--render_size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    print(f'[load] {args.ckpt}')
    model, info = load_model(args.ckpt, device)
    image_size = info['image_size']
    print(f'[load] val_psnr={info["val_psnr"]:.2f}dB  Ep{info["epoch"]} '
          f'image_size={image_size}  7d_anchor={info["use_7d_anchor"]}')
    print(f'[load] actions={list(ACTIONS)}')

    # 2. Expert C GT
    with open(args.gt_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    expert_records = {s['image_name']: s
                      for s in data['samples']
                      if s['expert'] == args.expert}
    print(f'[data] Expert {args.expert}: {len(expert_records)} records')

    # 3. JPEG matches
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

    # 4. Render loop
    previews = []
    all_params_log = []

    for i, (dng_name, jpg_path, gt_rec) in enumerate(samples):
        stem = dng_name.rsplit('.', 1)[0]
        print(f'[{i+1}/{len(samples)}] {stem}')

        orig_pil = Image.open(jpg_path).convert('RGB')

        # Model prediction (at model input size)
        enc_input, orig_model = preprocess(orig_pil, image_size, device)
        composite, per_action = predict_composite_params(
            model, enc_input, orig_model, device)

        # Render at larger size for visual quality
        resize_render = T.Resize((args.render_size, args.render_size))
        to_tensor = T.ToTensor()
        orig_render = to_tensor(resize_render(orig_pil)).unsqueeze(0).to(device)

        # Model composite params → apply_diff_isp
        model_params_t = {
            name: torch.tensor([composite[name]], dtype=torch.float32,
                               device=device)
            for name in PARAM_NAMES_7D
        }
        with torch.no_grad():
            model_render = apply_diff_isp(orig_render, model_params_t).clamp(0, 1)

        # Expert C GT params → apply_diff_isp
        gt_p = gt_rec['params']
        gt_params = {name: float(gt_p.get(name, 0) or 0)
                     for name in PARAM_NAMES_7D}
        gt_params_t = {
            name: torch.tensor([gt_params[name]], dtype=torch.float32,
                               device=device)
            for name in PARAM_NAMES_7D
        }
        with torch.no_grad():
            gt_render = apply_diff_isp(orig_render, gt_params_t).clamp(0, 1)

        # Build 3-panel: Original | Model Params | Expert C GT
        orig_pil_r = tensor_to_pil(orig_render)
        model_pil = tensor_to_pil(model_render)
        gt_pil = tensor_to_pil(gt_render)

        # Parameter text for labels
        model_txt = '  '.join(
            f'{n[:3]}={composite[n]:.0f}'
            if n == 'white_balance' else f'{n[:3]}={composite[n]:+.1f}'
            for n in PARAM_NAMES_7D)
        gt_txt = '  '.join(
            f'{n[:3]}={gt_params[n]:.0f}'
            if n == 'white_balance' else f'{n[:3]}={gt_params[n]:+.1f}'
            for n in PARAM_NAMES_7D)

        panels = [
            add_label(orig_pil_r, f'原图 {stem}'),
            add_label(model_pil, f'模型预测: {model_txt}'),
            add_label(gt_pil, f'Expert {args.expert} GT: {gt_txt}'),
        ]

        pw, ph = panels[0].size
        gap = 3
        strip = Image.new('RGB',
                          (pw * 3 + gap * 2, ph), (15, 23, 42))
        for j, p in enumerate(panels):
            strip.paste(p, (j * (pw + gap), 0))

        strip_path = out_dir / f'{i+1:02d}_{stem}.jpg'
        strip.save(strip_path, quality=92)
        previews.append(strip_path.name)

        # Log params
        all_params_log.append({
            'stem': stem,
            'model_composite': composite,
            'expert_c_gt': gt_params,
            'per_action_full': per_action,
        })

        # Print param comparison
        print(f'  Model:    {model_txt}')
        print(f'  Expert C: {gt_txt}')

    # 5. Contact sheet (vertical stack)
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

    # 6. Save detailed params log
    log_path = out_dir / 'params_comparison.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(all_params_log, f, indent=2, ensure_ascii=False, default=float)

    print(f'[done] params log: {log_path}')
    print(f'[done] output: {out_dir.resolve()}')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
