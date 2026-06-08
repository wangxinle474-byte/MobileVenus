"""v13a SOTA model vs Expert C GT 对比预览.

Pipeline:
  v11d base (frozen) + v13a refiner → refined (per action)
  Expert C params → apply_diff_isp → GT render

For each sample: 1 row of 9 panels:
  Original | 7 action outputs | Expert C GT

用法:
  D:\\anaconda\\envs\\Venus\\python.exe tools/data/data_prep/preview_v13a_vs_expert.py --n 12
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
    raise ImportError('需要 torchvision。用 Venus conda 环境: '
                      'D:\\anaconda\\envs\\Venus\\python.exe')

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Expert C 7D param names (for apply_diff_isp)
EXPERT_PARAM_NAMES = [
    'white_balance', 'brightness', 'contrast',
    'shadows', 'highlights', 'saturation', 'clarity',
]

ACTION_CN = {
    'contrast': '对比度', 'saturation': '饱和度', 'shadows': '暗部',
    'highlights': '高光', 'wb': '白平衡', 'brightness': '亮度',
    'clarity': '清晰度',
}


# ── model loading ────────────────────────────────────────────────

def load_v11d_base(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
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
        'per_action': ckpt.get('val_per_action', {}),
    }
    return refiner, info


# ── preprocessing ────────────────────────────────────────────────

def preprocess(img: Image.Image, image_size: int, device: torch.device):
    resize = T.Resize((image_size, image_size))
    to_tensor = T.ToTensor()
    norm = T.Normalize(NORM_MEAN, NORM_STD)
    img_resized = resize(img)
    orig_t = to_tensor(img_resized).unsqueeze(0).to(device)
    enc_input = norm(orig_t.squeeze(0)).unsqueeze(0).to(device)
    return enc_input, orig_t


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


# ── main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt',
                    default='checkpoints/lut_v11d_firered_7actions_6537/best.pt')
    ap.add_argument('--refiner_ckpt',
                    default='checkpoints/v13a_firered_refine_7actions/best.pt')
    ap.add_argument('--gt_json', default='data/fivek_expert_abcde_params.json')
    ap.add_argument('--jpeg_dir', default=r'E:\Data\dataset\fivek_jpeg')
    ap.add_argument('--expert', default='C')
    ap.add_argument('--out_dir', default='outputs/preview_v13a_vs_expert')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load models
    print(f'[load] v11d base: {args.base_ckpt}')
    base_model, base_args = load_v11d_base(Path(args.base_ckpt), device)
    image_size = base_args.get('image_size', 256)
    print(f'[load] actions={list(ACTIONS)} image_size={image_size}')

    print(f'[load] v13a refiner: {args.refiner_ckpt}')
    refiner, refiner_info = load_v13a_refiner(Path(args.refiner_ckpt), device)
    print(f'[load] refiner val_psnr={refiner_info["val_psnr"]:.2f}dB '
          f'@ Ep{refiner_info["epoch"]}')
    if refiner_info.get('per_action'):
        pa = refiner_info['per_action']
        print(f'[load] per-action: '
              f'{" ".join(f"{a[:3]}={pa[a]:.2f}" for a in ACTIONS if a in pa)}')

    # 2. Load Expert C GT params
    with open(args.gt_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    expert_records = {s['image_name']: s
                      for s in data['samples']
                      if s['expert'] == args.expert}
    print(f'[data] Expert {args.expert}: {len(expert_records)} GT records')

    # 3. Find JPEG matches
    jpeg_dir = Path(args.jpeg_dir)
    available = []
    for name, rec in expert_records.items():
        stem = name.rsplit('.', 1)[0]
        jpg = jpeg_dir / f'{stem}.jpg'
        if jpg.exists():
            available.append((name, str(jpg), rec))
    print(f'[data] JPEG matches: {len(available)}')

    rng = random.Random(args.seed)
    samples = rng.sample(available, min(args.n, len(available)))

    # 4. Inference loop
    previews = []
    for i, (dng_name, jpg_path, gt_rec) in enumerate(samples):
        stem = dng_name.rsplit('.', 1)[0]
        print(f'[{i+1}/{len(samples)}] {stem}')

        orig_pil = Image.open(jpg_path).convert('RGB')
        enc_input, orig_t = preprocess(orig_pil, image_size, device)

        # Run all 7 actions through v13a
        action_results = {}
        with torch.no_grad():
            for action in ACTIONS:
                action_idx = ACTION_TO_IDX[action]
                a_oh = torch.zeros(1, len(ACTIONS), dtype=torch.float32,
                                   device=device)
                a_oh[0, action_idx] = 1.0

                # v11d base
                base_out, _, _, _ = base_model(enc_input, orig_t, a_oh)
                # v13a refiner
                refined, delta = refiner(orig_t, base_out, a_oh)
                action_results[action] = {
                    'refined': refined[0].cpu(),
                    'base': base_out[0].cpu(),
                }

        # Expert C GT → apply_diff_isp
        gt_p = gt_rec['params']
        P_gt = {p: float(gt_p.get(p, 0) or 0) for p in EXPERT_PARAM_NAMES}
        gt_params_t = {p: torch.tensor([P_gt[p]], dtype=torch.float32,
                                        device=device)
                       for p in EXPERT_PARAM_NAMES}
        with torch.no_grad():
            gt_render = apply_diff_isp(orig_t, gt_params_t).clamp(0, 1)
        gt_pil = tensor_to_pil(gt_render)

        # Build panel: orig | 7 actions | Expert C GT
        panels = [add_label(tensor_to_pil(orig_t), '原图')]
        for action in ACTIONS:
            r = action_results[action]
            panels.append(add_label(
                tensor_to_pil(r['refined']),
                f'v13a {ACTION_CN.get(action, action)}'))
        panels.append(add_label(gt_pil, f'Expert {args.expert} GT'))

        # Horizontal strip
        pw, ph = panels[0].size
        gap = 2
        n_panels = len(panels)
        strip = Image.new('RGB',
                          (pw * n_panels + gap * (n_panels - 1), ph),
                          (15, 23, 42))
        for j, p in enumerate(panels):
            strip.paste(p, (j * (pw + gap), 0))

        strip_path = out_dir / f'{i+1:02d}_{stem}.jpg'
        strip.save(strip_path, quality=92)
        previews.append({'stem': stem, 'panel': strip_path.name})

    # 5. Contact sheet (2 cols since strips are wide)
    if previews:
        strips = [Image.open(out_dir / p['panel']) for p in previews]
        sw, sh = strips[0].size
        cols = 1  # strips are very wide, 1 col
        rows_count = len(strips)
        contact = Image.new('RGB', (sw, sh * rows_count + 6 * (rows_count - 1)),
                            (30, 30, 30))
        for idx, s in enumerate(strips):
            contact.paste(s, (0, idx * (sh + 6)))
        contact_path = out_dir / '_contact.jpg'
        contact.save(contact_path, quality=85)
        print(f'\n[done] contact sheet: {contact_path}')

    # 6. Summary
    summary = {
        'base_ckpt': args.base_ckpt,
        'refiner_ckpt': args.refiner_ckpt,
        'refiner_val_psnr': refiner_info['val_psnr'],
        'expert': args.expert,
        'n_samples': len(previews),
        'actions': list(ACTIONS),
    }
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f'[done] summary: {summary_path}')
    print(f'[done] output: {out_dir.resolve()}')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
