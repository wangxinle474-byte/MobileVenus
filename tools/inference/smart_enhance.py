"""Smart Enhance: 智能美化 — 组合多个 action 的参数预测，全分辨率渲染.

核心思路:
  1. 模型对 5 个 action 分别预测 7D ISP 参数
  2. 每个 action 是其"活动参数"的专家 (contrast→对比度, wb→色温, ...)
  3. 从每个 action 提取其专长参数，组合成统一的 7D 参数
  4. 用 apply_diff_isp 在原图全分辨率上渲染

Usage:
  python tools/smart_enhance.py --image photo.jpg
  python tools/smart_enhance.py --image photo.jpg --strength 0.7
  python tools/smart_enhance.py --image photo.jpg --strength 1.0 --no_wb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.firered_baseline.train_lut import (  # noqa: E402
    ACTIONS,
    ACTION_TO_IDX,
    ACTION_TO_PARAM_7D_INDICES,
    PARAM_NAMES_7D,
    PARAM_NORM_7D,
    NamedCurvesPredictor,
    set_actions,
)
from models.diff_isp import apply_diff_isp  # noqa: E402

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Default (neutral) physical values — no edit
DEFAULT_PARAMS = {
    'white_balance': 5500.0,
    'brightness': 0.0,
    'contrast': 0.0,
    'shadows': 0.0,
    'highlights': 0.0,
    'saturation': 0.0,
    'clarity': 0.0,
}

ACTION_CN = {
    'contrast': '对比度', 'saturation': '饱和度',
    'shadows': '暗部', 'highlights': '高光', 'wb': '白平衡',
    'brightness': '亮度', 'clarity': '清晰度',
}

PARAM_CN = {
    'white_balance': 'WB色温(K)', 'brightness': '亮度',
    'contrast': '对比度', 'shadows': '暗部',
    'highlights': '高光', 'saturation': '饱和度', 'clarity': '清晰度',
}


def denormalize_param(name: str, v_norm: float) -> float:
    cfg = PARAM_NORM_7D[name]
    return v_norm * cfg['scale'] + cfg['center']


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
    return model, ca.get('image_size', 256)


def predict_all_actions(model, image: Image.Image, image_size: int, device):
    """Run model on all checkpoint actions, return per-action 7D params (physical)."""
    resize = T.Resize((image_size, image_size))
    to_tensor = T.ToTensor()
    norm = T.Normalize(NORM_MEAN, NORM_STD)

    img_resized = resize(image)
    orig_t = to_tensor(img_resized).unsqueeze(0).to(device)
    enc_input = norm(orig_t[0]).unsqueeze(0).to(device)

    per_action_params = {}
    with torch.no_grad():
        for action in ACTIONS:
            action_idx = ACTION_TO_IDX[action]
            action_oh = torch.zeros(1, len(ACTIONS), dtype=torch.float32, device=device)
            action_oh[0, action_idx] = 1.0

            _, _, _, pred_p7 = model(enc_input, orig_t, action_oh)
            pred_norm = pred_p7[0].cpu().numpy()

            params = {}
            for i, name in enumerate(PARAM_NAMES_7D):
                params[name] = denormalize_param(name, float(pred_norm[i]))
            per_action_params[action] = params

    return per_action_params


def combine_params(per_action_params: dict, strength: float = 0.7,
                   disable_actions: set = None) -> dict:
    """Combine per-action predictions into a single 7D parameter set.

    Strategy:
      - For each param, use the prediction from its "expert" action
      - params without a dedicated expert action: average across actions
      - Apply strength scaling: blend between default and predicted
    """
    if disable_actions is None:
        disable_actions = set()

    # Map: param_name → expert_action
    param_expert = {}
    for ai, action in enumerate(ACTIONS):
        param_idx = ACTION_TO_PARAM_7D_INDICES[ai]
        param_name = PARAM_NAMES_7D[param_idx]
        param_expert[param_name] = action

    combined = {}
    for name in PARAM_NAMES_7D:
        default_val = DEFAULT_PARAMS[name]

        if name in param_expert:
            # This param has a dedicated expert action
            expert_action = param_expert[name]
            if expert_action in disable_actions:
                combined[name] = default_val
            else:
                raw_val = per_action_params[expert_action][name]
                # Blend: default + strength * (predicted - default)
                combined[name] = default_val + strength * (raw_val - default_val)
        else:
            # Secondary params (brightness, clarity): average across active actions
            vals = [per_action_params[a][name] for a in ACTIONS
                    if a not in disable_actions]
            if vals:
                avg = sum(vals) / len(vals)
                combined[name] = default_val + strength * (avg - default_val)
            else:
                combined[name] = default_val

    return combined


def render_full_res(image: Image.Image, params: dict, device) -> Image.Image:
    """Apply 7D ISP params to full-resolution image."""
    to_tensor = T.ToTensor()
    img_t = to_tensor(image).unsqueeze(0).to(device)  # (1, 3, H, W)

    # Build params dict with tensors
    params_t = {}
    for name in PARAM_NAMES_7D:
        params_t[name] = torch.tensor([params[name]], dtype=torch.float32, device=device)

    with torch.no_grad():
        output = apply_diff_isp(img_t, params_t)

    arr = (output[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_comparison(orig: Image.Image, enhanced: Image.Image,
                     params: dict, max_side: int = 800) -> Image.Image:
    """Side-by-side comparison with parameter overlay."""
    # Resize for display
    W, H = orig.size
    scale = min(max_side / W, max_side / H, 1.0)
    new_w, new_h = int(W * scale), int(H * scale)

    orig_disp = orig.resize((new_w, new_h), Image.LANCZOS)
    enh_disp = enhanced.resize((new_w, new_h), Image.LANCZOS)

    # Add labels
    bar_h = 36
    gap = 4

    canvas = Image.new('RGB', (new_w * 2 + gap, new_h + bar_h), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("msyh.ttc", 16)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

    # Draw labels
    draw.text((new_w // 2 - 30, 8), "原图", fill=(180, 180, 180), font=font)
    draw.text((new_w + gap + new_w // 2 - 50, 8), "智能增强",
              fill=(56, 189, 248), font=font)

    canvas.paste(orig_disp, (0, bar_h))
    canvas.paste(enh_disp, (new_w + gap, bar_h))

    return canvas


def main():
    ap = argparse.ArgumentParser(description='Smart Enhance: 智能美化')
    ap.add_argument('--image', required=True, help='Input image path')
    ap.add_argument('--strength', type=float, default=0.7,
                    help='Enhancement strength 0.0-1.0 (default: 0.7)')
    ap.add_argument('--ckpt',
                    default='checkpoints/lut_v11a_pathX_implicit_head/best.pt',
                    help='Model checkpoint')
    ap.add_argument('--out_dir', default=None,
                    help='Output directory (default: outputs/smart_enhance/<stem>)')
    ap.add_argument('--no_wb', action='store_true',
                    help='Disable white balance adjustment')
    ap.add_argument('--no_contrast', action='store_true',
                    help='Disable contrast adjustment')
    ap.add_argument('--no_saturation', action='store_true',
                    help='Disable saturation adjustment')
    ap.add_argument('--strengths', default=None,
                    help='Comma-separated per-action strengths in checkpoint action order')
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading model: {args.ckpt}")
    model, image_size = load_model(args.ckpt, device)
    print(f"  Device: {device}")

    # Load image
    orig_pil = Image.open(image_path).convert('RGB')
    W, H = orig_pil.size
    print(f"Input: {image_path.name} ({W}×{H})")

    # Predict per-action parameters
    print(f"Predicting parameters for all {len(ACTIONS)} actions...")
    per_action_params = predict_all_actions(model, orig_pil, image_size, device)

    # Determine disabled actions
    disable = set()
    if args.no_wb:
        disable.add('wb')
    if args.no_contrast:
        disable.add('contrast')
    if args.no_saturation:
        disable.add('saturation')

    # Combine parameters
    if args.strengths:
        # Per-action strengths mode
        s_vals = [float(x) for x in args.strengths.split(',')]
        assert len(s_vals) == len(ACTIONS), f"Need {len(ACTIONS)} strength values"
        combined = dict(DEFAULT_PARAMS)
        for ai, action in enumerate(ACTIONS):
            param_idx = ACTION_TO_PARAM_7D_INDICES[ai]
            param_name = PARAM_NAMES_7D[param_idx]
            raw = per_action_params[action][param_name]
            combined[param_name] = (DEFAULT_PARAMS[param_name]
                                    + s_vals[ai] * (raw - DEFAULT_PARAMS[param_name]))
        # Params without a dedicated expert action use mean strength.
        mean_s = sum(s_vals) / len(s_vals)
        expert_params = {
            PARAM_NAMES_7D[ACTION_TO_PARAM_7D_INDICES[ai]]
            for ai in range(len(ACTIONS))
        }
        for name in PARAM_NAMES_7D:
            if name in expert_params:
                continue
            vals = [per_action_params[a][name] for a in ACTIONS]
            avg = sum(vals) / len(vals)
            combined[name] = DEFAULT_PARAMS[name] + mean_s * (avg - DEFAULT_PARAMS[name])
    else:
        combined = combine_params(per_action_params, args.strength, disable)

    # Print combined parameters
    print(f"\n{'='*60}")
    print(f"  Smart Enhance 参数 (strength={args.strength})")
    print(f"{'='*60}")
    print(f"  {'参数':<12} {'默认值':<10} {'预测值':<10} {'调整后':<10}")
    print(f"  {'-'*48}")
    for name in PARAM_NAMES_7D:
        default = DEFAULT_PARAMS[name]
        # Find which action is the expert
        expert = None
        for ai, a in enumerate(ACTIONS):
            if ACTION_TO_PARAM_7D_INDICES[ai] == PARAM_NAMES_7D.index(name):
                expert = a
                break
        if expert:
            raw = per_action_params[expert][name]
        else:
            raw = sum(per_action_params[a][name] for a in ACTIONS) / len(ACTIONS)

        final = combined[name]
        if name == 'white_balance':
            print(f"  {PARAM_CN[name]:<10} {default:<10.0f} {raw:<10.0f} {final:<10.0f}")
        else:
            print(f"  {PARAM_CN[name]:<10} {default:<+10.1f} {raw:<+10.1f} {final:<+10.1f}")
    print(f"{'='*60}")

    # Render at full resolution
    print(f"\nRendering at full resolution ({W}×{H})...")
    enhanced = render_full_res(orig_pil, combined, device)

    # Save outputs
    stem = image_path.stem
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / 'outputs' / 'smart_enhance' / stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    enhanced.save(out_dir / 'enhanced.jpg', quality=95)
    orig_pil.save(out_dir / 'original.jpg', quality=95)

    # Side-by-side comparison
    comparison = build_comparison(orig_pil, enhanced, combined)
    comparison.save(out_dir / 'comparison.jpg', quality=92)

    # Multi-strength preview
    print("Generating strength sweep...")
    strengths = [0.3, 0.5, 0.7, 1.0]
    panels = []
    for s in strengths:
        p = combine_params(per_action_params, s, disable)
        img = render_full_res(orig_pil, p, device)
        panels.append((s, img))

    # Build sweep strip
    max_side = 400
    scale = min(max_side / W, max_side / H, 1.0)
    sw, sh = int(W * scale), int(H * scale)
    bar_h = 30
    n = len(panels) + 1  # +1 for original
    strip = Image.new('RGB', (sw * n + (n - 1) * 3, sh + bar_h), (15, 23, 42))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype("msyh.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    # Original
    draw.text((sw // 2 - 20, 6), "原图", fill=(180, 180, 180), font=font)
    strip.paste(orig_pil.resize((sw, sh), Image.LANCZOS), (0, bar_h))

    for i, (s, img) in enumerate(panels):
        x = (i + 1) * (sw + 3)
        label = f"×{s:.1f}"
        draw.text((x + sw // 2 - 15, 6), label, fill=(56, 189, 248), font=font)
        strip.paste(img.resize((sw, sh), Image.LANCZOS), (x, bar_h))

    strip.save(out_dir / 'strength_sweep.jpg', quality=90)

    # HTML viewer
    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Smart Enhance — {stem}</title>
<style>
  body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 0;
         padding: 20px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 22px; margin: 0 0 12px; color: #38bdf8; }}
  .info {{ font-size: 13px; color: #94a3b8; margin-bottom: 16px; }}
  .compare {{ display: flex; gap: 4px; margin-bottom: 20px; flex-wrap: wrap; }}
  .compare img {{ max-width: 48%; border-radius: 6px; }}
  .sweep img {{ width: 100%; max-width: 1200px; border-radius: 6px; }}
  table {{ border-collapse: collapse; font-size: 13px; margin: 16px 0; }}
  th, td {{ padding: 6px 14px; border: 1px solid #334155; text-align: center; }}
  th {{ background: #334155; color: #fbbf24; }}
  .highlight {{ color: #38bdf8; font-weight: bold; }}
  .slider-section {{ background: #1e293b; padding: 16px; border-radius: 8px;
                     margin: 16px 0; border: 1px solid #334155; }}
  .slider-section label {{ display: block; margin: 8px 0 4px; font-size: 13px; }}
  input[type="range"] {{ width: 200px; }}
</style>
</head>
<body>
<h1>🎨 Smart Enhance</h1>
<p class="info">输入: {image_path.name} ({W}×{H}) | strength={args.strength}</p>

<h3>对比</h3>
<div class="compare">
  <img src="original.jpg" title="原图">
  <img src="enhanced.jpg" title="智能增强 (strength={args.strength})">
</div>

<h3>不同强度对比</h3>
<div class="sweep"><img src="strength_sweep.jpg"></div>

<h3>组合参数</h3>
<table>
  <thead><tr><th>参数</th><th>默认</th><th>模型预测</th><th>最终 (×{args.strength})</th></tr></thead>
  <tbody>'''

    for name in PARAM_NAMES_7D:
        default = DEFAULT_PARAMS[name]
        expert = None
        for ai, a in enumerate(ACTIONS):
            if ACTION_TO_PARAM_7D_INDICES[ai] == PARAM_NAMES_7D.index(name):
                expert = a
                break
        raw = per_action_params[expert][name] if expert else (
            sum(per_action_params[a][name] for a in ACTIONS) / len(ACTIONS))
        final = combined[name]
        if name == 'white_balance':
            html += (f'<tr><td>{PARAM_CN[name]}</td><td>{default:.0f}K</td>'
                     f'<td>{raw:.0f}K</td>'
                     f'<td class="highlight">{final:.0f}K</td></tr>')
        else:
            html += (f'<tr><td>{PARAM_CN[name]}</td><td>{default:+.1f}</td>'
                     f'<td>{raw:+.1f}</td>'
                     f'<td class="highlight">{final:+.1f}</td></tr>')

    html += f'''
  </tbody>
</table>

<div class="slider-section">
  <b>💡 CLI 使用提示</b><br>
  <code style="color:#fbbf24">python tools/smart_enhance.py --image photo.jpg --strength 0.5</code><br>
  <code style="color:#fbbf24">python tools/smart_enhance.py --image photo.jpg --strengths "{','.join(['0.7'] * len(ACTIONS))}"</code><br>
  <span style="font-size:12px;color:#94a3b8">
    strengths 顺序: {', '.join(ACTIONS)}
  </span>
</div>
</body>
</html>'''

    (out_dir / 'viewer.html').write_text(html, encoding='utf-8')

    print(f"\n[saved] {out_dir}/")
    print(f"  enhanced.jpg      — 智能增强结果")
    print(f"  comparison.jpg    — 左右对比")
    print(f"  strength_sweep.jpg — 不同强度预览")
    print(f"  viewer.html       — 完整 viewer")

    # Auto open
    import os
    os.startfile(str(out_dir / 'viewer.html'))


if __name__ == '__main__':
    main()
