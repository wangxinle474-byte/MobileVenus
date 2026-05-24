"""Interactive inference: test model on your own images.

Usage:
  python tools/run_inference.py --image path/to/photo.jpg
  python tools/run_inference.py --image path/to/photo.jpg --action wb
  python tools/run_inference.py --image path/to/photo.jpg --ckpt checkpoints/lut_v11a_action_gated_context/best.pt

Outputs:
  - Side-by-side comparison (orig + all 5 actions) saved as PNG
  - HTML viewer with parameter details
  - Per-action output images
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
    PARAM_NAMES_7D,
    PARAM_NORM_7D,
    NamedCurvesPredictor,
    set_actions,
)

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

ACTION_CN = {
    'contrast': '对比度增强',
    'saturation': '饱和度增强',
    'shadows': '暗部提亮',
    'highlights': '高光压暗',
    'wb': '白平衡调整',
    'brightness': '亮度调整',
    'clarity': '清晰度调整',
}

PARAM_CN = {
    'white_balance': 'WB色温(K)',
    'brightness': '亮度',
    'contrast': '对比度',
    'shadows': '暗部',
    'highlights': '高光',
    'saturation': '饱和度',
    'clarity': '清晰度',
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

    info = {
        'val_psnr': ckpt.get('val_psnr', 0),
        'epoch': ckpt.get('epoch', '?'),
        'use_implicit_head': ca.get('use_implicit_head', False),
        'image_size': ca.get('image_size', 256),
        'actions': list(ACTIONS),
    }
    return model, info


def preprocess(img: Image.Image, image_size: int):
    """Preprocess a PIL image for model input."""
    resize = T.Resize((image_size, image_size))
    to_tensor = T.ToTensor()
    norm = T.Normalize(NORM_MEAN, NORM_STD)

    img_resized = resize(img)
    orig_t = to_tensor(img_resized)        # (3, H, W) [0, 1]
    enc_input = norm(orig_t)               # (3, H, W) normalized
    return enc_input.unsqueeze(0), orig_t.unsqueeze(0)


def run_all_actions(model, enc_input, orig, device, actions=None):
    """Run inference for selected actions, return results dict."""
    if actions is None:
        actions = ACTIONS
    results = {}
    with torch.no_grad():
        for action in actions:
            action_idx = ACTION_TO_IDX[action]
            action_oh = torch.zeros(1, len(ACTIONS), dtype=torch.float32, device=device)
            action_oh[0, action_idx] = 1.0

            refined, cn_avg, y_b, pred_p7 = model(enc_input, orig, action_oh)

            # Decode predicted parameters
            pred_norm = pred_p7[0].cpu().numpy()
            params = {}
            for i, name in enumerate(PARAM_NAMES_7D):
                params[name] = denormalize_param(name, float(pred_norm[i]))

            results[action] = {
                'refined': refined[0].cpu(),
                'params': params,
            }
    return results


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_label(img: Image.Image, text: str, bg_color=(30, 41, 59)):
    W, H = img.size
    bar_h = 32
    new = Image.new('RGB', (W, H + bar_h), bg_color)
    draw = ImageDraw.Draw(new)
    try:
        font = ImageFont.truetype("msyh.ttc", 16)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, 6), text, fill=(255, 255, 255), font=font)
    new.paste(img, (0, bar_h))
    return new


def build_comparison_strip(orig_pil, results, image_size, actions=None):
    """Build a horizontal strip: orig + selected action outputs."""
    if actions is None:
        actions = ACTIONS
    panels = [add_label(orig_pil.resize((image_size, image_size)), '原图 Original')]

    for action in actions:
        r = results[action]
        out_pil = tensor_to_pil(r['refined'])
        label = f"{ACTION_CN.get(action, action)}"
        panels.append(add_label(out_pil, label))

    W, H = panels[0].size
    gap = 3
    total_w = W * len(panels) + gap * (len(panels) - 1)
    strip = Image.new('RGB', (total_w, H), (15, 23, 42))
    for i, p in enumerate(panels):
        strip.paste(p, (i * (W + gap), 0))
    return strip


def build_html_viewer(image_path, results, info, out_dir, orig_pil, image_size,
                      actions=None):
    """Generate an HTML viewer with parameters table."""
    if actions is None:
        actions = ACTIONS
    # Save per-action output images
    img_dir = out_dir / 'imgs'
    img_dir.mkdir(exist_ok=True)

    orig_resized = orig_pil.resize((image_size, image_size))
    orig_resized.save(img_dir / 'original.jpg', quality=92)

    for action in actions:
        out_pil = tensor_to_pil(results[action]['refined'])
        out_pil.save(img_dir / f'{action}.jpg', quality=92)

    # Build parameter table rows
    param_rows = ''
    for action in actions:
        params = results[action]['params']
        row = f'<tr><td class="act">{ACTION_CN.get(action, action)}</td>'
        for name in PARAM_NAMES_7D:
            v = params[name]
            if name == 'white_balance':
                row += f'<td>{v:.0f}K</td>'
            else:
                row += f'<td>{v:+.1f}</td>'
        row += '</tr>'
        param_rows += row

    # Action image cards
    cards = ''
    for action in actions:
        cards += f'''
        <div class="card">
          <img src="imgs/{action}.jpg">
          <div class="label">{ACTION_CN.get(action, action)}</div>
        </div>'''

    stem = Path(image_path).stem

    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>推理结果 — {stem}</title>
<style>
  body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 0;
         padding: 20px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 22px; margin: 0 0 8px; }}
  .info {{ font-size: 13px; color: #94a3b8; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
           max-width: 900px; margin-bottom: 24px; }}
  .card {{ background: #1e293b; border-radius: 8px; overflow: hidden;
           border: 1px solid #334155; }}
  .card img {{ width: 100%; display: block; }}
  .card .label {{ padding: 8px; text-align: center; font-size: 14px;
                  font-weight: bold; color: #38bdf8; }}
  .orig-section {{ margin-bottom: 24px; }}
  .orig-section img {{ max-width: 300px; border-radius: 8px;
                       border: 2px solid #64748b; }}
  table {{ border-collapse: collapse; font-size: 13px; margin-top: 16px; }}
  th, td {{ padding: 8px 14px; border: 1px solid #334155; text-align: center; }}
  th {{ background: #334155; color: #fbbf24; }}
  .act {{ font-weight: bold; color: #38bdf8; text-align: left; }}
  .strip {{ margin: 16px 0; }}
  .strip img {{ width: 100%; max-width: 1500px; border-radius: 6px; }}
</style>
</head>
<body>
<h1>模型推理结果</h1>
<p class="info">
  输入: {image_path}<br>
  Checkpoint: val_psnr={info["val_psnr"]:.2f}dB @ Ep{info["epoch"]}
  | implicit_head={info["use_implicit_head"]}
</p>

<div class="orig-section">
  <h3>原图</h3>
  <img src="imgs/original.jpg">
</div>

<h3>{len(actions)} 种编辑动作输出</h3>
<div class="grid">{cards}</div>

<div class="strip">
  <h3>横向对比条</h3>
  <img src="comparison_strip.jpg">
</div>

<h3>预测的 7D ISP 参数</h3>
<table>
  <thead>
    <tr>
      <th>动作</th>
      {''.join(f"<th>{PARAM_CN[p]}</th>" for p in PARAM_NAMES_7D)}
    </tr>
  </thead>
  <tbody>{param_rows}</tbody>
</table>

<p class="info" style="margin-top: 24px;">
  参数含义: WB色温=白平衡色温(K), 其他参数单位为内部 scale 值 (约 ±100 范围).<br>
  "原图"对应的默认参数: WB≈5500K, 其他≈0.
</p>
</body>
</html>'''

    (out_dir / 'viewer.html').write_text(html, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description='Run model inference on custom images')
    ap.add_argument('--image', required=True, help='Input image path')
    ap.add_argument('--action', default=None,
                    help='Single action (default: run all checkpoint actions)')
    ap.add_argument('--ckpt',
                    default='checkpoints/lut_v11a_pathX_implicit_head/best.pt',
                    help='Checkpoint path')
    ap.add_argument('--out_dir', default=None,
                    help='Output directory (default: outputs/inference/<stem>)')
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading: {args.ckpt}")
    model, info = load_model(args.ckpt, device)
    image_size = info['image_size']
    if args.action is not None and args.action not in ACTIONS:
        sys.exit(f"Unknown action for this checkpoint: {args.action}. "
                 f"Available: {', '.join(ACTIONS)}")
    actions_to_run = [args.action] if args.action else list(ACTIONS)
    print(f"  val_psnr={info['val_psnr']:.2f}dB, "
          f"implicit_head={info['use_implicit_head']}")

    # Load and preprocess image
    orig_pil = Image.open(image_path).convert('RGB')
    print(f"Input: {image_path.name} ({orig_pil.size[0]}×{orig_pil.size[1]})")
    enc_input, orig_t = preprocess(orig_pil, image_size)
    enc_input = enc_input.to(device)
    orig_t = orig_t.to(device)

    # Run inference
    results = run_all_actions(model, enc_input, orig_t, device, actions_to_run)

    # Print parameters
    print(f"\n{'='*70}")
    print(f"{'动作':<12} {'WB(K)':<8} {'亮度':<8} {'对比':<8} "
          f"{'暗部':<8} {'高光':<8} {'饱和':<8} {'清晰':<8}")
    print(f"{'-'*70}")
    for action in actions_to_run:
        p = results[action]['params']
        print(f"{ACTION_CN.get(action, action):<10} "
              f"{p['white_balance']:<8.0f} "
              f"{p['brightness']:<+8.1f} "
              f"{p['contrast']:<+8.1f} "
              f"{p['shadows']:<+8.1f} "
              f"{p['highlights']:<+8.1f} "
              f"{p['saturation']:<+8.1f} "
              f"{p['clarity']:<+8.1f}")
    print(f"{'='*70}")

    # Output
    stem = image_path.stem
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / 'outputs' / 'inference' / stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison strip
    strip = build_comparison_strip(orig_pil, results, image_size, actions_to_run)
    strip.save(out_dir / 'comparison_strip.jpg', quality=92)
    print(f"\n[saved] comparison_strip.jpg")

    # Save individual outputs
    img_dir = out_dir / 'imgs'
    img_dir.mkdir(exist_ok=True)
    for action in actions_to_run:
        out_pil = tensor_to_pil(results[action]['refined'])
        out_pil.save(img_dir / f'{action}.jpg', quality=92)

    # Build HTML viewer
    build_html_viewer(str(image_path), results, info, out_dir, orig_pil,
                      image_size, actions_to_run)
    print(f"[saved] viewer.html")
    print(f"\nOutput: {out_dir}")

    # Auto-open viewer
    import os
    os.startfile(str(out_dir / 'viewer.html'))


if __name__ == '__main__':
    main()
