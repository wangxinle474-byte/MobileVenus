"""
Baseline vs 语义蒸馏 对比 Demo
同时加载两个模型，对相同图片预测参数并生成对比图。

Usage:
  python tools/demo_distill_compare.py
  python tools/demo_distill_compare.py --image path/to/img.jpg
"""
import sys
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.fivek_8param import FiveK8ParamModel, PARAM_NAMES, PARAM_RANGES
from training.semantic_distill.model import SemanticDistillModel, DistillParamModel

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_baseline(ckpt_path: str, device):
    model = FiveK8ParamModel(image_size=224)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device).eval()
    print(f"[Baseline] loaded, epoch={state.get('epoch','?')}")
    return model


def load_distill(stage_a_ckpt: str, stage_b_ckpt: str, device):
    stage_a_model = SemanticDistillModel(
        image_size=224, visual_dim=384, semantic_dim=256, text_dim=384,
    )
    sa_state = torch.load(stage_a_ckpt, map_location='cpu', weights_only=False)
    stage_a_model.load_state_dict(sa_state['model_state_dict'])

    model = DistillParamModel(stage_a_model, decoder_hidden=256)
    sb_state = torch.load(stage_b_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sb_state['model_state_dict'])
    model = model.to(device).eval()
    print(f"[Distill]  loaded, epoch={sb_state.get('epoch','?')}")
    return model


def predict(model, image_path: str, device) -> dict:
    img = Image.open(image_path).convert('RGB')
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    return {p: float(out['raw_params'][p].squeeze()) for p in PARAM_NAMES}


def apply_adjustments(img: Image.Image, pred: dict) -> Image.Image:
    from models.isp_pipeline import apply_lightroom_params
    return apply_lightroom_params(img, pred)


def create_triple_comparison(img_path, pred_base, pred_dist, gt=None, size=350):
    """生成 Original | Baseline | Distill 三列对比"""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    ratio = size / max(w, h)
    nw, nh = int(w * ratio), int(h * ratio)
    img_r = img.resize((nw, nh), Image.LANCZOS)

    adj_base = apply_adjustments(img_r, pred_base)
    adj_dist = apply_adjustments(img_r, pred_dist)

    gap = 6
    text_h = 100
    canvas_w = nw * 3 + gap * 2
    canvas_h = nh + text_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), (25, 25, 25))
    canvas.paste(img_r, (0, 0))
    canvas.paste(adj_base, (nw + gap, 0))
    canvas.paste(adj_dist, (nw * 2 + gap * 2, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        font_s = ImageFont.truetype("arial.ttf", 9)
    except OSError:
        font = ImageFont.load_default()
        font_s = font

    y = nh + 3
    name = Path(img_path).stem
    draw.text((5, y), f"Original ({name})", fill=(180, 180, 180), font=font)
    draw.text((nw + gap + 5, y), "Baseline", fill=(100, 200, 255), font=font)
    draw.text((nw * 2 + gap * 2 + 5, y), "Distill", fill=(120, 255, 120), font=font)

    y += 16
    for pred, x_off, color in [
        (pred_base, nw + gap + 5, (100, 200, 255)),
        (pred_dist, nw * 2 + gap * 2 + 5, (120, 255, 120)),
    ]:
        l1 = f"EV={pred['ev_compensation']:+.2f} WB={pred['white_balance']:.0f}"
        l2 = f"Con={pred['contrast']:.1f} Bri={pred['brightness']:.1f}"
        l3 = f"Sha={pred['shadows']:.1f} Hi={pred['highlights']:.1f}"
        l4 = f"Sat={pred['saturation']:.1f} Vib={pred['vibrance']:.1f}"
        draw.text((x_off, y), l1, fill=color, font=font_s)
        draw.text((x_off, y + 12), l2, fill=color, font=font_s)
        draw.text((x_off, y + 24), l3, fill=color, font=font_s)
        draw.text((x_off, y + 36), l4, fill=color, font=font_s)

    if gt:
        l1 = f"GT: EV={gt['ev_compensation']:+.2f} WB={gt['white_balance']:.0f}"
        l2 = f"Con={gt['contrast']:.1f} Bri={gt['brightness']:.1f}"
        l3 = f"Sha={gt['shadows']:.1f} Hi={gt['highlights']:.1f}"
        l4 = f"Sat={gt['saturation']:.1f} Vib={gt['vibrance']:.1f}"
        draw.text((5, y), l1, fill=(200, 160, 255), font=font_s)
        draw.text((5, y + 12), l2, fill=(200, 160, 255), font=font_s)
        draw.text((5, y + 24), l3, fill=(200, 160, 255), font=font_s)
        draw.text((5, y + 36), l4, fill=(200, 160, 255), font=font_s)

    return canvas


def load_expert_gt(params_file: Path) -> dict:
    if not params_file.exists():
        return {}
    with open(params_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_vals = defaultdict(lambda: defaultdict(list))
    for s in data['samples']:
        name = s['image_name'].replace('.dng', '')
        for p in PARAM_NAMES:
            if p in s:
                img_vals[name][p].append(s[p])
    return {name: {p: np.mean(v) for p, v in params.items()} for name, params in img_vals.items()}


def main():
    parser = argparse.ArgumentParser(description='Baseline vs Distill Comparison')
    parser.add_argument('--image', type=str)
    parser.add_argument('--num', type=int, default=6)
    parser.add_argument('--baseline_ckpt', default=str(PROJECT_ROOT / 'checkpoints/fivek_8param/best.pt'))
    parser.add_argument('--stage_a_ckpt', default=str(PROJECT_ROOT / 'checkpoints/semantic_distill/stage_a/best.pt'))
    parser.add_argument('--stage_b_ckpt', default=str(PROJECT_ROOT / 'checkpoints/semantic_distill/stage_b/best.pt'))
    parser.add_argument('--output', default=str(PROJECT_ROOT / 'outputs/demo/demo_distill_compare.png'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_base = load_baseline(args.baseline_ckpt, device)
    model_dist = load_distill(args.stage_a_ckpt, args.stage_b_ckpt, device)

    if args.image:
        images = [args.image]
    else:
        jpeg_dir = Path(r'E:\dataset\fivek_jpeg')
        if not jpeg_dir.exists():
            print(f"FiveK JPEG not found: {jpeg_dir}")
            return
        all_jpgs = sorted(jpeg_dir.glob('*.jpg'))
        random.seed(42)
        images = [str(f) for f in random.sample(all_jpgs, min(args.num, len(all_jpgs)))]

    gt_params = load_expert_gt(PROJECT_ROOT / 'data' / 'fivek_expert_params.json')

    print(f"\nProcessing {len(images)} images...\n")
    print(f"{'Name':>12s}  {'':>6s}  {'EV':>6s}  {'WB':>6s}  {'Con':>5s}  {'Bri':>5s}  {'Sha':>5s}  {'Hi':>5s}  {'Sat':>5s}  {'Vib':>5s}")
    print("-" * 85)

    comparisons = []
    mae_base = defaultdict(list)
    mae_dist = defaultdict(list)

    for img_path in images:
        name = Path(img_path).stem
        pred_b = predict(model_base, img_path, device)
        pred_d = predict(model_dist, img_path, device)
        gt = gt_params.get(name)

        for tag, pred in [("base", pred_b), ("dist", pred_d)]:
            vals = f"  {name:>12s}  {tag:>6s}"
            for p in PARAM_NAMES:
                v = pred[p]
                vals += f"  {v:>5.1f}" if p != 'white_balance' else f"  {v:>5.0f}"
            print(vals)
            if gt:
                for p in PARAM_NAMES:
                    err = abs(pred[p] - gt[p])
                    (mae_base if tag == "base" else mae_dist)[p].append(err)

        if gt:
            vals = f"  {name:>12s}  {'GT':>6s}"
            for p in PARAM_NAMES:
                v = gt[p]
                vals += f"  {v:>5.1f}" if p != 'white_balance' else f"  {v:>5.0f}"
            print(vals)
        print()

        comparisons.append(create_triple_comparison(img_path, pred_b, pred_d, gt))

    # MAE Summary
    if mae_base:
        print("\n" + "=" * 60)
        print(f"{'Parameter':>20s}  {'Baseline MAE':>12s}  {'Distill MAE':>12s}  {'Winner':>8s}")
        print("-" * 60)
        for p in PARAM_NAMES:
            mb = np.mean(mae_base[p]) if mae_base[p] else 0
            md = np.mean(mae_dist[p]) if mae_dist[p] else 0
            winner = "Base" if mb < md else "Dist" if md < mb else "Tie"
            mark = "✓" if winner == "Dist" else ""
            print(f"  {p:>20s}  {mb:>10.2f}    {md:>10.2f}    {winner:>6s} {mark}")

    # Grid output
    if comparisons:
        cols = min(2, len(comparisons))
        rows = (len(comparisons) + cols - 1) // cols
        cw, ch = comparisons[0].size
        g = 6
        grid = Image.new('RGB', (cols * cw + (cols - 1) * g, rows * ch + (rows - 1) * g), (15, 15, 15))
        for i, comp in enumerate(comparisons):
            r, c = divmod(i, cols)
            grid.paste(comp, (c * (cw + g), r * (ch + g)))

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        grid.save(str(out), quality=95)
        print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
