"""
FiveK 8参数推理 Demo
加载训练好的 8 参数模型，对任意图片预测 Lightroom 参数并可视化效果。

Usage:
  python tools/demo_8param.py                          # 随机抽 8 张 FiveK
  python tools/demo_8param.py --image path/to/img.jpg  # 指定单张
  python tools/demo_8param.py --dir path/to/folder     # 批量
"""
import sys
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from torchvision import transforms

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.fivek_8param import FiveK8ParamModel, PARAM_NAMES, PARAM_RANGES


TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(ckpt_path: str, device: torch.device) -> FiveK8ParamModel:
    model = FiveK8ParamModel(image_size=224)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {n/1e6:.2f}M params, epoch={state.get('epoch','?')}")
    return model


def predict_single(model, image_path: str, device: torch.device) -> dict:
    img = Image.open(image_path).convert('RGB')
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    result = {}
    for p in PARAM_NAMES:
        result[p] = float(out['raw_params'][p].squeeze())
    result['confidence'] = out['confidence'].squeeze().cpu().numpy()
    return result


def apply_adjustments(img: Image.Image, pred: dict) -> Image.Image:
    """模拟 Lightroom 参数效果 (使用改进的 gamma-aware ISP)"""
    from models.isp_pipeline import apply_lightroom_params
    return apply_lightroom_params(img, pred)


def create_comparison(img_path: str, pred: dict, gt: dict = None, size: int = 400) -> Image.Image:
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    ratio = size / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    adjusted = apply_adjustments(img_resized, pred)

    gap = 10
    text_h = 110
    canvas_w = new_w * 2 + gap
    canvas_h = new_h + text_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), (30, 30, 30))
    canvas.paste(img_resized, (0, 0))
    canvas.paste(adjusted, (new_w + gap, 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_s = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
        font_s = font

    y = new_h + 4
    draw.text((5, y), "Original", fill=(200, 200, 200), font=font)
    draw.text((new_w + gap + 5, y), "Adjusted (8-param)", fill=(120, 255, 120), font=font)

    y += 18
    line1 = f"EV={pred['ev_compensation']:+.2f}  WB={pred['white_balance']:.0f}K  Con={pred['contrast']:.1f}  Bri={pred['brightness']:.1f}"
    line2 = f"Sha={pred['shadows']:.1f}  Hi={pred['highlights']:.1f}  Sat={pred['saturation']:.1f}  Vib={pred['vibrance']:.1f}"
    draw.text((5, y), line1, fill=(255, 220, 100), font=font_s)
    y += 14
    draw.text((5, y), line2, fill=(255, 220, 100), font=font_s)

    if gt:
        y += 16
        gt_line = f"GT: EV={gt['ev_compensation']:+.2f}  WB={gt['white_balance']:.0f}K  Con={gt['contrast']:.1f}  Bri={gt['brightness']:.1f}"
        draw.text((5, y), gt_line, fill=(150, 150, 255), font=font_s)
        y += 14
        gt_line2 = f"    Sha={gt['shadows']:.1f}  Hi={gt['highlights']:.1f}  Sat={gt['saturation']:.1f}  Vib={gt['vibrance']:.1f}"
        draw.text((5, y), gt_line2, fill=(150, 150, 255), font=font_s)

    return canvas


def load_expert_gt(params_file: Path) -> dict:
    """加载专家参数均值作为 GT"""
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

    gt = {}
    for name, params in img_vals.items():
        gt[name] = {p: np.mean(vals) for p, vals in params.items()}
    return gt


def main():
    parser = argparse.ArgumentParser(description='FiveK 8-Param Inference Demo')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--dir', type=str, help='Directory of images')
    parser.add_argument('--num', type=int, default=8, help='Number of random FiveK samples')
    parser.add_argument('--ckpt', type=str,
                        default=str(PROJECT_ROOT / 'checkpoints' / 'fivek_8param' / 'best.pt'))
    parser.add_argument('--output', type=str,
                        default=str(PROJECT_ROOT / 'images' / 'inference' / 'demo_8param.png'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.ckpt, device)

    images = []
    gt_params = {}

    if args.image:
        images = [args.image]
    elif args.dir:
        d = Path(args.dir)
        images = sorted([str(f) for f in d.glob('*.jpg')])[:args.num]
    else:
        jpeg_dir = Path(r'E:\dataset\fivek_jpeg')
        if not jpeg_dir.exists():
            print(f"FiveK JPEG not found at {jpeg_dir}, use --image or --dir")
            return
        gt_params = load_expert_gt(PROJECT_ROOT / 'data' / 'fivek_expert_params.json')
        all_jpgs = sorted(jpeg_dir.glob('*.jpg'))
        random.seed(42)
        selected = random.sample(all_jpgs, min(args.num, len(all_jpgs)))
        images = [str(f) for f in selected]

    print(f"Processing {len(images)} images...")

    results = []
    for img_path in images:
        pred = predict_single(model, img_path, device)
        name = Path(img_path).stem
        gt = gt_params.get(name)
        results.append({'image': img_path, 'name': name, 'pred': pred, 'gt': gt})

        line = f"  {name}:"
        for p in PARAM_NAMES:
            v = pred[p]
            line += f" {p[:3]}={v:+.1f}" if p != 'white_balance' else f" wb={v:.0f}"
        print(line)

    # 生成对比图
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparisons = [create_comparison(r['image'], r['pred'], r['gt']) for r in results]

    if comparisons:
        cols = min(2, len(comparisons))
        rows = (len(comparisons) + cols - 1) // cols
        cw, ch = comparisons[0].size
        grid_gap = 6
        grid = Image.new('RGB',
                         (cols * cw + (cols-1) * grid_gap, rows * ch + (rows-1) * grid_gap),
                         (20, 20, 20))
        for i, comp in enumerate(comparisons):
            r, c = divmod(i, cols)
            grid.paste(comp, (c * (cw + grid_gap), r * (ch + grid_gap)))
        grid.save(str(output_path), quality=95)
        print(f"\nSaved: {output_path}")

    # MAE 汇总
    if any(r['gt'] for r in results):
        print(f"\nMAE Summary ({sum(1 for r in results if r['gt'])} samples with GT):")
        for p in PARAM_NAMES:
            errs = [abs(r['pred'][p] - r['gt'][p]) for r in results if r['gt'] and p in r['gt']]
            if errs:
                lo, hi = PARAM_RANGES[p]
                pct = np.mean(errs) / (hi - lo) * 100
                print(f"  {p:20s}: MAE={np.mean(errs):7.2f}  ({pct:.1f}%)")


if __name__ == '__main__':
    main()
