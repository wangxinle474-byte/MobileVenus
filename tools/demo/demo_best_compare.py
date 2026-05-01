"""
4 列可视化对比: Original | Baseline | Distill v2 (PSNR最优) | Distill v4 (SSIM最优)
Usage:
  python tools/demo_best_compare.py [--num 6] [--seed 42]
"""
import sys, random, argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from training.fivek_8param import FiveK8ParamModel, PARAM_NAMES
from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
from models.isp_pipeline import apply_lightroom_params

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_baseline():
    m = FiveK8ParamModel(image_size=224)
    s = torch.load(ROOT / 'checkpoints/fivek_8param/best.pt', map_location='cpu', weights_only=False)
    m.load_state_dict(s['model_state_dict'])
    return m.to(DEVICE).eval()


def load_distill(stage_a, stage_b):
    sa = SemanticDistillModel(image_size=224, visual_dim=384, semantic_dim=256, text_dim=384)
    sa.load_state_dict(torch.load(stage_a, map_location='cpu', weights_only=False)['model_state_dict'])
    m = DistillParamModel(sa, decoder_hidden=256)
    m.load_state_dict(torch.load(stage_b, map_location='cpu', weights_only=False)['model_state_dict'])
    return m.to(DEVICE).eval()


def predict(model, img):
    t = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)
    return {p: float(out['raw_params'][p].squeeze()) for p in PARAM_NAMES}


def make_row(img_path, models_dict, thumb=400):
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    r = thumb / max(w, h)
    nw, nh = int(w * r), int(h * r)
    img_t = img.resize((nw, nh), Image.LANCZOS)

    panels = [('Original', img_t, (180, 180, 180), None)]
    for label, model, color in models_dict:
        pred = predict(model, img_t)
        rendered = apply_lightroom_params(img_t, pred)
        panels.append((label, rendered, color, pred))

    gap, text_h = 4, 90
    row_w = nw * len(panels) + gap * (len(panels) - 1)
    row_h = nh + text_h
    row = Image.new('RGB', (row_w, row_h), (20, 20, 20))

    try:
        font = ImageFont.truetype("arial.ttf", 13)
        font_s = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font = font_s = ImageFont.load_default()

    draw = ImageDraw.Draw(row)
    for i, (label, panel, color, pred) in enumerate(panels):
        x = i * (nw + gap)
        row.paste(panel, (x, 0))
        draw.text((x + 4, nh + 3), label, fill=color, font=font)
        if pred:
            lines = [
                f"EV={pred['ev_compensation']:+.2f}  WB={pred['white_balance']:.0f}K",
                f"Con={pred['contrast']:.1f}  Bri={pred['brightness']:.1f}",
                f"Sha={pred['shadows']:.1f}  Hi={pred['highlights']:.1f}",
                f"Sat={pred['saturation']:.1f}  Vib={pred['vibrance']:.1f}",
            ]
            for j, line in enumerate(lines):
                draw.text((x + 4, nh + 20 + j * 14), line, fill=color, font=font_s)

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--size', type=int, default=600, help='每列图片的长边像素')
    parser.add_argument('--outdir', default=str(ROOT / 'images/inference/compare_best'))
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("加载模型...")
    m_base = load_baseline()
    m_v2 = load_distill(
        ROOT / 'checkpoints/semantic_distill/stage_a/best.pt',
        ROOT / 'checkpoints/semantic_distill_v2/stage_b/best.pt',
    )
    m_v4 = load_distill(
        ROOT / 'checkpoints/distill_v4/stage_a/best.pt',
        ROOT / 'checkpoints/distill_v4/stage_b/best.pt',
    )
    print("模型加载完成\n")

    models_info = [
        ('Baseline (32.05dB)', m_base, (100, 200, 255)),
        ('Distill v2 ★PSNR',  m_v2,   (255, 200, 80)),
        ('Distill v4 ★SSIM',  m_v4,   (120, 255, 120)),
    ]

    jpeg_dir = Path(r'E:\dataset\fivek_jpeg')
    all_jpgs = sorted(jpeg_dir.glob('*.jpg'))
    random.seed(args.seed)
    images = random.sample(all_jpgs, min(args.num, len(all_jpgs)))

    for i, img_path in enumerate(images):
        print(f"  [{i+1}/{len(images)}] {img_path.name}")
        row = make_row(img_path, models_info, thumb=args.size)
        out = out_dir / f"{img_path.stem}.png"
        row.save(str(out), quality=95)
        print(f"           → {out}")

    print(f"\n全部保存至: {out_dir}")


if __name__ == '__main__':
    main()
