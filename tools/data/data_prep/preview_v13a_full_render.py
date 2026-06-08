"""全量渲染结果预览: 原图 vs v13a auto vs Expert C GT (3-panel).

从 E:/Data/dataset/fivek_v13a_auto/ 随机抽 12 张,
v13a auto 直接读预渲染 PNG,
Expert C GT 通过 apply_diff_isp 实时渲染.

用法:
  D:\\anaconda\\envs\\Venus\\python.exe tools/data/data_prep/preview_v13a_full_render.py
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
from training.firered_baseline.train_lut import PARAM_NAMES_7D  # noqa: E402

ACTION_CN = {
    'contrast': '对比度', 'saturation': '饱和度', 'shadows': '暗部',
    'highlights': '高光', 'wb': '白平衡', 'brightness': '亮度',
    'clarity': '清晰度',
}


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
    draw.text((4, 5), text, fill=(255, 255, 255), font=font)
    new.paste(img, (0, bar_h))
    return new


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255
           ).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--render_dir', default=r'E:\Data\dataset\fivek_v13a_auto')
    ap.add_argument('--jpeg_dir', default=r'E:\Data\dataset\fivek_jpeg')
    ap.add_argument('--gt_json', default='data/fivek_expert_abcde_params.json')
    ap.add_argument('--expert', default='C')
    ap.add_argument('--out_dir', default='outputs/preview_v13a_full_render')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    render_dir = Path(args.render_dir)
    jpeg_dir = Path(args.jpeg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_path = render_dir / 'manifest.jsonl'
    records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    print(f'[data] manifest: {len(records)} records')

    # Load Expert C GT params
    with open(args.gt_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    expert_map = {}
    for s in data['samples']:
        if s['expert'] == args.expert:
            stem = s['image_name'].rsplit('.', 1)[0]
            expert_map[stem] = {n: float(s['params'].get(n, 0) or 0)
                                for n in PARAM_NAMES_7D}
    print(f'[data] Expert {args.expert} params: {len(expert_map)}')

    rng = random.Random(args.seed)
    samples = rng.sample(records, min(args.n, len(records)))

    previews = []
    for i, rec in enumerate(samples):
        stem = rec['name']
        action = rec['action']
        z_score = rec['z_score']

        orig_path = jpeg_dir / f'{stem}.jpg'
        render_path = render_dir / f'{stem}.png'

        if not orig_path.exists() or not render_path.exists():
            print(f'  [skip] {stem} (file missing)')
            continue
        if stem not in expert_map:
            print(f'  [skip] {stem} (no expert params)')
            continue

        orig = Image.open(orig_path).convert('RGB').resize(
            (args.size, args.size), Image.LANCZOS)
        rendered = Image.open(render_path).convert('RGB')

        # Expert C GT via apply_diff_isp
        from torchvision.transforms import ToTensor
        orig_t = ToTensor()(orig).unsqueeze(0).to(device)
        gt_params = expert_map[stem]
        gt_params_t = {n: torch.tensor([gt_params[n]], dtype=torch.float32,
                                        device=device)
                       for n in PARAM_NAMES_7D}
        with torch.no_grad():
            gt_render = apply_diff_isp(orig_t, gt_params_t).clamp(0, 1)
        gt_pil = tensor_to_pil(gt_render)

        # Compact GT label
        gt_txt = ' '.join(
            f'{n[:3]}={gt_params[n]:.0f}' if n == 'white_balance'
            else f'{n[:3]}={gt_params[n]:+.0f}'
            for n in PARAM_NAMES_7D)

        # 3-panel: Original | v13a Auto | Expert C GT
        p1 = add_label(orig, f'原图 {stem}')
        p2 = add_label(rendered,
                       f'v13a auto → {ACTION_CN.get(action, action)} '
                       f'(z={z_score:+.2f})')
        p3 = add_label(gt_pil, f'Expert {args.expert} GT: {gt_txt}')

        pw, ph = p1.size
        gap = 3
        strip = Image.new('RGB', (pw * 3 + gap * 2, ph), (15, 23, 42))
        strip.paste(p1, (0, 0))
        strip.paste(p2, (pw + gap, 0))
        strip.paste(p3, (2 * (pw + gap), 0))

        strip_path = out_dir / f'{i+1:02d}_{stem}.jpg'
        strip.save(strip_path, quality=92)
        previews.append(strip_path.name)
        print(f'[{i+1}/{len(samples)}] {stem}  action={action}  z={z_score:+.2f}')

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

    print(f'[done] output: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
