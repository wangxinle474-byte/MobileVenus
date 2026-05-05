"""\u591a\u6a21\u578b 5\u5f20\u5bf9\u6bd4 grid \u751f\u6210\u5668\u3002

\u8f93\u5165 (\u90e8\u5206\u53ef\u4e0d\u5b58\u5728, \u811a\u672c\u4f1a\u53d8\u901a):
  - outputs/sdxl_turbo_compare/        \u624b\u5199 caption + SDXL-Turbo
  - outputs/sdxl_turbo_vl_compare/     Qwen3-VL caption + SDXL-Turbo
  - outputs/longcat_compare/           LongCat-Turbo
  - data/compare_5_captions.json       \u539f\u56fe\u8def\u5f84 + idx
  - outputs/aug_ip2p_full_v1_reparsed.json \u5728\u8fd9 5 \u5f20\u4e0a\u7684 IP2P \u8bb0\u5f55 (\u53ef\u9009)

\u8f93\u51fa: docs/multi_model_comparison_5x*.jpg
\u4e2d\u95f4\u52a8\u6001\u8df3\u8fc7\u4e0d\u5b58\u5728\u7684\u5217 (\u5982 LongCat \u672a\u4e0b\u5b8c)\u3002

\u8fd0\u884c:
  python tools/data/viz/make_multi_model_grid.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# \u5217\u5b9a\u4e49: (label, glob_pattern, out_dir_relative)
# pattern \u4e2d {idx} \u4f1a\u88ab\u66ff\u6362\u4e3a 4-digit
COLUMNS = [
    ('Original',        '{idx:04d}_orig.png',      'outputs/sdxl_turbo_compare'),
    ('IP2P',            None,                       None),  # \u4ece IP2P \u8bb0\u5f55\u91cc\u53d6
    ('SDXL-Turbo\nhand cap. s=0.5', '{idx:04d}_sdxl_s05.png', 'outputs/sdxl_turbo_compare'),
    ('SDXL-Turbo\nVL cap. s=0.5',   '{idx:04d}_sdxl_vl_s05.png', 'outputs/sdxl_turbo_vl_compare'),
    ('LongCat-Turbo\n(8 NFE)',     '{idx:04d}_longcat.png',  'outputs/longcat_compare'),
]


def load_image_or_placeholder(path: Path, size=(512, 512)):
    """\u5982\u679c\u56fe\u4e0d\u5b58\u5728, \u8fd4\u56de\u4e00\u4e2a\u5360\u4f4d\u7070\u56fe."""
    if path is not None and path.exists():
        return Image.open(path).convert('RGB')
    placeholder = Image.new('RGB', size, '#888')
    d = ImageDraw.Draw(placeholder)
    msg = '(not\navailable)'
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size=24)
    except OSError:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), msg, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    d.text(((size[0] - tw) // 2, (size[1] - th) // 2),
           msg, fill='white', font=font)
    return placeholder


def find_ip2p_edit(idx, samples_meta, ip2p_records=None, ip2p_dir='outputs/ip2p_pilot_100'):
    """\u4ece IP2P \u8f93\u51fa\u76ee\u5f55\u91cc\u627e\u8fd9\u4e2a idx \u7684\u7f16\u8f91\u540e\u56fe.

    \u547d\u540d\u7ea6\u5b9a: {idx:04d}_edit.png  (test_ip2p.py \u53d1\u51fa)
    \u8be5\u76ee\u5f55\u540d\u53eb pilot_100 \u4f46\u5b9e\u9645\u5305\u542b\u5168 996 \u5f20.
    """
    p = PROJECT_ROOT / ip2p_dir / f'{idx:04d}_edit.png'
    return p if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', default='data/compare_5_captions.json')
    ap.add_argument('--ip2p_records',
                    default='outputs/aug_ip2p_full_v1_reparsed.json',
                    help='IP2P \u751f\u6210\u8bb0\u5f55 (\u53ef\u9009, \u4e0d\u5b58\u5728\u5219\u8df3\u8fc7 IP2P \u5217)')
    ap.add_argument('--out_path',
                    default='docs/multi_model_comparison_5x{n}.jpg')
    ap.add_argument('--cell_size', type=int, default=384,
                    help='\u6bcf\u4e2a cell \u7684\u7edf\u4e00\u8fb9\u957f (px)')
    ap.add_argument('--header_h', type=int, default=72,
                    help='\u9876\u90e8\u6807\u9898\u9ad8\u5ea6')
    args = ap.parse_args()

    cfg = json.load(open(PROJECT_ROOT / args.captions, encoding='utf-8'))
    samples = cfg['samples']
    n_samples = len(samples)
    print(f'[INFO] {n_samples} \u6837\u672c')

    # IP2P \u8bb0\u5f55 \u53ef\u9009
    ip2p_records = None
    ip2p_path = PROJECT_ROOT / args.ip2p_records
    if ip2p_path.exists():
        ip2p_data = json.load(open(ip2p_path, encoding='utf-8'))
        ip2p_records = ip2p_data.get('results', ip2p_data)
        print(f'[INFO] IP2P \u8bb0\u5f55 {len(ip2p_records)} \u6761')
    else:
        print(f'[INFO] IP2P \u8bb0\u5f55\u4e0d\u5b58\u5728: {ip2p_path} (\u5217\u5c06\u7528\u5360\u4f4d)')

    # \u9650\u5b9a\u53ef\u7528\u5217 (\u53bb\u6389\u5168\u90e8\u4e0d\u5b58\u5728\u7684\u5217)
    active_columns = []
    for col_label, pattern, sub_dir in COLUMNS:
        if col_label == 'IP2P':
            # IP2P \u9700\u8981\u8bb0\u5f55
            available = ip2p_records is not None
        elif col_label == 'Original':
            # \u539f\u56fe\u4e00\u822c\u5b58\u5728\u4e8e SDXL \u6216 LongCat \u8f93\u51fa
            available = True
        else:
            d = PROJECT_ROOT / sub_dir
            available = d.exists() and any(d.glob(pattern.format(idx=samples[0]['idx']).replace('05', '*')[:6] + '*'))
            # \u4fbf\u5b9c\u68c0\u67e5: \u770b\u5b50\u76ee\u5f55\u662f\u5426\u5b58\u5728\u4e14\u6709\u7b2c\u4e00\u4e2a\u4e3b\u5f55
            if d.exists():
                target = d / pattern.format(idx=samples[0]['idx'])
                available = target.exists()
        active_columns.append((col_label, pattern, sub_dir, available))
        flag = '\u2713' if available else '\u2717'
        print(f'  [{flag}] {col_label}')

    # \u62fc\u63a5
    n_cols = sum(1 for _, _, _, a in active_columns if a)
    cell = args.cell_size
    header_h = args.header_h
    label_w = 200  # \u5de6\u4fa7 idx \u6807\u7b7e\u5217\u5bbd

    grid_w = label_w + cell * n_cols
    grid_h = header_h + cell * n_samples
    grid = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid)

    try:
        font_h = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', size=18)
        font_l = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size=14)
    except OSError:
        font_h = ImageFont.load_default()
        font_l = ImageFont.load_default()

    # \u753b\u9876\u90e8\u6807\u9898
    col_x_positions = []
    cur_x = label_w
    for col_label, _, _, available in active_columns:
        if not available:
            continue
        col_x_positions.append((cur_x, col_label))
        bbox = draw.textbbox((0, 0), col_label, font=font_h)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.multiline_text(
            (cur_x + (cell - tw) // 2, (header_h - th) // 2 - 4),
            col_label, fill='black', font=font_h, align='center')
        cur_x += cell

    # \u753b\u6bcf\u884c
    for row_i, sample in enumerate(samples):
        idx = sample['idx']
        y = header_h + row_i * cell

        # \u5de6\u4fa7 idx \u6807\u7b7e
        info = f"#{idx}\n{sample['source_image']}"
        draw.multiline_text((10, y + 10), info, fill='black', font=font_l)

        # \u6bcf\u5217 cell
        cur_x = label_w
        for col_label, pattern, sub_dir, available in active_columns:
            if not available:
                continue
            if col_label == 'IP2P':
                p = find_ip2p_edit(idx, samples, ip2p_records)
            else:
                p = (PROJECT_ROOT / sub_dir / pattern.format(idx=idx)
                     if sub_dir else None)
            img = load_image_or_placeholder(p, size=(cell, cell))
            # resize\u4e3a\u6b63\u65b9 cell (\u4fdd\u6301\u957f\u5bbd\u6bd4 \u586b\u5145\u767d\u8fb9)
            ratio = min(cell / img.size[0], cell / img.size[1])
            new_w = int(img.size[0] * ratio)
            new_h = int(img.size[1] * ratio)
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            cell_canvas = Image.new('RGB', (cell, cell), 'white')
            cell_canvas.paste(resized,
                              ((cell - new_w) // 2, (cell - new_h) // 2))
            grid.paste(cell_canvas, (cur_x, y))
            cur_x += cell

    out_path = PROJECT_ROOT / args.out_path.format(n=n_cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, quality=92)
    print(f'\n[DONE] {out_path}')
    print(f'    {grid_w}x{grid_h}, {n_samples} rows x {n_cols} cols')


if __name__ == '__main__':
    main()
