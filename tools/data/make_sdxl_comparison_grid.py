"""\u62fc\u63a5 5\u00d75 \u5bf9\u6bd4\u56fe: orig | IP2P\u65e7 | SDXL s=0.3 | SDXL s=0.5 | SDXL s=0.7\u3002

\u8f93\u51fa: docs/sdxl_vs_ip2p_5x5.jpg
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


CELL_W = 384
CELL_H = 384
LABEL_H = 56
PAD = 6
HEADER_H = 40

ROOT = Path(__file__).resolve().parent.parent.parent


def fit_img(p, w, h, bg=(24, 24, 24)):
    if not p.exists():
        img_ph = Image.new('RGB', (w, h), bg)
        d = ImageDraw.Draw(img_ph)
        d.text((w // 2 - 40, h // 2), 'MISSING', fill=(200, 80, 80))
        return img_ph
    img = Image.open(p).convert('RGB')
    img.thumbnail((w, h), Image.LANCZOS)
    canvas = Image.new('RGB', (w, h), bg)
    canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return canvas


def main():
    cfg = json.load(open(ROOT / 'data' / 'compare_5_captions.json', encoding='utf-8'))
    samples = cfg['samples']
    N_ROWS = len(samples)
    N_COLS = 5

    total_w = N_COLS * CELL_W + (N_COLS + 1) * PAD
    total_h = HEADER_H + N_ROWS * (CELL_H + LABEL_H) + (N_ROWS + 1) * PAD

    canvas = Image.new('RGB', (total_w, total_h), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)

    try:
        font_lg = ImageFont.truetype('arialbd.ttf', 22)
        font_md = ImageFont.truetype('arial.ttf', 16)
        font_sm = ImageFont.truetype('arial.ttf', 13)
    except OSError:
        font_lg = font_md = font_sm = ImageFont.load_default()

    # Header
    col_headers = ['ORIGINAL', 'Old IP2P (instruction)',
                    'SDXL s=0.3 (light)',
                    'SDXL s=0.5 (medium)',
                    'SDXL s=0.7 (heavy)']
    col_colors = [(220, 220, 220), (200, 140, 80),
                   (120, 180, 200), (80, 200, 120), (220, 120, 80)]
    for c, (h, col) in enumerate(zip(col_headers, col_colors)):
        x = PAD + c * (CELL_W + PAD) + CELL_W // 2 - 70
        draw.text((x, 8), h, fill=col, font=font_lg)

    for r, s in enumerate(samples):
        idx = s['idx']
        y0 = HEADER_H + PAD + r * (CELL_H + LABEL_H + PAD)

        paths = [
            ROOT / 'outputs' / 'sdxl_turbo_compare' / f'{idx:04d}_orig.png',
            ROOT / 'outputs' / 'ip2p_pilot_100' / f'{idx:04d}_edit.png',
            ROOT / 'outputs' / 'sdxl_turbo_compare' / f'{idx:04d}_sdxl_s03.png',
            ROOT / 'outputs' / 'sdxl_turbo_compare' / f'{idx:04d}_sdxl_s05.png',
            ROOT / 'outputs' / 'sdxl_turbo_compare' / f'{idx:04d}_sdxl_s07.png',
        ]
        for c, p in enumerate(paths):
            x0 = PAD + c * (CELL_W + PAD)
            img = fit_img(p, CELL_W, CELL_H)
            canvas.paste(img, (x0, y0))

        # Row label
        label_y = y0 + CELL_H + 4
        row_info = (f'#{s["rank"]}  idx={idx}  {s["source_image"]}  '
                    f'(IP2P old edit score: {s["old_ip2p_edit_score"]})')
        draw.text((PAD + 4, label_y), row_info,
                  fill=(200, 200, 200), font=font_md)
        # Caption (new)
        caption = s['new_caption']
        if len(caption) > 135:
            caption = caption[:135] + '...'
        draw.text((PAD + 4, label_y + 22), 'NEW caption: ' + caption,
                  fill=(140, 180, 200), font=font_sm)
        # Target
        draw.text((PAD + 4, label_y + 40), 'Target: ' + s['aesthetic_target'],
                  fill=(200, 180, 120), font=font_sm)

    out_path = ROOT / 'docs' / 'sdxl_vs_ip2p_5x5.jpg'
    out_path.parent.mkdir(exist_ok=True)
    canvas.save(out_path, 'JPEG', quality=88, optimize=True)
    print(f'[DONE] {out_path}')
    print(f'  size: {canvas.size}')
    print(f'  file: {out_path.stat().st_size // 1024} KB')


if __name__ == '__main__':
    main()
