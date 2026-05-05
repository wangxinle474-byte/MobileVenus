"""\u62fc\u63a5 Top 5 \u7684 orig vs edit \u5bf9\u6bd4\u56fe (5\u884c x 2\u5217)\u3002

\u8f93\u51fa: docs/top5_comparison.jpg
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


CELL_W = 384
CELL_H = 384
LABEL_H = 48
PAD = 8
COL_LABEL_H = 36

d = json.load(open('data/aug_ip2p_full.json', encoding='utf-8'))
samples = sorted(d['samples'],
                 key=lambda s: (-s['score_edit'], -(s.get('score_delta') or 0)))[:5]


def fit_image(p, w, h):
    img = Image.open(p).convert('RGB')
    img.thumbnail((w, h), Image.LANCZOS)
    bg = Image.new('RGB', (w, h), (24, 24, 24))
    x = (w - img.width) // 2
    y = (h - img.height) // 2
    bg.paste(img, (x, y))
    return bg


# \u603b\u5c3a\u5bf8
N_ROWS = 5
total_w = 2 * CELL_W + 3 * PAD
total_h = COL_LABEL_H + N_ROWS * (CELL_H + LABEL_H) + (N_ROWS + 1) * PAD

canvas = Image.new('RGB', (total_w, total_h), (16, 16, 16))
draw = ImageDraw.Draw(canvas)

try:
    font_lg = ImageFont.truetype('arial.ttf', 22)
    font_sm = ImageFont.truetype('arial.ttf', 16)
except OSError:
    font_lg = ImageFont.load_default()
    font_sm = ImageFont.load_default()

# \u5217\u6807\u9898
draw.text((PAD + CELL_W // 2 - 30, PAD), 'ORIGINAL', fill=(220, 220, 220), font=font_lg)
draw.text((2 * PAD + CELL_W + CELL_W // 2 - 30, PAD), 'EDITED', fill=(120, 220, 120), font=font_lg)

for i, s in enumerate(samples):
    y0 = COL_LABEL_H + PAD + i * (CELL_H + LABEL_H + PAD)
    # orig
    op = Path(s['orig_path'])
    ep = Path(s['edit_path'])
    if op.exists():
        img_o = fit_image(op, CELL_W, CELL_H)
        canvas.paste(img_o, (PAD, y0))
    if ep.exists():
        img_e = fit_image(ep, CELL_W, CELL_H)
        canvas.paste(img_e, (2 * PAD + CELL_W, y0))

    # \u6807\u7b7e
    label_y = y0 + CELL_H + 4
    label_orig = (f'#{i+1}  idx={s["idx"]}  {s["source_image"]}  '
                  f'score={s["score_orig"]} ({s.get("orig_bucket")})')
    label_edit = (f'score={s["score_edit"]} ({s.get("edit_bucket")})  '
                  f'delta={s.get("score_delta"):+.1f}')
    draw.text((PAD + 4, label_y), label_orig, fill=(220, 220, 220), font=font_sm)
    draw.text((2 * PAD + CELL_W + 4, label_y), label_edit, fill=(120, 220, 120), font=font_sm)
    # prompt \u7b80\u8981\u653e\u7b2c\u4e8c\u884c
    prompt = (s.get('ip2p_prompt') or '').replace('\n', ' ')[:80]
    draw.text((PAD + 4, label_y + 22), f'prompt: "{prompt}..."',
              fill=(160, 160, 160), font=font_sm)


out_path = Path('docs/top5_comparison.jpg')
out_path.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out_path, 'JPEG', quality=88)
print(f'Saved: {out_path}  ({out_path.stat().st_size // 1024} KB)')
print(f'Size: {canvas.size}')
