"""Build a 3x3 visual comparison grid: orig | FireRed | Qwen (for each of 3 samples)."""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


SAMPLES = [
    {'idx': 850, 'firered': 'outputs/teacher_edits/fivek_wb_clean_warm/0850.png', 'tag': 'warm'},
    {'idx': 487, 'firered': 'outputs/teacher_edits/fivek_wb_clean_warm/0487.png', 'tag': 'warm'},
    {'idx': 1263, 'firered': 'outputs/teacher_edits/fivek_wb_clean_cool/1263.png', 'tag': 'cool'},
]

QWEN_DIR = Path('outputs/teacher_edits/fivek_wb_clean_qwen_pilot')
MASTER = 'outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl'
OUT = Path('outputs/qwen_vs_firered_grid.png')


def main():
    idx2orig = {}
    with open(MASTER, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            idx2orig[r['idx']] = r['orig_path']

    TILE = 384
    PAD = 10
    HDR = 30
    n = len(SAMPLES)
    W = 3 * TILE + 4 * PAD
    H = n * (TILE + HDR + PAD) + PAD

    grid = Image.new('RGB', (W, H), (32, 32, 32))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype('arial.ttf', 14)
        fhdr = ImageFont.truetype('arialbd.ttf', 16)
    except IOError:
        font = ImageFont.load_default()
        fhdr = font

    for row, s in enumerate(SAMPLES):
        orig_path = Path(idx2orig[s['idx']])
        fr_path = Path(s['firered'])
        qw_path = QWEN_DIR / f"{s['idx']:04d}.png"

        paths = [orig_path, fr_path, qw_path]
        labels = [f'orig (idx={s["idx"]}, {s["tag"]})', 'FireRed', 'Qwen-Image-Edit']

        y0 = PAD + row * (TILE + HDR + PAD)
        for col, (p, lbl) in enumerate(zip(paths, labels)):
            x0 = PAD + col * (TILE + PAD)
            draw.text((x0, y0), lbl, fill=(255, 255, 255), font=fhdr)
            if p.exists():
                img = Image.open(p).convert('RGB')
                img = img.resize((TILE, TILE), Image.LANCZOS)
                grid.paste(img, (x0, y0 + HDR))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    grid.save(OUT)
    print(f'[SAVED] {OUT}  ({grid.size[0]}x{grid.size[1]})')


if __name__ == '__main__':
    main()
