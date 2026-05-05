"""\u628a 8 \u5f20 heatmap \u62fc\u6210\u4e00\u5f20\u603b\u89c8\u56fe, \u4fdd\u5b58\u5230 docs/ (\u975e gitignore)\u3002"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / 'outputs' / 'diagnose_param_conflicts'
OUT = ROOT / 'docs' / 'diagnose_param_conflicts_montage.png'

# \u6309 ridge_pct \u964d\u5e8f \u6392\u5217
order = [
    'saturation__vs__vibrance.png',          # 10.7%  \u2b55
    'shadows__vs__highlights.png',           #  9.6%
    'white_balance__vs__ev_compensation.png',#  8.3% \u707e\u96beλratio
    'ev_compensation__vs__brightness.png',   #  8.0%
    'saturation__vs__contrast.png',          #  4.3%
    'contrast__vs__brightness.png',          #  3.7%
    'contrast__vs__clarity.png',             #  3.7%
    'vibrance__vs__contrast.png',            #  3.4%
]

imgs = [Image.open(SRC / f) for f in order]
w, h = imgs[0].size
# 4 \u5217 x 2 \u884c
cols, rows = 4, 2
W, H = cols * w, rows * h
canvas = Image.new('RGB', (W, H), 'white')
for i, im in enumerate(imgs):
    r, c = i // cols, i % cols
    canvas.paste(im, (c * w, r * h))

OUT.parent.mkdir(parents=True, exist_ok=True)
canvas.save(OUT, quality=92)
print(f'Saved montage: {OUT}')
print(f'Size: {W}x{H}')
