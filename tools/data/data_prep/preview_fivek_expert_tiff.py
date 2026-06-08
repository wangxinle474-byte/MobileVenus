"""Generate side-by-side previews: raw-decoded JPEG (before) vs Expert TIFF (after).

Inputs:
  - data/fivek_expert_<X>_tiff_manifest.jsonl  (build_fivek_expert_tiff_manifest.py)
  - E:\\Data\\dataset\\fivek_jpeg\\<stem>.jpg   (raw-decoded "before")
  - <tiff_path from manifest>                  (Expert-rendered "after")

Output:
  - outputs/preview_fivek_expert_<X>/<stem>.jpg     (per-sample side-by-side)
  - outputs/preview_fivek_expert_<X>/_contact.jpg   (grid overview, all in one)

Usage:
  python tools/data/data_prep/preview_fivek_expert_tiff.py --expert C --n 12
  python tools/data/data_prep/preview_fivek_expert_tiff.py --expert C --n 12 --random_seed 42
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont


def load_tiff_as_pil(tiff_path: Path) -> Image.Image:
    """16-bit TIFF -> 8-bit PIL.Image (RGB)."""
    arr = tifffile.imread(str(tiff_path))  # uint16 (H, W, 3)
    arr8 = (arr // 256).astype(np.uint8)
    return Image.fromarray(arr8, mode="RGB")


def fit_height(img: Image.Image, target_h: int) -> Image.Image:
    w, h = img.size
    if h == target_h:
        return img
    new_w = max(1, int(round(w * target_h / h)))
    return img.resize((new_w, target_h), Image.LANCZOS)


def get_font(size: int = 14):
    for f in ("arial.ttf", "DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def make_pair(
    before: Image.Image,
    after: Image.Image,
    title: str,
    subtitle: str,
    target_h: int = 600,
    pad: int = 8,
    label_h: int = 50,
) -> Image.Image:
    before = fit_height(before, target_h)
    after = fit_height(after, target_h)
    W = before.width + after.width + pad
    H = target_h + label_h
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    canvas.paste(before, (0, label_h))
    canvas.paste(after, (before.width + pad, label_h))

    draw = ImageDraw.Draw(canvas)
    title_font = get_font(16)
    sub_font = get_font(13)
    draw.text((6, 4), title, fill=(255, 255, 255), font=title_font)
    draw.text((6, 24), subtitle, fill=(180, 220, 255), font=sub_font)

    label_font = get_font(14)
    draw.text((6, label_h - 18), "BEFORE (raw decode)", fill=(255, 200, 80), font=label_font)
    draw.text(
        (before.width + pad + 6, label_h - 18),
        "AFTER (Expert C render)",
        fill=(80, 220, 120),
        font=label_font,
    )
    return canvas


def make_contact_sheet(
    pair_imgs: list, cols: int = 3, pad: int = 6
) -> Image.Image:
    rows = math.ceil(len(pair_imgs) / cols)
    cell_w = max(p.width for p in pair_imgs)
    cell_h = max(p.height for p in pair_imgs)
    W = cols * cell_w + (cols + 1) * pad
    H = rows * cell_h + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), (16, 16, 16))
    for i, p in enumerate(pair_imgs):
        r, c = divmod(i, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + pad)
        sheet.paste(p, (x, y))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None,
                    help="default: data/fivek_expert_<X>_tiff_manifest.jsonl")
    ap.add_argument("--input_jpeg_dir", default=r"E:\Data\dataset\fivek_jpeg")
    ap.add_argument("--output_dir", default=None,
                    help="default: outputs/preview_fivek_expert_<X>")
    ap.add_argument("--expert", default="C")
    ap.add_argument("--n", type=int, default=12, help="random samples")
    ap.add_argument("--target_h", type=int, default=600, help="preview height (px)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    expert_lc = args.expert.lower()
    manifest = Path(args.manifest or f"data/fivek_expert_{expert_lc}_tiff_manifest.jsonl")
    out_dir = Path(args.output_dir or f"outputs/preview_fivek_expert_{expert_lc}")
    in_dir = Path(args.input_jpeg_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[config] manifest = {manifest}")
    print(f"[config] input JPEG dir = {in_dir}")
    print(f"[config] output dir = {out_dir}")

    # Load manifest
    records = []
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[load] manifest: {len(records)} records")

    # Filter to those with input JPEG present
    pool = []
    for r in records:
        jpg = in_dir / f"{r['stem']}.jpg"
        if jpg.exists() and Path(r["tiff_path"]).exists():
            pool.append((r, jpg))
    print(f"[load] {len(pool)} pairs (input jpg + output tiff both present)")

    if not pool:
        print("[ERR] no valid pairs")
        return 1

    # Sample N
    rng = random.Random(args.seed)
    pool.sort(key=lambda x: x[0]["stem"])
    sample = rng.sample(pool, min(args.n, len(pool)))
    print(f"[sample] {len(sample)} pairs (seed={args.seed})")

    pair_imgs = []
    for i, (r, jpg) in enumerate(sample):
        try:
            before = Image.open(jpg).convert("RGB")
            after = load_tiff_as_pil(Path(r["tiff_path"]))
        except Exception as e:
            print(f"  [skip] {r['stem']}: {e.__class__.__name__}: {e}")
            continue

        p = r["params"]
        title = r["stem"]
        subtitle = (
            f"WB={p.get('white_balance', 0):.0f}K  "
            f"Bright={p.get('brightness', 0):+.0f}  "
            f"Contrast={p.get('contrast', 0):+.0f}  "
            f"Shadows={p.get('shadows', 0):+.0f}  "
            f"Highlights={p.get('highlights', 0):+.0f}  "
            f"Sat={p.get('saturation', 0):+.0f}"
        )
        pair = make_pair(before, after, title, subtitle, target_h=args.target_h)
        out_path = out_dir / f"{r['stem']}.jpg"
        pair.save(out_path, "JPEG", quality=92)
        pair_imgs.append(pair)
        print(f"  [{i+1:>2}/{len(sample)}] {r['stem']:35s} -> {out_path.name}  "
              f"({pair.width}x{pair.height}, {out_path.stat().st_size/1024:.0f} KB)")

    # Contact sheet
    if pair_imgs:
        # Resize each pair to a smaller height for contact sheet
        thumb_h = 280
        thumbs = []
        for p in pair_imgs:
            scale = thumb_h / p.height
            thumbs.append(p.resize((int(p.width * scale), thumb_h), Image.LANCZOS))
        sheet = make_contact_sheet(thumbs, cols=3, pad=8)
        contact_path = out_dir / "_contact.jpg"
        sheet.save(contact_path, "JPEG", quality=88)
        print(f"\n[contact] {contact_path}  ({sheet.width}x{sheet.height}, "
              f"{contact_path.stat().st_size/1024:.0f} KB)")

    print(f"\n[done] open in Explorer: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
