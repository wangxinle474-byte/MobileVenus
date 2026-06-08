"""FiveK DNG + lrcat Expert settings → 16-bit TIFF 渲染管线 (Path 3 smoke).

Pipeline:
  data/fivek_expert_abcde_params.json  (parse_fivek_lrcat.py 已产出)
    + E:\\Data\\dataset\\fivek_dataset\\raw_photos\\HQa*\\photos\\*.dng
    -> rawpy.postprocess (16-bit sRGB, camera WB, no auto-bright)
    -> models.isp_pipeline.render_params (6D LR ISP 近似)
    -> tifffile 16-bit RGB TIFF

NOTE: 这是一个 LR 近似渲染器, 不是 Adobe LR 像素级复刻。
      与官方 tiff16_c 对比时 PSNR 通常在 25-32 dB 区间。
      用于验证 Path 3 (real-LR target lifts ceiling) 的 in-house pipeline 可行性。

用法 (smoke):
  python tools/data/data_prep/render_fivek_expert_tiff.py --expert C --limit 10

全量:
  python tools/data/data_prep/render_fivek_expert_tiff.py --expert C --limit 0 \\
      --max_size 2048 --output_dir E:/Data/dataset/fivek_expert_c_tiff
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rawpy
import tifffile
import torch
from tqdm import tqdm

# import ISP from project root
_REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO))
from models.isp_pipeline import render_params  # noqa: E402


# --------------------------------------------------------------------------- #
# DNG locator
# --------------------------------------------------------------------------- #

def build_dng_index(fivek_root: Path) -> dict:
    """Map image_name (basename, with .dng) -> full Path. One-time scan."""
    idx = {}
    for sub in sorted(fivek_root.glob("HQa*/photos")):
        for dng in sub.glob("*.dng"):
            idx[dng.name] = dng
    return idx


# --------------------------------------------------------------------------- #
# Params extraction (Expert record -> render_params kwargs)
# --------------------------------------------------------------------------- #

def make_params_tensor(record: dict, device: str = "cpu") -> dict:
    """从 Expert record 提 6D ISP 输入张量 (B=1)."""
    p = record["params"]
    raw_lr = record.get("raw_lr", {})

    # WB: Temperature (K). params.white_balance 优先, fallback raw_lr.Temperature
    wb = p.get("white_balance") or raw_lr.get("Temperature") or 6500.0
    wb = float(wb) if wb is not None else 6500.0

    # EV: PV2012 用 Exposure (stops); PV2010 用 Brightness ([-150, +150] 感知中调)
    pv = str(raw_lr.get("ProcessVersion", ""))
    is_pv2012 = pv.startswith("6") or ("Exposure" in raw_lr and "Brightness" not in raw_lr)
    if is_pv2012:
        ev = float(raw_lr.get("Exposure", p.get("brightness", 0)) or 0.0)
    else:
        # PV2010 brightness: ~/50 stops approximation
        bright_pv2010 = raw_lr.get("Brightness", p.get("brightness", 0))
        ev = float(bright_pv2010 or 0.0) / 50.0

    def t(x):
        return torch.tensor([[float(x)]], device=device)

    return {
        "white_balance": t(wb),
        "ev_compensation": t(ev),
        "contrast": t(p.get("contrast", 0) or 0),
        "shadows": t(p.get("shadows", 0) or 0),
        "highlights": t(p.get("highlights", 0) or 0),
        "saturation": t(p.get("saturation", 0) or 0),
    }


# --------------------------------------------------------------------------- #
# Per-image render
# --------------------------------------------------------------------------- #

def _resize_uint16_rgb(arr: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """16-bit RGB (H,W,3) -> resized via PIL per-channel 'I;16' LANCZOS."""
    from PIL import Image
    out = np.empty((new_h, new_w, 3), dtype=np.uint16)
    for c in range(3):
        pil = Image.fromarray(arr[..., c], mode="I;16")
        out[..., c] = np.array(pil.resize((new_w, new_h), Image.LANCZOS), dtype=np.uint16)
    return out


def render_one(
    dng_path: Path,
    params_t: dict,
    max_size: Optional[int] = None,
    half_size: bool = False,
) -> np.ndarray:
    """Render one DNG + LR params -> uint16 (H, W, 3) sRGB ndarray."""
    with rawpy.imread(str(dng_path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            half_size=half_size,
            no_auto_bright=True,
            output_bps=16,
            output_color=rawpy.ColorSpace.sRGB,
        )  # (H, W, 3) uint16, sRGB-encoded

    img = rgb16.astype(np.float32) / 65535.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    with torch.no_grad():
        out_t = render_params(img_t, params_t)

    out = out_t.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    out16 = (out * 65535.0 + 0.5).astype(np.uint16)

    if max_size is not None:
        h, w = out16.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            out16 = _resize_uint16_rgb(out16, new_w, new_h)

    # tifffile requires C-contiguous; permute + numpy() returns non-contig
    return np.ascontiguousarray(out16)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params_json", default="data/fivek_expert_abcde_params.json")
    ap.add_argument("--fivek_root", default=r"E:\Data\dataset\fivek_dataset\raw_photos")
    ap.add_argument("--expert", default="C", choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--output_dir", default=r"E:\Data\dataset\fivek_expert_c_tiff_smoke")
    ap.add_argument("--limit", type=int, default=10, help="0 = all")
    ap.add_argument("--max_size", type=int, default=None, help="optional resize long edge")
    ap.add_argument("--half_size", action="store_true", help="rawpy half-res decode (4x faster)")
    # NOTE: lzw / packbits 需要 imagecodecs (默认未装), zlib/deflate 用 Python 内置
    ap.add_argument("--compression", default="zlib", choices=["none", "zlib", "deflate"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    print(f"[config] expert={args.expert}  limit={args.limit}  max_size={args.max_size}")
    print(f"[config] fivek_root={args.fivek_root}")
    print(f"[config] output_dir={args.output_dir}")

    # load expert records
    params_path = Path(args.params_json)
    if not params_path.exists():
        print(f"[ERR] params json not found: {params_path}")
        return 1
    with open(params_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = [s for s in data["samples"] if s["expert"] == args.expert]
    print(f"[load] expert {args.expert}: {len(records)} records")
    if args.limit:
        records = records[: args.limit]
        print(f"  smoke -> first {len(records)}")

    # build DNG index
    fivek_root = Path(args.fivek_root)
    if not fivek_root.exists():
        print(f"[ERR] fivek_root not found: {fivek_root}")
        return 1
    print(f"[scan] indexing DNG files under {fivek_root}/HQa*/photos ...")
    t0 = time.time()
    dng_idx = build_dng_index(fivek_root)
    print(f"[scan] indexed {len(dng_idx)} DNG files in {time.time()-t0:.1f}s")

    # output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compression = None if args.compression == "none" else args.compression

    n_ok = n_skip = n_err = n_missing = 0
    t0 = time.time()
    for r in tqdm(records, desc=f"render exp{args.expert}"):
        name = r["image_name"]
        dng = dng_idx.get(name)
        if dng is None:
            tqdm.write(f"[missing-dng] {name}")
            n_missing += 1
            continue
        stem = Path(name).stem
        out_path = out_dir / f"{stem}.tif"
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue
        try:
            params_t = make_params_tensor(r)
            tiff = render_one(dng, params_t, args.max_size, args.half_size)
            tifffile.imwrite(
                str(out_path),
                tiff,
                photometric="rgb",
                compression=compression,
            )
            n_ok += 1
        except Exception as e:
            tqdm.write(f"[err] {name}: {e.__class__.__name__}: {e}")
            n_err += 1

    dt = time.time() - t0
    print(
        f"\n[done] ok={n_ok}  skip={n_skip}  missing={n_missing}  err={n_err} "
        f"in {dt:.1f}s ({dt/max(n_ok,1):.2f}s/img)"
    )
    print(f"[done] output_dir = {out_dir.resolve()}")
    if n_ok:
        sizes = [p.stat().st_size for p in out_dir.glob("*.tif")][:n_ok]
        if sizes:
            print(
                f"[done] tiff size: min={min(sizes)/1e6:.1f}MB "
                f"avg={sum(sizes)/len(sizes)/1e6:.1f}MB "
                f"max={max(sizes)/1e6:.1f}MB"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
