"""Build manifest JSONL for FiveK Expert TIFF rendered targets.

Consumes:
  - data/fivek_expert_abcde_params.json   (parse_fivek_lrcat.py output)
  - <tiff_dir>/*.tif                       (render_fivek_expert_tiff.py output)

Produces:
  - data/fivek_expert_<X>_tiff_manifest.jsonl  (1 line per (image, expert) pair)
  - prints summary stats + sanity-check on N random TIFFs

Usage:
  python tools/data/data_prep/build_fivek_expert_tiff_manifest.py --expert C \
      --tiff_dir E:/Data/dataset/fivek_expert_c_tiff \
      --output data/fivek_expert_c_tiff_manifest.jsonl
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import tifffile


def find_dng_index(fivek_root: Path) -> dict:
    idx = {}
    for sub in sorted(fivek_root.glob("HQa*/photos")):
        for dng in sub.glob("*.dng"):
            idx[dng.name] = str(dng)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params_json", default="data/fivek_expert_abcde_params.json")
    ap.add_argument("--fivek_root", default=r"E:\Data\dataset\fivek_dataset\raw_photos")
    ap.add_argument("--tiff_dir", default=r"E:\Data\dataset\fivek_expert_c_tiff")
    ap.add_argument("--expert", default="C", choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--output", default=None,
                    help="default: data/fivek_expert_<X>_tiff_manifest.jsonl")
    ap.add_argument("--sanity_n", type=int, default=5,
                    help="random TIFFs to load + inspect")
    args = ap.parse_args()

    out_path = Path(
        args.output or f"data/fivek_expert_{args.expert.lower()}_tiff_manifest.jsonl"
    )

    print(f"[config] expert={args.expert}")
    print(f"[config] tiff_dir={args.tiff_dir}")
    print(f"[config] output={out_path}")

    # 1) Load params records
    with open(args.params_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = [s for s in data["samples"] if s["expert"] == args.expert]
    print(f"[load] expert {args.expert}: {len(records)} params records")

    # 2) Index DNG paths
    dng_idx = find_dng_index(Path(args.fivek_root))
    print(f"[load] indexed {len(dng_idx)} DNG files")

    # 3) Walk records, match TIFF + DNG
    tiff_dir = Path(args.tiff_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matched = []
    missing_tiff = []
    missing_dng = []
    for r in records:
        name = r["image_name"]
        stem = Path(name).stem
        tiff_path = tiff_dir / f"{stem}.tif"
        dng_path = dng_idx.get(name)
        if not tiff_path.exists():
            missing_tiff.append(name)
            continue
        if dng_path is None:
            missing_dng.append(name)
            continue
        matched.append({
            "image_name": name,
            "stem": stem,
            "expert": args.expert,
            "tiff_path": str(tiff_path).replace("\\", "/"),
            "source_dng": dng_path.replace("\\", "/"),
            "tiff_bytes": tiff_path.stat().st_size,
            "params": r["params"],
            "raw_lr": r.get("raw_lr", {}),
        })

    print(f"[match] {len(matched)} matched   {len(missing_tiff)} missing tiff   "
          f"{len(missing_dng)} missing dng")
    if missing_tiff[:5]:
        print(f"  first missing tiff (up to 5): {missing_tiff[:5]}")
    if missing_dng[:5]:
        print(f"  first missing dng (up to 5):  {missing_dng[:5]}")

    # 4) Sanity check N random TIFFs
    print(f"\n[sanity] loading {args.sanity_n} random TIFFs ...")
    random.seed(0)
    sample = random.sample(matched, min(args.sanity_n, len(matched)))
    sanity_ok = 0
    for s in sample:
        try:
            arr = tifffile.imread(s["tiff_path"])
            ok = (
                arr.dtype == np.uint16
                and arr.ndim == 3
                and arr.shape[2] == 3
                and arr.size > 0
            )
            print(f"  {s['stem']:35s}  shape={arr.shape}  dtype={arr.dtype}  "
                  f"min={arr.min():>5d}  max={arr.max():>5d}  "
                  f"mean={arr.mean():>7.0f}  "
                  f"{'OK' if ok else 'BAD'}")
            if ok:
                sanity_ok += 1
        except Exception as e:
            print(f"  {s['stem']:35s}  LOAD FAIL: {e.__class__.__name__}: {e}")
    print(f"[sanity] {sanity_ok}/{len(sample)} passed")

    # 5) Write manifest
    with open(out_path, "w", encoding="utf-8") as f:
        for m in matched:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"\n[write] manifest -> {out_path}  ({len(matched)} lines, "
          f"{out_path.stat().st_size/1024:.1f} KB)")

    # 6) Summary stats
    total_gb = sum(m["tiff_bytes"] for m in matched) / 1e9
    avg_mb = sum(m["tiff_bytes"] for m in matched) / max(len(matched), 1) / 1e6
    print(f"\n[stats] total: {len(matched)} TIFFs, {total_gb:.2f} GB, "
          f"avg {avg_mb:.2f} MB")

    # Param distribution (Expert C parameters)
    print(f"\n[stats] Expert {args.expert} param ranges:")
    keys = ["white_balance", "brightness", "contrast", "shadows", "highlights",
            "saturation", "clarity"]
    for k in keys:
        vals = [m["params"].get(k, 0) or 0 for m in matched]
        vals = np.array(vals, dtype=np.float64)
        print(f"  {k:14s}  n={len(vals)}  min={vals.min():>8.2f}  "
              f"max={vals.max():>8.2f}  mean={vals.mean():>8.2f}  "
              f"std={vals.std():>6.2f}")

    return 0 if sanity_ok == len(sample) else 1


if __name__ == "__main__":
    sys.exit(main())
