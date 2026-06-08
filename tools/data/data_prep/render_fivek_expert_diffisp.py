"""FiveK Expert C 参数 → apply_diff_isp 渲染 (与训练引擎一致).

Input:  fivek_jpeg/ (sRGB JPEG) + fivek_expert_abcde_params.json
Engine: models.diff_isp.apply_diff_isp (7D: wb, brightness, contrast,
        shadows, highlights, saturation, clarity)
Output: 16-bit TIFF (from 8-bit JPEG, low bits padded)

用法 (smoke):
  python tools/data/data_prep/render_fivek_expert_diffisp.py --expert C --limit 10

全量:
  python tools/data/data_prep/render_fivek_expert_diffisp.py --expert C --limit 0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tifffile
from PIL import Image
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO))
from models.diff_isp import apply_diff_isp  # noqa: E402

PARAM_NAMES_7D = [
    'white_balance', 'brightness', 'contrast',
    'shadows', 'highlights', 'saturation', 'clarity',
]


def make_params_tensor(record: dict) -> dict:
    """Expert record → 7D diff_isp params dict (each (1,) tensor)."""
    p = record['params']
    result = {}
    for name in PARAM_NAMES_7D:
        val = p.get(name, 0) or 0
        result[name] = torch.tensor([float(val)], dtype=torch.float32)
    return result


def render_one_jpeg(
    jpeg_path: Path,
    params_t: dict,
    max_size: int | None = None,
) -> np.ndarray:
    """JPEG → apply_diff_isp → uint16 (H, W, 3)."""
    img = Image.open(jpeg_path).convert('RGB')

    # optional resize
    if max_size is not None:
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.asarray(img).astype(np.float32) / 255.0
    img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    with torch.no_grad():
        out_t = apply_diff_isp(img_t, params_t).clamp(0.0, 1.0)

    out = out_t.squeeze(0).permute(1, 2, 0).numpy()
    out16 = (out * 65535.0 + 0.5).astype(np.uint16)
    return np.ascontiguousarray(out16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params_json', default='data/fivek_expert_abcde_params.json')
    ap.add_argument('--jpeg_dir', default=r'E:\Data\dataset\fivek_jpeg')
    ap.add_argument('--expert', default='C', choices=list('ABCDE'))
    ap.add_argument('--output_dir', default=r'E:\Data\dataset\fivek_expert_c_diffisp')
    ap.add_argument('--limit', type=int, default=10, help='0 = all')
    ap.add_argument('--max_size', type=int, default=None)
    ap.add_argument('--compression', default='zlib', choices=['none', 'zlib', 'deflate'])
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    print(f'[config] expert={args.expert}  limit={args.limit}  max_size={args.max_size}')
    print(f'[config] jpeg_dir={args.jpeg_dir}')
    print(f'[config] output_dir={args.output_dir}')
    print(f'[config] engine=apply_diff_isp (7D, differentiable)')

    # load expert records
    params_path = Path(args.params_json)
    if not params_path.exists():
        print(f'[ERR] params json not found: {params_path}')
        return 1
    with open(params_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    records = [s for s in data['samples'] if s['expert'] == args.expert]
    print(f'[load] expert {args.expert}: {len(records)} records')
    if args.limit:
        records = records[:args.limit]
        print(f'  smoke -> first {len(records)}')

    jpeg_dir = Path(args.jpeg_dir)
    if not jpeg_dir.exists():
        print(f'[ERR] jpeg_dir not found: {jpeg_dir}')
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compression = None if args.compression == 'none' else args.compression

    n_ok = n_skip = n_err = n_missing = 0
    t0 = time.time()
    for r in tqdm(records, desc=f'render exp{args.expert} (diff_isp)'):
        name = r['image_name']
        stem = Path(name).stem
        jpg = jpeg_dir / f'{stem}.jpg'
        if not jpg.exists():
            tqdm.write(f'[missing] {name}')
            n_missing += 1
            continue

        out_path = out_dir / f'{stem}.tif'
        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            params_t = make_params_tensor(r)
            tiff = render_one_jpeg(jpg, params_t, args.max_size)
            tifffile.imwrite(
                str(out_path), tiff,
                photometric='rgb', compression=compression,
            )
            n_ok += 1
        except Exception as e:
            tqdm.write(f'[err] {name}: {e.__class__.__name__}: {e}')
            n_err += 1

    dt = time.time() - t0
    print(
        f'\n[done] ok={n_ok}  skip={n_skip}  missing={n_missing}  err={n_err} '
        f'in {dt:.1f}s ({dt/max(n_ok,1):.2f}s/img)'
    )
    print(f'[done] output_dir = {out_dir.resolve()}')
    if n_ok:
        sizes = [p.stat().st_size for p in out_dir.glob('*.tif')][:n_ok]
        if sizes:
            total_gb = sum(sizes) / 1e9
            print(
                f'[done] tiff size: min={min(sizes)/1e6:.1f}MB '
                f'avg={sum(sizes)/len(sizes)/1e6:.1f}MB '
                f'max={max(sizes)/1e6:.1f}MB  total={total_gb:.2f}GB'
            )
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
