"""End-to-end pilot: 真实 (orig, teacher_edit) 对 + inverse_fit + 残差报告.

跟 micro_pilot_inverse_fit.py 不同: target 不是 diff_isp 合成的, 而是 FireRed 1.1 API 真实输出.
这检验 FireRed 输出能否被 7 维 ISP 参数 (diff_isp) 表达.

流程:
  1. 加载 orig (FiveK JPEG) + target (FireRed PNG)
  2. resize target 到 orig 分辨率 (FireRed 通常 upscale 到 ~1MP)
  3. inverse_fit(orig, target) → P_inferred
  4. rendered_back = apply_diff_isp(orig, P_inferred)
  5. 报告 pixel L1(rendered_back, target) + 各维参数
  6. 保存 orig/target/rendered_back/diff_x5 4 张图给肉眼对比

用法:
  python tools/data/data_prep/inverse_fit_real_pilot.py \\
      --orig E:\\Data\\dataset\\fivek_jpeg\\a0006-IMG_2787.jpg \\
      --target outputs/teacher_edits/fivek_a0006_pilot/0006.png \\
      --out_dir outputs/inverse_fit_pilot/a0006

通过准则:
  L1 < 0.01: 极佳 fit
  L1 < 0.02: 合理 fit, 进 N 张批跑
  L1 < 0.05: 一般 fit, FireRed 引入了部分非 ISP 内容
  L1 ≥ 0.05: fit 差, 需想策略 (例如 SimpleTone-only 模式 / 后处理 ISP 投影)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.diff_isp import apply_diff_isp  # noqa: E402
from tools.data.data_prep.inverse_fit import inverse_fit  # noqa: E402


def load_image(path: Path, max_size: int = 0) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if max_size and max(t.shape[-2:]) > max_size:
        scale = max_size / max(t.shape[-2:])
        h = int(t.shape[-2] * scale)
        w = int(t.shape[-1] * scale)
        t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
    return t


def save_image(t: torch.Tensor, path: Path):
    arr = (t.squeeze(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--out_dir', default='outputs/inverse_fit_pilot')
    ap.add_argument('--max_size', type=int, default=512,
                    help='orig + target 都 resize 长边到此 (越小越快)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--n_restarts', type=int, default=3)
    ap.add_argument('--maxiter', type=int, default=200)
    ap.add_argument('--ssim_weight', type=float, default=0.5)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    print(f'[CONFIG] orig={Path(args.orig).name}, target={Path(args.target).name}, '
          f'max_size={args.max_size}, device={args.device}')

    orig_path = Path(args.orig)
    target_path = Path(args.target)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not orig_path.exists():
        print(f'[ERR] orig 不存在: {orig_path}'); return 1
    if not target_path.exists():
        print(f'[ERR] target 不存在: {target_path}'); return 1

    # 加载 orig (按 max_size 缩)
    orig = load_image(orig_path, args.max_size).to(args.device)
    H, W = orig.shape[-2:]
    print(f'[ORIG]   {tuple(orig.shape)}  range=[{orig.min():.3f}, {orig.max():.3f}]')

    # 加载 target (full size) 然后 resize 到 orig 一致
    target_full = load_image(target_path, 0).to(args.device)
    if target_full.shape[-2:] != (H, W):
        target = F.interpolate(target_full, size=(H, W),
                               mode='bilinear', align_corners=False)
        print(f'[TARGET] {tuple(target_full.shape)} → {tuple(target.shape)} (resized)')
    else:
        target = target_full
        print(f'[TARGET] {tuple(target.shape)} (no resize)')
    target = target.clamp(0, 1)

    delta = (target - orig).abs().mean().item()
    print(f'[DELTA]  mean |target-orig| = {delta:.4f}')

    # inverse_fit
    print(f'\n[INVERSE_FIT] n_restarts={args.n_restarts}, maxiter={args.maxiter}, '
          f'ssim_w={args.ssim_weight}')
    t0 = time.time()
    P_inferred, final_loss = inverse_fit(
        orig, target,
        n_restarts=args.n_restarts,
        maxiter=args.maxiter,
        ssim_weight=args.ssim_weight,
        verbose=args.verbose,
    )
    dt = time.time() - t0
    print(f'[DONE] final_loss={final_loss:.6f}, {dt:.1f}s')

    print(f'\n[P_INFERRED]')
    for k, v in P_inferred.items():
        print(f'  {k:>16s} = {float(v):>10.3f}')

    # rendered_back
    P_t = {k: torch.tensor([float(v)], device=args.device, dtype=torch.float32)
           for k, v in P_inferred.items()}
    with torch.no_grad():
        rendered_back = apply_diff_isp(orig, P_t).clamp(0, 1)

    pixel_l1 = F.l1_loss(rendered_back, target).item()
    pixel_l2 = F.mse_loss(rendered_back, target).item() ** 0.5
    print(f'\n[FIT QUALITY]')
    print(f'  L1(rendered, target) = {pixel_l1:.5f}')
    print(f'  L2(rendered, target) = {pixel_l2:.5f}')

    # 保存对比图
    save_image(orig, out_dir / 'orig.png')
    save_image(target, out_dir / 'target.png')
    save_image(rendered_back, out_dir / 'rendered_back.png')
    diff_x5 = ((target - rendered_back).abs() * 5).clamp(0, 1)
    save_image(diff_x5, out_dir / 'diff_x5.png')

    # 保存报告
    with open(out_dir / 'fit_report.json', 'w', encoding='utf-8') as f:
        json.dump({
            'orig': str(orig_path),
            'target': str(target_path),
            'orig_shape': list(orig.shape),
            'target_orig_shape': list(target_full.shape),
            'fit_at_size': [H, W],
            'P_inferred': {k: float(v) for k, v in P_inferred.items()},
            'pixel_l1': pixel_l1,
            'pixel_l2': pixel_l2,
            'final_loss': final_loss,
            'runtime_sec': dt,
            'config': {
                'n_restarts': args.n_restarts, 'maxiter': args.maxiter,
                'ssim_weight': args.ssim_weight,
            },
        }, f, indent=2)
    print(f'\n[SAVED] {out_dir}/  (orig.png, target.png, rendered_back.png, '
          f'diff_x5.png, fit_report.json)')

    # 裁决
    print('\n[VERDICT]')
    if pixel_l1 < 0.01:
        print(f'  L1={pixel_l1:.5f}  PASS (极佳, FireRed 输出在 ISP 表达范围内)')
        rc = 0
    elif pixel_l1 < 0.02:
        print(f'  L1={pixel_l1:.5f}  PASS (合理, 可以 scale 到 N 张)')
        rc = 0
    elif pixel_l1 < 0.05:
        print(f'  L1={pixel_l1:.5f}  WARN (一般, FireRed 含部分非 ISP 内容)')
        rc = 0
    else:
        print(f'  L1={pixel_l1:.5f}  FAIL (差, 需想策略)')
        rc = 2
    return rc


if __name__ == '__main__':
    sys.exit(main())
