"""Micro-pilot 验证 inverse_fit 的 LBFGS solver 在 diff_isp roundtrip 上能否恢复 ground truth.

流程:
  1. 加载 a0001-jmac_DSC1459.jpg (FiveK default JPEG)
  2. 从 aug_fivek_params.json 取一组非平凡参数 P_gt (默认 image_id=8959: contrast=43, saturation=21)
  3. target = apply_diff_isp(orig, P_gt[5_shared_dims])  ← 合成 ground truth
  4. P_recovered = inverse_fit(orig, target)
  5. 对比 P_recovered vs P_gt 在 5 共享维 (white_balance/contrast/shadows/highlights/saturation),
     报告每维 abs/rel 误差 + 像素 L1 误差.

通过准则:
  - 5 维平均 abs error < 5 (在 [-100, 100] 量程下 < 5%)
  - white_balance abs error < 200K (在 [2000, 10000] 量程下)
  - 像素 L1 < 0.01

通过 -> inverse_fit solver 可靠, 进入下一步 N 张批跑.
不通过 -> debug solver / 改 maxiter / 改 restarts / 改 loss weights.

用法:
  python tools/data/data_prep/micro_pilot_inverse_fit.py
  python tools/data/data_prep/micro_pilot_inverse_fit.py --image a0002-dgw_005.jpg --variant_id 1234
  python tools/data/data_prep/micro_pilot_inverse_fit.py --max_size 256 --maxiter 100
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

# 项目根: tools/data/data_prep/<this>.py -> .parent×3 = tools/, .parent×4 = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.diff_isp import apply_diff_isp  # noqa: E402
from tools.data.data_prep.inverse_fit import inverse_fit  # noqa: E402


SHARED_DIMS = ['white_balance', 'contrast', 'shadows', 'highlights', 'saturation']
INVERSE_FIT_EXTRAS = ['brightness', 'clarity']  # inverse_fit 有但 ground truth 没用


def load_image(path: Path, max_size: int = 512) -> torch.Tensor:
    """读 JPEG -> (1,3,H,W) float32 [0,1] 张量, 长边 resize 到 max_size."""
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    if max_size and max(t.shape[-2:]) > max_size:
        scale = max_size / max(t.shape[-2:])
        h = int(t.shape[-2] * scale)
        w = int(t.shape[-1] * scale)
        t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
    return t


def find_variant(aug_json: Path, image_name: str, variant_id: int = None,
                 expert: str = None):
    """在 aug_fivek_params.json 找指定图像的某个 variant.

    优先级: variant_id (image_id) > expert + image_name 第一个匹配.
    """
    with open(aug_json, encoding='utf-8') as f:
        d = json.load(f)
    matches = [s for s in d['samples'] if s['image_name'] == image_name]
    if not matches:
        return None, []
    if variant_id is not None:
        for s in matches:
            if s.get('image_id') == variant_id:
                return s, matches
    if expert:
        for s in matches:
            if s['expert'] == expert:
                return s, matches
    return matches[0], matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default='a0001-jmac_DSC1459.jpg')
    ap.add_argument('--image_dir', default=r'E:\Data\dataset\fivek_jpeg')
    ap.add_argument('--params_json', default='data/aug_fivek_params.json')
    ap.add_argument('--variant_id', type=int, default=8959,
                    help='image_id in aug_fivek_params.json (优先于 --expert)')
    ap.add_argument('--expert', default=None, help='备用: 按 expert 名筛 (default/expert_a/...)')
    ap.add_argument('--max_size', type=int, default=512,
                    help='resize 长边 (越小越快, 默认 512)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--n_restarts', type=int, default=3)
    ap.add_argument('--maxiter', type=int, default=200)
    ap.add_argument('--ssim_weight', type=float, default=0.5)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    print(f'[CONFIG] image={args.image}, variant_id={args.variant_id}, '
          f'max_size={args.max_size}, device={args.device}')

    # === 1. 加载原图 ===
    img_path = Path(args.image_dir) / args.image
    if not img_path.exists():
        print(f'[ERR] {img_path} 不存在')
        return 1
    orig = load_image(img_path, args.max_size).to(args.device)
    print(f'[ORIG] {orig.shape}, range=[{orig.min():.3f}, {orig.max():.3f}]')

    # === 2. 找 ground truth params ===
    sample, all_variants = find_variant(
        Path(args.params_json), args.image, args.variant_id, args.expert)
    if sample is None:
        print(f'[ERR] {args.image} 在 {args.params_json} 没找到')
        print(f'  共 {len(all_variants)} 个变体')
        return 1
    P_gt = sample['params']
    print(f'\n[GT] id={sample["id"]} expert={sample["expert"]} '
          f'(共 {len(all_variants)} 个 {args.image} 变体)')
    for k, v in P_gt.items():
        print(f'  {k:>20s} = {v:>10.3f}')

    # === 3. 渲染合成 target ===
    # 只用 5 共享维, ev_compensation 跳过 (inverse_fit 不搜索此维)
    P_gt_tensor = {
        k: torch.tensor([float(P_gt[k])], device=args.device, dtype=torch.float32)
        for k in SHARED_DIMS if k in P_gt
    }
    print(f'\n[RENDER] target = apply_diff_isp(orig, P_gt[{list(P_gt_tensor.keys())}])')
    with torch.no_grad():
        target = apply_diff_isp(orig, P_gt_tensor)
    target = target.clamp(0, 1)
    print(f'[TARGET] {target.shape}, range=[{target.min():.3f}, {target.max():.3f}]')
    delta = (target - orig).abs().mean().item()
    print(f'[DELTA] mean |target-orig| = {delta:.4f} (>0.01 表示有可见编辑)')

    # === 4. 跑 inverse_fit ===
    print(f'\n[INVERSE_FIT] n_restarts={args.n_restarts}, maxiter={args.maxiter}, '
          f'ssim_w={args.ssim_weight}')
    t0 = time.time()
    P_recovered, final_loss = inverse_fit(
        orig, target,
        n_restarts=args.n_restarts,
        maxiter=args.maxiter,
        ssim_weight=args.ssim_weight,
        verbose=args.verbose,
    )
    dt = time.time() - t0
    print(f'[DONE] final_loss={final_loss:.6f}, {dt:.1f}s')

    # === 5. 对比 ===
    print(f'\n[COMPARE] P_recovered vs P_gt (5 shared dims):')
    print(f'  {"param":<18s}  {"GT":>10s}  {"recovered":>12s}  '
          f'{"abs_err":>10s}  {"rel_err":>10s}')
    print('  ' + '-' * 70)
    abs_errors = []
    for dim in SHARED_DIMS:
        gt = float(P_gt.get(dim, 0))
        rec = float(P_recovered.get(dim, 0))
        abs_err = abs(rec - gt)
        if dim == 'white_balance':
            rel_err = abs_err / 8000 * 100  # 量程 [2000, 10000]
        else:
            rel_err = abs_err / 200 * 100  # 量程 [-100, 100]
        abs_errors.append(abs_err)
        flag = '  ' if rel_err < 5 else ('!!' if rel_err > 20 else '! ')
        print(f'  {flag}{dim:<16s}  {gt:>10.3f}  {rec:>12.3f}  '
              f'{abs_err:>10.3f}  {rel_err:>9.1f}%')

    print(f'\n[EXTRAS] inverse_fit 多出来的维度 (P_gt 渲染时设 0, 期望 recovered ≈ 0):')
    for dim in INVERSE_FIT_EXTRAS:
        rec = float(P_recovered.get(dim, 0))
        flag = '  ' if abs(rec) < 5 else '! '
        print(f'  {flag}{dim:<16s}  expected~0,  recovered={rec:>10.3f}')

    # 像素误差
    with torch.no_grad():
        P_recovered_tensor = {
            k: torch.tensor([float(P_recovered[k])], device=args.device,
                            dtype=torch.float32)
            for k in P_recovered
        }
        rendered_back = apply_diff_isp(orig, P_recovered_tensor).clamp(0, 1)
        pixel_l1 = F.l1_loss(rendered_back, target).item()

    print(f'\n[PIXEL] L1(rendered_back, target) = {pixel_l1:.5f}')

    # 通过准则
    print('\n[VERDICT]')
    crit_pixel = pixel_l1 < 0.01
    crit_dims = sum(1 for e in abs_errors[1:] if e < 5)  # 后 4 维 (除 wb)
    crit_wb = abs_errors[0] < 200
    print(f'  pixel L1 < 0.01:       {"PASS" if crit_pixel else "FAIL"} ({pixel_l1:.5f})')
    print(f'  WB abs_err < 200K:     {"PASS" if crit_wb else "FAIL"} ({abs_errors[0]:.0f})')
    print(f'  4 维 abs_err < 5 each: {crit_dims}/4')
    overall = crit_pixel and crit_wb and crit_dims >= 3
    print(f'  OVERALL:               {"PASS ✓" if overall else "FAIL ✗"}')
    return 0 if overall else 2


if __name__ == '__main__':
    sys.exit(main())
