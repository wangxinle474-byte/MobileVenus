"""
诊断 diff_isp 参数之间的冲突性 / 非唯一性。

做法:
  1. 加载一张 FiveK 图片, 用随机 GT 参数渲染一个 target
  2. 对每个关键参数对, 固定其他 7 个参数 at GT, 在 2D 网格上扫描该对
     计算 L1 loss (rendered vs target)
  3. 画出 loss landscape heatmap
  4. 根据"谷底延伸方向"定量非唯一性

指标:
  - ridge_10%: loss 在 [min, min+10%*range] 内的网格点占比
                高 = 宽谷, 低 = 窄谷
  - ratio_along_vs_perp: 主方向 eigenvalue 比值 (Hessian 分析)

输出: outputs/diagnose_param_conflicts/*.png + CSV 指标
"""
import os
# Windows 上 PyTorch + matplotlib 的 OpenMP 冲突 workaround
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.diff_isp import apply_diff_isp
from tools.data.inverse_fit import PARAM_SPEC

# ─────────────────── 关键参数对 (猜想有冲突的) ───────────────────
PAIRS = [
    ('saturation', 'vibrance'),          # 最怀疑: 都是色度缩放
    ('contrast', 'brightness'),          # 都影响中间调亮度
    ('shadows', 'highlights'),           # 互补但都调 luminance
    ('ev_compensation', 'brightness'),   # 都是整体亮度
    ('contrast', 'clarity'),             # 都是对比度 (全局 vs 局部)
    ('saturation', 'contrast'),          # 耦合?
    ('white_balance', 'ev_compensation'),# WB 也影响亮度
    ('vibrance', 'contrast'),            # 可能耦合
]

GT_PARAMS = {
    'ev_compensation': 0.3,
    'white_balance':   6000.0,
    'contrast':        20.0,
    'brightness':      10.0,
    'shadows':         15.0,
    'highlights':     -20.0,
    'saturation':      15.0,
    'vibrance':        10.0,
    'clarity':         5.0,
}


def render_loss(orig, target, params_dict, device):
    params = {k: torch.tensor([float(v)], device=device, dtype=torch.float32)
              for k, v in params_dict.items()}
    with torch.no_grad():
        rendered = apply_diff_isp(orig, params)
        rendered = torch.nan_to_num(rendered, nan=0.5, posinf=1.0, neginf=0.0)
        return F.l1_loss(rendered, target).item()


def sweep_pair(orig, target, p1, p2, gt_params, grid_n, device, radius_scale=0.6):
    """固定 7 参数, 扫 (p1, p2) 在以 GT 为中心的网格."""
    spec1 = PARAM_SPEC[p1]
    spec2 = PARAM_SPEC[p2]
    r1 = (spec1['hi'] - spec1['lo']) * radius_scale / 2
    r2 = (spec2['hi'] - spec2['lo']) * radius_scale / 2
    p1_vals = np.linspace(
        max(spec1['lo'], gt_params[p1] - r1),
        min(spec1['hi'], gt_params[p1] + r1), grid_n)
    p2_vals = np.linspace(
        max(spec2['lo'], gt_params[p2] - r2),
        min(spec2['hi'], gt_params[p2] + r2), grid_n)

    fixed_params = {k: v for k, v in gt_params.items() if k not in (p1, p2)}

    loss_grid = np.zeros((grid_n, grid_n))
    for i, v1 in enumerate(p1_vals):
        for j, v2 in enumerate(p2_vals):
            params = dict(fixed_params)
            params[p1] = float(v1)
            params[p2] = float(v2)
            loss_grid[i, j] = render_loss(orig, target, params, device)
    return p1_vals, p2_vals, loss_grid


def ridge_stats(loss_grid, rel_thresh=0.1):
    """量化 '谷底' 宽度.

    rel_thresh=0.1 表示 loss 在 [min, min+10%*(max-min)] 之内的点算"谷底"。
    高占比 = 宽而平的山谷 = 非唯一性严重。
    """
    lmin, lmax = loss_grid.min(), loss_grid.max()
    rng = lmax - lmin
    if rng < 1e-8:
        return dict(min=lmin, max=lmax, ridge_pct=100.0)
    thresh = lmin + rel_thresh * rng
    ridge_mask = loss_grid < thresh
    return dict(
        min=lmin, max=lmax, ridge_pct=100.0 * ridge_mask.sum() / ridge_mask.size,
        dynamic_range=rng,
    )


def hessian_ratio(p1_vals, p2_vals, loss_grid):
    """2 阶 Taylor 展开中心 Hessian 的特征值比 (大/小).

    比值越大, loss 越"各向异性" (只对一个方向敏感, 另一方向 flat = 非唯一性)。
    """
    ci, cj = len(p1_vals) // 2, len(p2_vals) // 2
    # 正方形窗
    d1 = (p1_vals[1] - p1_vals[0])
    d2 = (p2_vals[1] - p2_vals[0])
    if ci <= 1 or cj <= 1:
        return 1.0
    f = loss_grid
    # 二阶偏导 (中心差分)
    fxx = (f[ci+1, cj] - 2*f[ci, cj] + f[ci-1, cj]) / (d1 * d1 + 1e-12)
    fyy = (f[ci, cj+1] - 2*f[ci, cj] + f[ci, cj-1]) / (d2 * d2 + 1e-12)
    fxy = ((f[ci+1, cj+1] - f[ci+1, cj-1]
           - f[ci-1, cj+1] + f[ci-1, cj-1]) / (4 * d1 * d2 + 1e-12))
    H = np.array([[fxx, fxy], [fxy, fyy]])
    # 对称化
    H = 0.5 * (H + H.T)
    try:
        eigs = np.linalg.eigvalsh(H)
        emax, emin = eigs.max(), eigs.min()
        if abs(emin) < 1e-12:
            return float('inf')
        return float(abs(emax) / abs(emin))
    except np.linalg.LinAlgError:
        return float('nan')


def plot_heatmap(p1_vals, p2_vals, loss_grid, p1, p2, gt_params, out_path, stats):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    # pcolormesh 需要 (Y, X) 布局
    im = ax.pcolormesh(p1_vals, p2_vals, loss_grid.T,
                        cmap='viridis', shading='auto')
    cs = ax.contour(p1_vals, p2_vals, loss_grid.T,
                    levels=6, colors='white', alpha=0.5, linewidths=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%.4f')
    plt.colorbar(im, ax=ax, label='L1 loss')
    ax.axvline(gt_params[p1], color='red', linestyle='--', alpha=0.6, linewidth=0.8)
    ax.axhline(gt_params[p2], color='red', linestyle='--', alpha=0.6, linewidth=0.8)
    ax.scatter([gt_params[p1]], [gt_params[p2]],
               color='red', marker='*', s=200, zorder=5, label='GT')
    ax.set_xlabel(p1)
    ax.set_ylabel(p2)
    ax.set_title(
        f'{p1} vs {p2}\n'
        f'ridge@10%={stats["ridge_pct"]:.1f}%   '
        f'λmax/λmin={stats.get("hess_ratio", float("nan")):.1f}'
    )
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img', default=None,
                    help='测试图片路径 (默认从 E:\\Data\\dataset\\fivek_jpeg 取第一张)')
    ap.add_argument('--out_dir', default='outputs/diagnose_param_conflicts')
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--grid', type=int, default=25, help='网格大小 (grid x grid)')
    ap.add_argument('--radius', type=float, default=0.6,
                    help='扫描半径 (相对参数 range)')
    args = ap.parse_args()

    # 找一张图
    if args.img is None:
        jpeg_dir = Path(r'E:\Data\dataset\fivek_jpeg')
        jpgs = sorted(jpeg_dir.glob('*.jpg'))
        if not jpgs:
            print(f'No JPEGs in {jpeg_dir}')
            sys.exit(1)
        img_path = jpgs[100]  # 取中间一张避免边界
    else:
        img_path = Path(args.img)
    print(f'Test image: {img_path.name}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载图 + 渲染 target
    pil = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    orig = transform(pil).unsqueeze(0).to(device)
    gt_torch = {k: torch.tensor([v], device=device, dtype=torch.float32)
                for k, v in GT_PARAMS.items()}
    with torch.no_grad():
        target = apply_diff_isp(orig, gt_torch)
        target = torch.nan_to_num(target, nan=0.5).clamp(0, 1)

    print(f'GT params: {GT_PARAMS}')
    print(f'Grid size: {args.grid} x {args.grid} per pair')
    print(f'Radius: {args.radius * 100:.0f}% of param range')
    print(f'Output: {out_dir}\n')

    # 每对都跑
    results = []
    for p1, p2 in PAIRS:
        print(f'Sweeping ({p1}, {p2})...', end=' ', flush=True)
        p1_vals, p2_vals, loss_grid = sweep_pair(
            orig, target, p1, p2, GT_PARAMS, args.grid, device, args.radius)

        stats = ridge_stats(loss_grid)
        stats['hess_ratio'] = hessian_ratio(p1_vals, p2_vals, loss_grid)

        out_png = out_dir / f'{p1}__vs__{p2}.png'
        plot_heatmap(p1_vals, p2_vals, loss_grid, p1, p2, GT_PARAMS, out_png, stats)

        print(f"min={stats['min']:.5f}  rng={stats['dynamic_range']:.5f}  "
              f"ridge@10%={stats['ridge_pct']:5.1f}%  λratio={stats['hess_ratio']:.1f}")
        results.append(dict(pair=f'{p1}__vs__{p2}', **stats))

    # 排序 summary
    print('\n' + '='*70)
    print('非唯一性排名 (ridge_pct 越大 = 越多参数组合能产生近似相同输出)')
    print('='*70)
    results_sorted = sorted(results, key=lambda r: -r['ridge_pct'])
    for r in results_sorted:
        print(f'  {r["pair"]:40s}  ridge@10%={r["ridge_pct"]:5.1f}%  '
              f'λratio={r["hess_ratio"]:7.1f}')

    # 保存 CSV
    import csv
    csv_path = out_dir / 'summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['pair', 'min_loss', 'max_loss', 'dynamic_range',
                    'ridge_pct_at_10', 'hessian_ratio'])
        for r in results_sorted:
            w.writerow([r['pair'], f"{r['min']:.6f}", f"{r['max']:.6f}",
                        f"{r.get('dynamic_range', 0):.6f}",
                        f"{r['ridge_pct']:.2f}",
                        f"{r['hess_ratio']:.3f}"])
    print(f'\nSummary CSV -> {csv_path}')
    print(f'Heatmaps in  -> {out_dir}')


if __name__ == '__main__':
    main()
