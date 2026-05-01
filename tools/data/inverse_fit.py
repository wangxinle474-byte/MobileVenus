"""
ISP 参数反推 (Inverse Fitting)
给定 (原图, 目标图)，通过梯度优化反推最优 6 维 ISP 参数。

用途:
  1. 数据扩充: 编辑模型产出的好图 → 反推 ISP 参数标签
  2. 验证: 已知参数 round-trip 测试

原理:
  params* = argmin_{p} || apply_diff_isp(original, p) - target ||
  使用 L1 + SSIM 联合损失，Adam 优化器。
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.diff_isp import apply_diff_isp, ssim_loss

logger = logging.getLogger(__name__)

# 参数范围 & 初始值
PARAM_SPEC = {
    'ev_compensation': {'init': 0.0,    'lo': -3.0,    'hi': 3.0,     'lr_scale': 1.0},
    'white_balance':   {'init': 5500.0, 'lo': 2000.0,  'hi': 10000.0, 'lr_scale': 50.0},
    'contrast':        {'init': 0.0,    'lo': -100.0,  'hi': 100.0,   'lr_scale': 5.0},
    'shadows':         {'init': 0.0,    'lo': -100.0,  'hi': 100.0,   'lr_scale': 5.0},
    'highlights':      {'init': 0.0,    'lo': -100.0,  'hi': 100.0,   'lr_scale': 5.0},
    'saturation':      {'init': 0.0,    'lo': -100.0,  'hi': 100.0,   'lr_scale': 5.0},
}


def inverse_fit(
    original: torch.Tensor,
    target: torch.Tensor,
    n_restarts: int = 3,
    maxiter: int = 200,
    verbose: bool = False,
    init_params: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], float]:
    """
    给定 (原图, 目标图)，通过 scipy L-BFGS-B 反推最优 ISP 参数。

    Args:
        original:   (1, 3, H, W) [0, 1] sRGB tensor
        target:     (1, 3, H, W) [0, 1] sRGB tensor
        n_restarts: 多起点重启次数 (第1次用默认初始值，其余随机)
        maxiter:    L-BFGS-B 最大迭代次数
        verbose:    是否打印优化过程
        init_params: 可选的初始参数字典 (物理值)

    Returns:
        (params_dict, final_loss)
        params_dict: {param_name: float} 物理范围值
    """
    from scipy.optimize import minimize
    import numpy as np

    device = original.device
    param_names = list(PARAM_SPEC.keys())
    lo = np.array([PARAM_SPEC[n]['lo'] for n in param_names])
    hi = np.array([PARAM_SPEC[n]['hi'] for n in param_names])
    span = hi - lo  # 参数跨度

    def _to_physical(x_norm: np.ndarray) -> np.ndarray:
        """[0, 1] → 物理范围"""
        return lo + x_norm * span

    def _to_norm(x_phys: np.ndarray) -> np.ndarray:
        """物理范围 → [0, 1]"""
        return (x_phys - lo) / span

    def _eval(x_norm: np.ndarray) -> float:
        """归一化参数 → 前向渲染 → L1 loss"""
        x_phys = _to_physical(x_norm)
        params = {}
        for i, name in enumerate(param_names):
            params[name] = torch.tensor([float(x_phys[i])], device=device, dtype=torch.float32)
        with torch.no_grad():
            rendered = apply_diff_isp(original, params)
            loss = F.l1_loss(rendered, target).item()
        return loss if not (np.isnan(loss) or np.isinf(loss)) else 1e6

    # 默认初始点 (归一化)
    x0_phys = np.array([PARAM_SPEC[n]['init'] for n in param_names])
    if init_params:
        for i, n in enumerate(param_names):
            if n in init_params:
                x0_phys[i] = init_params[n]
    x0_default = _to_norm(x0_phys)

    # 归一化空间的 bounds = [0, 1]
    norm_bounds = [(0.0, 1.0)] * len(param_names)

    best_loss = float('inf')
    best_x_norm = x0_default.copy()
    rng = np.random.RandomState(42)

    for restart in range(n_restarts):
        if restart == 0:
            x0 = x0_default.copy()
        else:
            x0 = rng.uniform(0.0, 1.0, size=len(param_names))

        result = minimize(
            _eval, x0, method='L-BFGS-B', bounds=norm_bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'eps': 1e-4},
        )

        x_phys = _to_physical(result.x)
        if verbose:
            param_str = ', '.join(f'{param_names[i]}={x_phys[i]:.2f}' for i in range(len(param_names)))
            logger.info(f'  restart {restart}  loss={result.fun:.6f}  nfev={result.nfev}  [{param_str}]')

        if result.fun < best_loss:
            best_loss = result.fun
            best_x_norm = result.x.copy()

    best_phys = _to_physical(best_x_norm)
    best_params = {param_names[i]: float(best_phys[i]) for i in range(len(param_names))}
    return best_params, best_loss


def batch_inverse_fit(
    originals: torch.Tensor,
    targets: torch.Tensor,
    **kwargs,
) -> list:
    """
    逐张反推 ISP 参数。
    originals, targets: (N, 3, H, W)
    Returns: list of (params_dict, loss)
    """
    results = []
    N = originals.shape[0]
    for i in range(N):
        params, loss = inverse_fit(
            originals[i:i+1], targets[i:i+1], **kwargs
        )
        results.append((params, loss))
    return results


# ─────────────────── 验证脚本 ───────────────────
if __name__ == '__main__':
    import argparse
    import json
    import numpy as np
    from PIL import Image
    from torchvision import transforms

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    ap = argparse.ArgumentParser(description='Inverse Fit 验证')
    ap.add_argument('--jpeg_dir', default=r'E:\dataset\fivek_jpeg',
                    help='FiveK JPEG 目录')
    ap.add_argument('--data_file', default='data/fivek_expert_params.json',
                    help='专家参数 JSON')
    ap.add_argument('--num_test', type=int, default=5,
                    help='测试图片数量')
    ap.add_argument('--steps', type=int, default=500,
                    help='优化步数')
    ap.add_argument('--image_size', type=int, default=256,
                    help='图像缩放尺寸')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # 加载 GT 参数
    with open(args.data_file, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    samples = raw['samples'] if isinstance(raw, dict) and 'samples' in raw else raw

    # 按图片聚合专家均值
    from collections import defaultdict
    img_params = defaultdict(lambda: defaultdict(list))
    for s in samples:
        name = s['image_name'].replace('.dng', '')
        for p in PARAM_SPEC:
            img_params[name][p].append(float(s.get(p, 0)))

    # 找到有 JPEG 的图
    jpeg_dir = Path(args.jpeg_dir)
    valid = []
    for name, pdict in img_params.items():
        jpg = jpeg_dir / f'{name}.jpg'
        if jpg.exists():
            mean_params = {p: float(np.mean(vs)) for p, vs in pdict.items()}
            valid.append((name, str(jpg), mean_params))
    logger.info(f'可用图片: {len(valid)} 张')

    # 随机选 N 张
    rng = np.random.RandomState(42)
    rng.shuffle(valid)
    test_set = valid[:args.num_test]

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    logger.info(f'\n{"="*70}')
    logger.info(f'Round-trip 验证: 已知参数 → 渲染目标 → 反推参数 → 比较')
    logger.info(f'{"="*70}')

    all_mae = {p: [] for p in PARAM_SPEC}
    all_losses = []

    for idx, (name, jpg_path, gt_params) in enumerate(test_set):
        logger.info(f'\n[{idx+1}/{len(test_set)}] {name}')

        # 加载原图
        pil = Image.open(jpg_path).convert('RGB')
        original = transform(pil).unsqueeze(0).to(device)  # (1,3,H,W)

        # 用 GT 参数渲染目标
        gt_torch = {p: torch.tensor([v], device=device, dtype=torch.float32)
                    for p, v in gt_params.items()}
        with torch.no_grad():
            target = apply_diff_isp(original, gt_torch)

        logger.info(f'  GT params: {", ".join(f"{k}={v:.2f}" for k, v in gt_params.items())}')

        # 反推
        recovered, final_loss = inverse_fit(
            original, target,
            n_restarts=3, maxiter=300,
            verbose=True,
        )

        logger.info(f'  Recovered: {", ".join(f"{k}={v:.2f}" for k, v in recovered.items())}')

        # 计算 MAE
        logger.info(f'  Parameter MAE:')
        for p in PARAM_SPEC:
            mae = abs(recovered[p] - gt_params[p])
            all_mae[p].append(mae)
            logger.info(f'    {p:20s}  GT={gt_params[p]:8.2f}  Rec={recovered[p]:8.2f}  MAE={mae:.4f}')

        all_losses.append(final_loss)

        # 渲染对比: PSNR
        rec_torch = {p: torch.tensor([v], device=device, dtype=torch.float32)
                     for p, v in recovered.items()}
        with torch.no_grad():
            rec_rendered = apply_diff_isp(original, rec_torch)
        mse = F.mse_loss(rec_rendered, target).item()
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        logger.info(f'  Rendered PSNR: {psnr:.2f} dB  (loss={final_loss:.6f})')

    # 汇总
    logger.info(f'\n{"="*70}')
    logger.info(f'汇总 ({len(test_set)} 张):')
    logger.info(f'{"="*70}')
    for p in PARAM_SPEC:
        avg = np.mean(all_mae[p])
        spec = PARAM_SPEC[p]
        range_pct = avg / (spec['hi'] - spec['lo']) * 100
        logger.info(f'  {p:20s}  MAE={avg:.4f}  (范围占比 {range_pct:.2f}%)')
    logger.info(f'  Mean render loss: {np.mean(all_losses):.6f}')
    logger.info(f'  Mean render PSNR: {10 * np.log10(1.0 / (np.mean(all_losses) + 1e-10)):.2f} dB (approx)')
    logger.info(f'\n[DONE] Inverse fit 验证完成')
