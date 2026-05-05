"""
ISP 参数反推 (Inverse Fitting) — PyTorch autograd + Adam 版
给定 (原图, 目标图), 通过梯度优化反推最优 9 维 ISP 参数:
  ev_compensation, white_balance, contrast, brightness,
  shadows, highlights, saturation, vibrance, clarity

用途:
  1. 数据扩充: 编辑模型产出的好图 → 反推 ISP 参数标签
  2. 验证: 已知参数 round-trip 测试

原理:
  params* = argmin_p  L1(apply_diff_isp(orig, p), target)
                    + ssim_w * SSIM_loss(...)
  Sigmoid 参数化强制 bounds, Adam 自动梯度, 多起点重启。
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

# 参数范围 & 初始值 (精简到 7 维)
# 基于 diagnose_param_conflicts 诊断删除了 2 个冗余参数:
#   ev_compensation (与 brightness/WB 重度冗余, λratio=16M/2866)
#   vibrance       (与 saturation 重度冗余, ridge@10%=10.7%)
# diff_isp.forward() 仍通过 .get() 接受这两个 key (向后兼容, 默认 0)
PARAM_SPEC = {
    'white_balance':   {'init': 5500.0, 'lo': 2000.0,  'hi': 10000.0},
    'brightness':      {'init': 0.0,    'lo': -100.0,  'hi': 100.0},
    'contrast':        {'init': 0.0,    'lo': -100.0,  'hi': 100.0},
    'shadows':         {'init': 0.0,    'lo': -100.0,  'hi': 100.0},
    'highlights':      {'init': 0.0,    'lo': -100.0,  'hi': 100.0},
    'saturation':      {'init': 0.0,    'lo': -100.0,  'hi': 100.0},
    'clarity':         {'init': 0.0,    'lo': -100.0,  'hi': 100.0},
}


def inverse_fit(
    original: torch.Tensor,
    target: torch.Tensor,
    n_restarts: int = 3,
    maxiter: int = 200,
    lr: float = 0.05,
    l1_weight: float = 1.0,
    ssim_weight: float = 0.5,
    early_stop_tol: float = 1e-7,
    grad_clip: float = 1.0,
    raw_clip: float = 8.0,
    verbose: bool = False,
    init_params: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], float]:
    """
    PyTorch Adam 反推 ISP 参数。

    Args:
        original:      (1, 3, H, W) [0, 1] sRGB tensor
        target:        (1, 3, H, W) [0, 1] sRGB tensor
        n_restarts:    多起点重启次数 (第1次默认初始值, 其余随机)
        maxiter:       Adam 最大迭代次数
        lr:            Adam 学习率 (作用于 sigmoid raw 空间)
        l1_weight:     L1 权重
        ssim_weight:   SSIM 损失权重 (0 表示不用 SSIM)
        early_stop_tol: loss 变化小于此值时早停
        verbose:       打印迭代过程
        init_params:   可选的初始参数 (物理值 dict)

    Returns:
        (params_dict, final_loss)
    """
    device = original.device
    dtype = torch.float32
    param_names = list(PARAM_SPEC.keys())
    K = len(param_names)

    lo = torch.tensor([PARAM_SPEC[n]['lo'] for n in param_names],
                      device=device, dtype=dtype)
    hi = torch.tensor([PARAM_SPEC[n]['hi'] for n in param_names],
                      device=device, dtype=dtype)
    span = hi - lo  # 参数跨度

    # Sigmoid 参数化: x_phys = lo + sigmoid(x_raw) * span
    # 自动强制 bounds, 梯度在 raw 空间均匀化
    def _to_phys(x_raw: torch.Tensor) -> torch.Tensor:
        return lo + torch.sigmoid(x_raw) * span

    def _to_raw(x_phys: torch.Tensor) -> torch.Tensor:
        x_norm = ((x_phys - lo) / span).clamp(1e-6, 1.0 - 1e-6)
        return torch.log(x_norm / (1.0 - x_norm))

    def _render(x_raw: torch.Tensor) -> torch.Tensor:
        x_phys = _to_phys(x_raw)
        # 构造 params dict: name → (1,) tensor 且可回传梯度
        params = {param_names[i]: x_phys[i:i+1] for i in range(K)}
        return apply_diff_isp(original, params)

    def _loss_fn(x_raw: torch.Tensor) -> torch.Tensor:
        rendered = _render(x_raw)
        loss = l1_weight * F.l1_loss(rendered, target)
        if ssim_weight > 0:
            loss = loss + ssim_weight * ssim_loss(rendered, target)
        return loss

    # 默认初始点
    x0_phys = torch.tensor(
        [PARAM_SPEC[n]['init'] for n in param_names],
        device=device, dtype=dtype,
    )
    if init_params:
        for i, n in enumerate(param_names):
            if n in init_params:
                x0_phys[i] = float(init_params[n])
    x0_raw_default = _to_raw(x0_phys).detach()

    best_loss = float('inf')
    best_phys = x0_phys.clone()
    gen = torch.Generator(device='cpu').manual_seed(42)

    for restart in range(n_restarts):
        if restart == 0:
            x_raw = x0_raw_default.clone().detach().requires_grad_(True)
        else:
            # 随机初始 (正态 σ=1.5 覆盖中间 80% 范围)
            x_raw = (torch.randn(K, generator=gen) * 1.5).to(
                device=device, dtype=dtype).requires_grad_(True)

        # LBFGS + 线搜索 对此类光滑连续目标远优于 Adam
        optimizer = torch.optim.LBFGS(
            [x_raw],
            lr=1.0,
            max_iter=20,
            history_size=10,
            line_search_fn='strong_wolfe',
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        # 保存"最好"的 x_phys (在此 restart 内) 防止末尾 NaN 污染
        best_in_restart_phys = None
        best_in_restart_loss = float('inf')
        prev_loss = float('inf')

        def _closure():
            """LBFGS closure: 前向 + 反向, 返回 loss."""
            optimizer.zero_grad()
            loss = _loss_fn(x_raw)
            if torch.isnan(loss) or torch.isinf(loss):
                # NaN -> 返回一个大数, 让 LBFGS 回退
                return torch.tensor(1e6, device=device, dtype=dtype, requires_grad=True)
            loss.backward()
            # 清理 NaN 梯度 (通过 pow/log 的不稳定路径)
            if x_raw.grad is not None and torch.isnan(x_raw.grad).any():
                x_raw.grad = torch.nan_to_num(
                    x_raw.grad, nan=0.0, posinf=0.0, neginf=0.0)
            # 可选梯度裁剪
            if grad_clip > 0 and x_raw.grad is not None:
                torch.nn.utils.clip_grad_norm_([x_raw], max_norm=grad_clip)
            return loss

        # 外层循环: 每次 optimizer.step(closure) 内部跑 max_iter=20 次 LBFGS
        n_outer = max(1, maxiter // 20)
        for outer in range(n_outer):
            try:
                optimizer.step(_closure)
            except Exception as e:
                if verbose:
                    logger.warning(f'  restart {restart} outer {outer} LBFGS err: {e}')
                break

            # 记录当前
            with torch.no_grad():
                # 限制 raw 范围防止 sigmoid 饱和
                x_raw.data.clamp_(-raw_clip, raw_clip)
                cur_loss = _loss_fn(x_raw).item()

            if cur_loss == cur_loss and cur_loss != float('inf'):  # not NaN
                if cur_loss < best_in_restart_loss:
                    best_in_restart_loss = cur_loss
                    with torch.no_grad():
                        best_in_restart_phys = _to_phys(x_raw).detach().clone()

            if verbose:
                logger.info(f'  restart {restart} outer {outer:2d}  loss={cur_loss:.6f}')

            # 早停
            if abs(prev_loss - cur_loss) < early_stop_tol:
                if verbose:
                    logger.info(f'  restart {restart} early-stop @ outer {outer}')
                break
            prev_loss = cur_loss

        # 用 "restart 内最优" 作为此 restart 的最终结果
        if best_in_restart_phys is None:
            # 完全失败 (第 0 步就 NaN) - 跳过此 restart
            if verbose:
                logger.warning(f'  restart {restart}  failed (no valid step)')
            continue

        x_phys_final = best_in_restart_phys
        final_loss = best_in_restart_loss

        if verbose:
            param_str = ', '.join(f'{param_names[i]}={x_phys_final[i].item():.2f}'
                                  for i in range(K))
            logger.info(f'  restart {restart}  final_loss={final_loss:.6f}  [{param_str}]')

        if final_loss < best_loss:
            best_loss = final_loss
            best_phys = x_phys_final.clone()

    best_params = {param_names[i]: float(best_phys[i].item()) for i in range(K)}
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
