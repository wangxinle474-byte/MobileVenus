"""批量 inverse_fit on (FiveK orig, FireRed teacher edit) pairs.

输入:
  --captions data/teacher_edits_fivek_top20_instr.json  (有 idx + source_image + orig_path)
  --target_dir outputs/teacher_edits/fivek_top20_instr/   (含 <idx:04d>.png teacher edits)

输出:
  <out_dir>/pseudo_labels.jsonl   每行一组 (orig, target, P_inferred, pixel_l1, ...)
  <out_dir>/summary.json          L1 分布 + PASS/WARN/FAIL 统计
  <out_dir>/comparison/<idx:04d>.png  4-panel 对比图 (orig | target | rendered_back | diff_x5)

用法:
  python tools/data/data_prep/inverse_fit_batch.py \\
      --captions data/teacher_edits_fivek_top20_instr.json \\
      --target_dir outputs/teacher_edits/fivek_top20_instr \\
      --out_dir outputs/inverse_fit_pilot/fivek_top20
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

from models.diff_isp import apply_diff_isp, ssim_loss  # noqa: E402
from tools.data.data_prep.inverse_fit import inverse_fit, PARAM_SPEC  # noqa: E402


def inverse_fit_adam(
    original: torch.Tensor,
    target: torch.Tensor,
    n_restarts: int = 3,
    n_iters: int = 300,
    lr: float = 0.05,
    l1_weight: float = 1.0,
    ssim_weight: float = 0.5,
    early_stop_tol: float = 1e-7,
    grad_clip: float = 1.0,
    raw_clip: float = 8.0,
    init_params: dict = None,
    verbose: bool = False,
):
    """Adam 求解器替代 inverse_fit 的 LBFGS, 对非光滑 loss (FireRed 输出) 更鲁棒.

    与 inverse_fit 的 LBFGS 版本逻辑一致 (sigmoid 参数化 + 多起点重启 + 早停),
    但用 Adam 替代 LBFGS+strong_wolfe, 避免 line search 在不平滑梯度下提前 fail.
    """
    device = original.device
    dtype = torch.float32
    param_names = list(PARAM_SPEC.keys())
    K = len(param_names)

    lo = torch.tensor([PARAM_SPEC[n]['lo'] for n in param_names], device=device, dtype=dtype)
    hi = torch.tensor([PARAM_SPEC[n]['hi'] for n in param_names], device=device, dtype=dtype)
    span = hi - lo

    def _to_phys(x_raw):
        return lo + torch.sigmoid(x_raw) * span

    def _to_raw(x_phys):
        x_norm = ((x_phys - lo) / span).clamp(1e-6, 1.0 - 1e-6)
        return torch.log(x_norm / (1.0 - x_norm))

    def _loss_fn(x_raw):
        x_phys = _to_phys(x_raw)
        params = {param_names[i]: x_phys[i:i+1] for i in range(K)}
        rendered = apply_diff_isp(original, params)
        loss = l1_weight * F.l1_loss(rendered, target)
        if ssim_weight > 0:
            loss = loss + ssim_weight * ssim_loss(rendered, target)
        return loss

    # 默认初始点
    x0_phys = torch.tensor([PARAM_SPEC[n]['init'] for n in param_names],
                           device=device, dtype=dtype)
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
            x_raw = (torch.randn(K, generator=gen) * 1.5).to(
                device=device, dtype=dtype).requires_grad_(True)

        optimizer = torch.optim.Adam([x_raw], lr=lr)
        best_in_restart_phys = None
        best_in_restart_loss = float('inf')
        prev_loss = float('inf')

        for it in range(n_iters):
            optimizer.zero_grad()
            loss = _loss_fn(x_raw)
            if torch.isnan(loss) or torch.isinf(loss):
                if verbose:
                    print(f'  restart {restart} iter {it} NaN loss, break')
                break
            loss.backward()
            if x_raw.grad is not None and torch.isnan(x_raw.grad).any():
                x_raw.grad = torch.nan_to_num(x_raw.grad, nan=0.0, posinf=0.0, neginf=0.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([x_raw], max_norm=grad_clip)
            optimizer.step()

            with torch.no_grad():
                x_raw.data.clamp_(-raw_clip, raw_clip)
                cur_loss = float(_loss_fn(x_raw).item())

            if cur_loss == cur_loss and cur_loss != float('inf'):
                if cur_loss < best_in_restart_loss:
                    best_in_restart_loss = cur_loss
                    with torch.no_grad():
                        best_in_restart_phys = _to_phys(x_raw).detach().clone()

            if verbose and it % 50 == 0:
                print(f'  restart {restart} iter {it:3d}  loss={cur_loss:.6f}')

            if abs(prev_loss - cur_loss) < early_stop_tol and it > 10:
                if verbose:
                    print(f'  restart {restart} early-stop @ iter {it}')
                break
            prev_loss = cur_loss

        if best_in_restart_phys is None:
            continue
        if best_in_restart_loss < best_loss:
            best_loss = best_in_restart_loss
            best_phys = best_in_restart_phys.clone()

    best_params = {param_names[i]: float(best_phys[i].item()) for i in range(K)}
    return best_params, best_loss


def load_image(path: Path, max_size: int = 0) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if max_size and max(t.shape[-2:]) > max_size:
        scale = max_size / max(t.shape[-2:])
        h = int(t.shape[-2] * scale); w = int(t.shape[-1] * scale)
        t = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)
    return t


def estimate_init_params(orig: torch.Tensor, target: torch.Tensor) -> dict:
    """从 (orig, target) 像素统计估计 inverse_fit 初值, 避免 LBFGS 卡在 default init.

    启发式:
      brightness    ← (mean(target) - mean(orig)) * 200    # 量程 [-100, 100]
      contrast      ← (std(target)  - std(orig))  * 500    # 量程 [-100, 100]
      saturation    ← (sat(target)  - sat(orig))  * 200
      white_balance ← 5500 + 4000 * log(R/B_ratio_target / R/B_ratio_orig)
      shadows/highlights/clarity ← 0 (让 LBFGS 微调)

    Returns:
      dict 兼容 inverse_fit 的 init_params 入参.
    """
    with torch.no_grad():
        # mean / std per channel
        mean_o = orig.mean().item()
        mean_t = target.mean().item()
        std_o = orig.std().item()
        std_t = target.std().item()

        # saturation 估计 = mean(max(rgb) - min(rgb))
        max_o = orig.max(dim=1).values; min_o = orig.min(dim=1).values
        max_t = target.max(dim=1).values; min_t = target.min(dim=1).values
        sat_o = (max_o - min_o).mean().item()
        sat_t = (max_t - min_t).mean().item()

        # R/B ratio (color temperature proxy)
        # 越大表示 R 比 B 强 → 暖色 → 较高色温值（注: Lightroom 色温越高图越暖）
        eps = 1e-6
        rb_o = (orig[:, 0].mean() / (orig[:, 2].mean() + eps)).item()
        rb_t = (target[:, 0].mean() / (target[:, 2].mean() + eps)).item()

    init = {
        'brightness':    float(np.clip((mean_t - mean_o) * 200, -80, 80)),
        'contrast':      float(np.clip((std_t - std_o) * 500, -80, 80)),
        'saturation':    float(np.clip((sat_t - sat_o) * 200, -80, 80)),
        'white_balance': float(np.clip(
            5500.0 + 4000.0 * np.log(max(rb_t / max(rb_o, eps), eps)),
            2500, 9500)),
        'shadows':    0.0,
        'highlights': 0.0,
        'clarity':    0.0,
    }
    return init


def save_image(t: torch.Tensor, path: Path):
    arr = (t.squeeze(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    arr = arr.transpose(1, 2, 0)
    Image.fromarray(arr).save(path)


def make_4panel(orig, target, rendered_back, save_path: Path):
    """4-panel 横向拼接: orig | target | rendered_back | diff_x5."""
    diff = ((target - rendered_back).abs() * 5).clamp(0, 1)
    row = torch.cat([orig, target, rendered_back, diff], dim=-1)  # (1,3,H,4W)
    save_image(row, save_path)


def fit_one(orig_path: Path, target_path: Path, device: str, max_size: int,
            n_restarts: int, maxiter: int, ssim_weight: float,
            use_heuristic_init: bool = True,
            optim_type: str = 'adam'):
    """跑 inverse_fit 一对 (orig, target), 返回 dict.

    optim_type: 'adam' (默认, 对 FireRed 输出鲁棒) 或 'lbfgs' (原版).
    """
    orig = load_image(orig_path, max_size).to(device)
    H, W = orig.shape[-2:]
    target_full = load_image(target_path, 0).to(device)
    if target_full.shape[-2:] != (H, W):
        target = F.interpolate(target_full, size=(H, W),
                               mode='bilinear', align_corners=False)
    else:
        target = target_full
    target = target.clamp(0, 1)

    delta = (target - orig).abs().mean().item()
    init_params = estimate_init_params(orig, target) if use_heuristic_init else None
    t0 = time.time()
    if optim_type == 'adam':
        P, final_loss = inverse_fit_adam(
            orig, target,
            n_restarts=n_restarts, n_iters=maxiter, ssim_weight=ssim_weight,
            verbose=False, init_params=init_params,
        )
    else:
        P, final_loss = inverse_fit(
            orig, target,
            n_restarts=n_restarts, maxiter=maxiter, ssim_weight=ssim_weight,
            verbose=False, init_params=init_params,
        )
    dt = time.time() - t0

    P_t = {k: torch.tensor([float(v)], device=device, dtype=torch.float32)
           for k, v in P.items()}
    with torch.no_grad():
        rendered_back = apply_diff_isp(orig, P_t).clamp(0, 1)
    pixel_l1 = F.l1_loss(rendered_back, target).item()
    pixel_l2 = F.mse_loss(rendered_back, target).item() ** 0.5

    return {
        'orig': orig, 'target': target, 'rendered_back': rendered_back,
        'P_inferred': {k: float(v) for k, v in P.items()},
        'P_init_heuristic': init_params,
        'pixel_l1': pixel_l1, 'pixel_l2': pixel_l2,
        'final_loss': final_loss, 'delta': delta, 'runtime_sec': dt,
        'fit_size': [H, W],
    }


def verdict(l1: float) -> str:
    if l1 < 0.01: return 'PASS_GREAT'
    if l1 < 0.02: return 'PASS'
    if l1 < 0.05: return 'WARN'
    return 'FAIL'


def percentile(arr: list, p: float) -> float:
    if not arr: return 0.0
    return float(np.percentile(arr, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', required=True,
                    help='caption JSON (有 idx + source_image + orig_path)')
    ap.add_argument('--target_dir', required=True,
                    help='teacher edits dir (含 <idx:04d>.png)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_size', type=int, default=512)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--n_restarts', type=int, default=3)
    ap.add_argument('--maxiter', type=int, default=200)
    ap.add_argument('--ssim_weight', type=float, default=0.5)
    ap.add_argument('--save_panels', action='store_true', default=True,
                    help='保存 4-panel 对比图 (默认 on)')
    ap.add_argument('--no_heuristic_init', action='store_true',
                    help='禁用 (orig, target) 像素统计估算的智能初值 (默认 enabled)')
    ap.add_argument('--optim', choices=['adam', 'lbfgs'], default='adam',
                    help='优化器: adam (默认, 鲁棒) / lbfgs (原版, 速度快但脆弱)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = out_dir / 'comparison'
    if args.save_panels:
        panel_dir.mkdir(exist_ok=True)

    with open(args.captions, encoding='utf-8') as f:
        cap = json.load(f)
    samples = cap['samples']
    print(f'[INFO] {len(samples)} samples to fit, device={args.device}, '
          f'max_size={args.max_size}')

    target_dir = Path(args.target_dir)
    jsonl_path = out_dir / 'pseudo_labels.jsonl'
    records = []

    t_start = time.time()
    with open(jsonl_path, 'w', encoding='utf-8') as fjsonl:
        for i, s in enumerate(samples, start=1):
            idx = s['idx']
            orig_path = Path(s['orig_path'])
            target_path = target_dir / f'{idx:04d}.png'

            if not orig_path.exists():
                print(f'[SKIP {i}/{len(samples)}] idx={idx} orig 不存在: {orig_path}')
                continue
            if not target_path.exists():
                print(f'[SKIP {i}/{len(samples)}] idx={idx} target 不存在: {target_path}')
                continue

            try:
                r = fit_one(orig_path, target_path, args.device, args.max_size,
                            args.n_restarts, args.maxiter, args.ssim_weight,
                            use_heuristic_init=not args.no_heuristic_init,
                            optim_type=args.optim)
            except Exception as e:
                print(f'[ERR  {i}/{len(samples)}] idx={idx}: {e}')
                continue

            v = verdict(r['pixel_l1'])
            print(f'[{i:>2}/{len(samples)}] idx={idx:<5} L1={r["pixel_l1"]:.4f} '
                  f'L2={r["pixel_l2"]:.4f} dt={r["runtime_sec"]:.1f}s  {v}')

            rec = {
                'rank': s.get('rank'),
                'idx': idx,
                'source_image': s['source_image'],
                'orig_path': str(orig_path).replace('\\', '/'),
                'target_path': str(target_path).replace('\\', '/'),
                'caption': s['new_caption'],
                'tone_target': s.get('tone_target'),
                'P_inferred': r['P_inferred'],
                'P_init_heuristic': r.get('P_init_heuristic'),
                'pixel_l1': r['pixel_l1'],
                'pixel_l2': r['pixel_l2'],
                'final_loss': r['final_loss'],
                'delta_target_orig': r['delta'],
                'fit_size': r['fit_size'],
                'runtime_sec': r['runtime_sec'],
                'verdict': v,
            }
            fjsonl.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fjsonl.flush()
            records.append(rec)

            if args.save_panels:
                make_4panel(r['orig'], r['target'], r['rendered_back'],
                            panel_dir / f'{idx:04d}.png')

    total = time.time() - t_start
    print(f'\n[DONE] {len(records)}/{len(samples)} fitted in {total:.1f}s')

    # 统计
    l1s = [r['pixel_l1'] for r in records]
    verdicts = [r['verdict'] for r in records]
    summary = {
        'n_total': len(samples),
        'n_fitted': len(records),
        'total_runtime_sec': total,
        'pixel_l1': {
            'min': min(l1s) if l1s else None,
            'max': max(l1s) if l1s else None,
            'mean': float(np.mean(l1s)) if l1s else None,
            'median': float(np.median(l1s)) if l1s else None,
            'p25': percentile(l1s, 25),
            'p75': percentile(l1s, 75),
        },
        'verdict_counts': {v: verdicts.count(v) for v in set(verdicts)},
        'params_stats': {},
    }
    # 各参数维统计
    if records:
        param_names = list(records[0]['P_inferred'].keys())
        for p in param_names:
            vals = [r['P_inferred'][p] for r in records]
            summary['params_stats'][p] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
            }

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('\n[SUMMARY]')
    print(f'  pixel_l1: min={summary["pixel_l1"]["min"]:.4f}  '
          f'median={summary["pixel_l1"]["median"]:.4f}  '
          f'max={summary["pixel_l1"]["max"]:.4f}')
    print(f'  verdicts: {summary["verdict_counts"]}')
    print(f'\n[SAVED] {jsonl_path}')
    print(f'[SAVED] {out_dir / "summary.json"}')
    if args.save_panels:
        print(f'[SAVED] {panel_dir}/ (4-panel comparisons)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
