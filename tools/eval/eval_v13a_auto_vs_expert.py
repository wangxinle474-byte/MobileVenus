"""定量评估: v13a auto-rendered vs Expert C GT (apply_diff_isp).

对 5000 张图计算 PSNR / SSIM, 输出 overall + per-action 统计.

用法:
  D:\\anaconda\\envs\\Venus\\python.exe tools/eval/eval_v13a_auto_vs_expert.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.diff_isp import apply_diff_isp  # noqa: E402
from training.firered_baseline.train_lut import PARAM_NAMES_7D  # noqa: E402

from torchvision.transforms import ToTensor


def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR between two [0,1] tensors (BCHW)."""
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * np.log10(mse)


def calc_ssim(pred: torch.Tensor, target: torch.Tensor,
              window_size: int = 11) -> float:
    """Simplified SSIM on [0,1] BCHW tensors (per-channel then average)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    # Use average pooling as window
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size,
                        device=pred.device) / (window_size ** 2)
    ssim_vals = []
    for c in range(pred.shape[1]):
        p = pred[:, c:c+1, :, :]
        t = target[:, c:c+1, :, :]
        mu_p = torch.nn.functional.conv2d(p, kernel, padding=pad)
        mu_t = torch.nn.functional.conv2d(t, kernel, padding=pad)
        mu_pp = mu_p * mu_p
        mu_tt = mu_t * mu_t
        mu_pt = mu_p * mu_t
        sigma_pp = torch.nn.functional.conv2d(p * p, kernel, padding=pad) - mu_pp
        sigma_tt = torch.nn.functional.conv2d(t * t, kernel, padding=pad) - mu_tt
        sigma_pt = torch.nn.functional.conv2d(p * t, kernel, padding=pad) - mu_pt
        num = (2 * mu_pt + C1) * (2 * sigma_pt + C2)
        den = (mu_pp + mu_tt + C1) * (sigma_pp + sigma_tt + C2)
        ssim_map = num / den
        ssim_vals.append(ssim_map.mean().item())
    return float(np.mean(ssim_vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--render_dir', default=r'E:\Data\dataset\fivek_v13a_auto')
    ap.add_argument('--jpeg_dir', default=r'E:\Data\dataset\fivek_jpeg')
    ap.add_argument('--gt_json', default='data/fivek_expert_abcde_params.json')
    ap.add_argument('--expert', default='C')
    ap.add_argument('--size', type=int, default=256,
                    help='resize for eval (v13a renders are 256px)')
    ap.add_argument('--max', type=int, default=None)
    ap.add_argument('--out', default='outputs/eval_v13a_auto_vs_expert')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    render_dir = Path(args.render_dir)
    jpeg_dir = Path(args.jpeg_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    to_tensor = ToTensor()

    # Load manifest
    manifest_path = render_dir / 'manifest.jsonl'
    records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    print(f'[data] manifest: {len(records)} records')

    # Load Expert C GT params
    with open(args.gt_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    expert_map = {}
    for s in data['samples']:
        if s['expert'] == args.expert:
            stem = s['image_name'].rsplit('.', 1)[0]
            expert_map[stem] = {n: float(s['params'].get(n, 0) or 0)
                                for n in PARAM_NAMES_7D}
    print(f'[data] Expert {args.expert} params: {len(expert_map)}')

    if args.max:
        records = records[:args.max]

    # Per-action accumulators
    action_psnrs = defaultdict(list)
    action_ssims = defaultdict(list)
    all_psnrs = []
    all_ssims = []
    skipped = 0

    t0 = time.time()
    for i, rec in enumerate(records):
        stem = rec['name']
        action = rec['action']

        orig_path = jpeg_dir / f'{stem}.jpg'
        render_path = render_dir / f'{stem}.png'

        if not orig_path.exists() or not render_path.exists():
            skipped += 1
            continue
        if stem not in expert_map:
            skipped += 1
            continue

        # Load original → tensor (256×256)
        orig_pil = Image.open(orig_path).convert('RGB').resize(
            (args.size, args.size), Image.LANCZOS)
        orig_t = to_tensor(orig_pil).unsqueeze(0).to(device)

        # v13a auto output
        v13a_pil = Image.open(render_path).convert('RGB')
        v13a_t = to_tensor(v13a_pil).unsqueeze(0).to(device)

        # Expert C GT via apply_diff_isp
        gt_params = expert_map[stem]
        gt_params_t = {n: torch.tensor([gt_params[n]], dtype=torch.float32,
                                        device=device)
                       for n in PARAM_NAMES_7D}
        with torch.no_grad():
            gt_t = apply_diff_isp(orig_t, gt_params_t).clamp(0, 1)

        # Compute metrics
        with torch.no_grad():
            p = calc_psnr(v13a_t, gt_t)
            s = calc_ssim(v13a_t, gt_t)

        all_psnrs.append(p)
        all_ssims.append(s)
        action_psnrs[action].append(p)
        action_ssims[action].append(s)

        if (i + 1) % 500 == 0 or (i + 1) == len(records):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(records) - i - 1) / rate
            cur_psnr = np.mean(all_psnrs)
            cur_ssim = np.mean(all_ssims)
            print(f'[{i+1}/{len(records)}] PSNR={cur_psnr:.2f}dB  '
                  f'SSIM={cur_ssim:.4f}  ({rate:.1f} img/s, '
                  f'ETA {eta:.0f}s)  skip={skipped}')

    # Summary
    elapsed_total = time.time() - t0
    print(f'\n{"="*60}')
    print(f'v13a auto vs Expert {args.expert} GT (apply_diff_isp)')
    print(f'{"="*60}')
    print(f'Total: {len(all_psnrs)} images  (skipped={skipped})  '
          f'time={elapsed_total:.1f}s')
    print(f'\n  Overall PSNR: {np.mean(all_psnrs):.2f} dB  '
          f'(std={np.std(all_psnrs):.2f})')
    print(f'  Overall SSIM: {np.mean(all_ssims):.4f}  '
          f'(std={np.std(all_ssims):.4f})')

    print(f'\n  Per-action breakdown:')
    print(f'  {"Action":<12} {"N":>5} {"PSNR(dB)":>10} {"SSIM":>8}')
    print(f'  {"-"*40}')
    for act in sorted(action_psnrs.keys()):
        ap_list = action_psnrs[act]
        as_list = action_ssims[act]
        print(f'  {act:<12} {len(ap_list):>5} '
              f'{np.mean(ap_list):>10.2f} {np.mean(as_list):>8.4f}')

    # Save results
    results = {
        'n_images': len(all_psnrs),
        'skipped': skipped,
        'overall_psnr': float(np.mean(all_psnrs)),
        'overall_ssim': float(np.mean(all_ssims)),
        'psnr_std': float(np.std(all_psnrs)),
        'ssim_std': float(np.std(all_ssims)),
        'per_action': {
            act: {
                'n': len(action_psnrs[act]),
                'psnr': float(np.mean(action_psnrs[act])),
                'ssim': float(np.mean(action_ssims[act])),
            }
            for act in sorted(action_psnrs.keys())
        },
        'config': vars(args),
    }
    results_path = out_dir / 'results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n[saved] {results_path}')


if __name__ == '__main__':
    main()
