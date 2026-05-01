"""主评估脚本: PSNR / SSIM / MS-SSIM。

对比各版本模型在 FiveK 测试集上的图像还原质量。

用法:
    python tools/eval/eval_psnr_ssim.py \
        --model all \
        --jpeg_dir E:/dataset/fivek_jpeg \
        --params_json data/fivek_expert_params.json \
        --num_images 500 \
        --output outputs/eval/eval_psnr_ssim.txt
"""

import argparse
import os
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
from training.fivek_8param.dataset import FiveKRawDataset
from models.isp_pipeline import render_params


def compute_psnr_ssim(pred_img, gt_img):
    """计算 PSNR 和 SSIM。

    Args:
        pred_img: (3, H, W) tensor [0, 1]
        gt_img: (3, H, W) tensor [0, 1]
    Returns:
        psnr, ssim
    """
    pred = pred_img.cpu().numpy().transpose(1, 2, 0)
    gt = gt_img.cpu().numpy().transpose(1, 2, 0)

    pred = np.clip(pred, 0.0, 1.0)
    gt = np.clip(gt, 0.0, 1.0)

    psnr = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim = structural_similarity(gt, pred, channel_axis=2, data_range=1.0)

    return psnr, ssim


def evaluate_model(model, dataset, device='cuda', num_images=500):
    """评估模型的 PSNR/SSIM。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    psnr_list = []
    ssim_list = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=min(num_images, len(dataset)))):
            if i >= num_images:
                break

            images = batch['image'].to(device)
            raw_images = batch['raw_image'].to(device)
            target_params = batch['params']

            # 预测参数
            out = model(images)

            # 渲染
            pred_rendered = render_params(raw_images, out['raw_params'])

            # GT 渲染
            gt_params = {}
            for j, name in enumerate(PARAM_NAMES):
                lo, hi = PARAM_RANGES[name]
                val = (target_params[0, j].item() + 1.0) / 2.0 * (hi - lo) + lo
                gt_params[name] = torch.tensor([[val]], device=device)
            gt_rendered = render_params(raw_images, gt_params)

            psnr, ssim = compute_psnr_ssim(
                pred_rendered[0], gt_rendered[0]
            )
            psnr_list.append(psnr)
            ssim_list.append(ssim)

    return {
        'psnr_mean': float(np.mean(psnr_list)),
        'psnr_std': float(np.std(psnr_list)),
        'ssim_mean': float(np.mean(ssim_list)),
        'ssim_std': float(np.std(ssim_list)),
        'num_images': len(psnr_list),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='all')
    parser.add_argument('--jpeg_dir', type=str, required=True)
    parser.add_argument('--params_json', type=str, required=True)
    parser.add_argument('--num_images', type=int, default=500)
    parser.add_argument('--output', type=str, default='outputs/eval/eval_psnr_ssim.txt')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dataset = FiveKRawDataset(
        jpeg_dir=args.jpeg_dir,
        params_json=args.params_json,
        split='test',
        image_size=224,
        augment=False,
    )

    print(f"Evaluating on {min(args.num_images, len(dataset))} images...")

    # 查找所有 checkpoints
    checkpoint_dir = 'checkpoints'
    checkpoints = {}
    if os.path.isdir(checkpoint_dir):
        for name in sorted(os.listdir(checkpoint_dir)):
            best = os.path.join(checkpoint_dir, name, 'best.pt')
            stage_b = os.path.join(checkpoint_dir, name, 'stage_b', 'best.pt')
            if os.path.exists(stage_b):
                checkpoints[name] = stage_b
            elif os.path.exists(best):
                checkpoints[name] = best

    results = {}
    for name, ckpt in checkpoints.items():
        if args.model != 'all' and args.model not in name:
            continue
        print(f"\n--- {name} ({ckpt}) ---")
        try:
            from training.semantic_distill.model import DistillParamModel
            model = DistillParamModel()
            state = torch.load(ckpt, map_location=args.device)
            model.load_state_dict(state['model_state_dict'], strict=False)
            model = model.to(args.device)

            r = evaluate_model(model, dataset, args.device, args.num_images)
            results[name] = r
            print(f"  PSNR: {r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}")
            print(f"  SSIM: {r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}")
        except Exception as e:
            print(f"  Error: {e}")

    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("PSNR/SSIM Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        for name, r in results.items():
            f.write(f"{name}:\n")
            f.write(f"  PSNR: {r['psnr_mean']:.2f} ± {r['psnr_std']:.2f}\n")
            f.write(f"  SSIM: {r['ssim_mean']:.4f} ± {r['ssim_std']:.4f}\n")
            f.write(f"  N: {r['num_images']}\n\n")
    print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
