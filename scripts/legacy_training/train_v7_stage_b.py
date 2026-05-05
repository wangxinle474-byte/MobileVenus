"""v7 Stage B: 退化增强 + 对比学习。

在 v6 Stage A 基础上，加入退化增强和一致性损失。
- 对输入图片随机施加 EV/WB/对比度偏移 (退化)
- 原图和退化图的语义 embedding 应保持一致 (consistency_loss)
- 对比学习增强语义区分性

用法 (AutoDL):
    python scripts/train_v7_stage_b.py \
        --stage_a_ckpt /root/autodl-tmp/checkpoints/distill_v6/stage_a/best.pt \
        --jpeg_dir /root/autodl-tmp/fivek_jpeg \
        --params_json /root/autodl-tmp/data/fivek_expert_params.json \
        --output_dir /root/autodl-tmp/checkpoints/distill_v7/stage_b
"""

import argparse
import os
import sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.semantic_distill.model import DistillParamModel
from training.fivek_8param.dataset import FiveKDataset
from training.semantic_distill.config import DistillConfig


def random_degrade(images, prob=0.5):
    """随机退化图片 (EV/WB/对比度偏移)。"""
    if random.random() > prob:
        return images, False

    B = images.shape[0]
    degraded = images.clone()

    # 随机亮度偏移
    ev_shift = (torch.rand(B, 1, 1, 1, device=images.device) - 0.5) * 0.6
    degraded = degraded + ev_shift

    # 随机色彩偏移 (模拟 WB 偏移)
    color_shift = (torch.rand(B, 3, 1, 1, device=images.device) - 0.5) * 0.2
    degraded = degraded + color_shift

    # 随机对比度
    contrast = 0.7 + torch.rand(B, 1, 1, 1, device=images.device) * 0.6
    mean = degraded.mean(dim=[2, 3], keepdim=True)
    degraded = mean + (degraded - mean) * contrast

    return degraded.clamp(-3, 3), True  # clamp to reasonable range


def main():
    parser = argparse.ArgumentParser(description='v7 Stage B: Degradation Augmentation')
    parser.add_argument('--stage_a_ckpt', type=str, required=True)
    parser.add_argument('--jpeg_dir', type=str, required=True)
    parser.add_argument('--params_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/distill_v7/stage_b')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--consistency_weight', type=float, default=0.1)
    parser.add_argument('--degrade_prob', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 模型
    model = DistillParamModel.from_stage_a(
        args.stage_a_ckpt, device=args.device,
    )
    model = model.to(args.device)

    # 数据
    train_ds = FiveKDataset(
        jpeg_dir=args.jpeg_dir, params_json=args.params_json,
        split='train', augment=True,
    )
    val_ds = FiveKDataset(
        jpeg_dir=args.jpeg_dir, params_json=args.params_json,
        split='val', augment=False,
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    param_loss_fn = nn.L1Loss()
    best_val = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_param = 0
        total_cons = 0
        n = 0
        t0 = time.time()

        for batch in train_loader:
            images = batch['image'].to(args.device)
            target = batch['params'].to(args.device)

            # 原图前向
            out = model(images, return_embedding=True)
            param_loss = param_loss_fn(out['norm_params'], target)

            loss = param_loss

            # 退化增强 + 一致性
            degraded, did_degrade = random_degrade(images, args.degrade_prob)
            if did_degrade:
                out_deg = model(degraded, return_embedding=True)
                cons_loss = F.mse_loss(out['semantic_emb'].detach(),
                                       out_deg['semantic_emb'])
                loss = loss + args.consistency_weight * cons_loss
                total_cons += cons_loss.item() * images.shape[0]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            total_param += param_loss.item() * images.shape[0]
            n += images.shape[0]

        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(args.device)
                target = batch['params'].to(args.device)
                out = model(images)
                vl = param_loss_fn(out['norm_params'], target)
                val_loss += vl.item() * images.shape[0]
                val_n += images.shape[0]
        val_loss /= val_n

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} [{dt:.0f}s] "
              f"loss={total_loss/n:.4f} param={total_param/n:.4f} "
              f"cons={total_cons/n:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val,
            }, os.path.join(args.output_dir, 'best.pt'))
            print(f"  → New best: {best_val:.4f}")

    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
    }, os.path.join(args.output_dir, 'last.pt'))
    print(f"Done. Best val_loss: {best_val:.4f}")


if __name__ == '__main__':
    main()
