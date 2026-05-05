"""v6 Stage A: Venus NL 语义对齐训练。

使用 FiveK 域内图片 + Venus 生成的美学文本 embedding 进行视觉-语义对齐。
与 v1-v5 使用 COCO 不同，v6 使用 FiveK 域内数据，消除域差异。

用法 (AutoDL):
    python scripts/train_v6_stage_a.py \
        --image_root /root/autodl-tmp/fivek_jpeg \
        --embedding_path /root/autodl-tmp/data/fivek_text_embeddings.npz \
        --output_dir /root/autodl-tmp/checkpoints/distill_v6/stage_a
"""

import argparse
import os
import sys

import torch

# 项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.semantic_distill.model import SemanticDistillModel
from training.semantic_distill.text_dataset import TextImageDataset
from training.semantic_distill.trainer import StageATrainer
from training.semantic_distill.config import DistillConfig


def main():
    parser = argparse.ArgumentParser(description='v6 Stage A Training')
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/distill_v6/stage_a')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = DistillConfig()
    config.stage_a_epochs = args.epochs
    config.stage_a_batch_size = args.batch_size
    config.stage_a_lr = args.lr

    # 数据集
    train_ds = TextImageDataset(
        image_root=args.image_root,
        embedding_path=args.embedding_path,
        image_size=config.image_size,
        split='train',
        augment=True,
    )
    val_ds = TextImageDataset(
        image_root=args.image_root,
        embedding_path=args.embedding_path,
        image_size=config.image_size,
        split='val',
        augment=False,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # 模型
    model = SemanticDistillModel(
        image_size=config.image_size,
        visual_dim=config.visual_dim,
        semantic_dim=config.semantic_dim,
        text_dim=config.text_dim,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params / 1e6:.2f}M")

    # 训练
    trainer = StageATrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        device=args.device,
        output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == '__main__':
    main()
