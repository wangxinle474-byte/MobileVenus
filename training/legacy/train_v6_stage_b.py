"""v6 Stage B: 6参数预测训练。

基于 v6 Stage A 的冻结 backbone，训练 LightroomDecoder 预测 6 个 ISP 参数。
使用 Expert C 单专家精准监督。

用法 (AutoDL):
    python training/legacy/train_v6_stage_b.py \
        --stage_a_ckpt /root/autodl-tmp/checkpoints/distill_v6/stage_a/best.pt \
        --jpeg_dir /root/autodl-tmp/fivek_jpeg \
        --params_json /root/autodl-tmp/data/fivek_expert_params.json \
        --output_dir /root/autodl-tmp/checkpoints/distill_v6/stage_b
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.semantic_distill.model import DistillParamModel
from training.fivek_8param.dataset import FiveKDataset
from training.semantic_distill.trainer import StageBTrainer
from training.semantic_distill.config import DistillConfig


def main():
    parser = argparse.ArgumentParser(description='v6 Stage B Training')
    parser.add_argument('--stage_a_ckpt', type=str, required=True)
    parser.add_argument('--jpeg_dir', type=str, required=True)
    parser.add_argument('--params_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/distill_v6/stage_b')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    config = DistillConfig()
    config.stage_b_epochs = args.epochs
    config.stage_b_batch_size = args.batch_size
    config.stage_b_lr = args.lr

    # 从 Stage A 构建模型
    model = DistillParamModel.from_stage_a(
        args.stage_a_ckpt, device=args.device,
        image_size=config.image_size,
        visual_dim=config.visual_dim,
        semantic_dim=config.semantic_dim,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable / 1e3:.1f}K")

    # 数据集
    train_ds = FiveKDataset(
        jpeg_dir=args.jpeg_dir,
        params_json=args.params_json,
        split='train', augment=True,
    )
    val_ds = FiveKDataset(
        jpeg_dir=args.jpeg_dir,
        params_json=args.params_json,
        split='val', augment=False,
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    trainer = StageBTrainer(
        model=model, train_dataset=train_ds, val_dataset=val_ds,
        config=config, device=args.device, output_dir=args.output_dir,
    )
    trainer.train()


if __name__ == '__main__':
    main()
