"""v8 Stage C: 文本条件化 ISP 参数预测训练。

基于 v7 Stage B 冻结 backbone，新增 LightTextEncoder + FiLM 融合。
训练模型根据口语化指令调整参数预测。

用法 (AutoDL):
    python scripts/train_stage_c.py \
        --stage_b_ckpt /root/autodl-tmp/checkpoints/distill_v7/stage_b/best.pt \
        --jpeg_dir /root/autodl-tmp/fivek_jpeg \
        --instruction_data data/instruction_data.json \
        --output_dir /root/autodl-tmp/checkpoints/distill_v8/stage_c
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.semantic_distill.model import DistillParamModel
from training.text_condition.model import TextConditionedModel
from training.text_condition.config import TextCondConfig
from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES


class InstructionDataset(Dataset):
    """口语化指令数据集。"""

    def __init__(self, data_path, jpeg_dir, image_size=224, split='train',
                 max_text_len=64):
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        n = len(all_data)
        split_idx = int(n * 0.9)
        self.data = all_data[:split_idx] if split == 'train' else all_data[split_idx:]
        self.jpeg_dir = jpeg_dir
        self.max_text_len = max_text_len

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        # 简易字符词表
        self._build_vocab()

    def _build_vocab(self):
        chars = set()
        for item in self.data:
            text = item.get('instruction', '')
            chars.update(text)
        self.char2id = {c: i + 2 for i, c in enumerate(sorted(chars))}
        self.char2id['[PAD]'] = 0
        self.char2id['[CLS]'] = 1

    def tokenize(self, text):
        ids = [1]  # CLS
        for ch in text[:self.max_text_len - 1]:
            ids.append(self.char2id.get(ch, 1))
        while len(ids) < self.max_text_len:
            ids.append(0)
        return ids

    def _normalize_params(self, params_dict):
        result = []
        for name in PARAM_NAMES:
            val = params_dict.get(name, 0.0)
            lo, hi = PARAM_RANGES[name]
            norm = 2.0 * (val - lo) / (hi - lo) - 1.0
            result.append(max(-1.0, min(1.0, norm)))
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_id = item['image_id']
        instruction = item.get('instruction', '')
        target_params = item.get('target_params', {})

        # 图片
        img_path = os.path.join(self.jpeg_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.jpeg_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # 文本
        text_ids = torch.tensor(self.tokenize(instruction), dtype=torch.long)

        # 参数
        params = torch.tensor(self._normalize_params(target_params),
                              dtype=torch.float32)

        return {
            'image': image,
            'text_ids': text_ids,
            'params': params,
            'instruction': instruction,
        }


def main():
    parser = argparse.ArgumentParser(description='Stage C Training')
    parser.add_argument('--stage_b_ckpt', type=str, required=True)
    parser.add_argument('--jpeg_dir', type=str, required=True)
    parser.add_argument('--instruction_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str,
                        default='checkpoints/distill_v8/stage_c')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--consistency_weight', type=float, default=0.1)
    parser.add_argument('--base_align_weight', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = TextCondConfig()

    # 加载 Stage B 模型
    stage_b = DistillParamModel()
    state = torch.load(args.stage_b_ckpt, map_location=args.device)
    stage_b.load_state_dict(state['model_state_dict'], strict=False)

    # 构建 Stage C 模型
    model = TextConditionedModel(
        stage_b_model=stage_b,
        vocab_size=cfg.vocab_size,
        text_embed_dim=cfg.text_embed_dim,
        text_hidden_dim=cfg.text_hidden_dim,
        text_num_heads=cfg.text_num_heads,
        text_num_layers=cfg.text_num_layers,
        max_text_len=cfg.max_text_len,
        semantic_dim=cfg.semantic_dim,
        fusion_type=cfg.fusion_type,
    ).to(args.device)

    trainable = model.trainable_param_count()
    print(f"Trainable: {trainable / 1e6:.2f}M")

    # 数据
    train_ds = InstructionDataset(
        args.instruction_data, args.jpeg_dir, split='train',
    )
    val_ds = InstructionDataset(
        args.instruction_data, args.jpeg_dir, split='val',
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4)

    optimizer = AdamW(model.get_trainable_params(), lr=args.lr,
                      weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    param_loss_fn = nn.L1Loss()
    best_val = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n = 0
        t0 = time.time()

        for batch in train_loader:
            images = batch['image'].to(args.device)
            text_ids = batch['text_ids'].to(args.device)
            target = batch['params'].to(args.device)

            # 有文本前向
            out = model(images, text_ids)
            param_loss = param_loss_fn(out['norm_params'], target)

            # 一致性: 无文本也应合理
            out_no_text = model(images, text_ids=None)
            consistency = F.mse_loss(out_no_text['norm_params'],
                                     out['base_params'])

            # 基线对齐: 无文本时应接近 Stage B
            base_align = F.mse_loss(out_no_text['norm_params'],
                                    out['base_params'].detach())

            loss = param_loss + \
                   args.consistency_weight * consistency + \
                   args.base_align_weight * base_align

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            n += images.shape[0]

        scheduler.step()

        # 验证
        model.eval()
        val_loss = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(args.device)
                text_ids = batch['text_ids'].to(args.device)
                target = batch['params'].to(args.device)
                out = model(images, text_ids)
                vl = param_loss_fn(out['norm_params'], target)
                val_loss += vl.item() * images.shape[0]
                val_n += images.shape[0]
        val_loss /= max(val_n, 1)

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} [{dt:.0f}s] "
              f"loss={total_loss/n:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_loss': best_val,
            }, os.path.join(args.output_dir, 'best.pt'))
            print(f"  → Best: {best_val:.4f}")

    print(f"Stage C done. Best val: {best_val:.4f}")


if __name__ == '__main__':
    main()
