"""FiveK 数据集加载器 (基础版)。

加载 MIT-Adobe FiveK 图片和 Expert C 专家参数标注。

数据格式:
    - 图片: JPEG (fivek_jpeg/ 目录)
    - 参数: fivek_expert_params.json (46K 条)
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .config import PARAM_NAMES, PARAM_RANGES, DEFAULT_EXPERT


class FiveKDataset(Dataset):
    """FiveK 参数预测数据集。

    Args:
        jpeg_dir: JPEG 图片目录
        params_json: 参数标注 JSON 路径
        expert: 使用哪个专家的标注 (默认 expert_c)
        split: 'train' 或 'val' 或 'test'
        image_size: 图片缩放大小
        augment: 是否数据增强
        max_samples: 最大样本数 (调试用)
    """

    def __init__(self, jpeg_dir, params_json, expert=DEFAULT_EXPERT,
                 split='train', image_size=224, augment=True,
                 max_samples=None):
        super().__init__()
        self.jpeg_dir = jpeg_dir
        self.expert = expert
        self.image_size = image_size
        self.augment = augment and (split == 'train')

        # 加载参数
        with open(params_json, 'r', encoding='utf-8') as f:
            all_params = json.load(f)

        # 按 split 划分
        self.samples = self._split_data(all_params, split)
        if max_samples:
            self.samples = self.samples[:max_samples]

        # 图片变换
        if self.augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])

    def _split_data(self, all_params, split):
        """按 80/10/10 划分 train/val/test。"""
        items = list(all_params.items()) if isinstance(all_params, dict) \
            else [(item.get('image_id', str(i)), item)
                  for i, item in enumerate(all_params)]

        # 固定随机种子确保可复现
        rng = random.Random(42)
        rng.shuffle(items)

        n = len(items)
        if split == 'train':
            return items[:int(n * 0.8)]
        elif split == 'val':
            return items[int(n * 0.8):int(n * 0.9)]
        else:  # test
            return items[int(n * 0.9):]

    def _get_expert_params(self, param_data):
        """提取指定专家的 6 个参数并归一化到 [-1, 1]。"""
        if isinstance(param_data, dict) and self.expert in param_data:
            expert_data = param_data[self.expert]
        else:
            expert_data = param_data

        params = []
        for name in PARAM_NAMES:
            val = expert_data.get(name, 0.0)
            lo, hi = PARAM_RANGES[name]
            # 归一化到 [-1, 1]
            norm_val = 2.0 * (val - lo) / (hi - lo) - 1.0
            params.append(np.clip(norm_val, -1.0, 1.0))

        return np.array(params, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, param_data = self.samples[idx]

        # 加载图片
        img_path = os.path.join(self.jpeg_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.jpeg_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # 参数
        params = self._get_expert_params(param_data)
        params = torch.from_numpy(params)

        return {
            'image': image,
            'params': params,
            'image_id': img_id,
        }


class FiveKRawDataset(FiveKDataset):
    """FiveK 数据集 — 同时返回原始未归一化图片 (用于 ISP 渲染)。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),  # [0, 1], 不做 normalize
        ])

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        # 额外加载原始图片 (不做 ImageNet 归一化)
        img_id = item['image_id']
        img_path = os.path.join(self.jpeg_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.jpeg_dir, img_id)
        raw_image = Image.open(img_path).convert('RGB')
        item['raw_image'] = self.raw_transform(raw_image)

        return item
