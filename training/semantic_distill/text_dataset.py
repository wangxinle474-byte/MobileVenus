"""Stage A 图文对数据集。

加载图片 + 预计算的 MiniLM 文本 embedding，用于视觉-语义对齐训练。

数据来源:
    - 图片: FiveK / COCO / Venus Stage1
    - 文本 embedding: *.npz (由 embed_texts.py 生成)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class TextImageDataset(Dataset):
    """Stage A 图文对数据集。

    Args:
        image_root: 图片根目录
        embedding_path: .npz 文件路径 (含 image_ids, embeddings)
        image_size: 图片缩放大小
        split: 'train' 或 'val'
        augment: 是否数据增强
    """

    def __init__(self, image_root, embedding_path, image_size=224,
                 split='train', augment=True):
        super().__init__()
        self.image_root = image_root
        self.image_size = image_size

        # 加载预计算 embedding
        data = np.load(embedding_path, allow_pickle=True)
        self.image_ids = data['image_ids'].tolist()
        self.embeddings = data['embeddings'].astype(np.float32)

        # 过滤不存在的图片
        valid = []
        for i, img_id in enumerate(self.image_ids):
            path = self._find_image(img_id)
            if path is not None:
                valid.append(i)

        self.valid_indices = valid

        # 划分
        n = len(self.valid_indices)
        split_idx = int(n * 0.9)
        if split == 'train':
            self.valid_indices = self.valid_indices[:split_idx]
        else:
            self.valid_indices = self.valid_indices[split_idx:]

        # 图片变换
        if augment and split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
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

    def _find_image(self, img_id):
        """查找图片路径。"""
        for ext in ['', '.jpg', '.jpeg', '.png']:
            path = os.path.join(self.image_root, f"{img_id}{ext}")
            if os.path.exists(path):
                return path
        return None

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_id = self.image_ids[real_idx]
        text_emb = self.embeddings[real_idx]

        # 加载图片
        img_path = self._find_image(img_id)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return {
            'image': image,
            'text_emb': torch.from_numpy(text_emb),
            'image_id': img_id,
        }


class MultiSourceTextImageDataset(Dataset):
    """多数据源 Stage A 数据集 (FiveK + COCO + Venus)。"""

    def __init__(self, sources, image_size=224, split='train', augment=True):
        """
        Args:
            sources: list of dict, each with:
                - image_root: str
                - embedding_path: str
        """
        self.datasets = [
            TextImageDataset(
                image_root=src['image_root'],
                embedding_path=src['embedding_path'],
                image_size=image_size,
                split=split,
                augment=augment,
            )
            for src in sources
        ]
        # 计算累积长度
        self.cumlen = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cumlen.append(total)

    def __len__(self):
        return self.cumlen[-1] if self.cumlen else 0

    def __getitem__(self, idx):
        for i, cl in enumerate(self.cumlen):
            if idx < cl:
                offset = self.cumlen[i - 1] if i > 0 else 0
                return self.datasets[i][idx - offset]
        raise IndexError(f"Index {idx} out of range")
