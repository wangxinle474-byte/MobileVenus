"""
Stage C 数据集: (图片, 文本指令, 目标参数) 三元组
"""
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES, normalize_param
from training.text_condition.model import build_char_vocab, tokenize

logger = logging.getLogger(__name__)


class InstructionDataset(Dataset):
    """
    Stage C 训练数据集

    每个样本:
      - image:       (3, H, W)  ImageNet 归一化
      - raw_image:   (3, H, W)  [0, 1]
      - text_ids:    (L,)       文本 token ids
      - params:      (6,)       目标归一化参数 [-1, 1]
      - raw_params:  (6,)       目标物理参数
      - base_params: (6,)       基准归一化参数 (Stage B 预测 / source)
      - has_text:    scalar     1.0 = 有文本, 0.0 = 无文本 (text dropout)
    """

    def __init__(
        self,
        samples: List[dict],
        image_dirs: Dict[str, str],
        vocab: dict,
        image_size: int = 224,
        max_text_len: int = 64,
        is_train: bool = True,
        text_dropout: float = 0.2,
    ):
        super().__init__()
        self.samples = samples
        self.image_dirs = image_dirs  # {'fivek': '/path/to/fivek_jpeg', 'ppr10k': '...'}
        self.vocab = vocab
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.is_train = is_train
        self.text_dropout = text_dropout if is_train else 0.0

        self.resize = T.Resize((image_size, image_size))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        logger.info(
            f"InstructionDataset {'train' if is_train else 'val'}: "
            f"{len(samples)} samples, text_dropout={self.text_dropout}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 加载图片
        image_name = s['image_name']
        source = s.get('source', 'fivek')

        img_path = self._find_image(image_name, source)
        try:
            pil_img = Image.open(str(img_path)).convert('RGB')
        except Exception as e:
            logger.warning(f"图片加载失败: {img_path}: {e}")
            return self._dummy_sample()

        pil_img = self.resize(pil_img)
        raw_img = self.to_tensor(pil_img)
        img_normed = self.normalize(raw_img.clone())

        # 文本
        instruction = s.get('instruction', '')
        has_text = 1.0

        # 训练时 text dropout
        if self.is_train and random.random() < self.text_dropout:
            instruction = ''
            has_text = 0.0

        text_ids = torch.tensor(
            tokenize(instruction, self.vocab, self.max_text_len),
            dtype=torch.long,
        )

        # 目标参数
        target = s.get('target_params', {})
        params = torch.tensor(
            [normalize_param(p, target.get(p, 0.0)) for p in PARAM_NAMES],
            dtype=torch.float32,
        )
        raw_params = torch.tensor(
            [float(target.get(p, 0.0)) for p in PARAM_NAMES],
            dtype=torch.float32,
        )

        # 基准参数
        base = s.get('base_params', {})
        base_params = torch.tensor(
            [normalize_param(p, base.get(p, 0.0)) for p in PARAM_NAMES],
            dtype=torch.float32,
        )

        return {
            'image': img_normed,
            'raw_image': raw_img,
            'text_ids': text_ids,
            'params': params,
            'raw_params': raw_params,
            'base_params': base_params,
            'has_text': torch.tensor(has_text),
        }

    def _find_image(self, image_name: str, source: str) -> Path:
        """查找图片路径"""
        stem = Path(image_name).stem

        # FiveK
        if 'fivek' in source and 'fivek' in self.image_dirs:
            p = Path(self.image_dirs['fivek']) / image_name
            if p.exists():
                return p
            # 尝试不同扩展名
            for ext in ['.jpg', '.jpeg', '.png', '.tif']:
                p = Path(self.image_dirs['fivek']) / f'{stem}{ext}'
                if p.exists():
                    return p

        # PPR10K
        if 'ppr10k' in source and 'ppr10k' in self.image_dirs:
            p = Path(self.image_dirs['ppr10k']) / 'source' / f'{stem}.tif'
            if p.exists():
                return p

        # fallback: 在所有目录中查找 (含子目录)
        for dir_path in self.image_dirs.values():
            for ext in ['.jpg', '.jpeg', '.png', '.tif']:
                p = Path(dir_path) / f'{stem}{ext}'
                if p.exists():
                    return p
            # PPR10K source 子目录
            for ext in ['.tif', '.TIF', '.png', '.jpg']:
                p = Path(dir_path) / 'source' / f'{stem}{ext}'
                if p.exists():
                    return p

        return Path(self.image_dirs.get('fivek', '')) / image_name

    def _dummy_sample(self):
        dummy = torch.zeros(3, self.image_size, self.image_size)
        zeros = torch.zeros(len(PARAM_NAMES))
        text_ids = torch.zeros(self.max_text_len, dtype=torch.long)
        return {
            'image': dummy, 'raw_image': dummy,
            'text_ids': text_ids,
            'params': zeros, 'raw_params': zeros,
            'base_params': zeros,
            'has_text': torch.tensor(0.0),
        }


def build_instruction_datasets(
    data_file: str,
    image_dirs: Dict[str, str],
    vocab: dict = None,
    image_size: int = 224,
    max_text_len: int = 64,
    val_ratio: float = 0.1,
    seed: int = 42,
    text_dropout: float = 0.2,
) -> Tuple[InstructionDataset, InstructionDataset, dict]:
    """
    构建 Stage C 训练/验证数据集

    Returns:
        train_ds, val_ds, vocab
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = data['samples']
    logger.info(f"指令数据加载: {len(samples)} 条")

    # 构建词表
    if vocab is None:
        all_texts = [s['instruction'] for s in samples]
        vocab = build_char_vocab(all_texts)
        logger.info(f"词表大小: {len(vocab)}")

    # 分割
    random.seed(seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)
    val_count = int(len(samples) * val_ratio)

    val_samples = [samples[i] for i in indices[:val_count]]
    train_samples = [samples[i] for i in indices[val_count:]]

    train_ds = InstructionDataset(
        train_samples, image_dirs, vocab, image_size, max_text_len,
        is_train=True, text_dropout=text_dropout,
    )
    val_ds = InstructionDataset(
        val_samples, image_dirs, vocab, image_size, max_text_len,
        is_train=False, text_dropout=0.0,
    )

    return train_ds, val_ds, vocab
