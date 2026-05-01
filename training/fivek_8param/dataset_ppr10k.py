"""
PPR10K 数据集加载器 — 与 FiveK 训练 pipeline 兼容

输出格式与 FiveKExpertPairDataset 一致:
  - image:        (3, H, W)  ImageNet 归一化  ← 模型输入
  - raw_image:    (3, H, W)  [0, 1] 无归一化  ← diff ISP 输入
  - expert_image: (3, H, W)  [0, 1] 无归一化  ← 真实专家 GT
  - params:       (6,)       归一化参数 [-1,1]
  - raw_params:   (6,)       物理参数
  - weights:      (6,)       一致性权重

支持:
  - 3 个专家 (a/b/c) 随机采样训练
  - 验证时固定单专家
  - 与 FiveK 数据集混合训练 (ConcatDataset)
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms as T

from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES, normalize_param

logger = logging.getLogger(__name__)


class PPR10KDataset(Dataset):
    """
    PPR10K 配对数据集。

    加载 (原始360p图, 专家修图后360p图, 参数标注) 三元组。
    训练时: 随机采样 Expert A/B/C 中一个作为 GT
    验证时: 固定用指定 Expert
    """

    def __init__(
        self,
        samples: List[dict],
        image_dir: str,
        image_size: int = 224,
        is_train: bool = True,
        val_expert: str = 'a',
    ):
        super().__init__()
        self.samples = samples
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.is_train = is_train
        self.val_expert = val_expert

        self.resize = T.Resize((image_size, image_size))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.hflip = T.RandomHorizontalFlip(1.0)

        experts_available = set()
        for s in samples:
            experts_available.update(s.get('experts', {}).keys())
        logger.info(
            f"PPR10K {'训练' if is_train else '验证'}集: {len(self.samples)} 对  "
            f"experts={sorted(experts_available)}  val_expert={val_expert}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 选择专家
        experts = s.get('experts', {})
        if self.is_train and experts:
            chosen_expert = random.choice(list(experts.keys()))
        else:
            chosen_expert = self.val_expert

        expert_data = experts.get(chosen_expert, {})

        # 加载图片
        image_name = s['image_name']
        stem = Path(image_name).stem

        # PPR10K 的 360p 图片在 train_val_images_tif_360p/ 下
        # source: train_val_images_tif_360p/source/  或直接在目录下
        # target: train_val_images_tif_360p/target_a/ target_b/ target_c/
        orig_path = self._find_image(stem, 'source')
        expert_path = self._find_image(stem, f'target_{chosen_expert}')

        try:
            pil_orig = Image.open(str(orig_path)).convert('RGB')
            pil_expert = Image.open(str(expert_path)).convert('RGB')
        except Exception as e:
            logger.warning(f"图片加载失败: {e} (orig={orig_path}, expert={expert_path})")
            dummy = torch.zeros(3, self.image_size, self.image_size)
            zeros = torch.zeros(len(PARAM_NAMES))
            return {
                'image': dummy, 'raw_image': dummy, 'expert_image': dummy,
                'params': zeros, 'raw_params': zeros,
                'weights': torch.ones(len(PARAM_NAMES)) * 0.5,
            }

        # 空间变换
        pil_orig = self.resize(pil_orig)
        pil_expert = self.resize(pil_expert)
        if self.is_train and random.random() < 0.5:
            pil_orig = self.hflip(pil_orig)
            pil_expert = self.hflip(pil_expert)

        raw_orig = self.to_tensor(pil_orig)       # [0, 1]
        raw_expert = self.to_tensor(pil_expert)    # [0, 1]
        img_normed = self.normalize(raw_orig.clone())

        # 参数: 使用专家的绝对参数
        phys_params = expert_data.get('absolute', s.get('mean_params', {}))

        params = torch.tensor(
            [normalize_param(p, phys_params.get(p, 0.0)) for p in PARAM_NAMES],
            dtype=torch.float32,
        )
        raw_params = torch.tensor(
            [float(phys_params.get(p, 0.0)) for p in PARAM_NAMES],
            dtype=torch.float32,
        )

        # 一致性权重: 3 专家间一致性越高权重越大
        weights = self._compute_consistency_weights(s)

        return {
            'image': img_normed,
            'raw_image': raw_orig,
            'expert_image': raw_expert,
            'params': params,
            'raw_params': raw_params,
            'weights': weights,
        }

    def _find_image(self, stem: str, subdir: str) -> Path:
        """查找图片文件，支持多种后缀"""
        for ext in ['.tif', '.TIF', '.tiff', '.png', '.jpg', '.jpeg']:
            p = self.image_dir / subdir / f'{stem}{ext}'
            if p.exists():
                return p
        # fallback: 直接在 image_dir 下搜索
        for ext in ['.tif', '.TIF', '.tiff', '.png', '.jpg', '.jpeg']:
            p = self.image_dir / f'{stem}{ext}'
            if p.exists():
                return p
        return self.image_dir / subdir / f'{stem}.tif'

    def _compute_consistency_weights(self, sample: dict) -> torch.Tensor:
        """从 3 个专家的参数差异计算一致性权重"""
        experts = sample.get('experts', {})
        if len(experts) < 2:
            return torch.ones(len(PARAM_NAMES)) * 0.5

        weights = []
        for param in PARAM_NAMES:
            vals = [experts[ex]['absolute'].get(param, 0.0)
                    for ex in experts if 'absolute' in experts[ex]]
            if len(vals) >= 2:
                std = np.std(vals)
                lo, hi = PARAM_RANGES[param]
                range_size = hi - lo
                # 标准差越小 → 一致性越高 → 权重越大
                normalized_std = std / (range_size + 1e-8)
                w = max(0.1, 1.0 - normalized_std * 5.0)
                weights.append(w)
            else:
                weights.append(0.5)

        return torch.tensor(weights, dtype=torch.float32)


def build_ppr10k_datasets(
    params_file: str,
    image_dir: str,
    image_size: int = 224,
    val_ratio: float = 0.2,
    seed: int = 42,
    val_expert: str = 'a',
    train_split_count: int = 8875,
) -> Tuple[PPR10KDataset, PPR10KDataset]:
    """
    构建 PPR10K 训练/验证数据集。

    Args:
        params_file: parse_ppr10k_xmp.py 生成的 JSON 文件路径
        image_dir: PPR10K 360p 图片根目录
        image_size: 输入尺寸
        val_ratio: 验证集比例 (若 train_split_count > 0 则忽略)
        seed: 随机种子
        val_expert: 验证时固定用的专家
        train_split_count: PPR10K 官方 train/val 分割点 (前8875训练, 后2286验证)
    """
    with open(params_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = data['samples']
    logger.info(f"PPR10K 加载: {len(samples)} 张, experts={data['meta']['experts']}")

    # PPR10K 官方按文件名排序，前 8875 训练，后 2286 验证
    if train_split_count > 0 and train_split_count < len(samples):
        train_samples = samples[:train_split_count]
        val_samples = samples[train_split_count:]
    else:
        random.seed(seed)
        indices = list(range(len(samples)))
        random.shuffle(indices)
        val_count = int(len(samples) * val_ratio)
        val_samples = [samples[i] for i in indices[:val_count]]
        train_samples = [samples[i] for i in indices[val_count:]]

    train_ds = PPR10KDataset(
        train_samples, image_dir, image_size,
        is_train=True, val_expert=val_expert)
    val_ds = PPR10KDataset(
        val_samples, image_dir, image_size,
        is_train=False, val_expert=val_expert)

    return train_ds, val_ds


def build_combined_datasets(
    fivek_train: Dataset,
    fivek_val: Dataset,
    ppr10k_train: Dataset,
    ppr10k_val: Dataset,
) -> Tuple[ConcatDataset, ConcatDataset]:
    """
    合并 FiveK + PPR10K 数据集。

    两个数据集输出格式兼容 (image, raw_image, expert_image, params, weights)，
    可以直接 ConcatDataset 混合训练。
    """
    combined_train = ConcatDataset([fivek_train, ppr10k_train])
    combined_val = ConcatDataset([fivek_val, ppr10k_val])
    logger.info(
        f"合并数据集: train={len(combined_train)} "
        f"(FiveK={len(fivek_train)} + PPR10K={len(ppr10k_train)}), "
        f"val={len(combined_val)} "
        f"(FiveK={len(fivek_val)} + PPR10K={len(ppr10k_val)})"
    )
    return combined_train, combined_val
