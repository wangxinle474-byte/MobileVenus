"""
FiveK Expert 配对数据集 — 支持全部 5 个 Expert

每个样本返回:
  - image:        (3, H, W)  ImageNet 归一化  ← 模型输入
  - raw_image:    (3, H, W)  [0, 1] 无归一化  ← diff ISP 输入
  - expert_image: (3, H, W)  [0, 1] 无归一化  ← 真实专家 GT
  - params:       (8,)       归一化参数 [-1,1]  (可选辅助监督)
  - weights:      (8,)       一致性权重

训练时: 随机抱取一个可用 Expert 的 GT（数据增强）
验证时: 固定用 Expert C（标准 benchmark）
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
from torch.utils.data import Dataset
from torchvision import transforms as T

from training.fivek_8param.config import PARAM_NAMES, normalize_param, PARAM_RANGES

logger = logging.getLogger(__name__)


class FiveKExpertPairDataset(Dataset):
    """
    加载 (原始图, Expert GT 图, 参数标注) 三元组。
    训练时: 随机抱取 Expert A~E 一作 GT
    验证时: 固定用 Expert C
    """

    def __init__(
        self,
        samples: List[dict],           # [{orig_path, expert_paths:{a:,b:,...}, mean_params, dng_name}, ...]
        consensus_scores: Dict[str, dict],
        image_size: int = 224,
        is_train: bool = True,
        val_expert: str = 'c',         # 验证时固定用哪个 Expert
    ):
        super().__init__()
        self.samples    = samples
        self.consensus  = consensus_scores
        self.image_size = image_size
        self.is_train   = is_train
        self.val_expert = val_expert

        self.resize    = T.Resize((image_size, image_size))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.hflip     = T.RandomHorizontalFlip(1.0)

        # 统计有多少个 Expert的图对数
        available = set()
        for s in samples:
            available.update(s.get('expert_paths', {}).keys())
        logger.info(
            f"{'训练' if is_train else '验证'}集: {len(self.samples)} 对 "
            f"experts={sorted(available)}  val_expert={val_expert}  image_size={image_size}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 训练时随机抱取一个 Expert；验证时固定 val_expert
        expert_paths = s.get('expert_paths', {})
        if self.is_train and expert_paths:
            available = list(expert_paths.keys())
            chosen_expert = random.choice(available)
        else:
            chosen_expert = self.val_expert
        expert_path = expert_paths.get(chosen_expert) or s.get('expert_path', '')

        try:
            pil_orig   = Image.open(s['orig_path']).convert('RGB')
            pil_expert = Image.open(expert_path).convert('RGB')
        except Exception as e:
            logger.warning(f"图片加载失败: {e}")
            dummy  = torch.zeros(3, self.image_size, self.image_size)
            zeros8 = torch.zeros(len(PARAM_NAMES))
            return {
                'image': dummy, 'raw_image': dummy, 'expert_image': dummy,
                'params': zeros8, 'raw_params': zeros8,
                'weights': torch.ones(len(PARAM_NAMES)) * 0.5,
            }

        # 空间变换：两路保持相同（先 resize，再随机翻转）
        pil_orig   = self.resize(pil_orig)
        pil_expert = self.resize(pil_expert)
        if self.is_train and random.random() < 0.5:
            pil_orig   = self.hflip(pil_orig)
            pil_expert = self.hflip(pil_expert)

        raw_orig   = self.to_tensor(pil_orig)    # [0, 1]
        raw_expert = self.to_tensor(pil_expert)  # [0, 1]
        img_normed = self.normalize(raw_orig.clone())

        # 参数标注 (归一化)
        params = torch.tensor(
            [normalize_param(p, s['mean_params'][p]) for p in PARAM_NAMES],
            dtype=torch.float32,
        )

        # 原始物理参数（用于 diff ISP 渲染时的参数上界/下界对齐）
        raw_params = torch.tensor(
            [float(s['mean_params'][p]) for p in PARAM_NAMES],
            dtype=torch.float32,
        )

        # 一致性权重
        dng_name = s['dng_name']
        if dng_name in self.consensus:
            per_param = self.consensus[dng_name]['per_param']
            weights = torch.tensor(
                [per_param.get(p, 0.5) for p in PARAM_NAMES],
                dtype=torch.float32,
            )
        else:
            weights = torch.ones(len(PARAM_NAMES), dtype=torch.float32) * 0.5

        return {
            'image':        img_normed,     # (3, H, W) ImageNet 归一化
            'raw_image':    raw_orig,       # (3, H, W) [0, 1]
            'expert_image': raw_expert,     # (3, H, W) [0, 1]  ← 真实 GT
            'params':       params,         # (8,) 归一化参数
            'raw_params':   raw_params,     # (8,) 物理参数
            'weights':      weights,        # (8,)
        }


def build_expert_datasets(
    data_file: str,
    consensus_file: str,
    orig_jpeg_dir: str,
    expert_dirs: Dict[str, str],      # {'a': '/path/a', 'b': '/path/b', ...}
    image_size: int = 224,
    val_ratio: float = 0.1,
    seed: int = 42,
    val_expert: str = 'c',
) -> Tuple[FiveKExpertPairDataset, FiveKExpertPairDataset]:
    """
    构建 (\u539f\u59cb, Expert GT) 配对数据集

    Args:
        data_file:     fivek_expert_params.json
        consensus_file: fivek_expert_consensus.json
        orig_jpeg_dir: 原始图目录 (fivek_jpeg/)
        expert_dirs:   {'a': 'path/to/a', 'c': 'path/to/c', ...}
        val_expert:    验证时固定用的 Expert（默认 'c'）
    """
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_params = defaultdict(lambda: defaultdict(list))
    for s in data['samples']:
        name = s['image_name']
        for p in PARAM_NAMES:
            img_params[name][p].append(float(s.get(p, 0)))

    consensus = {}
    if Path(consensus_file).exists():
        with open(consensus_file, 'r', encoding='utf-8') as f:
            consensus = json.load(f).get('scores', {})

    # 构建 expert_dirs 的 Path 对象
    exp_paths = {ex: Path(d) for ex, d in expert_dirs.items() if Path(d).exists()}
    if not exp_paths:
        raise RuntimeError(
            f'没有找到任何 Expert 图彐!\n'
            f'expert_dirs={expert_dirs}\n'
            f'请先运行: python tools/download_fivek_expert_c.py'
        )
    logger.info(f'可用 Expert: {sorted(exp_paths.keys())}  (验证固定={val_expert})')

    # val 必须有 val_expert
    if val_expert not in exp_paths:
        raise RuntimeError(f'val_expert="{val_expert}" 未下载, 请先下载 Expert {val_expert}')

    orig_dir = Path(orig_jpeg_dir)
    samples  = []
    skipped  = 0

    for dng_name, param_dict in img_params.items():
        stem      = dng_name.replace('.dng', '')
        orig_path = orig_dir / f'{stem}.jpg'
        if not orig_path.exists():
            continue

        # 收集所有可用 Expert 的路径
        ep = {}
        for ex, ex_dir in exp_paths.items():
            p = ex_dir / f'{stem}.jpg'
            if p.exists():
                ep[ex] = str(p)

        # val_expert 必须存在
        if val_expert not in ep:
            skipped += 1
            continue

        mean_params = {p: float(np.mean(vs)) for p, vs in param_dict.items()}
        samples.append({
            'dng_name':    dng_name,
            'orig_path':   str(orig_path),
            'expert_paths': ep,          # {'a': ..., 'c': ..., ...}
            'mean_params': mean_params,
        })

    logger.info(f'配对成功: {len(samples)} 对  跳过(val_expert缺失): {skipped}')
    if not samples:
        raise RuntimeError('没有找到任何有效配对')

    random.seed(seed)
    random.shuffle(samples)
    val_count     = int(len(samples) * val_ratio)
    val_samples   = samples[:val_count]
    train_samples = samples[val_count:]

    train_ds = FiveKExpertPairDataset(train_samples, consensus, image_size,
                                      is_train=True,  val_expert=val_expert)
    val_ds   = FiveKExpertPairDataset(val_samples,   consensus, image_size,
                                      is_train=False, val_expert=val_expert)
    return train_ds, val_ds
