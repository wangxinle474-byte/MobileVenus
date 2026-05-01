"""
统一 DataLoader (5参数版)

支持的数据源:
  1. FiveK 标签数据 (convert_fivek.py 输出)
  2. AVA 美学评分数据 (可选)
  3. 演示/合成数据

训练样本格式:
  {
    "id": str,
    "image": str (图像路径),
    "targets": {
      "ev_compensation": float,
      "white_balance": int,
      "focus_point": [x, y],
      "hdr": int (0/1),
      "mode": int (0-4),
    },
    "problem_labels": [9 ints],
    "severity_labels": [9 ints],
  }
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import torchvision.transforms as T
except ImportError:
    logger.warning("PIL/torchvision not available, image loading disabled")


# ============================================================
# 数据增强
# ============================================================

def get_train_transforms(image_size: int = 224):
    """训练时数据增强"""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: int = 224):
    """验证时数据变换"""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


# ============================================================
# 参数归一化
# ============================================================

class ParameterNormalizer:
    """
    将相机参数归一化到 [0, 1] 或 [-1, 1] 范围, 方便训练
    """
    
    # 各参数的原始范围
    RANGES = {
        'ev_compensation': (-3.0, 3.0),
        'white_balance': (2000.0, 10000.0),
    }
    
    @staticmethod
    def normalize(name: str, value: float) -> float:
        """将原始值归一化到 [-1, 1]"""
        if name in ParameterNormalizer.RANGES:
            lo, hi = ParameterNormalizer.RANGES[name]
            return 2.0 * (value - lo) / (hi - lo) - 1.0
        return value
    
    @staticmethod
    def denormalize(name: str, value: float) -> float:
        """将归一化值还原到原始范围"""
        if name in ParameterNormalizer.RANGES:
            lo, hi = ParameterNormalizer.RANGES[name]
            return lo + (value + 1.0) / 2.0 * (hi - lo)
        return value


# ============================================================
# 统一数据集
# ============================================================

class MobileVenusDataset(Dataset):
    """
    MobileVenus 统一训练数据集 (5参数版)
    
    Args:
        data_file: JSON 数据文件路径 (convert_fivek.py 输出)
        image_root: 图像根目录 (如果 data_file 中的路径是相对的)
        transform: 图像变换
        normalize_params: 是否归一化参数到 [-1, 1]
        use_dummy_images: 使用随机张量代替真实图像 (用于测试)
        image_size: 图像尺寸
    """
    
    def __init__(
        self,
        data_file: str,
        image_root: Optional[str] = None,
        transform=None,
        normalize_params: bool = True,
        use_dummy_images: bool = False,
        image_size: int = 224
    ):
        super().__init__()
        
        self.image_root = Path(image_root) if image_root else None
        self.transform = transform
        self.normalize_params = normalize_params
        self.use_dummy_images = use_dummy_images
        self.image_size = image_size
        
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        logger.info(f"加载 {len(self.samples)} 个样本 from {data_file}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # ---------- 图像 ----------
        if self.use_dummy_images:
            image = torch.randn(3, self.image_size, self.image_size)
        else:
            image = self._load_image(sample['image'])
        
        # ---------- 5 参数目标值 ----------
        targets = sample['targets']
        
        ev = float(targets['ev_compensation'])
        wb = float(targets['white_balance'])
        focus = targets['focus_point']  # [x, y]
        hdr = int(targets['hdr'])
        mode = int(targets['mode'])
        
        # 可选: 归一化连续参数
        if self.normalize_params:
            ev = ParameterNormalizer.normalize('ev_compensation', ev)
            wb = ParameterNormalizer.normalize('white_balance', wb)
        
        # ---------- 问题检测标签 ----------
        problem_labels = torch.tensor(
            sample.get('problem_labels', [0] * 9), dtype=torch.float32
        )
        severity_labels = torch.tensor(
            sample.get('severity_labels', [0] * 9), dtype=torch.long
        )
        
        return {
            'image': image,
            'ev_compensation': torch.tensor(ev, dtype=torch.float32),
            'white_balance': torch.tensor(wb, dtype=torch.float32),
            'focus_point': torch.tensor(focus, dtype=torch.float32),
            'hdr': torch.tensor(hdr, dtype=torch.long),
            'mode': torch.tensor(mode, dtype=torch.long),
            'problem_labels': problem_labels,
            'severity_labels': severity_labels,
        }
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """加载并预处理图像"""
        path = Path(image_path)
        
        # 尝试多种路径
        if not path.is_absolute() and self.image_root:
            path = self.image_root / path
        
        if not path.exists():
            logger.warning(f"图像不存在: {path}, 使用随机张量")
            return torch.randn(3, self.image_size, self.image_size)
        
        try:
            img = Image.open(str(path)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                img = get_val_transforms(self.image_size)(img)
            return img
        except Exception as e:
            logger.warning(f"加载图像失败: {path}, {e}")
            return torch.randn(3, self.image_size, self.image_size)


# ============================================================
# DataLoader 创建工具
# ============================================================

def create_dataloaders(
    train_file: str,
    val_file: Optional[str] = None,
    image_root: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    normalize_params: bool = True,
    use_dummy_images: bool = False,
    val_split: float = 0.1
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建训练和验证 DataLoader
    
    Args:
        train_file: 训练数据 JSON 文件
        val_file: 验证数据 JSON 文件 (如果为 None, 从训练集中划分)
        image_root: 图像根目录
        batch_size: 批大小
        num_workers: 数据加载线程数
        image_size: 图像尺寸
        normalize_params: 是否归一化参数
        use_dummy_images: 使用随机张量代替真实图像
        val_split: 验证集比例 (当 val_file 为 None 时使用)
        
    Returns:
        train_loader, val_loader
    """
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    if val_file is not None:
        # 有独立验证集
        train_dataset = MobileVenusDataset(
            train_file, image_root, train_transform,
            normalize_params, use_dummy_images, image_size
        )
        val_dataset = MobileVenusDataset(
            val_file, image_root, val_transform,
            normalize_params, use_dummy_images, image_size
        )
    else:
        # 从训练集划分
        full_dataset = MobileVenusDataset(
            train_file, image_root, train_transform,
            normalize_params, use_dummy_images, image_size
        )
        
        total = len(full_dataset)
        val_size = max(int(total * val_split), 1)
        train_size = total - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"数据集划分: 训练 {train_size}, 验证 {val_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    print("=" * 60)
    print("MobileVenus DataLoader 测试 (5参数版)")
    print("=" * 60)
    
    # 先生成演示数据
    from tools.convert_fivek import create_demo_data
    demo_dir = "data/test_dataloader"
    create_demo_data(demo_dir, num_samples=50)
    
    demo_file = f"{demo_dir}/demo_5params_train.json"
    
    # 创建 DataLoader (使用 dummy images)
    train_loader, val_loader = create_dataloaders(
        train_file=demo_file,
        batch_size=8,
        num_workers=0,
        use_dummy_images=True,
        val_split=0.2
    )
    
    print(f"\n训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} 批次")
    print(f"验证集: {len(val_loader.dataset)} 样本, {len(val_loader)} 批次")
    
    # 检查一个 batch
    batch = next(iter(train_loader))
    print(f"\n--- Batch 内容 ---")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}: shape={str(v.shape):20s} dtype={v.dtype}")
    
    # 验证参数范围
    print(f"\n--- 参数范围 ---")
    print(f"  EV (normalized):  [{batch['ev_compensation'].min():.3f}, {batch['ev_compensation'].max():.3f}]")
    print(f"  WB (normalized):  [{batch['white_balance'].min():.3f}, {batch['white_balance'].max():.3f}]")
    print(f"  Focus point:      [{batch['focus_point'].min():.3f}, {batch['focus_point'].max():.3f}]")
    print(f"  HDR:              {batch['hdr'].unique().tolist()}")
    print(f"  Mode:             {batch['mode'].unique().tolist()}")
    print(f"  Problem labels:   shape={batch['problem_labels'].shape}")
    
    # 反归一化验证
    ev_raw = ParameterNormalizer.denormalize('ev_compensation', float(batch['ev_compensation'][0]))
    wb_raw = ParameterNormalizer.denormalize('white_balance', float(batch['white_balance'][0]))
    print(f"\n--- 反归一化验证 ---")
    print(f"  EV: normalized={batch['ev_compensation'][0]:.3f} → raw={ev_raw:.3f}")
    print(f"  WB: normalized={batch['white_balance'][0]:.3f} → raw={wb_raw:.0f}K")
    
    print("\n" + "=" * 60)
    print("DataLoader 测试通过!")
    print("=" * 60)
