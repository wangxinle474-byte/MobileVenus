"""
AADB 美学评分器训练脚本

使用 AADB 数据集的 11 维属性标注训练 aesthetic_scorer，
验证 MobileVenus 美学评分方案的可行性。

AADB 属性 → 5 维美学评分映射:
  composition ← RuleOfThirds, Symmetry, BalacingElements, Repetition
  lighting    ← Light
  color       ← ColorHarmony, VividColor
  clarity     ← MotionBlur, DoF
  subject     ← Object, Content

用法:
  python training/train_aadb_aesthetic.py --aadb_root E:/dataset/AADB --epochs 30
  python training/train_aadb_aesthetic.py --aadb_root E:/dataset/AADB --epochs 5 --quick  # 快速验证
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vision_encoder import MobileViTSmall
from models.aesthetic_scorer import AestheticScorer

try:
    from PIL import Image
    import torchvision.transforms as T
except ImportError:
    raise ImportError("需要安装 Pillow 和 torchvision")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# AADB 属性 → 5 维美学评分映射
# ============================================================

# AADB 原始 11 个属性名
AADB_ATTRIBUTES = [
    'BalacingElements',  # 注意: AADB 原始拼写如此
    'ColorHarmony',
    'Content',
    'DoF',
    'Light',
    'MotionBlur',
    'Object',
    'Repetition',
    'RuleOfThirds',
    'Symmetry',
    'VividColor',
]

# 映射: 5 维美学评分 ← AADB 属性
# 每个维度由相关 AADB 属性的加权平均得到
DIMENSION_MAPPING = {
    'composition': {  # 构图
        'RuleOfThirds': 0.35,
        'Symmetry': 0.25,
        'BalacingElements': 0.25,
        'Repetition': 0.15,
    },
    'lighting': {  # 光线
        'Light': 1.0,
    },
    'color': {  # 色彩
        'ColorHarmony': 0.5,
        'VividColor': 0.5,
    },
    'clarity': {  # 清晰度
        'DoF': 0.5,
        'MotionBlur': 0.5,
    },
    'subject': {  # 主体
        'Object': 0.5,
        'Content': 0.5,
    },
}

DIMENSION_NAMES = ['composition', 'lighting', 'color', 'clarity', 'subject']


# ============================================================
# AADB 数据集
# ============================================================

class AADBDataset(Dataset):
    """
    AADB 数据集加载器
    
    加载图像 + 11 维属性标注 + 总分，
    并将 11 维属性映射为 5 维美学评分。
    """
    
    def __init__(
        self,
        aadb_root: str,
        split: str = 'train',
        image_size: int = 224,
        use_original_size: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.aadb_root = Path(aadb_root)
        self.split = split
        self.image_size = image_size
        
        # 图像目录
        if use_original_size:
            self.image_dir = self.aadb_root / 'datasetImages_originalSize'
        else:
            self.image_dir = self.aadb_root / 'datasetImages_warp256'
        
        # 加载标注
        self.samples = self._load_labels()
        
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
        
        # 数据增强
        if split == 'train':
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            ])
        
        logger.info(f"AADB {split}: {len(self.samples)} 样本")
    
    def _load_labels(self) -> List[Dict]:
        """加载所有属性标注"""
        label_dir = self.aadb_root / 'imgListFiles_label'
        
        # 确定文件前缀
        if self.split == 'train':
            prefix = 'imgListTrainRegression'
        elif self.split == 'val':
            prefix = 'imgListValidationRegression'
        elif self.split == 'test':
            prefix = 'imgListTestRegression'
        elif self.split == 'test_new':
            prefix = 'imgListTestNewRegression'
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # 先加载总分获取文件名列表
        score_file = label_dir / f'{prefix}_score.txt'
        filenames = []
        scores = []
        with open(score_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filenames.append(parts[0])
                    scores.append(float(parts[1]))
        
        # 加载每个属性
        attr_scores = {}
        for attr_name in AADB_ATTRIBUTES:
            attr_file = label_dir / f'{prefix}_{attr_name}.txt'
            attr_values = {}
            with open(attr_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        attr_values[parts[0]] = float(parts[1])
            attr_scores[attr_name] = attr_values
        
        # 合并为样本列表
        samples = []
        for fname, score in zip(filenames, scores):
            img_path = self.image_dir / fname
            if not img_path.exists():
                continue
            
            # 获取 11 维属性
            attrs = {}
            valid = True
            for attr_name in AADB_ATTRIBUTES:
                if fname in attr_scores[attr_name]:
                    attrs[attr_name] = attr_scores[attr_name][fname]
                else:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # 映射为 5 维美学评分
            dim_scores = self._map_to_dimensions(attrs)
            
            samples.append({
                'filename': fname,
                'image_path': str(img_path),
                'overall_score': score,           # [0, 1]
                'attributes': attrs,              # 11 维原始属性
                'dimension_scores': dim_scores,   # 5 维映射评分
            })
        
        return samples
    
    def _map_to_dimensions(self, attrs: Dict[str, float]) -> List[float]:
        """
        将 AADB 11 维属性映射为 5 维美学评分
        
        AADB 属性范围: 约 [-1, 1]
        输出范围: [0, 1] (后续 * 10 得到 [0, 10])
        """
        dim_scores = []
        for dim_name in DIMENSION_NAMES:
            mapping = DIMENSION_MAPPING[dim_name]
            score = 0.0
            for attr_name, weight in mapping.items():
                score += attrs[attr_name] * weight
            # AADB 属性范围约 [-1, 1], 映射到 [0, 1]
            score = (score + 1.0) / 2.0
            score = max(0.0, min(1.0, score))
            dim_scores.append(score)
        return dim_scores
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            # 返回随机图像作为 fallback
            logger.warning(f"图像加载失败: {sample['filename']}: {e}")
            image = torch.randn(3, self.image_size, self.image_size)
        
        # 标签: 5 维评分 [0, 1] 和总分 [0, 1]
        dim_scores = torch.tensor(sample['dimension_scores'], dtype=torch.float32)
        overall_score = torch.tensor(sample['overall_score'], dtype=torch.float32)
        
        return {
            'image': image,
            'dim_scores': dim_scores,          # (5,) [0, 1]
            'overall_score': overall_score,    # scalar [0, 1]
        }


# ============================================================
# 训练模型: 视觉编码器 + 投影 + 美学评分器
# ============================================================

class AestheticModel(nn.Module):
    """
    用于 AADB 训练的美学评分模型
    包含: 视觉编码器 + 投影层 + AestheticScorer
    """
    
    def __init__(
        self,
        image_size: int = 224,
        vision_output_dim: int = 384,
        projection_dim: int = 512,
        scorer_hidden_dim: int = 256,
        num_dimensions: int = 5,
    ):
        super().__init__()
        
        self.vision_encoder = MobileViTSmall(
            image_size=image_size,
            num_classes=0,
            use_se=True,
            use_fpn=True,
            output_dim=vision_output_dim
        )
        
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_output_dim, projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        self.aesthetic_scorer = AestheticScorer(
            input_dim=projection_dim,
            num_dimensions=num_dimensions,
            hidden_dim=scorer_hidden_dim
        )
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_features = self.vision_encoder(images)
        visual_features = self.vision_projection(raw_features)
        scores, weighted_score = self.aesthetic_scorer(visual_features)
        return {
            'scores': scores,                  # (B, 5) [0, 10]
            'weighted_score': weighted_score,   # (B,) [0, 10]
            'features': visual_features,        # (B, 512)
        }


# ============================================================
# 损失函数
# ============================================================

class AestheticLoss(nn.Module):
    """
    美学评分联合损失
    
    L = α * L_dim + β * L_overall + γ * L_rank
    """
    
    def __init__(self, dim_weight=1.0, overall_weight=1.0, rank_weight=0.5):
        super().__init__()
        self.dim_weight = dim_weight
        self.overall_weight = overall_weight
        self.rank_weight = rank_weight
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
    
    def forward(
        self,
        pred_scores: torch.Tensor,       # (B, 5) [0, 10]
        pred_weighted: torch.Tensor,      # (B,) [0, 10]
        target_dims: torch.Tensor,        # (B, 5) [0, 1]
        target_overall: torch.Tensor,     # (B,) [0, 1]
    ) -> Dict[str, torch.Tensor]:
        
        # 将预测缩放到 [0, 1] 与标签对齐
        pred_norm = pred_scores / 10.0        # (B, 5) [0, 1]
        pred_overall_norm = pred_weighted / 10.0  # (B,) [0, 1]
        
        # 1. 维度评分损失 (MSE)
        loss_dim = self.mse(pred_norm, target_dims)
        
        # 2. 总分损失 (Smooth L1)
        loss_overall = self.smooth_l1(pred_overall_norm, target_overall)
        
        # 3. 排序损失 (pairwise ranking)
        loss_rank = torch.tensor(0.0, device=pred_scores.device)
        B = pred_scores.size(0)
        if B >= 2:
            # 随机采样一些对进行排序约束
            num_pairs = min(B * 2, B * (B - 1) // 2)
            idx_i = torch.randint(0, B, (num_pairs,), device=pred_scores.device)
            idx_j = torch.randint(0, B, (num_pairs,), device=pred_scores.device)
            mask = idx_i != idx_j
            idx_i = idx_i[mask]
            idx_j = idx_j[mask]
            
            if len(idx_i) > 0:
                diff_pred = pred_overall_norm[idx_i] - pred_overall_norm[idx_j]
                diff_target = target_overall[idx_i] - target_overall[idx_j]
                sign_target = torch.sign(diff_target)
                # MarginRankingLoss
                loss_rank = torch.clamp(0.1 - sign_target * diff_pred, min=0).mean()
        
        total = (self.dim_weight * loss_dim +
                 self.overall_weight * loss_overall +
                 self.rank_weight * loss_rank)
        
        return {
            'total': total,
            'dim': loss_dim,
            'overall': loss_overall,
            'rank': loss_rank,
        }


# ============================================================
# 评估指标
# ============================================================

def compute_metrics(
    pred_scores: np.ndarray,
    target_scores: np.ndarray,
) -> Dict[str, float]:
    """
    计算评估指标: SRCC, LCC, MSE
    """
    from scipy import stats
    
    metrics = {}
    
    # 总分
    srcc, _ = stats.spearmanr(pred_scores, target_scores)
    lcc, _ = stats.pearsonr(pred_scores, target_scores)
    mse = np.mean((pred_scores - target_scores) ** 2)
    
    metrics['srcc'] = srcc
    metrics['lcc'] = lcc
    metrics['mse'] = mse
    
    return metrics


# ============================================================
# 训练器
# ============================================================

class AADBTrainer:
    """AADB 美学评分训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        # 输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据集
        logger.info("加载 AADB 数据集...")
        max_samples = args.max_samples if args.quick else None
        
        self.train_dataset = AADBDataset(
            args.aadb_root, split='train',
            image_size=args.image_size,
            max_samples=max_samples
        )
        self.val_dataset = AADBDataset(
            args.aadb_root, split='val',
            image_size=args.image_size
        )
        self.test_dataset = AADBDataset(
            args.aadb_root, split='test',
            image_size=args.image_size
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # 模型
        logger.info("构建模型...")
        self.model = AestheticModel(
            image_size=args.image_size,
            vision_output_dim=384,
            projection_dim=512,
            scorer_hidden_dim=256,
            num_dimensions=5,
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")
        
        # 损失
        self.criterion = AestheticLoss(
            dim_weight=1.0,
            overall_weight=1.5,
            rank_weight=0.5
        )
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
        
        # FP16
        self.scaler = None
        if args.fp16 and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        
        # 训练状态
        self.best_srcc = -1.0
        self.history = []
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        losses = {'total': 0, 'dim': 0, 'overall': 0, 'rank': 0}
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            dim_targets = batch['dim_scores'].to(self.device)
            overall_targets = batch['overall_score'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss_dict = self.criterion(
                        outputs['scores'],
                        outputs['weighted_score'],
                        dim_targets,
                        overall_targets
                    )
                self.scaler.scale(loss_dict['total']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(
                    outputs['scores'],
                    outputs['weighted_score'],
                    dim_targets,
                    overall_targets
                )
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            for k in losses:
                losses[k] += loss_dict[k].item()
            num_batches += 1
            
            if (batch_idx + 1) % self.args.log_every == 0:
                avg_loss = losses['total'] / num_batches
                logger.info(
                    f"  Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] "
                    f"loss={avg_loss:.4f}"
                )
        
        self.scheduler.step()
        
        return {k: v / max(num_batches, 1) for k, v in losses.items()}
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, name: str = 'val') -> Dict[str, float]:
        """评估"""
        self.model.eval()
        
        all_pred_overall = []
        all_target_overall = []
        all_pred_dims = []
        all_target_dims = []
        total_loss = 0
        num_batches = 0
        
        for batch in loader:
            images = batch['image'].to(self.device)
            dim_targets = batch['dim_scores'].to(self.device)
            overall_targets = batch['overall_score'].to(self.device)
            
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss_dict = self.criterion(
                        outputs['scores'],
                        outputs['weighted_score'],
                        dim_targets,
                        overall_targets
                    )
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(
                    outputs['scores'],
                    outputs['weighted_score'],
                    dim_targets,
                    overall_targets
                )
            
            total_loss += loss_dict['total'].item()
            num_batches += 1
            
            # 收集预测值 (归一化到 [0,1])
            pred_overall = (outputs['weighted_score'] / 10.0).cpu().numpy()
            target_overall = overall_targets.cpu().numpy()
            pred_dims = (outputs['scores'] / 10.0).cpu().numpy()
            target_dims = dim_targets.cpu().numpy()
            
            all_pred_overall.append(pred_overall)
            all_target_overall.append(target_overall)
            all_pred_dims.append(pred_dims)
            all_target_dims.append(target_dims)
        
        all_pred_overall = np.concatenate(all_pred_overall)
        all_target_overall = np.concatenate(all_target_overall)
        all_pred_dims = np.concatenate(all_pred_dims)
        all_target_dims = np.concatenate(all_target_dims)
        
        # 总分指标
        metrics = compute_metrics(all_pred_overall, all_target_overall)
        metrics['loss'] = total_loss / max(num_batches, 1)
        
        # 每个维度的 SRCC
        from scipy import stats
        for i, dim_name in enumerate(DIMENSION_NAMES):
            srcc_i, _ = stats.spearmanr(all_pred_dims[:, i], all_target_dims[:, i])
            metrics[f'srcc_{dim_name}'] = srcc_i
        
        return metrics
    
    def train(self):
        """完整训练流程"""
        logger.info("=" * 60)
        logger.info("AADB 美学评分器训练")
        logger.info(f"  训练集: {len(self.train_dataset)}")
        logger.info(f"  验证集: {len(self.val_dataset)}")
        logger.info(f"  测试集: {len(self.test_dataset)}")
        logger.info(f"  Epochs: {self.args.epochs}")
        logger.info(f"  Batch size: {self.args.batch_size}")
        logger.info(f"  LR: {self.args.lr}")
        logger.info(f"  Device: {self.device}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, self.args.epochs + 1):
            epoch_start = time.time()
            
            # 训练
            train_losses = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.evaluate(self.val_loader, 'val')
            
            epoch_time = time.time() - epoch_start
            
            # 记录
            record = {
                'epoch': epoch,
                'train_loss': train_losses['total'],
                'train_loss_dim': train_losses['dim'],
                'train_loss_overall': train_losses['overall'],
                'train_loss_rank': train_losses['rank'],
                'val_loss': val_metrics['loss'],
                'val_srcc': val_metrics['srcc'],
                'val_lcc': val_metrics['lcc'],
                'val_mse': val_metrics['mse'],
                'lr': self.optimizer.param_groups[0]['lr'],
                'time': epoch_time,
            }
            for dim_name in DIMENSION_NAMES:
                record[f'val_srcc_{dim_name}'] = val_metrics[f'srcc_{dim_name}']
            
            self.history.append(record)
            
            # 日志
            logger.info(
                f"Epoch {epoch}/{self.args.epochs} "
                f"| train_loss={train_losses['total']:.4f} "
                f"| val_loss={val_metrics['loss']:.4f} "
                f"| SRCC={val_metrics['srcc']:.4f} "
                f"| LCC={val_metrics['lcc']:.4f} "
                f"| MSE={val_metrics['mse']:.4f} "
                f"| {epoch_time:.1f}s"
            )
            
            # 各维度 SRCC
            dim_srcc_str = " | ".join(
                f"{d[:4]}={val_metrics[f'srcc_{d}']:.3f}"
                for d in DIMENSION_NAMES
            )
            logger.info(f"  维度 SRCC: {dim_srcc_str}")
            
            # 保存最佳模型
            if val_metrics['srcc'] > self.best_srcc:
                self.best_srcc = val_metrics['srcc']
                self._save_checkpoint('best.pt', epoch)
                logger.info(f"  ★ 新最佳 SRCC: {self.best_srcc:.4f}")
        
        # 保存最终模型
        self._save_checkpoint('last.pt', self.args.epochs)
        
        total_time = time.time() - start_time
        logger.info(f"\n训练完成! 总时间: {total_time/60:.1f} 分钟")
        logger.info(f"最佳验证 SRCC: {self.best_srcc:.4f}")
        
        # 测试集评估
        logger.info("\n" + "=" * 60)
        logger.info("测试集评估")
        logger.info("=" * 60)
        
        # 加载最佳模型
        best_ckpt = self.output_dir / 'best.pt'
        if best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state['model'])
            logger.info(f"已加载最佳模型 (epoch {state['epoch']})")
        
        test_metrics = self.evaluate(self.test_loader, 'test')
        
        logger.info(f"测试集 SRCC:  {test_metrics['srcc']:.4f}")
        logger.info(f"测试集 LCC:   {test_metrics['lcc']:.4f}")
        logger.info(f"测试集 MSE:   {test_metrics['mse']:.4f}")
        for dim_name in DIMENSION_NAMES:
            logger.info(f"  {dim_name:12s} SRCC: {test_metrics[f'srcc_{dim_name}']:.4f}")
        
        # 保存历史和测试结果 (转换 numpy float32 → Python float)
        def to_python_float(obj):
            if isinstance(obj, dict):
                return {k: to_python_float(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_python_float(v) for v in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, float):
                return float(obj)
            return obj
        
        results = {
            'history': to_python_float(self.history),
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'best_val_srcc': float(self.best_srcc),
            'config': vars(self.args),
        }
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n结果已保存到: {self.output_dir}")
        
        return test_metrics
    
    def _save_checkpoint(self, filename: str, epoch: int):
        """保存检查点"""
        path = self.output_dir / filename
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_srcc': self.best_srcc,
        }, path)


# ============================================================
# 主函数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='AADB 美学评分器训练')
    
    # 数据
    parser.add_argument('--aadb_root', type=str, default='E:/dataset/AADB',
                       help='AADB 数据集根目录')
    parser.add_argument('--image_size', type=int, default=224)
    
    # 训练
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader workers (Windows 建议 0)')
    
    # 快速验证模式
    parser.add_argument('--quick', action='store_true',
                       help='快速验证模式，只用部分训练数据')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='快速模式下的最大训练样本数')
    
    # 设备
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fp16', action='store_true', default=False)
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='checkpoints/aadb_aesthetic')
    parser.add_argument('--log_every', type=int, default=20)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("AADB 美学评分器方案验证")
    logger.info(f"参数: {vars(args)}")
    
    trainer = AADBTrainer(args)
    test_metrics = trainer.train()
    
    # 方案可行性判断
    logger.info("\n" + "=" * 60)
    logger.info("方案可行性评估")
    logger.info("=" * 60)
    
    srcc = test_metrics['srcc']
    if srcc >= 0.65:
        logger.info(f"✓ SRCC={srcc:.4f} >= 0.65, 方案可行!")
        logger.info("  美学评分器能有效学习图像质量排序")
    elif srcc >= 0.50:
        logger.info(f"△ SRCC={srcc:.4f} >= 0.50, 方案基本可行")
        logger.info("  有改进空间，可尝试增大数据集或调整超参数")
    else:
        logger.info(f"✗ SRCC={srcc:.4f} < 0.50, 需要改进")
        logger.info("  建议: 增大数据集、调整模型结构或训练策略")


if __name__ == '__main__':
    main()
