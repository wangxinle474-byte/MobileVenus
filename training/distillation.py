"""
知识蒸馏训练脚本
从 Venus 教师模型蒸馏到 MobileVenus 学生模型
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import yaml
from tqdm import tqdm
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models import MobileVenus, MobileVenusConfig


class DistillationLoss(nn.Module):
    """
    多目标蒸馏损失
    """
    
    def __init__(self, temperature=4.0, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # 评分损失权重
        self.beta = beta    # 特征损失权重
        self.gamma = gamma  # 响应损失权重
        
    def forward(self, student_outputs, teacher_outputs):
        """
        计算蒸馏损失
        
        Args:
            student_outputs: dict with 'scores', 'features'
            teacher_outputs: dict with 'scores', 'features'
            
        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失字典
        """
        # 1. 评分蒸馏 (MSE)
        loss_score = F.mse_loss(
            student_outputs['scores'],
            teacher_outputs['scores']
        )
        
        # 2. 特征蒸馏 (Cosine Embedding)
        student_feat = F.normalize(student_outputs['features'], dim=-1)
        teacher_feat = F.normalize(teacher_outputs['features'], dim=-1)
        loss_feature = 1 - (student_feat * teacher_feat).sum(dim=-1).mean()
        
        # 3. 响应蒸馏 (KL Divergence)
        student_probs = F.log_softmax(
            student_outputs['scores'] / self.temperature, dim=-1
        )
        teacher_probs = F.softmax(
            teacher_outputs['scores'] / self.temperature, dim=-1
        )
        loss_response = F.kl_div(
            student_probs, teacher_probs, reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 总损失
        total_loss = (
            self.alpha * loss_score +
            self.beta * loss_feature +
            self.gamma * loss_response
        )
        
        loss_dict = {
            'loss_total': total_loss.item(),
            'loss_score': loss_score.item(),
            'loss_feature': loss_feature.item(),
            'loss_response': loss_response.item()
        }
        
        return total_loss, loss_dict


class DistillationTrainer:
    """
    知识蒸馏训练器
    """
    
    def __init__(self, config_path):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self._init_models()
        
        # 初始化优化器
        self._init_optimizer()
        
        # 初始化损失函数
        self.criterion = DistillationLoss(
            temperature=self.config['distillation']['temperature'],
            alpha=self.config['distillation']['alpha_score'],
            beta=self.config['distillation']['alpha_feature'],
            gamma=self.config['distillation']['alpha_response']
        )
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def _init_models(self):
        """初始化教师和学生模型"""
        print("Initializing models...")
        
        # 教师模型 (Venus)
        teacher_path = self.config['distillation']['teacher']['model_path']
        print(f"Loading teacher model from {teacher_path}")
        
        # 这里需要加载 Venus 模型
        # 由于 Venus 代码在另一个目录，这里用占位符
        # 实际使用时需要导入 Venus 模型
        self.teacher = None  # 占位符
        
        if self.teacher is not None:
            self.teacher.eval()
            self.teacher.to(self.device)
            # 冻结教师模型
            for param in self.teacher.parameters():
                param.requires_grad = False
        
        # 学生模型 (MobileVenus)
        print("Creating student model...")
        student_config = MobileVenusConfig(
            image_size=224,
            dropout=0.1,
            scorer_hidden_dim=256,
            use_language_model=True
        )
        self.student = MobileVenus(student_config)
        self.student.to(self.device)
        
        print(f"Student model parameters: {self.student.get_num_params()['total']:,}")
        
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        train_config = self.config['distillation']['training']
        
        # 优化器
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        # 学习率调度器
        num_epochs = train_config['num_epochs']
        steps_per_epoch = 1000  # 占位符，实际应根据数据集大小计算
        total_steps = num_epochs * steps_per_epoch
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=train_config['warmup_steps'],
            num_training_steps=total_steps
        )
        
    def train_epoch(self, train_loader):
        """训练一个 epoch"""
        self.student.train()
        
        epoch_loss = 0.0
        epoch_metrics = {
            'loss_score': 0.0,
            'loss_feature': 0.0,
            'loss_response': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            
            # 教师模型前向传播 (无梯度)
            with torch.no_grad():
                if self.teacher is not None:
                    teacher_scores, _, teacher_features = self.teacher(
                        images, return_features=True
                    )
                else:
                    # 占位符：使用随机值模拟教师输出
                    teacher_scores = torch.rand(images.shape[0], 5).to(self.device) * 10
                    teacher_features = torch.randn(images.shape[0], 512).to(self.device)
            
            # 学生模型前向传播
            student_scores, _, student_features = self.student(
                images, return_features=True
            )
            
            # 计算损失
            loss, loss_dict = self.criterion(
                {'scores': student_scores, 'features': student_features},
                {'scores': teacher_scores, 'features': teacher_features}
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 更新指标
            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += loss_dict[key]
            
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
        
        # 计算平均指标
        num_batches = len(train_loader)
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_loss, epoch_metrics
    
    def validate(self, val_loader):
        """验证"""
        self.student.eval()
        
        val_loss = 0.0
        val_metrics = {
            'loss_score': 0.0,
            'loss_feature': 0.0,
            'loss_response': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                
                # 教师模型
                if self.teacher is not None:
                    teacher_scores, _, teacher_features = self.teacher(
                        images, return_features=True
                    )
                else:
                    teacher_scores = torch.rand(images.shape[0], 5).to(self.device) * 10
                    teacher_features = torch.randn(images.shape[0], 512).to(self.device)
                
                # 学生模型
                student_scores, _, student_features = self.student(
                    images, return_features=True
                )
                
                # 计算损失
                loss, loss_dict = self.criterion(
                    {'scores': student_scores, 'features': student_features},
                    {'scores': teacher_scores, 'features': teacher_features}
                )
                
                val_loss += loss.item()
                for key in val_metrics:
                    val_metrics[key] += loss_dict[key]
        
        # 计算平均
        num_batches = len(val_loader)
        val_loss /= num_batches
        for key in val_metrics:
            val_metrics[key] /= num_batches
        
        return val_loss, val_metrics
    
    def save_checkpoint(self, save_path, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = save_path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train(self, train_loader, val_loader, output_dir):
        """完整训练流程"""
        os.makedirs(output_dir, exist_ok=True)
        
        num_epochs = self.config['distillation']['training']['num_epochs']
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Output directory: {output_dir}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # 训练
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
            for key, value in train_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # 验证
            val_loss, val_metrics = self.validate(val_loader)
            
            print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            self.save_checkpoint(checkpoint_path, is_best=is_best)
            
            print(f"Saved checkpoint to {checkpoint_path}")
            print("-" * 50)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MobileVenus Distillation Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    args = parser.parse_args()
    
    # 创建训练器
    trainer = DistillationTrainer(args.config)
    
    # 创建数据加载器（占位符）
    # 实际使用时需要实现数据集类
    print("Note: Using placeholder data loaders. Implement actual dataset for real training.")
    
    # 这里应该创建真实的数据加载器
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 占位符数据加载器
    class DummyDataset:
        def __len__(self):
            return 100
        
        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, 224, 224),
                'scores': torch.rand(5) * 10
            }
    
    train_dataset = DummyDataset()
    val_dataset = DummyDataset()
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 开始训练
    trainer.train(train_loader, val_loader, args.output_dir)


if __name__ == "__main__":
    main()
