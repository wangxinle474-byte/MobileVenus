"""Stage A / Stage B 训练器。

支持:
- Stage A: 视觉-语义对齐训练
- Stage B: 参数预测训练 (冻结/解冻 backbone)
- 退化增强 + 一致性损失 (v7)
- Cosine 学习率调度 + Warmup
- 梯度裁剪 + EMA (可选)
"""

import os
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .config import DistillConfig
from .loss import AlignmentLoss, ConsistencyLoss


class StageATrainer:
    """Stage A 语义对齐训练器。"""

    def __init__(self, model, train_dataset, val_dataset=None,
                 config=None, device='cuda', output_dir='checkpoints/stage_a'):
        self.model = model.to(device)
        self.device = device
        self.config = config or DistillConfig()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.stage_a_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.stage_a_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        ) if val_dataset else None

        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.stage_a_lr,
            weight_decay=self.config.weight_decay,
        )

        # Warmup + Cosine
        warmup = LinearLR(self.optimizer, start_factor=0.1,
                          total_iters=self.config.stage_a_warmup)
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.stage_a_epochs - self.config.stage_a_warmup,
        )
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, cosine],
            milestones=[self.config.stage_a_warmup],
        )

        self.loss_fn = AlignmentLoss()
        self.best_cos_sim = -1.0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_cos = 0
        n = 0

        for batch in self.train_loader:
            images = batch['image'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)

            out = self.model(images, text_emb)
            loss, cos_sim = self.loss_fn(out['student_emb'], out['teacher_emb'])

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * images.shape[0]
            total_cos += cos_sim.item() * images.shape[0]
            n += images.shape[0]

        return total_loss / n, total_cos / n

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return 0, 0
        self.model.eval()
        total_loss = 0
        total_cos = 0
        n = 0

        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)

            out = self.model(images, text_emb)
            loss, cos_sim = self.loss_fn(out['student_emb'], out['teacher_emb'])

            total_loss += loss.item() * images.shape[0]
            total_cos += cos_sim.item() * images.shape[0]
            n += images.shape[0]

        return total_loss / n, total_cos / n

    def train(self):
        print(f"Stage A Training - {self.config.stage_a_epochs} epochs")
        for epoch in range(self.config.stage_a_epochs):
            t0 = time.time()
            train_loss, train_cos = self.train_epoch(epoch)
            val_loss, val_cos = self.validate()
            self.scheduler.step()

            dt = time.time() - t0
            print(f"Epoch {epoch+1}/{self.config.stage_a_epochs} "
                  f"[{dt:.0f}s] "
                  f"train_loss={train_loss:.4f} cos={train_cos:.4f} "
                  f"val_loss={val_loss:.4f} cos={val_cos:.4f}")

            # 保存最优
            metric = val_cos if self.val_loader else train_cos
            if metric > self.best_cos_sim:
                self.best_cos_sim = metric
                self.save_checkpoint(epoch, 'best.pt')
                print(f"  → New best cos_sim: {metric:.4f}")

        self.save_checkpoint(self.config.stage_a_epochs - 1, 'last.pt')
        print(f"Stage A done. Best cos_sim: {self.best_cos_sim:.4f}")

    def save_checkpoint(self, epoch, filename):
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_cos_sim': self.best_cos_sim,
        }, path)


class StageBTrainer:
    """Stage B 参数预测训练器。"""

    def __init__(self, model, train_dataset, val_dataset=None,
                 config=None, device='cuda',
                 output_dir='checkpoints/stage_b'):
        self.model = model.to(device)
        self.device = device
        self.config = config or DistillConfig()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.stage_b_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.stage_b_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        ) if val_dataset else None

        # 先冻结 backbone, 只训练 decoder
        self.model.freeze_backbone()
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.stage_b_lr,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config.stage_b_epochs,
        )

        self.loss_fn = nn.L1Loss()
        self.consistency_loss = ConsistencyLoss()
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        n = 0

        for batch in self.train_loader:
            images = batch['image'].to(self.device)
            target = batch['params'].to(self.device)

            out = self.model(images, return_embedding=True)
            loss = self.loss_fn(out['norm_params'], target)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * images.shape[0]
            n += images.shape[0]

        return total_loss / n

    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return float('inf')
        self.model.eval()
        total_loss = 0
        n = 0

        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            target = batch['params'].to(self.device)

            out = self.model(images)
            loss = self.loss_fn(out['norm_params'], target)

            total_loss += loss.item() * images.shape[0]
            n += images.shape[0]

        return total_loss / n

    def train(self):
        print(f"Stage B Training - {self.config.stage_b_epochs} epochs")
        for epoch in range(self.config.stage_b_epochs):
            # 可选: 解冻 backbone
            if epoch == self.config.stage_b_freeze_epochs:
                print(f"  → Unfreezing backbone at epoch {epoch}")
                self.model.unfreeze_backbone()
                # 重建 optimizer 加入 backbone 参数
                self.optimizer = AdamW(
                    self.model.get_param_groups(
                        self.config.stage_b_lr,
                        self.config.stage_b_backbone_lr_scale,
                    ),
                    weight_decay=self.config.weight_decay,
                )

            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()

            dt = time.time() - t0
            print(f"Epoch {epoch+1}/{self.config.stage_b_epochs} "
                  f"[{dt:.0f}s] "
                  f"train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, 'best.pt')
                print(f"  → New best val_loss: {val_loss:.4f}")

        self.save_checkpoint(self.config.stage_b_epochs - 1, 'last.pt')
        print(f"Stage B done. Best val_loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, epoch, filename):
        path = os.path.join(self.output_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, path)
