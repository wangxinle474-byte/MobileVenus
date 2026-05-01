"""MUSIQ 美学损失封装。

用于 RefinementNet 训练时直接最大化 MUSIQ-AVA 分数。

用法:
    loss_fn = AestheticLoss(device='cuda', target=8.5)
    loss = loss_fn(rendered_images)  # 越高越好 → loss = -score
"""

import torch
import torch.nn as nn


class AestheticLoss(nn.Module):
    """MUSIQ-AVA 美学损失。

    两种模式:
    - 'maximize': loss = -score.mean() (直接最大化)
    - 'hinge': loss = max(0, target - score).mean() (达到目标后停止)

    Args:
        device: 计算设备
        mode: 损失模式 ('maximize' 或 'hinge')
        target: hinge 模式的目标分数
        metric_name: pyiqa 指标名称
    """

    def __init__(self, device='cuda', mode='maximize', target=8.5,
                 metric_name='musiq'):
        super().__init__()
        self.mode = mode
        self.target = target
        self.device = device
        self._model = None
        self._metric_name = metric_name

    def _load_model(self):
        """延迟加载 MUSIQ 模型。"""
        if self._model is None:
            import pyiqa
            self._model = pyiqa.create_metric(
                self._metric_name, device=self.device
            )
            # 冻结参数但允许梯度流过
            for p in self._model.parameters():
                p.requires_grad = False

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) sRGB [0, 1], 建议 H,W >= 512
        Returns:
            loss: 标量
            scores: (B,) 每张图的 MUSIQ 分数
        """
        self._load_model()

        scores = self._model(images).squeeze()  # (B,)

        if self.mode == 'maximize':
            loss = -scores.mean()
        elif self.mode == 'hinge':
            loss = torch.clamp(self.target - scores, min=0).mean()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return loss, scores

    @torch.no_grad()
    def score(self, images):
        """只评分，不计算梯度。"""
        self._load_model()
        return self._model(images).squeeze()


class CLIPIQALoss(nn.Module):
    """CLIPIQA+ 自然度损失 (辅助)。

    Args:
        device: 计算设备
        weight: 损失权重
    """

    def __init__(self, device='cuda', weight=0.1):
        super().__init__()
        self.weight = weight
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            import pyiqa
            self._model = pyiqa.create_metric('clipiqa+', device=self.device)
            for p in self._model.parameters():
                p.requires_grad = False

    def forward(self, images):
        self._load_model()
        scores = self._model(images).squeeze()
        loss = -scores.mean() * self.weight
        return loss, scores
