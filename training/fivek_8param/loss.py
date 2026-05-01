"""FiveK 训练损失函数。

ConsensusWeightedLoss: 基于专家一致性加权的参数损失
CombinedLoss: 参数损失 + 图像损失 (可选)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsensusWeightedLoss(nn.Module):
    """专家共识加权参数损失。

    不同参数的专家一致性不同 (如 EV 方差大, Saturation 方差小),
    用一致性分数加权各参数的 MAE 损失。

    Args:
        num_params: 参数数量
        reduction: 'mean' 或 'sum'
    """

    def __init__(self, num_params=6, reduction='mean'):
        super().__init__()
        self.num_params = num_params
        self.reduction = reduction
        # 默认等权重, 可通过 set_weights 设置
        self.register_buffer('weights',
                             torch.ones(num_params) / num_params)

    def set_weights(self, consensus_scores):
        """设置基于专家一致性的权重。

        Args:
            consensus_scores: (num_params,) 一致性分数, 越高越一致
        """
        w = torch.tensor(consensus_scores, dtype=torch.float32)
        w = w / w.sum()
        self.weights.copy_(w)

    def forward(self, pred, target):
        """
        Args:
            pred: (B, num_params) 预测参数 (归一化)
            target: (B, num_params) GT 参数 (归一化)
        Returns:
            加权 MAE 标量
        """
        mae = torch.abs(pred - target)  # (B, num_params)
        weighted = mae * self.weights.unsqueeze(0)

        if self.reduction == 'mean':
            return weighted.mean()
        return weighted.sum()


class CombinedLoss(nn.Module):
    """组合损失: 参数损失 + 图像损失 (可选)。

    Args:
        param_weight: 参数损失权重
        image_weight: 图像损失权重 (L1)
        ssim_weight: SSIM 损失权重
        confidence_weight: 置信度正则化权重
    """

    def __init__(self, param_weight=1.0, image_weight=0.0,
                 ssim_weight=0.0, confidence_weight=0.01):
        super().__init__()
        self.param_weight = param_weight
        self.image_weight = image_weight
        self.ssim_weight = ssim_weight
        self.confidence_weight = confidence_weight
        self.param_loss_fn = ConsensusWeightedLoss()

    def forward(self, pred_params, target_params, confidence=None,
                rendered=None, target_image=None):
        """
        Args:
            pred_params: (B, 6) 预测参数
            target_params: (B, 6) GT 参数
            confidence: (B, 6) 置信度 (可选)
            rendered: (B, 3, H, W) 渲染图像 (可选)
            target_image: (B, 3, H, W) GT 图像 (可选)
        Returns:
            dict: total, param_loss, image_loss, ssim_loss, conf_loss
        """
        losses = {}

        # 参数损失
        param_loss = self.param_loss_fn(pred_params, target_params)
        losses['param_loss'] = param_loss
        total = self.param_weight * param_loss

        # 图像损失
        if self.image_weight > 0 and rendered is not None and target_image is not None:
            img_loss = F.l1_loss(rendered, target_image)
            losses['image_loss'] = img_loss
            total = total + self.image_weight * img_loss

        # SSIM 损失
        if self.ssim_weight > 0 and rendered is not None and target_image is not None:
            ssim_loss = 1.0 - _ssim(rendered, target_image)
            losses['ssim_loss'] = ssim_loss
            total = total + self.ssim_weight * ssim_loss

        # 置信度正则化 (鼓励高置信度)
        if self.confidence_weight > 0 and confidence is not None:
            conf_loss = -confidence.mean()
            losses['conf_loss'] = conf_loss
            total = total + self.confidence_weight * conf_loss

        losses['total'] = total
        return losses


def _ssim(x, y, window_size=11):
    """简化版 SSIM 计算。"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size // 2)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(x ** 2, window_size, stride=1,
                               padding=window_size // 2) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y ** 2, window_size, stride=1,
                               padding=window_size // 2) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1,
                             padding=window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    return ssim_map.mean()
