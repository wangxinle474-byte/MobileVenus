"""美学评分器模块。

基于 AADB 数据集训练的多属性美学评分器。
用于训练过程中的美学质量监控。

输入: (B, 3, 224, 224) sRGB 图片
输出: overall score + 11 维属性分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class AestheticScorer(nn.Module):
    """AADB 美学评分器。

    基于 ResNet-18 backbone + 多属性预测头。
    AADB 11 维属性: BalancingElements, ColorHarmony, Content,
    DoF, Light, Object, RuleOfThirds, Symmetry, VividColor,
    Repetition, TextureDetail

    Args:
        backbone: 预训练 backbone 类型
        num_attributes: 属性数量
        pretrained: 是否使用 ImageNet 预训练权重
    """

    ATTRIBUTE_NAMES = [
        'BalancingElements', 'ColorHarmony', 'Content',
        'DoF', 'Light', 'Object', 'RuleOfThirds',
        'Symmetry', 'VividColor', 'Repetition', 'TextureDetail',
    ]

    def __init__(self, backbone='resnet18', num_attributes=11,
                 pretrained=True):
        super().__init__()
        self.num_attributes = num_attributes

        # Backbone
        if backbone == 'resnet18':
            base = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 512
        elif backbone == 'resnet50':
            base = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 去掉分类头
        self.features = nn.Sequential(*list(base.children())[:-1])

        # 属性预测头
        self.attribute_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_attributes),
            nn.Sigmoid(),
        )

        # 整体评分头 (从属性聚合)
        self.score_head = nn.Sequential(
            nn.Linear(num_attributes + feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        """
        Args:
            images: (B, 3, 224, 224) sRGB 图片 (ImageNet 归一化)
        Returns:
            dict:
                'score': (B, 1) 整体美学分 [0, 1]
                'attributes': (B, 11) 各属性分 [0, 1]
                'features': (B, feat_dim) 特征向量
        """
        feat = self.features(images).flatten(1)  # (B, feat_dim)
        attributes = self.attribute_head(feat)     # (B, 11)

        # 整体分 = f(attributes, features)
        score_input = torch.cat([attributes, feat], dim=-1)
        score = self.score_head(score_input)        # (B, 1)

        return {
            'score': score,
            'attributes': attributes,
            'features': feat,
        }

    def predict_score(self, images):
        """简化接口: 只返回整体分。"""
        return self.forward(images)['score']


class MUSIQWrapper(nn.Module):
    """MUSIQ-AVA 模型封装。

    封装 pyiqa 的 MUSIQ 模型，用于训练时的美学监控。
    注意: MUSIQ 需要 512×512+ 输入才准确。

    Args:
        device: 计算设备
        metric_name: pyiqa 指标名称
    """

    def __init__(self, device='cuda', metric_name='musiq'):
        super().__init__()
        try:
            import pyiqa
            self.model = pyiqa.create_metric(metric_name, device=device)
        except ImportError:
            print("Warning: pyiqa not installed, MUSIQWrapper disabled")
            self.model = None
        self.device = device

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) sRGB [0, 1]
        Returns:
            (B,) MUSIQ-AVA 分数
        """
        if self.model is None:
            return torch.zeros(images.shape[0], device=images.device)
        with torch.no_grad():
            scores = self.model(images)
        return scores.squeeze()

    @torch.no_grad()
    def score_batch(self, images):
        """批量评分 (no_grad)。"""
        return self.forward(images)
