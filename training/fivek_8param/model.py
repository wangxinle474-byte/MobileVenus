"""FiveK 8参数预测模型 (Stage B)。

组装 MobileViTSmall + SemanticProjector + LightroomDecoder
实现端到端的图像 → 6 个 Lightroom ISP 参数预测。

参数量: ~2.1M (不含 frozen backbone 时 ~0.2M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision_encoder import MobileViTSmall
from models.semantic_bridge import SemanticProjector, LightroomDecoder
from .config import PARAM_NAMES


class FiveK8ParamModel(nn.Module):
    """FiveK 参数预测完整模型。

    Pipeline:
        Image → MobileViTSmall → SemanticProjector → LightroomDecoder → 6 params

    Args:
        image_size: 输入图片大小
        visual_dim: 视觉编码器输出维度
        semantic_dim: 语义空间维度
        use_se: 视觉编码器是否使用 SE 注意力
        use_fpn: 视觉编码器是否使用 FPN
    """

    def __init__(self, image_size=224, visual_dim=384, semantic_dim=256,
                 use_se=True, use_fpn=True):
        super().__init__()
        self.vision_encoder = MobileViTSmall(
            image_size=image_size,
            output_dim=visual_dim,
            use_se=use_se,
            use_fpn=use_fpn,
        )
        self.semantic_projector = SemanticProjector(
            visual_dim=visual_dim,
            semantic_dim=semantic_dim,
        )
        self.decoder = LightroomDecoder(
            semantic_dim=semantic_dim,
        )

    def forward(self, images, return_embedding=False):
        """
        Args:
            images: (B, 3, H, W) 输入图片
            return_embedding: 是否同时返回语义向量
        Returns:
            dict with 'raw_params', 'norm_params', 'confidence'
            如果 return_embedding=True, 额外包含 'semantic_emb'
        """
        visual_feat = self.vision_encoder(images)         # (B, 384)
        semantic_emb = self.semantic_projector(visual_feat)  # (B, 256)
        outputs = self.decoder(semantic_emb)                 # dict

        if return_embedding:
            outputs['semantic_emb'] = semantic_emb

        return outputs

    def freeze_backbone(self):
        """冻结视觉编码器和语义投影层 (Stage B 训练时)。"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.semantic_projector.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, lr_scale=0.1):
        """解冻 backbone，通常用更小的学习率。"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        for param in self.semantic_projector.parameters():
            param.requires_grad = True

    def get_param_groups(self, lr, backbone_lr_scale=0.1):
        """返回分组学习率的参数组。"""
        backbone_params = list(self.vision_encoder.parameters()) + \
                          list(self.semantic_projector.parameters())
        decoder_params = list(self.decoder.parameters())

        return [
            {'params': backbone_params, 'lr': lr * backbone_lr_scale},
            {'params': decoder_params, 'lr': lr},
        ]


if __name__ == '__main__':
    model = FiveK8ParamModel(image_size=224)
    x = torch.randn(2, 3, 224, 224)
    out = model(x, return_embedding=True)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Semantic embedding: {out['semantic_emb'].shape}")
    for name in PARAM_NAMES:
        print(f"  {name}: {out['raw_params'][name].item():.3f}")
    print(f"Confidence: {out['confidence']}")
