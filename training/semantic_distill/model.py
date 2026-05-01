"""语义蒸馏模型定义。

SemanticDistillModel: Stage A 视觉-语义对齐模型
DistillParamModel:    Stage B 参数预测模型 (复用 Stage A backbone)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vision_encoder import MobileViTSmall
from models.semantic_bridge import SemanticProjector, TextProjector, LightroomDecoder


class SemanticDistillModel(nn.Module):
    """Stage A: 视觉-语义对齐模型。

    双塔结构:
        Student: Image → MobileViT → SemanticProjector → visual_emb (256-D)
        Teacher: Text  → TextProjector → text_emb (256-D)
    训练目标: cosine_similarity(visual_emb, text_emb) → 1

    Args:
        image_size: 输入图片大小
        visual_dim: 视觉编码器输出维度
        semantic_dim: 语义空间维度
        text_dim: 文本 embedding 维度 (MiniLM = 384)
    """

    def __init__(self, image_size=224, visual_dim=384, semantic_dim=256,
                 text_dim=384):
        super().__init__()
        self.vision_encoder = MobileViTSmall(
            image_size=image_size,
            output_dim=visual_dim,
        )
        self.semantic_projector = SemanticProjector(
            visual_dim=visual_dim,
            semantic_dim=semantic_dim,
        )
        self.text_projector = TextProjector(
            text_dim=text_dim,
            semantic_dim=semantic_dim,
        )

    def forward(self, images, text_emb=None):
        """
        Args:
            images: (B, 3, H, W) 输入图片
            text_emb: (B, text_dim) 预计算的 MiniLM 文本 embedding
        Returns:
            dict with:
                'student_emb': (B, semantic_dim) 视觉语义向量
                'teacher_emb': (B, semantic_dim) 文本语义向量 (如果提供 text_emb)
                'cos_sim': 余弦相似度标量 (如果提供 text_emb)
        """
        visual_feat = self.vision_encoder(images)
        student_emb = self.semantic_projector(visual_feat)

        result = {'student_emb': student_emb}

        if text_emb is not None:
            teacher_emb = self.text_projector(text_emb)
            result['teacher_emb'] = teacher_emb

            # 余弦相似度
            cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
            result['cos_sim'] = cos_sim.mean()

        return result

    def alignment_loss(self, student_emb, teacher_emb):
        """计算对齐损失 (1 - cosine_similarity)。"""
        cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
        return (1.0 - cos_sim).mean()


class DistillParamModel(nn.Module):
    """Stage B: 参数预测模型。

    复用 Stage A 训练好的 vision_encoder + semantic_projector，
    新增 LightroomDecoder 预测 6 个 ISP 参数。

    Pipeline:
        Image → MobileViT(冻结) → SemanticProjector(冻结) → LightroomDecoder → 6 params

    Args:
        stage_a_model: 训练好的 SemanticDistillModel (或 None)
        image_size: 输入图片大小
        visual_dim: 视觉编码器输出维度
        semantic_dim: 语义空间维度
    """

    def __init__(self, stage_a_model=None, image_size=224, visual_dim=384,
                 semantic_dim=256, decoder_hidden=256):
        super().__init__()

        if stage_a_model is not None:
            self.vision_encoder = stage_a_model.vision_encoder
            self.semantic_projector = stage_a_model.semantic_projector
        else:
            self.vision_encoder = MobileViTSmall(
                image_size=image_size,
                output_dim=visual_dim,
            )
            self.semantic_projector = SemanticProjector(
                visual_dim=visual_dim,
                semantic_dim=semantic_dim,
            )

        self.decoder = LightroomDecoder(
            semantic_dim=semantic_dim,
            hidden_dim=decoder_hidden,
        )

    def forward(self, images, return_embedding=False):
        """
        Args:
            images: (B, 3, H, W) 输入图片
            return_embedding: 是否返回语义向量
        Returns:
            dict with 'raw_params', 'norm_params', 'confidence'
        """
        visual_feat = self.vision_encoder(images)
        semantic_emb = self.semantic_projector(visual_feat)
        outputs = self.decoder(semantic_emb)

        if return_embedding:
            outputs['semantic_emb'] = semantic_emb

        return outputs

    def freeze_backbone(self):
        """冻结 Stage A 的 backbone。"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.semantic_projector.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻 backbone (通常在训练后期)。"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        for param in self.semantic_projector.parameters():
            param.requires_grad = True

    def get_param_groups(self, lr, backbone_lr_scale=0.1):
        """分组学习率参数。"""
        backbone_params = list(self.vision_encoder.parameters()) + \
                          list(self.semantic_projector.parameters())
        decoder_params = list(self.decoder.parameters())
        return [
            {'params': backbone_params, 'lr': lr * backbone_lr_scale},
            {'params': decoder_params, 'lr': lr},
        ]

    @classmethod
    def from_stage_a(cls, checkpoint_path, device='cpu', **kwargs):
        """从 Stage A checkpoint 构建 Stage B 模型。"""
        # 分离 Stage A 和 Stage B 专属参数
        decoder_hidden = kwargs.pop('decoder_hidden', 256)
        stage_a = SemanticDistillModel(**kwargs)
        state = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state:
            stage_a.load_state_dict(state['model_state_dict'])
        else:
            stage_a.load_state_dict(state)

        model = cls(stage_a_model=stage_a, decoder_hidden=decoder_hidden, **kwargs)
        model.freeze_backbone()
        return model


if __name__ == '__main__':
    # Stage A
    model_a = SemanticDistillModel()
    images = torch.randn(2, 3, 224, 224)
    text_emb = torch.randn(2, 384)
    out_a = model_a(images, text_emb)
    print(f"Stage A - cos_sim: {out_a['cos_sim']:.4f}")
    print(f"  student_emb: {out_a['student_emb'].shape}")
    print(f"  teacher_emb: {out_a['teacher_emb'].shape}")

    # Stage B
    model_b = DistillParamModel(stage_a_model=model_a)
    model_b.freeze_backbone()
    out_b = model_b(images, return_embedding=True)
    print(f"\nStage B params: {sum(p.numel() for p in model_b.parameters()) / 1e6:.2f}M")
    print(f"  trainable: {sum(p.numel() for p in model_b.parameters() if p.requires_grad) / 1e3:.1f}K")
    for name, val in out_b['raw_params'].items():
        print(f"  {name}: {val.item():.3f}")
