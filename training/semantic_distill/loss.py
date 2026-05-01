"""Stage A / Stage B 损失函数。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLoss(nn.Module):
    """Stage A 视觉-语义对齐损失。

    loss = 1 - cosine_similarity(student_emb, teacher_emb)
    """

    def __init__(self):
        super().__init__()

    def forward(self, student_emb, teacher_emb):
        cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
        return (1.0 - cos_sim).mean(), cos_sim.mean()


class ContrastiveAlignmentLoss(nn.Module):
    """对比学习对齐损失 (InfoNCE)。

    在 batch 内做正负样本对比，增强区分性。

    Args:
        temperature: 温度系数
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_emb, teacher_emb):
        """
        Args:
            student_emb: (B, D) L2 归一化
            teacher_emb: (B, D) L2 归一化
        Returns:
            loss, cos_sim (正样本对平均)
        """
        # 相似度矩阵
        logits = student_emb @ teacher_emb.T / self.temperature  # (B, B)
        labels = torch.arange(logits.shape[0], device=logits.device)

        # 双向对比
        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.T, labels)
        loss = (loss_s2t + loss_t2s) / 2

        # 正样本对余弦相似度
        cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1).mean()

        return loss, cos_sim


class ConsistencyLoss(nn.Module):
    """退化一致性损失 (v7)。

    原图和退化图的语义 embedding 应该相近。

    loss = ||emb_orig - emb_degraded||_2
    """

    def __init__(self):
        super().__init__()

    def forward(self, emb_orig, emb_degraded):
        return F.mse_loss(emb_orig, emb_degraded)
