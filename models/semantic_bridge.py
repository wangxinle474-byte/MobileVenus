"""语义桥接模块: SemanticProjector + LightroomDecoder + TextProjector。

核心创新组件，连接视觉特征空间与语义空间，并从语义空间解码 ISP 参数。

SemanticProjector: 384-D 视觉特征 → 256-D L2 归一化语义向量
TextProjector:     384-D 文本 embedding → 256-D L2 归一化语义向量
LightroomDecoder:  256-D 语义向量 → 6 个 Lightroom 参数 + 置信度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticProjector(nn.Module):
    """视觉特征 → 语义空间投影。

    渐进式 MLP 投影，将视觉编码器的全局特征映射到
    与文本对齐的统一语义空间。

    Args:
        visual_dim: 输入视觉特征维度 (默认 384)
        semantic_dim: 输出语义空间维度 (默认 256)
        hidden_dim: 中间层维度 (默认 None, 取两者均值)
        dropout: Dropout 比率
    """

    def __init__(self, visual_dim=384, semantic_dim=256, hidden_dim=None,
                 dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (visual_dim + semantic_dim) // 2

        self.projector = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, visual_feat):
        """
        Args:
            visual_feat: (B, visual_dim) 视觉编码器输出
        Returns:
            (B, semantic_dim) L2 归一化的语义向量
        """
        projected = self.projector(visual_feat)
        return F.normalize(projected, p=2, dim=-1)


class TextProjector(nn.Module):
    """文本 embedding → 语义空间投影 (Stage A 教师侧)。

    将预训练 MiniLM-L6-v2 的 384-D 文本 embedding
    投影到与视觉侧相同的 256-D 语义空间。

    Args:
        text_dim: 输入文本 embedding 维度 (默认 384, MiniLM)
        semantic_dim: 输出语义空间维度 (默认 256)
        hidden_dim: 中间层维度
        dropout: Dropout 比率
    """

    def __init__(self, text_dim=384, semantic_dim=256, hidden_dim=None,
                 dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (text_dim + semantic_dim) // 2

        self.projector = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, semantic_dim),
            nn.LayerNorm(semantic_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, text_emb):
        """
        Args:
            text_emb: (B, text_dim) MiniLM 文本 embedding
        Returns:
            (B, semantic_dim) L2 归一化的语义向量
        """
        projected = self.projector(text_emb)
        return F.normalize(projected, p=2, dim=-1)


class LightroomDecoder(nn.Module):
    """语义向量 → 6 个 Lightroom ISP 参数 + 置信度。

    共享 MLP backbone + 6 个独立参数头 + 置信度头。

    6 个参数:
        - ev_compensation:  [-3, +3]       (Tanh × 3)
        - white_balance:    [2000, 10000] K (Sigmoid × 8000 + 2000)
        - contrast:         [-100, 100]    (Tanh × 100)
        - shadows:          [-100, 100]    (Tanh × 100)
        - highlights:       [-100, 100]    (Tanh × 100)
        - saturation:       [-100, 100]    (Tanh × 100)

    Args:
        semantic_dim: 输入语义向量维度 (默认 256)
        hidden_dim: 共享 MLP 隐藏层维度 (默认 256)
    """

    PARAM_NAMES = [
        'ev_compensation', 'white_balance', 'contrast',
        'shadows', 'highlights', 'saturation',
    ]

    def __init__(self, semantic_dim=256, hidden_dim=256):
        super().__init__()
        self.num_params = 6

        # 共享 backbone
        self.shared = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 6 个独立参数头
        self.param_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )
            for _ in range(self.num_params)
        ])

        # 置信度头
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, self.num_params),
            nn.Sigmoid(),
        )

    def forward(self, semantic_emb):
        """
        Args:
            semantic_emb: (B, semantic_dim) 语义向量
        Returns:
            dict with:
                'raw_params': {name: (B, 1) tensor} 实际范围参数
                'norm_params': (B, 6) 归一化参数 [-1, 1]
                'confidence': (B, 6) 每个参数的置信度
        """
        h = self.shared(semantic_emb)

        # 各参数预测
        raw_outputs = [head(h) for head in self.param_heads]
        norm_params = torch.cat(raw_outputs, dim=-1)  # (B, 6) tanh 前

        # 置信度
        confidence = self.confidence_head(h)  # (B, 6)

        # 映射到实际范围
        raw_params = {}
        activations = norm_params.split(1, dim=-1)

        # EV: tanh × 3 → [-3, 3]
        raw_params['ev_compensation'] = torch.tanh(activations[0]) * 3.0

        # WB: sigmoid × 8000 + 2000 → [2000, 10000]
        raw_params['white_balance'] = torch.sigmoid(activations[1]) * 8000.0 + 2000.0

        # Contrast: tanh × 100 → [-100, 100]
        raw_params['contrast'] = torch.tanh(activations[2]) * 100.0

        # Shadows: tanh × 100 → [-100, 100]
        raw_params['shadows'] = torch.tanh(activations[3]) * 100.0

        # Highlights: tanh × 100 → [-100, 100]
        raw_params['highlights'] = torch.tanh(activations[4]) * 100.0

        # Saturation: tanh × 100 → [-100, 100]
        raw_params['saturation'] = torch.tanh(activations[5]) * 100.0

        return {
            'raw_params': raw_params,
            'norm_params': torch.tanh(norm_params),
            'confidence': confidence,
        }


class ParameterDecoder(LightroomDecoder):
    """LightroomDecoder 的别名，向后兼容。"""
    pass


if __name__ == '__main__':
    proj = SemanticProjector(384, 256)
    dec = LightroomDecoder(256)
    text_proj = TextProjector(384, 256)

    visual_feat = torch.randn(2, 384)
    text_emb = torch.randn(2, 384)

    sem = proj(visual_feat)
    print(f"SemanticProjector: {visual_feat.shape} → {sem.shape}")
    print(f"  L2 norm: {sem.norm(dim=-1)}")

    teacher = text_proj(text_emb)
    print(f"TextProjector: {text_emb.shape} → {teacher.shape}")

    out = dec(sem)
    print(f"LightroomDecoder output:")
    for name, val in out['raw_params'].items():
        print(f"  {name}: {val.shape}, range [{val.min():.2f}, {val.max():.2f}]")
    print(f"  confidence: {out['confidence'].shape}")

    total = sum(p.numel() for p in proj.parameters()) + \
            sum(p.numel() for p in dec.parameters())
    print(f"Projector + Decoder params: {total / 1e3:.1f}K")
