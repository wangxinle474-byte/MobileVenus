"""Stage C: 文本条件化 ISP 参数预测模型。

核心组件:
- LightTextEncoder: 轻量 Transformer 文本编码器 (vocab=5000, 2层4头)
- FiLMFusion: FiLM 调制融合 (文本条件化视觉特征)
- CrossAttentionFusion: 交叉注意力融合 (备选方案)
- TextConditionedModel: 完整 Stage C 模型

设计原则:
- 冻结 Stage B backbone, 只训练新增模块
- FiLM 初始化为恒等变换, 保证初始行为等价于 Stage B
- 无文本输入时优雅退化为纯视觉预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.semantic_bridge import LightroomDecoder


class LightTextEncoder(nn.Module):
    """轻量文本编码器 — 直接理解口语化中文指令。

    不依赖预训练语言模型, 完全自主训练。
    使用字符级 tokenize, 2 层 Transformer, CLS pooling。

    Args:
        vocab_size: 词表大小 (字符级)
        embed_dim: 嵌入维度
        hidden_dim: Transformer FFN 隐藏维度
        num_heads: 注意力头数
        num_layers: Transformer 层数
        max_len: 最大序列长度
        output_dim: 输出特征维度
        dropout: Dropout 比率
    """

    def __init__(self, vocab_size=5000, embed_dim=128, hidden_dim=256,
                 num_heads=4, num_layers=2, max_len=64, output_dim=256,
                 dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # 字符嵌入 + 位置编码
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 投影到语义空间
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        for p in self.projector.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, token_ids):
        """
        Args:
            token_ids: (B, L) 字符 token IDs, 0 为 padding
        Returns:
            (B, output_dim) L2 归一化的文本特征
        """
        B, L = token_ids.shape

        # 嵌入
        positions = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.token_embed(token_ids) + self.pos_embed(positions)

        # Padding mask
        padding_mask = (token_ids == 0)  # True = 忽略

        # Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)

        # CLS pooling (取第 0 位)
        cls_feat = x[:, 0, :]  # (B, embed_dim)

        # 投影
        out = self.projector(cls_feat)  # (B, output_dim)
        return F.normalize(out, p=2, dim=-1)


class FiLMFusion(nn.Module):
    """FiLM (Feature-wise Linear Modulation) 融合层。

    用文本特征生成 scale (γ) 和 shift (β) 来调制视觉特征:
        fused = γ ⊙ visual_emb + β

    关键设计:
    - 初始化为恒等变换 (γ≈1, β≈0), 保证初始行为等价于无文本
    - γ 范围 [0.5, 1.5], β 范围 [-0.3, 0.3], 防止极端调制
    - 无文本输入时直接返回 visual_emb

    Args:
        text_dim: 文本特征维度
        visual_dim: 视觉特征维度
        hidden_dim: 中间层维度
    """

    def __init__(self, text_dim=256, visual_dim=256, hidden_dim=128):
        super().__init__()

        self.gamma_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, visual_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, visual_dim),
        )

        # 初始化为恒等变换
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.zeros_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, visual_emb, text_emb=None):
        """
        Args:
            visual_emb: (B, visual_dim) 视觉语义向量
            text_emb: (B, text_dim) 文本特征 (可选)
        Returns:
            (B, visual_dim) 融合后特征
        """
        if text_emb is None:
            return visual_emb

        # γ ∈ [0.5, 1.5], β ∈ [-0.3, 0.3]
        gamma = 1.0 + self.gamma_net(text_emb) * 0.5
        beta = self.beta_net(text_emb) * 0.3

        return gamma * visual_emb + beta


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合 (备选方案)。

    文本作 query, 视觉作 key/value, 门控残差连接。
    初始 gate ≈ 0, 保持 Stage B 行为。

    Args:
        dim: 特征维度
        num_heads: 注意力头数
    """

    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=0.1
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # 门控参数, 初始化为小值 → 初始行为 ≈ 纯视觉
        self.gate = nn.Parameter(torch.zeros(1) - 3.0)  # sigmoid(-3) ≈ 0.05

    def forward(self, visual_emb, text_emb=None):
        """
        Args:
            visual_emb: (B, dim) 视觉特征
            text_emb: (B, dim) 文本特征
        Returns:
            (B, dim) 融合特征
        """
        if text_emb is None:
            return visual_emb

        # 扩展为序列维度 (B, 1, dim)
        q = self.norm_q(text_emb.unsqueeze(1))
        kv = self.norm_kv(visual_emb.unsqueeze(1))

        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = attn_out.squeeze(1)  # (B, dim)

        gate = torch.sigmoid(self.gate)
        return visual_emb + gate * attn_out


class TextConditionedModel(nn.Module):
    """Stage C: 文本条件化 ISP 参数预测模型。

    在 Stage B 基础上新增文本分支:
    - 冻结 vision_encoder + semantic_projector (从 Stage B)
    - 新增 LightTextEncoder + FiLMFusion + cond_decoder

    Pipeline:
        Image → MobileViT(冻结) → SemanticProjector(冻结) → visual_emb
                                                                ↓
        Text  → LightTextEncoder(训练) → text_emb → FiLM融合 → fused_emb
                                                                ↓
                                                        cond_decoder → 6 params

    Args:
        stage_b_model: 训练好的 Stage B 模型
        vocab_size: 文本词表大小
        text_embed_dim: 文本嵌入维度
        text_hidden_dim: 文本 Transformer FFN 维度
        text_num_heads: 文本注意力头数
        text_num_layers: 文本 Transformer 层数
        max_text_len: 最大文本长度
        semantic_dim: 语义空间维度
        fusion_type: 融合方式 ('film' 或 'cross_attention')
    """

    def __init__(self, stage_b_model=None, vocab_size=5000,
                 text_embed_dim=128, text_hidden_dim=256,
                 text_num_heads=4, text_num_layers=2,
                 max_text_len=64, semantic_dim=256,
                 fusion_type='film'):
        super().__init__()
        self.fusion_type = fusion_type

        # 从 Stage B 复制 backbone (冻结)
        if stage_b_model is not None:
            self.vision_encoder = stage_b_model.vision_encoder
            self.semantic_projector = stage_b_model.semantic_projector
            # 保存 Stage B decoder 作为基准 (冻结)
            self.base_decoder = stage_b_model.decoder
        else:
            from models.vision_encoder import MobileViTSmall
            from models.semantic_bridge import SemanticProjector
            self.vision_encoder = MobileViTSmall(output_dim=384)
            self.semantic_projector = SemanticProjector(384, semantic_dim)
            self.base_decoder = LightroomDecoder(semantic_dim)

        # 新增: 文本编码器
        self.text_encoder = LightTextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            hidden_dim=text_hidden_dim,
            num_heads=text_num_heads,
            num_layers=text_num_layers,
            max_len=max_text_len,
            output_dim=semantic_dim,
        )

        # 新增: 融合层
        if fusion_type == 'film':
            self.fusion = FiLMFusion(
                text_dim=semantic_dim,
                visual_dim=semantic_dim,
            )
        elif fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(
                dim=semantic_dim,
                num_heads=text_num_heads,
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # 新增: 条件化 decoder (从 Stage B decoder 初始化)
        self.cond_decoder = LightroomDecoder(semantic_dim)
        if stage_b_model is not None:
            self.cond_decoder.load_state_dict(
                stage_b_model.decoder.state_dict()
            )

        # 冻结 backbone
        self.freeze_backbone()

    def freeze_backbone(self):
        """冻结 Stage B 的视觉 backbone 和基准 decoder。"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.semantic_projector.parameters():
            param.requires_grad = False
        for param in self.base_decoder.parameters():
            param.requires_grad = False

    def forward(self, images, text_ids=None):
        """
        Args:
            images: (B, 3, H, W) 输入图片
            text_ids: (B, L) 文本 token IDs (可选)
        Returns:
            dict with 'raw_params', 'norm_params', 'confidence',
                       'base_params' (Stage B 基准输出)
        """
        # 冻结的视觉 backbone
        with torch.no_grad():
            visual_feat = self.vision_encoder(images)
            semantic_emb = self.semantic_projector(visual_feat)
            base_output = self.base_decoder(semantic_emb)

        # 文本编码 (如果有)
        text_emb = None
        if text_ids is not None:
            text_emb = self.text_encoder(text_ids)

        # 融合
        fused_emb = self.fusion(semantic_emb, text_emb)

        # 条件化解码
        outputs = self.cond_decoder(fused_emb)
        outputs['base_params'] = base_output['norm_params']
        outputs['semantic_emb'] = semantic_emb

        return outputs

    def get_trainable_params(self):
        """返回可训练参数 (文本编码器 + 融合层 + 条件 decoder)。"""
        params = []
        params.extend(self.text_encoder.parameters())
        params.extend(self.fusion.parameters())
        params.extend(self.cond_decoder.parameters())
        return params

    def trainable_param_count(self):
        """可训练参数数量。"""
        return sum(p.numel() for p in self.get_trainable_params())


if __name__ == '__main__':
    from training.semantic_distill.model import DistillParamModel

    # 模拟 Stage B 模型
    stage_b = DistillParamModel()

    # Stage C
    model = TextConditionedModel(
        stage_b_model=stage_b,
        fusion_type='film',
    )

    images = torch.randn(2, 3, 224, 224)
    text_ids = torch.randint(1, 5000, (2, 20))

    # 有文本
    out = model(images, text_ids)
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable params: {model.trainable_param_count() / 1e6:.2f}M")
    for name, val in out['raw_params'].items():
        print(f"  {name}: {val.item():.3f}")

    # 无文本 (退化为 Stage B)
    out_no_text = model(images, text_ids=None)
    print(f"\nNo text - should be similar to base:")
    print(f"  base norm_params[0]: {out_no_text['base_params'][0, :3]}")
    print(f"  cond norm_params[0]: {out_no_text['norm_params'][0, :3]}")
