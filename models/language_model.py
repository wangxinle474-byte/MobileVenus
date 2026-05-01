"""
[DEPRECATED] TinyLLaMA 轻量语言模型 — 旧架构，已废弃

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
此文件属于早期方案 (1B模型 + TinyLLaMA 建议生成)，
已被当前 LightTextEncoder (字符级 Transformer) 替代。

当前文本编码请参考:
  - training/text_condition/model.py::LightTextEncoder

保留此文件仅作为历史参考，不参与当前训练管线。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原始设计: TinyLLaMA 轻量语言模型，用于生成美学建议文本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class TinyLLaMAConfig:
    """TinyLLaMA 模型配置"""
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 16
    num_heads: int = 16
    intermediate_size: int = 5632
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5


class TinyLLaMA(nn.Module):
    """
    TinyLLaMA-1B 轻量语言模型
    参数量: ~1B
    """
    
    def __init__(self, config: TinyLLaMAConfig):
        super().__init__()
        self.config = config
        
        # Token Embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Positional Embedding
        self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerLayer(config)
            for _ in range(config.num_layers)
        ])
        
        # Final Layer Norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # LM Head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        visual_prefix: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        前向传播
        
        Args:
            input_ids: (B, seq_len) token IDs
            visual_prefix: (B, hidden_size) 视觉特征前缀
            attention_mask: (B, seq_len) 注意力掩码
            
        Returns:
            logits: (B, seq_len, vocab_size)
        """
        B, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token + Position Embeddings
        token_emb = self.token_embedding(input_ids)  # (B, seq_len, hidden_size)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(pos_ids)
        
        hidden_states = token_emb + pos_emb
        
        # 如果有视觉前缀，拼接到序列开头
        if visual_prefix is not None:
            visual_prefix = visual_prefix.unsqueeze(1)  # (B, 1, hidden_size)
            hidden_states = torch.cat([visual_prefix, hidden_states], dim=1)
            
            # 更新 attention_mask
            if attention_mask is not None:
                prefix_mask = torch.ones(B, 1, device=device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Transformer Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.norm(hidden_states)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def encode(self, visual_features: torch.Tensor):
        """
        编码视觉特征为语言模型隐藏状态
        
        Args:
            visual_features: (B, feature_dim)
        Returns:
            hidden_states: (B, hidden_size)
        """
        # 简单的线性投影
        if visual_features.shape[-1] != self.config.hidden_size:
            projection = nn.Linear(
                visual_features.shape[-1],
                self.config.hidden_size
            ).to(visual_features.device)
            hidden_states = projection(visual_features)
        else:
            hidden_states = visual_features
        
        return hidden_states


class TransformerLayer(nn.Module):
    """Transformer 层"""
    
    def __init__(self, config: TinyLLaMAConfig):
        super().__init__()
        
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (B, seq_len, hidden_size)
            attention_mask: (B, seq_len)
        Returns:
            hidden_states: (B, seq_len, hidden_size)
        """
        # Self-Attention with residual
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        
        # Convert attention_mask to proper format
        if attention_mask is not None:
            # (B, seq_len) -> (B, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        attn_output, _ = self.self_attn(
            hidden_states, hidden_states, hidden_states,
            attn_mask=attention_mask
        )
        hidden_states = residual + attn_output
        
        # FFN with residual
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = residual + self.ffn(hidden_states)
        
        return hidden_states


if __name__ == "__main__":
    # 测试 TinyLLaMA
    config = TinyLLaMAConfig()
    model = TinyLLaMA(config)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (FP32): {total_params * 4 / (1024**3):.2f} GB")
    print(f"Model size (FP16): {total_params * 2 / (1024**3):.2f} GB")
    
    # 测试前向传播
    batch_size = 2
    seq_len = 32
    
    dummy_input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    dummy_visual_prefix = torch.randn(batch_size, config.hidden_size)
    
    with torch.no_grad():
        logits = model(dummy_input_ids, visual_prefix=dummy_visual_prefix)
    
    print(f"\nInput IDs shape: {dummy_input_ids.shape}")
    print(f"Visual prefix shape: {dummy_visual_prefix.shape}")
    print(f"Output logits shape: {logits.shape}")
