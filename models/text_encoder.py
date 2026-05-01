"""
[DEPRECATED] 文本编码器 — 旧架构，已废弃

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
此文件属于早期方案 (sentence-transformers 外部编码器)，
已被当前两阶段方案替代:
  - Stage A: MiniLM-L6-v2 预计算 → TextProjector 投影
  - Stage C: LightTextEncoder 字符级自主编码

当前文本编码请参考:
  - training/semantic_distill/embed_texts.py (MiniLM 预计算)
  - training/text_condition/model.py::LightTextEncoder

保留此文件仅作为历史参考，不参与当前训练管线。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

原始设计: 将 Venus 的自然语言分析编码为语义 embedding
支持两种模式:
  1. sentence-transformers (推荐, 轻量级)
  2. 预计算模式 (离线编码, 训练时直接加载)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class TextEncoder:
    """
    将 Venus 文本分析编码为固定维度的语义向量
    
    使用 sentence-transformers 的 MiniLM 模型:
    - 轻量: ~33M 参数
    - 快速: CPU 上也很快
    - 效果好: 能捕获语义相似度
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        output_dim: int = 384,  # MiniLM 输出维度
    ):
        self.device = device
        self.output_dim = output_dim
        self._model = None
        self._model_name = model_name
    
    def _load_model(self):
        """延迟加载模型"""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading text encoder: {self._model_name}")
            self._model = SentenceTransformer(self._model_name, device=self.device)
            logger.info(f"Text encoder loaded (dim={self.output_dim})")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        编码文本列表
        
        Args:
            texts: Venus 文本分析列表
            batch_size: 编码批大小
            
        Returns:
            embeddings: (N, output_dim) numpy array, L2-normalized
        """
        self._load_model()
        embeddings = self._model.encode(
            texts, batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单条文本"""
        return self.encode([text])[0]


class SemanticProjectionHead(nn.Module):
    """
    将文本编码器的输出维度投影到语义桥接维度
    
    text_encoder_dim (384) → semantic_dim (256)
    """
    
    def __init__(self, input_dim: int = 384, output_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    
    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        projected = self.projection(text_embedding)
        return torch.nn.functional.normalize(projected, dim=-1)


def precompute_text_embeddings(
    pseudo_labels_file: str,
    output_file: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu",
):
    """
    离线预计算 Venus 文本分析的 embedding
    
    读取 venus_pseudo_labels.json, 对每张图片的文字分析编码,
    保存为 .npz 文件供训练时直接加载。
    
    Args:
        pseudo_labels_file: venus_pseudo_labels.json 路径
        output_file: 输出 .npz 路径
    """
    logger.info(f"Loading pseudo labels from {pseudo_labels_file}")
    with open(pseudo_labels_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels = data.get('labels', [])
    if not labels:
        logger.error("No labels found")
        return
    
    # 提取文本分析
    texts = []
    image_names = []
    scores = []
    
    for label in labels:
        raw = label.get('raw_response', '')
        # 分离文字分析和评分部分
        if ' ||| SCORES:' in raw:
            text_part = raw.split(' ||| SCORES:')[0]
        else:
            text_part = raw
        
        if text_part.strip():
            texts.append(text_part.strip())
            image_names.append(label['image'])
            scores.append(label.get('scores', {}))
    
    logger.info(f"Found {len(texts)} text analyses to encode")
    
    # 编码
    encoder = TextEncoder(model_name=model_name, device=device)
    embeddings = encoder.encode(texts, batch_size=64)
    
    logger.info(f"Encoded shape: {embeddings.shape}")
    
    # 保存
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        image_names=np.array(image_names),
    )
    
    # 同时保存对应的分数和图片名为 JSON (方便训练时匹配)
    mapping_file = str(output_file).replace('.npz', '_mapping.json')
    mapping = []
    for i, name in enumerate(image_names):
        mapping.append({
            'image': name,
            'index': i,
            'scores': scores[i] if i < len(scores) else {},
        })
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved embeddings to {output_file}")
    logger.info(f"Saved mapping to {mapping_file}")
    
    # 统计
    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(f"Embedding norms: mean={norms.mean():.3f}, std={norms.std():.6f}")
    
    # 检查语义多样性 (通过 cosine similarity)
    if len(embeddings) > 10:
        sample_idx = np.random.choice(len(embeddings), min(100, len(embeddings)), replace=False)
        sample = embeddings[sample_idx]
        sim_matrix = sample @ sample.T
        off_diag = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        logger.info(f"Cosine similarity: mean={off_diag.mean():.3f}, "
                    f"std={off_diag.std():.3f}, "
                    f"range=[{off_diag.min():.3f}, {off_diag.max():.3f}]")


if __name__ == '__main__':
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="预计算 Venus 文本 embedding")
    parser.add_argument("--input", type=str, required=True,
                        help="venus_pseudo_labels.json 路径")
    parser.add_argument("--output", type=str, default="data/venus_text_embeddings.npz",
                        help="输出 .npz 路径")
    parser.add_argument("--model", type=str, 
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    precompute_text_embeddings(args.input, args.output, args.model, args.device)
