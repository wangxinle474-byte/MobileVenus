"""生成文本 embedding (MiniLM-L6-v2)。

将 Venus 生成的美学文本描述编码为 384-D 向量，
用于 Stage A 语义对齐训练。

用法:
    python training/semantic_distill/embed_texts.py \
        --json outputs/fivek_stage_a.json \
        --image_root E:/dataset/fivek_jpeg \
        --output data/fivek_text_embeddings.npz
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm


def load_model(model_name='all-MiniLM-L6-v2', device='cpu'):
    """加载 sentence-transformers 模型。"""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=device)
    return model


def extract_texts_from_json(json_path):
    """从 Venus 分析 JSON 提取文本。

    Args:
        json_path: Venus 输出 JSON 路径
    Returns:
        image_ids: list of str
        texts: list of str
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    image_ids = []
    texts = []

    for item in data:
        img_id = item.get('image_id', item.get('filename', ''))
        # 合并所有文本字段
        text_parts = []
        for key in ['analysis', 'description', 'aesthetic', 'suggestion']:
            if key in item and item[key]:
                text_parts.append(str(item[key]))
        text = ' '.join(text_parts) if text_parts else ''

        if text.strip():
            image_ids.append(img_id)
            texts.append(text)

    return image_ids, texts


def embed_texts(model, texts, batch_size=64, show_progress=True):
    """批量编码文本。

    Args:
        model: SentenceTransformer 模型
        texts: list of str
        batch_size: 批大小
    Returns:
        embeddings: (N, 384) numpy array
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
    )
    return embeddings


def main():
    parser = argparse.ArgumentParser(description='生成文本 embedding')
    parser.add_argument('--json', type=str, required=True,
                        help='Venus 分析输出 JSON')
    parser.add_argument('--image_root', type=str, default='',
                        help='图片根目录 (用于验证)')
    parser.add_argument('--output', type=str, required=True,
                        help='输出 .npz 路径')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence-transformers 模型名')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model, args.device)

    print(f"Loading texts from: {args.json}")
    image_ids, texts = extract_texts_from_json(args.json)
    print(f"  Found {len(texts)} texts")

    print("Encoding texts...")
    embeddings = embed_texts(model, texts, args.batch_size)
    print(f"  Embeddings shape: {embeddings.shape}")

    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        image_ids=np.array(image_ids),
        embeddings=embeddings.astype(np.float32),
        model_name=args.model,
    )
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()
