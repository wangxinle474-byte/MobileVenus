"""
从 venus_pseudo_labels.json 的 raw_response 字段生成 COCO 文本 embedding
输出: data/coco5k_text_embeddings.npz

用法:
  python tools/data/gen_coco_embeddings.py
"""
import json, argparse
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', default='data/venus_pseudo_labels.json')
    parser.add_argument('--output', default='data/coco5k_text_embeddings.npz')
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    parser.add_argument('--max_len', type=int, default=1000)
    args = parser.parse_args()

    print('1. 读取 COCO Venus 标注...')
    raw = json.load(open(args.labels, encoding='utf-8', errors='replace'))
    labels = raw['labels']
    print(f'   共 {len(labels)} 条')

    names, texts = [], []
    for item in labels:
        text = item.get('raw_response', '').strip()
        if not text:
            scores = item.get('scores', {})
            text = f"Composition:{scores.get('composition',5)} Lighting:{scores.get('lighting',5)} Color:{scores.get('color',5)} Clarity:{scores.get('clarity',5)} Subject:{scores.get('subject',5)}"
        names.append(Path(item['image']).stem)
        texts.append(text[:args.max_len])

    print(f'   有效样本: {len(names)}')
    print(f'   样本示例: {names[0]} | text_len={len(texts[0])}')

    print(f'2. MiniLM 编码 ({args.model})...')
    model = SentenceTransformer(args.model)
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True,
                               normalize_embeddings=True)
    print(f'   shape: {embeddings.shape}')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out),
                        image_names=np.array(names),
                        embeddings=embeddings.astype(np.float32),
                        text_dim=embeddings.shape[1])
    print(f'3. 保存: {out} ({out.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
