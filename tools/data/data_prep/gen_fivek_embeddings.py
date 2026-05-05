"""
从 FiveK 专家参数生成结构化文本 embedding
不需要运行 Venus，直接用参数值生成语义描述

输出: data/fivek_text_embeddings.npz

用法:
  python tools/data/data_prep/gen_fivek_embeddings.py
"""
import json, argparse
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


PARAM_UNITS = {
    'exposure':      ('EV',    (-3.0, 3.0)),
    'temperature':   ('K',     (2000, 10000)),
    'tint':          ('',      (-100, 100)),
    'contrast':      ('',      (-100, 100)),
    'highlights':    ('',      (-100, 100)),
    'shadows':       ('',      (-100, 100)),
    'whites':        ('',      (-100, 100)),
    'blacks':        ('',      (-100, 100)),
}


def params_to_text(params: dict) -> str:
    """将 8 个参数值转成自然语言描述"""
    parts = []

    ev = params.get('exposure', 0)
    if abs(ev) > 0.1:
        direction = 'brighter' if ev > 0 else 'darker'
        parts.append(f"exposure adjusted {direction} by {abs(ev):.1f} EV")

    temp = params.get('temperature', 5500)
    if temp > 6000:
        parts.append(f"warm color temperature ({temp:.0f}K)")
    elif temp < 5000:
        parts.append(f"cool color temperature ({temp:.0f}K)")
    else:
        parts.append(f"neutral white balance ({temp:.0f}K)")

    tint = params.get('tint', 0)
    if abs(tint) > 5:
        t_dir = 'green' if tint < 0 else 'magenta'
        parts.append(f"tint shifted toward {t_dir}")

    contrast = params.get('contrast', 0)
    if abs(contrast) > 5:
        parts.append(f"{'high' if contrast > 0 else 'low'} contrast ({contrast:+.0f})")

    hl = params.get('highlights', 0)
    sh = params.get('shadows', 0)
    if hl < -20:
        parts.append("highlights recovered")
    if sh > 20:
        parts.append("shadows lifted")

    wh = params.get('whites', 0)
    bk = params.get('blacks', 0)
    if abs(wh) > 10 or abs(bk) > 10:
        parts.append(f"tonal range adjusted (whites {wh:+.0f}, blacks {bk:+.0f})")

    if not parts:
        return "photo with minimal adjustments needed, well-balanced exposure and color"

    return "Photo enhancement: " + ", ".join(parts) + "."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default='data/fivek_expert_params.json')
    parser.add_argument('--expert', default='ExpertC',
                        help='Expert to use: ExpertA/B/C/D/E or consensus')
    parser.add_argument('--output', default='data/fivek_text_embeddings.npz')
    parser.add_argument('--model', default='all-MiniLM-L6-v2')
    parser.add_argument('--max_samples', type=int, default=-1,
                        help='Max samples (-1 = all)')
    args = parser.parse_args()

    print(f'1. 读取 FiveK 专家参数 ({args.expert})...')
    raw = json.load(open(args.data_file, encoding='utf-8'))
    samples = raw['samples'] if isinstance(raw, dict) and 'samples' in raw else raw
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f'   共 {len(samples)} 条')

    names, texts = [], []
    for item in samples:
        img_name = item.get('image_name', item.get('image', '')).replace('.dng', '')
        # 尝试多种参数字段格式
        params = (item.get(args.expert)
                  or item.get('params')
                  or item.get('expert_params')
                  or {k: item.get(k, 0) for k in PARAM_UNITS})
        if not params:
            continue
        text = params_to_text(params)
        names.append(img_name)
        texts.append(text)

    print(f'   有效样本: {len(names)}')
    print(f'   文本示例: {texts[0]}')

    print(f'2. MiniLM 编码 ({args.model})...')
    model = SentenceTransformer(args.model)
    embeddings = model.encode(texts, batch_size=512, show_progress_bar=True,
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
