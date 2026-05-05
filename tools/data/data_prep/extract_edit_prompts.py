"""
从 Benchmark_AesGuide 的美学描述中提取可用于图像编辑模型的 prompt.

输入: Benchmark_AesGuide.json (1000 条专业美学描述, 300-1000 字)
输出: venus_prompts.json (image_path, original_desc, edit_prompt, suggestion_full)

策略:
  1. 找出"建议"部分 (在 "could/may/consider/suggesting/However" 后面)
  2. 转换为祈使句 ("moderating the exposure" -> "moderate the exposure")
  3. 简化为 IP2P 风格短指令 (< 30 词)
"""
import json
import re
import argparse
from pathlib import Path


# 触发"建议"段落的关键短语 (按优先级排列)
SUGGESTION_TRIGGERS = [
    r'It is noted that',
    r'However, (?:there is )?',
    r'Aesthetic refinement (?:may be|can be|could be) (?:achieved|attained)',
    r'A suggestion for (?:further |additional )?enhancement is',
    r'(?:Aesthetic )?refinement (?:may|can|could) be',
    r'(?:could|may|might) benefit from',
    r'(?:could|may|might) (?:potentially )?(?:enhance|improve)',
    r'consider(?:ation could be given to)?',
    r'(?:possibly |potentially )?(?:enhanced|improved) by',
    r'opportunity to',
    r'recommend(?:ed|ation)?',
]

# 把动名词 / 名词化转祈使句的模式
GERUND_PATTERNS = [
    (r'\b[Mm]oderating the\b',       'moderate the'),
    (r'\b[Rr]epositioning the\b',    'reposition the'),
    (r'\b[Rr]ealigning the\b',       'realign the'),
    (r'\b[Aa]djusting the\b',        'adjust the'),
    (r'\b[Ee]nhancing the\b',        'enhance the'),
    (r'\b[Bb]alancing the\b',        'balance the'),
    (r'\b[Bb]rightening the\b',      'brighten the'),
    (r'\b[Ss]oftening the\b',        'soften the'),
    (r'\b[Bb]lurring the\b',         'blur the'),
    (r'\b[Ss]haring the\b',          'sharpen the'),
    (r'\b[Cc]ropping the\b',         'crop the'),
    (r'\b[Rr]efining the\b',         'refine the'),
    (r'\b[Aa]ddition(?:al)? of\b',   'add'),
    (r'\b[Rr]eduction (?:of|in) the\b', 'reduce the'),
    (r'\b[Ii]ntroducing\b',          'introduce'),
    (r'\b[Ii]ncorporating\b',        'incorporate'),
]

# 简化噪声短语
NOISE_PHRASES = [
    r'a more harmonious aesthetic outcome',
    r'a more enriched visual composition',
    r'(?:the )?overall (?:visual )?(?:composition|impact|appeal|aesthetic appeal)',
    r'within the (?:image|frame|composition)',
    r'thereby (?:elevating|allowing|enhancing|providing)',
    r'(?:the )?(?:perception|sense) of',
    r'a (?:more )?(?:nuanced|sophisticated|elevated|balanced)',
    r'aesthetic refinement',
    r'visual (?:impact|interest|coherence)',
    r'(?:the )?delineation of',
    r'It is (?:noted|observed|suggested) that\s*',
    r'\boverall\b',
    r'\bsomewhat\b',
    r'\bslightly\b',
    r'\bperhaps\b',
    r'\bcould potentially\b',
    r'\bmight\b',
    r'\bmay\b',
]


def extract_suggestion(text: str) -> str:
    """从美学描述中找出建议段落 (从触发词到结尾)."""
    for pattern in SUGGESTION_TRIGGERS:
        m = re.search(pattern, text)
        if m:
            return text[m.start():].strip()
    # fallback: 取最后一句
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return sentences[-1] if sentences else text


def to_imperative(text: str) -> str:
    """把动名词转为祈使句."""
    for pat, rep in GERUND_PATTERNS:
        text = re.sub(pat, rep, text)
    return text


def simplify(text: str) -> str:
    """去除冗余短语, 简化为短指令."""
    for noise in NOISE_PHRASES:
        text = re.sub(noise, '', text, flags=re.IGNORECASE)
    # 清理多余空格和标点
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*,\s*,\s*', ', ', text)
    text = re.sub(r'\s+([.,;])', r'\1', text)
    text = re.sub(r'^[\s,.;]+', '', text)
    text = re.sub(r'[\s,;]+$', '', text)
    return text.strip()


def to_edit_prompt(suggestion: str, max_words: int = 30) -> str:
    """转换成 IP2P 风格的简洁编辑指令."""
    s = to_imperative(suggestion)
    s = simplify(s)
    # 截断到 max_words
    words = s.split()
    if len(words) > max_words:
        s = ' '.join(words[:max_words])
        # 尽量在标点处截断
        last_punct = max((s.rfind(c) for c in ',.;'), default=-1)
        if last_punct > len(s) * 0.6:
            s = s[:last_punct]
    return s.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=r'E:/Data/dataset/Venus_data/Benchmark_AesGuide/json/Benchmark_AesGuide.json')
    parser.add_argument('--image_dir', default=r'E:/Data/dataset/Venus_data/Benchmark_AesGuide/images')
    parser.add_argument('--output', default='data/venus_edit_prompts.json')
    parser.add_argument('--max_words', type=int, default=30)
    parser.add_argument('--show_samples', type=int, default=10)
    args = parser.parse_args()

    in_path = Path(args.input)
    img_dir = Path(args.image_dir)
    print(f'Loading {in_path}...')
    d = json.load(open(in_path, 'r', encoding='utf-8'))
    print(f'  {len(d)} entries')

    out = []
    n_short = 0
    for img_name, full_desc in d.items():
        suggestion = extract_suggestion(full_desc)
        edit_prompt = to_edit_prompt(suggestion, max_words=args.max_words)

        if len(edit_prompt.split()) < 3:
            n_short += 1
            continue

        out.append({
            'image': img_name,
            'image_path': str(img_dir / img_name),
            'edit_prompt': edit_prompt,
            'suggestion_full': suggestion,
            'description_full': full_desc,
        })

    print(f'\nExtracted: {len(out)} prompts (skipped {n_short} too-short)')

    # Show samples
    print(f'\n=== {min(args.show_samples, len(out))} Samples ===')
    for i in range(min(args.show_samples, len(out))):
        e = out[i]
        print(f'\n[{i}] {e["image"]}')
        print(f'  prompt: "{e["edit_prompt"]}"')
        print(f'  full suggestion: {e["suggestion_full"][:200]}...')

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'source': 'Benchmark_AesGuide',
                'num_prompts': len(out),
                'max_words': args.max_words,
            },
            'prompts': out,
        }, f, ensure_ascii=False, indent=2)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
