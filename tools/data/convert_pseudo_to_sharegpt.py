"""\u628a JarvisEvo \u8f93\u51fa\u7684 pseudo_labels.jsonl \u8f6c\u6210 LLaMA-Factory ShareGPT \u591a\u6a21\u6001\u683c\u5f0f.

\u8f93\u5165:
  pseudo_labels.jsonl \u6bcf\u884c = {image_dir, user_want, think, tool_call, ...}

\u8f93\u51fa:
  artedit_pseudo_sharegpt_train.json  (\u6570\u7ec4 of JSON, 90%)
  artedit_pseudo_sharegpt_val.json    (10%)
  dataset_info_snippet.json           (\u6ce8\u518c\u7247\u6bb5, append \u5230 LLaMA-Factory/data/dataset_info.json)

\u683c\u5f0f\u89c4\u8303 (LLaMA-Factory ShareGPT + multimodal):
  {
    "conversations": [
      {"from": "human", "value": "<image>\\n{user_want}"},
      {"from": "gpt",   "value": "<think>{think}</think>\\n<tool_call>{tool_call}</tool_call>"}
    ],
    "images": ["<abs path>"]
  }
"""
import argparse
import json
import random
from pathlib import Path


def build_assistant_text(think: str | None, tool_call: str | None) -> str:
    """\u91cd\u65b0\u7ec4\u88c5 assistant \u7684 \u8f93\u51fa (\u4fdd\u6301 <think>+<tool_call> \u7ed3\u6784)."""
    parts = []
    if think:
        parts.append(f'<think>\n{think.strip()}\n</think>')
    if tool_call:
        parts.append(f'<tool_call>\n{tool_call.strip()}\n</tool_call>')
    return '\n'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_file',
                    default='/root/autodl-tmp/datasets/ArtEdit-Bench/pseudo_labels.jsonl')
    ap.add_argument('--out_dir',
                    default='/root/autodl-tmp/datasets/ArtEdit-Bench/sharegpt')
    ap.add_argument('--val_ratio', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--require_tool_call', action='store_true', default=True,
                    help='\u53ea\u4fdd\u7559\u6709 tool_call \u7684\u6837\u672c (\u63a8\u8350)')
    ap.add_argument('--min_think_chars', type=int, default=20,
                    help='think \u6700\u5c0f\u957f\u5ea6 (\u8fc7\u6ee4\u592a\u77ed\u7684\u5783\u573e\u6807\u7b7e)')
    args = ap.parse_args()

    in_path = Path(args.in_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # \u8bfb\u5168\u90e8 pseudo labels
    raw = []
    with open(in_path, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                raw.append(json.loads(ln))
            except Exception as e:
                print(f'[WARN] bad json line: {e}')

    print(f'[INFO] Total rows in {in_path.name}: {len(raw)}')

    # \u8fc7\u6ee4
    good = []
    n_no_tc = n_short_think = n_no_img = 0
    for r in raw:
        if args.require_tool_call and not r.get('tool_call'):
            n_no_tc += 1
            continue
        if len(r.get('think') or '') < args.min_think_chars:
            n_short_think += 1
            continue
        img_path = Path(r['image_dir']) / 'input.jpg'
        # \u6ce8: \u6211\u4eec\u4e0d\u80fd\u5728\u672c\u5730\u4e4b\u63a5\u8bbf\u95ee AutoDL \u6587\u4ef6, \u53ea\u68c0\u67e5\u5b57\u6bb5\u5408\u7406
        if 'image_dir' not in r or not str(r['image_dir']).strip():
            n_no_img += 1
            continue
        good.append(r)

    print(f'[FILTER] kept {len(good)} / {len(raw)}  '
          f'(drop: no_tool_call={n_no_tc}, short_think={n_short_think}, no_img={n_no_img})')

    # \u6784\u9020 ShareGPT \u6837\u672c
    records = []
    for r in good:
        img_path = str(Path(r['image_dir']) / 'input.jpg')
        user_text = (r.get('user_want') or '').strip()
        assistant_text = build_assistant_text(r.get('think'), r.get('tool_call'))
        records.append({
            'conversations': [
                {'from': 'human', 'value': f'<image>\n{user_text}'},
                {'from': 'gpt',   'value': assistant_text},
            ],
            'images': [img_path],
        })

    # Train/Val split (\u56fa\u5b9a seed, \u7a33\u5b9a \u590d\u73b0)
    random.seed(args.seed)
    random.shuffle(records)
    n_val = max(1, int(len(records) * args.val_ratio))
    val = records[:n_val]
    train = records[n_val:]
    print(f'[SPLIT] train={len(train)}  val={len(val)}  (val_ratio={args.val_ratio})')

    train_path = out_dir / 'artedit_pseudo_sharegpt_train.json'
    val_path = out_dir / 'artedit_pseudo_sharegpt_val.json'

    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f'[OUT] {path}  ({len(data)} records)')

    # dataset_info.json \u5ef6\u7247\u6bb5 (\u7528\u6237 append \u5230 LLaMA-Factory/data/dataset_info.json)
    snippet = {
        'artedit_pseudo_sft': {
            'file_name': str(train_path),
            'formatting': 'sharegpt',
            'columns': {
                'messages': 'conversations',
                'images': 'images',
            },
            'tags': {
                'role_tag': 'from',
                'content_tag': 'value',
                'user_tag': 'human',
                'assistant_tag': 'gpt',
            },
        },
        'artedit_pseudo_val': {
            'file_name': str(val_path),
            'formatting': 'sharegpt',
            'columns': {
                'messages': 'conversations',
                'images': 'images',
            },
            'tags': {
                'role_tag': 'from',
                'content_tag': 'value',
                'user_tag': 'human',
                'assistant_tag': 'gpt',
            },
        },
    }
    snippet_path = out_dir / 'dataset_info_snippet.json'
    with open(snippet_path, 'w', encoding='utf-8') as f:
        json.dump(snippet, f, ensure_ascii=False, indent=2)
    print(f'[OUT] {snippet_path}  (merge into LLaMA-Factory data/dataset_info.json)')

    # \u4e00\u4e2a\u6837\u4f8b\u4f9b\u4eba\u8089\u68c0\u9a8c
    print('\n=== SAMPLE (first train record) ===')
    sample = train[0]
    print(json.dumps(sample, ensure_ascii=False, indent=2)[:1500])


if __name__ == '__main__':
    main()
