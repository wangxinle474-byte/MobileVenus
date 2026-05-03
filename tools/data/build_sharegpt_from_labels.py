"""\u5c06 pseudo_labels.jsonl \u8f6c\u4e3a LLaMA-Factory \u517c\u5bb9\u7684 ShareGPT \u683c\u5f0f.

\u8f93\u5165: pseudo_labels.jsonl  (generate_cot_pseudo_labels.py \u8f93\u51fa)
\u8f93\u51fa:
  ArtEdit_LoRA_train.json  (90%)
  ArtEdit_LoRA_val.json    (10%)
  \u683c\u5f0f:
    [
      {
        "messages": [
          {"role": "user", "content": "<image>...\u7528\u6237 prompt"},
          {"role": "assistant", "content": "<think>...</think>\\n<tool_call>...</tool_call>"}
        ],
        "images": ["/abs/path/to/input.jpg"],
        "system": "..."
      },
      ...
    ]

\u7b80\u5355\u65cb\u8f6c: \u7528 \u524d 90%% \u4e3a train, \u540e 10%% \u4e3a val (\u6309 \u6392\u5e8f\u540e\u6587\u4ef6\u5217\u8868)\u3002
"""
import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = """You are a Lightroom adjustment specialist. Given an input image and user instruction, you output 4-8 Lightroom slider values.

Output format:
<think>2-3 sentences in user's language: state photo's current look and what sliders to move.</think>
<tool_call>{"key1": number, "key2": number, ...}</tool_call>

Allowed keys (use only these): temp, tint, exposure, contrast, highlights, shadows, whites, blacks, clarity, vibrance, saturation, texture, dehaze.
The tool_call must be valid JSON with double-quoted keys and numeric values only.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_jsonl', default='/root/autodl-tmp/datasets/ArtEdit-Bench/pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='/root/autodl-tmp/datasets/ArtEdit-Bench/sharegpt')
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min_tool_call_len', type=int, default=10,
                    help='\u6700\u5c0f tool_call \u957f\u5ea6 \u2014 \u8fc7\u6ee4\u5e9f\u6807\u7b7e')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # \u52a0\u8f7d pseudo \u6807\u7b7e + dedupe (\u591a\u6b21\u65ad\u70b9\u91cd\u542f\u4f1a\u7559 \u91cd\u590d\u884c)
    # \u7b56\u7565: \u6309 image_dir \u53bb\u91cd, \u4f18\u5148\u4fdd\u7559 parsed_ok=True \u7684\u884c
    by_dir: dict[str, dict] = {}
    n_raw = n_bad_json = 0
    with open(args.in_jsonl, encoding='utf-8') as f:
        for line in f:
            try:
                s = json.loads(line)
            except Exception:
                n_bad_json += 1
                continue
            n_raw += 1
            key = s.get('image_dir', '')
            if not key:
                continue
            prev = by_dir.get(key)
            if prev is None:
                by_dir[key] = s
            else:
                # \u4fdd\u7559 parsed_ok=True \u7684\u90a3\u884c (\u82e5\u90fd True, \u7559\u65b0\u7684)
                prev_ok = prev.get('parsed_ok', False)
                cur_ok = s.get('parsed_ok', False)
                if cur_ok and not prev_ok:
                    by_dir[key] = s

    print(f'[INFO] jsonl raw lines={n_raw}  bad_json={n_bad_json}  '
          f'unique image_dir={len(by_dir)}')

    # \u8fc7\u6ee4\u4e0d\u5408\u683c\u6837\u672c
    samples = []
    n_no_parsed = n_no_think = 0
    for s in by_dir.values():
        parsed = s.get('parsed_tool_call') or {}
        if not parsed or len(parsed) < 2:
            n_no_parsed += 1
            continue
        if not s.get('think'):
            n_no_think += 1
            continue
        samples.append(s)
    print(f'[INFO] after filter: valid={len(samples)}  '
          f'no_parsed={n_no_parsed}  no_think={n_no_think}')

    # \u8f6c\u4e3a ShareGPT (assistant \u8f93\u51fa\u7528 \u5e72\u51c0\u7684 parsed_tool_call)
    def convert_one(s):
        input_img = str(Path(s['image_dir']) / 'input.jpg')
        user_content = f"<image>{s['user_want']}"

        parsed = s['parsed_tool_call']
        clean_json = json.dumps(parsed, ensure_ascii=False)  # \u53cc\u5f15\u53f7 \u6807\u51c6 JSON
        assistant_content = (
            f"<think>{s['think']}</think>\n"
            f"<tool_call>{clean_json}</tool_call>"
        )

        return {
            'messages': [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': assistant_content},
            ],
            'images': [input_img],
            'system': SYSTEM_PROMPT,
            'sample_id': s.get('sample_id', ''),
            'lang': s.get('lang', ''),
        }

    converted = [convert_one(s) for s in samples]
    print(f'[INFO] {len(converted)} samples converted to ShareGPT')

    # \u5212\u5206 train/val
    random.seed(args.seed)
    random.shuffle(converted)
    n_val = max(1, int(len(converted) * args.val_ratio))
    val = converted[:n_val]
    train = converted[n_val:]

    with open(out_dir / 'ArtEdit_LoRA_train.json', 'w', encoding='utf-8') as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(out_dir / 'ArtEdit_LoRA_val.json', 'w', encoding='utf-8') as f:
        json.dump(val, f, indent=2, ensure_ascii=False)

    print(f'[DONE] train={len(train)} / val={len(val)}')
    print(f'[OUT] {out_dir}')

    # \u751f\u6210 dataset_info.json snippet \u4ee5\u4f9b LLaMA-Factory \u6ce8\u518c
    dataset_info = {
        'ArtEdit_LoRA_train': {
            'file_name': str(out_dir / 'ArtEdit_LoRA_train.json'),
            'formatting': 'sharegpt',
            'columns': {'messages': 'messages', 'images': 'images', 'system': 'system'},
            'tags': {
                'role_tag': 'role', 'content_tag': 'content',
                'user_tag': 'user', 'assistant_tag': 'assistant',
            },
        },
        'ArtEdit_LoRA_val': {
            'file_name': str(out_dir / 'ArtEdit_LoRA_val.json'),
            'formatting': 'sharegpt',
            'columns': {'messages': 'messages', 'images': 'images', 'system': 'system'},
            'tags': {
                'role_tag': 'role', 'content_tag': 'content',
                'user_tag': 'user', 'assistant_tag': 'assistant',
            },
        },
    }
    with open(out_dir / 'dataset_info_snippet.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f'[HINT] \u5c06\u8fd9\u4e24\u6761 \u5408\u5e76\u5230 LLaMA-Factory \u7684 data/dataset_info.json')


if __name__ == '__main__':
    main()
