"""加载 Qwen3-VL-4B + LoRA adapter, 在 ShareGPT val 集上跑推理, 保存输出 JSON.

用法:
  # smoke test (前 2 个样本)
  python tools/eval/infer_lora.py --limit 2 --out outputs/eval_smoke.json

  # full eval (全部 75 个)
  python tools/eval/infer_lora.py --out outputs/eval_full.json

环境要求 (AutoDL):
  - /root/autodl-tmp/models/Qwen3-VL-4B-Instruct
  - /root/autodl-tmp/checkpoints/intelligence_camera/lora_v1/  (adapter)
  - /root/autodl-tmp/datasets/ArtEdit-Bench/sharegpt/ArtEdit_LoRA_val.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_MODEL = '/root/autodl-tmp/models/Qwen3-VL-4B-Instruct'
ADAPTER_DIR = '/root/autodl-tmp/checkpoints/intelligence_camera/lora_v1'
VAL_JSON = '/root/autodl-tmp/datasets/ArtEdit-Bench/sharegpt/ArtEdit_LoRA_val.json'

# allowed Lightroom keys (与 generate_cot_pseudo_labels.py 保持一致)
ALLOWED_KEYS = {
    'temp', 'tint', 'exposure', 'contrast', 'highlights', 'shadows',
    'whites', 'blacks', 'clarity', 'vibrance', 'saturation', 'texture', 'dehaze',
}

THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)


def parse_assistant(text: str) -> dict:
    """解析模型输出: {think, tool_call_raw, tool_call, parsed_ok, parse_issue}."""
    res = {
        'raw': text,
        'think': None,
        'tool_call_raw': None,
        'tool_call': None,
        'parsed_ok': False,
        'parse_issue': None,
    }
    m = THINK_RE.search(text)
    if m:
        res['think'] = m.group(1).strip()

    m = TOOL_CALL_RE.search(text)
    if not m:
        res['parse_issue'] = 'no_tool_call_block'
        return res
    raw = m.group(1).strip()
    res['tool_call_raw'] = raw

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        res['parse_issue'] = f'invalid_json:{e.msg}'
        return res

    if not isinstance(obj, dict):
        res['parse_issue'] = 'tool_call_not_object'
        return res

    bad_keys = [k for k in obj.keys() if k not in ALLOWED_KEYS]
    bad_vals = [k for k, v in obj.items() if not isinstance(v, (int, float))]
    if bad_keys:
        res['parse_issue'] = f'bad_keys:{bad_keys}'
    if bad_vals:
        res['parse_issue'] = (res['parse_issue'] or '') + f'|bad_vals:{bad_vals}'

    if not bad_keys and not bad_vals and len(obj) >= 1:
        res['parsed_ok'] = True
    res['tool_call'] = obj
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_model', default=BASE_MODEL)
    ap.add_argument('--adapter', default=ADAPTER_DIR)
    ap.add_argument('--val_json', default=VAL_JSON)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 个 (0=全部)')
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--temperature', type=float, default=0.0)
    args = ap.parse_args()

    print(f'[load] processor + model from {args.base_model}')
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map='cuda',
        trust_remote_code=True,
    )
    print(f'[load] LoRA adapter from {args.adapter}')
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    print(f'[load] val from {args.val_json}')
    with open(args.val_json, encoding='utf-8') as f:
        samples = json.load(f)
    if args.limit:
        samples = samples[:args.limit]
    print(f'[run] {len(samples)} samples')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results = []

    t0 = time.time()
    for i, s in enumerate(samples):
        sample_id = s.get('sample_id', f'idx_{i}')
        lang = s.get('lang', '?')
        img_path = s['images'][0]
        system = s['system']
        user_msg = s['messages'][0]['content']
        gt_assistant = s['messages'][1]['content']

        # 构造 chat template
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': system}]},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': img_path},
                {'type': 'text', 'text': user_msg.replace('<image>', '').strip()},
            ]},
        ]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'[{i}] FAIL load image: {e}')
            continue

        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[image], return_tensors='pt').to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        out_ids = gen[0, inputs['input_ids'].shape[-1]:]
        out_text = processor.tokenizer.decode(out_ids, skip_special_tokens=False).strip()

        parsed = parse_assistant(out_text)
        gt_parsed = parse_assistant(gt_assistant)

        results.append({
            'i': i,
            'sample_id': sample_id,
            'lang': lang,
            'image': img_path,
            'user': user_msg,
            'pred': parsed,
            'gt': gt_parsed,
        })

        ok = '✓' if parsed['parsed_ok'] else '✗'
        keys = list(parsed['tool_call'].keys()) if parsed['tool_call'] else []
        print(f'[{i+1}/{len(samples)}] {ok} {sample_id} {lang}  keys={keys[:5]}  '
              f'issue={parsed["parse_issue"]}')

    dt = time.time() - t0
    n_ok = sum(1 for r in results if r['pred']['parsed_ok'])
    print(f'\n[DONE] {n_ok}/{len(results)} parsed_ok ({n_ok/max(1, len(results))*100:.1f}%) '
          f'in {dt:.1f}s')

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({
            'meta': {
                'base_model': args.base_model,
                'adapter': args.adapter,
                'val_json': args.val_json,
                'n_total': len(results),
                'n_parsed_ok': n_ok,
                'parse_rate': n_ok / max(1, len(results)),
                'runtime_sec': dt,
            },
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f'[OUT] {args.out}')


if __name__ == '__main__':
    main()
