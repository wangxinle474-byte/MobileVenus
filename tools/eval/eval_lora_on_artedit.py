"""\u8bc4\u4f30 LoRA fine-tuned Qwen3-VL-4B \u5728 ArtEdit-Bench \u4e0a\u7684\u8f93\u51fa\u8d28\u91cf.

\u6307\u6807:
  1. \u683c\u5f0f\u6709\u6548\u7387 (\u80fd\u89e3\u51fa <think>/<tool_call>)
  2. tool_call JSON \u53ef\u89e3\u6790\u7387
  3. \u53c2\u6570\u5206\u5e03 (\u6a21\u578b\u7ed9\u7684 temp / exposure / \u7b49 \u5728\u5408\u7406\u8303\u56f4)
  4. (\u53ef\u9009) \u4e0e JarvisEvo-8B teacher \u7684 \u4e00\u81f4\u6027 \u5bf9\u6bd4

\u4f9d\u8d56:
  - LoRA \u6743\u91cd\u5df2\u4fdd\u5b58: /root/autodl-tmp/checkpoints/intelligence_camera/lora_v1/
  - \u57fa\u5ea7 Qwen3-VL-4B: /root/autodl-tmp/models/Qwen3-VL-4B-Instruct
  - \u9a8c\u8bc1\u6570\u636e: ArtEdit_LoRA_val.json
"""
import argparse
import base64
import json
import re
from collections import Counter
from pathlib import Path


def extract_tag(text: str, tag: str):
    m = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    return m.group(1).strip() if m else None


def check_tool_call_valid(tool_call_str: str) -> dict:
    """\u68c0\u67e5 tool_call \u5185\u5bb9\u662f\u5426\u662f\u6709\u6548 JSON + \u542b\u6cd5\u53c2\u6570."""
    if not tool_call_str:
        return {'valid_json': False, 'error': 'empty'}
    try:
        # \u53ef\u80fd\u6709 markdown code fence, \u526a\u6389
        t = re.sub(r'^```(?:json)?\s*', '', tool_call_str.strip())
        t = re.sub(r'\s*```$', '', t)
        obj = json.loads(t)
        return {'valid_json': True, 'obj': obj, 'n_keys': len(obj) if isinstance(obj, dict) else 0}
    except Exception as e:
        return {'valid_json': False, 'error': str(e)[:100]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', default='/root/autodl-tmp/checkpoints/intelligence_camera/lora_v1',
                    help='LoRA adapter directory')
    ap.add_argument('--base', default='/root/autodl-tmp/models/Qwen3-VL-4B-Instruct')
    ap.add_argument('--val_json',
                    default='/root/autodl-tmp/datasets/ArtEdit-Bench/sharegpt/ArtEdit_LoRA_val.json')
    ap.add_argument('--out', default='/root/autodl-tmp/IntelligenceCamera/outputs/lora_eval_results.json')
    ap.add_argument('--max_new_tokens', type=int, default=1024)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    # \u52a0\u8f7d\u6a21\u578b
    print(f'[INFO] Load base from {args.base}')
    processor = AutoProcessor.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='cuda:0',
    )
    if Path(args.adapter).exists() and any(Path(args.adapter).glob('adapter_*.bin'),
                                            Path(args.adapter).glob('adapter_*.safetensors')):
        print(f'[INFO] Load LoRA adapter from {args.adapter}')
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        mode = 'lora_merged'
    else:
        print(f'[INFO] No adapter at {args.adapter}, \u8bc4\u6d4b base model only')
        mode = 'base'
    model.eval()

    # \u52a0\u8f7d\u9a8c\u8bc1\u6570\u636e
    val = json.load(open(args.val_json, encoding='utf-8'))
    if args.limit:
        val = val[:args.limit]
    print(f'[INFO] {len(val)} val samples to evaluate')

    # \u9010\u4e2a\u63a8\u7406
    results = []
    n_valid_format = n_valid_json = 0
    for i, ex in enumerate(val):
        img_path = ex['images'][0]
        pil = Image.open(img_path).convert('RGB')
        user_msg = ex['messages'][0]['content'].replace('<image>', '').strip()
        gt_asst = ex['messages'][1]['content']

        # \u6784\u9020 chat messages (Qwen3-VL \u683c\u5f0f)
        msgs = [
            {'role': 'system', 'content': ex.get('system', '')},
            {'role': 'user', 'content': [{'type': 'image', 'image': pil},
                                          {'type': 'text', 'text': user_msg}]},
        ]
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_tensors='pt', return_dict=True,
        ).to('cuda:0')

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, temperature=None, top_p=None)
        # \u622a\u65ad\u53ea\u8981\u65b0\u751f\u6210\u90e8\u5206
        input_len = inputs['input_ids'].shape[1]
        resp_text = processor.decode(out_ids[0, input_len:], skip_special_tokens=True)

        think = extract_tag(resp_text, 'think')
        tool_call = extract_tag(resp_text, 'tool_call')
        has_format = bool(think and tool_call)
        json_check = check_tool_call_valid(tool_call) if tool_call else {'valid_json': False}

        if has_format:
            n_valid_format += 1
        if json_check.get('valid_json'):
            n_valid_json += 1

        results.append({
            'sample_id': ex.get('sample_id', str(i)),
            'lang': ex.get('lang', ''),
            'user_want': user_msg,
            'response': resp_text,
            'has_think': bool(think),
            'has_tool_call': bool(tool_call),
            'valid_json': json_check.get('valid_json', False),
            'json_keys': json_check.get('n_keys', 0),
            'json_error': json_check.get('error'),
        })

        if (i + 1) % 10 == 0:
            print(f'  [{i+1}/{len(val)}] format={n_valid_format}/{i+1} '
                  f'json={n_valid_json}/{i+1}')

    # \u6c47\u603b
    summary = {
        'mode': mode,
        'n_samples': len(val),
        'format_validity': n_valid_format / len(val) if val else 0,
        'json_validity': n_valid_json / len(val) if val else 0,
        'avg_json_keys': sum(r['json_keys'] for r in results) / max(len(results), 1),
        'per_lang_format_rate': {},
    }
    for lang in ['CN', 'EN']:
        lang_results = [r for r in results if r['lang'] == lang]
        if lang_results:
            summary['per_lang_format_rate'][lang] = \
                sum(1 for r in lang_results if r['has_think'] and r['has_tool_call']) / len(lang_results)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'results': results}, f,
                  indent=2, ensure_ascii=False)

    print('\n' + '=' * 70)
    print(f"LoRA Eval Summary (mode={mode})")
    print('=' * 70)
    print(f"  N samples:       {summary['n_samples']}")
    print(f"  Format validity: {summary['format_validity']*100:.1f}% (has both <think> and <tool_call>)")
    print(f"  JSON validity:   {summary['json_validity']*100:.1f}% (tool_call parses as JSON)")
    print(f"  Avg JSON keys:   {summary['avg_json_keys']:.1f}")
    for lang, rate in summary['per_lang_format_rate'].items():
        print(f"  {lang} format rate: {rate*100:.1f}%")
    print(f'\n[OUT] {args.out}')


if __name__ == '__main__':
    main()
