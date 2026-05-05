"""对改善的 (orig, edit) 对用 AesExpert 反推 Lightroom 参数 + 理由.

输入: outputs/aug_ip2p_pilot_v1_reparsed.json (含 delta_v2)
输出: outputs/aug_ip2p_with_vlm_params.json (含 vlm_params, vlm_reason)

流程: 对每个 delta_v2 >= THRESH 的对:
    1. 给 AesExpert 看编辑后的图 + Venus 原描述
    2. 问: 这张图采用了哪些摄影参数调整 (JSON + 理由)
    3. 解析输出, 存储
"""
import argparse
import json
import re
import time
from pathlib import Path
import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'

import torch
from PIL import Image

MODEL_PATH = '/root/autodl-tmp/models/AesExpert'


def build_prompt(venus_desc, venus_suggestion):
    """构造对 AesExpert 的询问 prompt."""
    ctx = ''
    if venus_suggestion:
        ctx = (f"\nContext: The original photo had this aesthetic critique: "
               f'"{venus_suggestion[:300]}"\n')
    return (
        "Analyze this photograph carefully. "
        + ctx +
        "\nBased on what you see, estimate the photographic post-processing adjustments "
        "that were likely applied to achieve this look. "
        "Think about: exposure (stops), white balance (Kelvin), contrast, "
        "shadows, highlights, and saturation.\n"
        "First, provide your analysis in 1-2 sentences. "
        "Then output a JSON object on a new line with these exact keys:\n"
        '{"ev_compensation": <number in [-3,3]>, '
        '"white_balance": <Kelvin in [2000,10000]>, '
        '"contrast": <[-100,100]>, '
        '"shadows": <[-100,100]>, '
        '"highlights": <[-100,100]>, '
        '"saturation": <[-100,100]>}'
    )


def load_model(model_path):
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, load_8bit=False, load_4bit=False)
    vision_tower = model.get_vision_tower()
    if not getattr(vision_tower, 'is_loaded', True):
        vision_tower.load_model()
    try:
        vt_device = next(vision_tower.parameters()).device
        if str(vt_device) == 'cpu':
            vision_tower.half().cuda()
    except StopIteration:
        vision_tower.load_model()
        vision_tower.half().cuda()
    if image_processor is None:
        image_processor = getattr(vision_tower, 'image_processor', None)
    if image_processor is None:
        from transformers import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained(
            'openai/clip-vit-large-patch14-336')
    return tokenizer, model, image_processor


def run_inference(tokenizer, model, image_processor, image_path, prompt,
                  max_new_tokens=512):
    from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                                  DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token, process_images
    image = Image.open(image_path).convert('RGB')
    image_sizes = [image.size]
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    if getattr(model.config, 'mm_use_im_start_end', False):
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv = conv_templates['llava_v1'].copy()
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
        return_tensors='pt').unsqueeze(0).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids, images=image_tensor, image_sizes=image_sizes,
            do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
    return tokenizer.batch_decode(
        output_ids, skip_special_tokens=True)[0].strip()


def parse_vlm_output(text):
    """从 VLM 输出里抽取 JSON 和理由."""
    if not text:
        return None, text
    # 找最后一个 JSON-like 块 (贪婪)
    blocks = re.findall(r'\{[^{}]*\}', text, flags=re.DOTALL)
    params = None
    for b in reversed(blocks):  # 取最后一个
        try:
            candidate = json.loads(b)
            if isinstance(candidate, dict) and len(candidate) >= 3:
                params = candidate
                break
        except Exception:
            continue
    # 理由 = JSON 之前的文字
    if params and blocks:
        last_block = blocks[-1]
        reason = text.split(last_block)[0].strip()
    else:
        reason = text.strip()
    return params, reason[:500]


def clamp_params(p):
    """把参数 clamp 到合法范围, 返回规范化后的 dict."""
    if not isinstance(p, dict):
        return None
    ranges = {
        'ev_compensation': (-3, 3),
        'white_balance': (2000, 10000),
        'contrast': (-100, 100),
        'shadows': (-100, 100),
        'highlights': (-100, 100),
        'saturation': (-100, 100),
    }
    out = {}
    for k, (lo, hi) in ranges.items():
        v = p.get(k)
        if v is None:
            continue
        try:
            v = float(v)
            out[k] = max(lo, min(hi, v))
        except Exception:
            continue
    return out if len(out) >= 3 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        default='/root/autodl-tmp/outputs/aug_ip2p_pilot_v1_reparsed.json')
    parser.add_argument('--pairs_dir', default='/root/autodl-tmp/outputs')
    parser.add_argument('--output',
                        default='/root/autodl-tmp/outputs/aug_ip2p_with_vlm_params.json')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='delta_v2 >= threshold 的才做反推')
    args = parser.parse_args()

    d = json.load(open(args.input, encoding='utf-8'))
    rs = d['results']
    improved = [r for r in rs if r.get('delta_v2', -99) >= args.threshold]
    print(f'Total: {len(rs)}  |  Improved (delta>={args.threshold}): {len(improved)}')

    pairs_dir = Path(args.pairs_dir)

    print('=' * 60)
    print('Loading AesExpert...')
    t0 = time.time()
    tokenizer, model, image_processor = load_model(MODEL_PATH)
    print(f'  loaded in {time.time() - t0:.1f}s')
    print('=' * 60)

    results = []
    t_start = time.time()
    n_parsed = 0

    for i, r in enumerate(improved):
        edit_path = pairs_dir / f'{r["idx"]:04d}_edit.png'
        if not edit_path.exists():
            print(f'[skip {i}] missing {edit_path}')
            continue

        # 找原始 manifest 里的 venus_suggestion
        venus_desc = r.get('description_full', '') if 'description_full' in r else ''
        venus_suggestion = r.get('suggestion_full', '') if 'suggestion_full' in r else ''
        if not venus_suggestion and 'edit_prompt' in r:
            venus_suggestion = r['edit_prompt']

        prompt = build_prompt(venus_desc, venus_suggestion)
        raw = run_inference(tokenizer, model, image_processor,
                            str(edit_path), prompt, max_new_tokens=512)
        vlm_params, vlm_reason = parse_vlm_output(raw)
        clamped = clamp_params(vlm_params)

        entry = {
            'idx': r['idx'],
            'image': r['image'],
            'edit_prompt': r['edit_prompt'],
            'score_orig': r['score_orig_v2'],
            'score_edit': r['score_edit_v2'],
            'delta': r['delta_v2'],
            'vlm_params_raw': vlm_params,
            'vlm_params': clamped,
            'vlm_reason': vlm_reason,
            'vlm_output_full': raw[:1000],
            'pixel_params': r.get('pixel_params'),
        }
        results.append(entry)
        if clamped:
            n_parsed += 1

        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(improved) - i - 1)
        status = 'OK' if clamped else 'FAIL'
        print(f'[{i + 1:2d}/{len(improved)}] {r["image"]:15s}  delta={r["delta_v2"]:+.1f}  '
              f'{status}  parsed={n_parsed}  '
              f'elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m',
              flush=True)

        # 每 5 条存一次
        if (i + 1) % 5 == 0:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({
                    'source_pipeline': 'Venus->IP2P->AesExpert->VLM_reverse',
                    'threshold_delta': args.threshold,
                    'num_processed': len(results),
                    'num_vlm_parsed': n_parsed,
                    'results': results,
                }, f, ensure_ascii=False, indent=2)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'source_pipeline': 'Venus->IP2P->AesExpert->VLM_reverse',
            'threshold_delta': args.threshold,
            'num_processed': len(results),
            'num_vlm_parsed': n_parsed,
            'results': results,
        }, f, ensure_ascii=False, indent=2)

    print('=' * 60)
    print(f'Done. Processed {len(results)}, VLM parsed {n_parsed}.')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
