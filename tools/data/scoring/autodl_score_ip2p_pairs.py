"""AesExpert 对 IP2P 生成的 (原图, 编辑图) 成对评分 + VLM 反推。

运行在 AutoDL 上:
    python autodl_score_ip2p_pairs.py \
        --pairs_dir /root/autodl-tmp/outputs \
        --output /root/autodl-tmp/outputs/pair_scores.json

流程:
    1. 载入 AesExpert 一次
    2. 对每一对 (orig, edit):
       - 评分 orig (score)
       - 评分 edit (score)
       - 描述 orig (desc_orig)
       - 描述 edit (desc_edit)
    3. 对 edit_score > orig_score + DELTA_THRESH 的对:
       - 请 AesExpert 分析编辑图, 给出参数建议 (exposure/wb/contrast/...)
    4. 所有结果 → JSON

约 100 对, ~600 次推理, 预计 30~60 min (fp16 7B)
"""
import os
import sys
import json
import re
import argparse
import time
from pathlib import Path

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/hf_cache'

import torch
from PIL import Image

MODEL_PATH = '/root/autodl-tmp/models/AesExpert'

PROMPT_SCORE = ("Rate the overall aesthetic quality of this photograph from 1 to 10. "
                "Only output one number.")

PROMPT_DESC = ("Describe the aesthetic quality of this photograph in detail, "
               "covering composition, lighting, color balance, and overall impression. "
               "Be concise (2-3 sentences).")

PROMPT_PARAM = (
    "Analyze this photograph and suggest photographic adjustments to improve it. "
    "Output a JSON with the following 6 keys, each a number in the specified range:\n"
    "- ev_compensation: exposure adjustment in stops, [-3, 3]\n"
    "- white_balance: color temperature in Kelvin, [2000, 10000]\n"
    "- contrast: [-100, 100]\n"
    "- shadows: lift/lower shadows, [-100, 100]\n"
    "- highlights: reduce/boost highlights, [-100, 100]\n"
    "- saturation: [-100, 100]\n"
    "Only output the JSON object, nothing else."
)

DELTA_THRESH = 0.5  # edit 比 orig 高多少才做参数反推


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


def extract_score(text):
    if not text:
        return None
    m = re.match(r'^\s*([0-9]+\.?[0-9]*)\s*(/10)?', text)
    if m:
        return float(m.group(1))
    m = re.search(r'(\d+\.?\d*)\s*(?:/\s*10|out of 10)', text)
    if m:
        return float(m.group(1))
    m = re.search(r'(?:score|rating|rate)[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    for m in re.finditer(r'\b(\d+\.?\d*)\b', text):
        v = float(m.group(1))
        if 1 <= v <= 10:
            return v
    return None


def extract_params(text):
    """从 VLM 输出里抽出 JSON 参数。"""
    if not text:
        return None
    # 抓第一个 { ... } 块
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs_dir', default='/root/autodl-tmp/outputs')
    parser.add_argument('--output',
                        default='/root/autodl-tmp/outputs/pair_scores.json')
    parser.add_argument('--limit', type=int, default=-1,
                        help='只跑前 N 对 (调试用, -1 = 全量)')
    parser.add_argument('--skip_desc', action='store_true',
                        help='跳过描述, 只评分 + 参数反推 (更快)')
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    manifest_path = pairs_dir / 'manifest.json'
    if not manifest_path.exists():
        print(f'[ERR] manifest not found: {manifest_path}')
        sys.exit(1)

    manifest = json.load(open(manifest_path))
    pairs = manifest['pairs']
    if args.limit > 0:
        pairs = pairs[:args.limit]
    print(f'Pairs: {len(pairs)}')

    print('=' * 60)
    print('Loading AesExpert...')
    t0 = time.time()
    tokenizer, model, image_processor = load_model(MODEL_PATH)
    print(f'  loaded in {time.time() - t0:.1f}s')
    print('=' * 60)

    results = []
    n_improved = 0
    t_start = time.time()

    for i, p in enumerate(pairs):
        orig_path = pairs_dir / p['orig_png']
        edit_path = pairs_dir / p['edit_png']
        if not orig_path.exists() or not edit_path.exists():
            print(f'[skip {i}] missing files')
            continue

        # 评分
        raw_o = run_inference(tokenizer, model, image_processor,
                              str(orig_path), PROMPT_SCORE, max_new_tokens=32)
        raw_e = run_inference(tokenizer, model, image_processor,
                              str(edit_path), PROMPT_SCORE, max_new_tokens=32)
        s_o = extract_score(raw_o)
        s_e = extract_score(raw_e)

        entry = {
            'idx': p['idx'],
            'image': p['image'],
            'edit_prompt': p['edit_prompt'],
            'score_orig': s_o,
            'score_edit': s_e,
            'score_raw_orig': raw_o[:80],
            'score_raw_edit': raw_e[:80],
        }

        # 描述 (可选)
        if not args.skip_desc:
            entry['desc_orig'] = run_inference(
                tokenizer, model, image_processor, str(orig_path),
                PROMPT_DESC, max_new_tokens=256)[:500]
            entry['desc_edit'] = run_inference(
                tokenizer, model, image_processor, str(edit_path),
                PROMPT_DESC, max_new_tokens=256)[:500]

        # 参数反推 (针对改进了的对)
        delta = None
        if s_o is not None and s_e is not None:
            delta = s_e - s_o
            entry['delta'] = delta
            if delta >= DELTA_THRESH:
                n_improved += 1
                raw_p = run_inference(tokenizer, model, image_processor,
                                      str(edit_path), PROMPT_PARAM,
                                      max_new_tokens=512)
                entry['params_raw'] = raw_p[:600]
                entry['params'] = extract_params(raw_p)

        results.append(entry)

        # 进度
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (len(pairs) - i - 1)
        extra = f' improved={n_improved}'
        so_str = f'{s_o:.1f}' if s_o is not None else 'None'
        se_str = f'{s_e:.1f}' if s_e is not None else 'None'
        print(f'[{i + 1:3d}/{len(pairs)}] {p["image"]:20s}  '
              f'orig={so_str}  edit={se_str}{extra}  '
              f'elapsed={elapsed / 60:.1f}m  eta={eta / 60:.1f}m',
              flush=True)

        # 每 10 对存一次 (断点续保)
        if (i + 1) % 10 == 0:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump({
                    'manifest': manifest,
                    'threshold_delta': DELTA_THRESH,
                    'num_processed': len(results),
                    'num_improved': n_improved,
                    'results': results,
                }, f, ensure_ascii=False, indent=2)

    # 最终保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'manifest': manifest,
            'threshold_delta': DELTA_THRESH,
            'num_processed': len(results),
            'num_improved': n_improved,
            'results': results,
        }, f, ensure_ascii=False, indent=2)

    print('=' * 60)
    print(f'Done. {len(results)} pairs, {n_improved} improved.')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
