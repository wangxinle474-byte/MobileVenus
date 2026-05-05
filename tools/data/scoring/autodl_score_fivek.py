"""AesExpert \u5bf9 FiveK JPEG \u5168\u91cf\u5355\u56fe\u8bc4\u5206\u3002

\u8fd0\u884c\u5728 AutoDL \u4e0a:
    python autodl_score_fivek.py \\
        --jpeg_dir /root/autodl-tmp/fivek_jpeg \\
        --output /root/autodl-tmp/outputs/fivek_aesexpert_scores.json

5121 \u5f20 \u00d7 ~0.18 \u79d2 = ~15 min on RTX 5090.
\u652f\u6301 --resume \u65ad\u70b9\u7eed\u8dd1.
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
                  max_new_tokens=32):
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


# \u5b9a\u6027\u8bcd\u2192\u6570\u503c\u6620\u5c04 (\u4e0e reparse_pair_scores \u4fdd\u6301\u4e00\u81f4)
QUAL_MAP = {
    'excellent': 9.0, 'outstanding': 9.0, 'exceptional': 9.0,
    'very good': 8.0, 'great': 8.0, 'beautiful': 7.5,
    'quite beautiful': 7.0, 'good': 7.0, 'nice': 6.5, 'pleasing': 6.5,
    'decent': 5.5, 'acceptable': 5.0, 'average': 5.0, 'okay': 5.0, 'ok': 5.0,
    'mediocre': 4.0, 'fair': 4.0, 'poor': 3.0, 'bad': 2.0,
    'very poor': 1.5, 'terrible': 1.0,
}


def extract_score(text):
    if not text:
        return None, 'empty'
    # \u5148\u8bd5\u7eaf\u6570\u5b57
    m = re.match(r'^\s*([0-9]+\.?[0-9]*)\s*(/10)?', text)
    if m:
        v = float(m.group(1))
        if 0 <= v <= 10:
            return v, 'numeric'
    m = re.search(r'(\d+\.?\d*)\s*(?:/\s*10|out of 10)', text)
    if m:
        return float(m.group(1)), 'fraction'
    m = re.search(r'(?:score|rating|rate)[:\s]*(\d+\.?\d*)', text, re.IGNORECASE)
    if m:
        return float(m.group(1)), 'keyword'
    for m in re.finditer(r'\b(\d+\.?\d*)\b', text):
        v = float(m.group(1))
        if 1 <= v <= 10:
            return v, 'inline'
    # \u5b9a\u6027\u8bcd
    text_low = text.lower()
    for phrase, val in sorted(QUAL_MAP.items(), key=lambda kv: -len(kv[0])):
        if phrase in text_low:
            return val, f'qual:{phrase}'
    return None, 'unparsed'


def bucket_of(score):
    if score is None:
        return None
    for lo, hi, lab in [(0, 2, '0-2'), (2, 4, '2-4'), (4, 6, '4-6'),
                         (6, 8, '6-8'), (8, 10.01, '8-10')]:
        if lo <= score < hi:
            return lab
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpeg_dir', default='/root/autodl-tmp/fivek_jpeg')
    parser.add_argument('--output', default='/root/autodl-tmp/outputs/fivek_aesexpert_scores.json')
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--save_every', type=int, default=200)
    args = parser.parse_args()

    jpeg_dir = Path(args.jpeg_dir)
    jpg_files = sorted(jpeg_dir.glob('*.jpg'))
    if args.limit > 0:
        jpg_files = jpg_files[:args.limit]
    print(f'Found {len(jpg_files)} JPEG files')

    # Resume
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.resume and out_path.exists():
        prev = json.load(open(out_path, 'r', encoding='utf-8'))
        results = prev.get('scores', {})
        print(f'Resume: loaded {len(results)} prior scores')

    todo = [f for f in jpg_files if f.name not in results]
    print(f'TODO: {len(todo)}')

    if not todo:
        print('All images scored, nothing to do')
        _save(out_path, results)
        return

    print('=' * 60)
    print('Loading AesExpert...')
    t0 = time.time()
    tokenizer, model, image_processor = load_model(MODEL_PATH)
    print(f'  loaded in {time.time() - t0:.1f}s')
    print('=' * 60)

    t_start = time.time()
    for i, img_path in enumerate(todo):
        try:
            raw = run_inference(tokenizer, model, image_processor,
                                str(img_path), PROMPT_SCORE, max_new_tokens=32)
            score, method = extract_score(raw)
            results[img_path.name] = {
                'score': score,
                'parse_method': method,
                'raw': raw[:120],
            }
        except Exception as e:
            print(f'  FAIL {img_path.name}: {type(e).__name__}: {e}', flush=True)
            results[img_path.name] = {'score': None, 'parse_method': 'error',
                                       'error': str(e)[:200]}

        # \u8fdb\u5ea6
        elapsed = time.time() - t_start
        speed = (i + 1) / max(elapsed, 0.001)
        eta_min = (len(todo) - i - 1) / max(speed, 0.001) / 60

        if (i + 1) % 50 == 0 or i == 0 or i == len(todo) - 1:
            s = results[img_path.name].get('score')
            ss = f'{s:.1f}' if s is not None else 'None'
            print(f'[{i + 1:5d}/{len(todo)}] {img_path.name:30s}  '
                  f'score={ss:5s}  speed={speed:.1f}img/s  '
                  f'eta={eta_min:.1f}m', flush=True)

        if (i + 1) % args.save_every == 0:
            _save(out_path, results)

    _save(out_path, results)

    # \u7edf\u8ba1
    valid = [r['score'] for r in results.values() if r.get('score') is not None]
    if valid:
        from collections import Counter
        bc = Counter(bucket_of(v) for v in valid)
        print('=' * 60)
        print(f'Done. {len(results)} images, {len(valid)} parseable')
        print(f'  mean={sum(valid)/len(valid):.2f}  '
              f'min={min(valid):.1f}  max={max(valid):.1f}')
        print('  bucket distribution:')
        for b in ['0-2', '2-4', '4-6', '6-8', '8-10']:
            n = bc.get(b, 0)
            print(f'    {b:6s}  {n:5d}  {n / len(valid) * 100:5.1f}%')
    print(f'Saved: {out_path}')


def _save(out_path, results):
    valid = [r['score'] for r in results.values() if r.get('score') is not None]
    meta = {
        'scorer': 'AesExpert (LLaVA-1.5-7B fp16)',
        'num_images': len(results),
        'num_parseable': len(valid),
    }
    if valid:
        meta['mean'] = round(sum(valid) / len(valid), 3)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'meta': meta, 'scores': results}, f,
                  ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
