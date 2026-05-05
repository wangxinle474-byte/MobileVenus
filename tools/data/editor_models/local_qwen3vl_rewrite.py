"""\u7528 Qwen3-VL-4B-Instruct \u5728 5 \u5f20\u56fe\u4e0a\u751f\u6210 'a photo of ...' caption\u3002

\u8f93\u5165: \u6240\u6709 orig_path (5 \u5f20)
\u8f93\u51fa: data/qwen3vl_captions.json + \u63a7\u53f0\u6253\u5370

\u8d44\u6e90:
  Qwen3-VL-4B: ~8 GB fp16 / bf16
  RTX 4060 Laptop 8 GB: \u7528 device_map='auto' + bf16 \u8fd8\u662f\u4f1a\u6ea2\u51fa\uff0c
  \u6539\u6210 load_in_4bit \u6216 sequential_cpu_offload
\u8fd0\u884c:
  python tools/data/editor_models/local_qwen3vl_rewrite.py
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

# HF \u76f4\u8fde + 7897 \u4ee3\u7406 (\u907f\u514d hf-mirror SSL EOF)
os.environ.pop('HF_ENDPOINT', None)
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HF_HOME'] = r'E:\cache\huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = r'E:\cache\huggingface\hub'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import torch
from PIL import Image

MODEL_ID = 'Qwen/Qwen3-VL-4B-Instruct'
CACHE_DIR = r'E:\cache\huggingface'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


SYSTEM_PROMPT = """You are a professional photography art director. Given an image, \
output ONE detailed caption for AI image editing in this exact format:

"a photo of [main subject and setting details], [2-3 aesthetic qualifiers]"

Rules:
- 15-30 words total
- MUST start with "a photo of" (or "a black and white photo of" if the image is B&W)
- Describe the concrete subject and setting faithfully (preserve existing content)
- Add 2-3 qualifiers from: warm/cool/golden-hour/dramatic lighting, vivid/saturated/muted colors, high contrast, sharp focus, professional photography, cinematic mood
- NO abstract words (no "aesthetic", "harmonious", "elegant")
- NO editing verbs (no "adjust", "enhance", "brighten")
- NO "However", no "could", no "suggests"
- Output ONLY the caption line, no explanation, no quotes."""


def load_qwen3vl():
    """\u52a0\u8f7d Qwen3-VL-4B \u6a21\u578b (\u8c03\u6574 dtype + offload \u9002\u914d 8GB)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f'[LOAD] {MODEL_ID} (bf16 + device_map auto)')
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    print(f'    loaded in {time.time()-t0:.0f}s')
    return model, processor


@torch.no_grad()
def generate_caption(model, processor, image, system_prompt, max_new_tokens=80):
    """\u5bf9\u5355\u5f20\u56fe\u751f\u6210 caption."""
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': system_prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image],
        padding=True, return_tensors='pt',
    ).to(model.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )
    # \u53ea\u89e3\u7801\u751f\u6210\u7684\u90e8\u5206
    gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
    caption = processor.batch_decode(
        gen_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0]
    return caption.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions_in', default='data/compare_5_captions.json')
    ap.add_argument('--out_json', default='data/qwen3vl_captions.json')
    ap.add_argument('--max_side', type=int, default=896,
                    help='\u8f93\u5165\u56fe\u957f\u8fb9\u4e0a\u9650 (VL \u7528, 896 \u591f)')
    args = ap.parse_args()

    # GPU \u68c0\u67e5
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f'[GPU] {torch.cuda.get_device_name(0)} '
              f'free={free/1e9:.1f}GB / total={total/1e9:.1f}GB')

    cfg_data = json.load(open(PROJECT_ROOT / args.captions_in, encoding='utf-8'))
    samples = cfg_data['samples']
    print(f'[INFO] {len(samples)} images')

    model, processor = load_qwen3vl()

    out_records = []
    for s in samples:
        idx = s['idx']
        orig_path = PROJECT_ROOT / s['orig_path']
        if not orig_path.exists():
            print(f'[SKIP] idx={idx}: {orig_path} missing')
            continue
        image = Image.open(orig_path).convert('RGB')
        # Resize for VL
        w, h = image.size
        if max(w, h) > args.max_side:
            scale = args.max_side / max(w, h)
            image = image.resize((int(w*scale)//14*14, int(h*scale)//14*14),
                                  Image.LANCZOS)

        t0 = time.time()
        try:
            caption = generate_caption(model, processor, image, SYSTEM_PROMPT)
        except Exception as e:
            print(f'  FAIL idx={idx}: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            caption = None
        dt = time.time() - t0

        rec = {
            'idx': idx,
            'source_image': s['source_image'],
            'orig_path': s['orig_path'],
            'hand_caption': s['new_caption'],           # \u6211\u5199\u7684
            'qwen3vl_caption': caption,                  # VL \u751f\u7684
            'aesthetic_target': s['aesthetic_target'],
            'gen_time_s': round(dt, 1),
        }
        out_records.append(rec)
        print(f'\n[#{s["rank"]}] idx={idx}  {s["source_image"]}  ({dt:.1f}s)')
        print(f'  \u624b\u5199 : {s["new_caption"][:100]}')
        print(f'  VL   : {caption[:100] if caption else "(failed)"}')

    out = {
        'metadata': {
            'model_id': MODEL_ID,
            'n_samples': len(out_records),
            'system_prompt': SYSTEM_PROMPT,
        },
        'records': out_records,
    }
    out_path = PROJECT_ROOT / args.out_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n[DONE] captions saved to {out_path}')


if __name__ == '__main__':
    main()
