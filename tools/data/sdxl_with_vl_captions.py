"""\u7528 Qwen3-VL \u751f\u6210\u7684 caption \u8dd1 SDXL-Turbo img2img\u3002

\u8bfb\u53d6: data/qwen3vl_captions.json
\u8c03\u7528: \u590d\u7528 local_run_sdxl_turbo.py \u7684\u903b\u8f91, \u4f46 caption \u4ece VL \u8bb0\u5f55\u91cc\u62ff
\u8f93\u51fa: outputs/sdxl_turbo_vl_compare/

\u4e0e local_run_sdxl_turbo.py \u7684\u5dee\u522b:
  - \u624b\u5199\u7248: outputs/sdxl_turbo_compare/
  - VL \u751f\u6210\u7248: outputs/sdxl_turbo_vl_compare/  (\u672c\u811a\u672c)
  \u4e24\u8005\u540c\u4e00 5 \u5f20\u539f\u56fe + 3 \u4e2a strength, \u552f\u4e00\u533a\u522b\u662f caption \u6765\u6e90
  \u6700\u540e\u62fc grid \u80fd\u770b\u51fa "\u624b\u5199 vs VL \u751f\u6210" \u5728 SDXL-Turbo \u4e0a\u8c01\u8d62

\u8fd0\u884c:
  python tools/data/sdxl_with_vl_captions.py
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

# \u7eaf\u672c\u5730\u7f13\u5b58 (SDXL-Turbo \u5df2\u4e0b)
os.environ.pop('HF_ENDPOINT', None)
os.environ['HF_HOME'] = r'E:\cache\huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = r'E:\cache\huggingface\hub'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image


def load_pipeline():
    print('[Load] SDXL-Turbo img2img pipeline...')
    t0 = time.time()
    pipe = AutoPipelineForImage2Image.from_pretrained(
        'stabilityai/sdxl-turbo',
        torch_dtype=torch.float16,
        variant='fp16',
        use_safetensors=True,
    )
    pipe.enable_model_cpu_offload()
    print(f'[OK] loaded in {time.time()-t0:.1f}s')
    return pipe


def edit_one(pipe, orig_img, caption, strength, steps=4, seed=42):
    generator = torch.Generator('cuda').manual_seed(seed)
    out = pipe(
        prompt=caption,
        image=orig_img,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=0.0,
        generator=generator,
    )
    return out.images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vl_json', default='data/qwen3vl_captions.json',
                    help='Qwen3-VL \u751f\u7684 caption JSON')
    ap.add_argument('--out_dir', default='outputs/sdxl_turbo_vl_compare')
    ap.add_argument('--strengths', nargs='+', type=float,
                    default=[0.3, 0.5, 0.7])
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--short_side', type=int, default=768)
    ap.add_argument('--caption_field', default='qwen3vl_caption',
                    help='\u8bfb\u54ea\u4e2a\u5b57\u6bb5\u4f5c\u4e3a caption (\u53ef\u9009 hand_caption)')
    args = ap.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    vl_path = PROJECT_ROOT / args.vl_json
    if not vl_path.exists():
        print(f'[ERR] {vl_path} \u4e0d\u5b58\u5728\u3002\u8bf7\u5148\u8dd1 local_qwen3vl_rewrite.py')
        sys.exit(1)

    data = json.load(open(vl_path, encoding='utf-8'))
    records = data['records']
    print(f'[INFO] {len(records)} \u6837\u672c, caption_field={args.caption_field}, '
          f'strengths={args.strengths}, steps={args.steps}')

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f'[GPU] {torch.cuda.get_device_name(0)} '
              f'free={free/1e9:.1f}GB / total={total/1e9:.1f}GB')

    pipe = load_pipeline()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in records:
        idx = r['idx']
        orig_path = PROJECT_ROOT / r['orig_path']
        if not orig_path.exists():
            print(f'[SKIP] idx={idx}: missing {orig_path}')
            continue
        caption = r.get(args.caption_field) or r.get('hand_caption')
        if not caption:
            print(f'[SKIP] idx={idx}: caption \u4e3a\u7a7a')
            continue

        img = Image.open(orig_path).convert('RGB')
        # SDXL \u7684\u8f93\u5165\u8fb9 \u22651024 \u8f83\u4f73, \u6211\u4eec\u7528 768 \u5151\u73b0\u8d44\u6e90
        w, h = img.size
        if min(w, h) != args.short_side:
            scale = args.short_side / min(w, h)
            new_w = int(w * scale) // 8 * 8
            new_h = int(h * scale) // 8 * 8
            img = img.resize((new_w, new_h), Image.LANCZOS)

        print(f'\n[idx={idx}] {r["source_image"]}  ({img.size[0]}x{img.size[1]})')
        print(f'  VL caption: "{caption[:100]}"')
        # \u4fdd\u5b58\u539f\u56fe (\u62fc grid \u7528)
        img.save(out_dir / f'{idx:04d}_orig.png')

        for st in args.strengths:
            t0 = time.time()
            try:
                result = edit_one(pipe, img, caption,
                                  strength=st, steps=args.steps, seed=42)
            except Exception as e:
                print(f'  FAIL strength={st}: {type(e).__name__}: {e}')
                continue
            dt = time.time() - t0
            out_path = out_dir / f'{idx:04d}_sdxl_vl_s{int(st*10):02d}.png'
            result.save(out_path)
            print(f'  strength={st}  ({dt:.1f}s)  -> {out_path.name}')

    # \u4fdd\u5b58\u4f7f\u7528\u7684 caption \u9001\u4ef6 (\u4f9b\u540e\u7eed\u62fc grid \u8d34\u6587\u672c)
    summary = {
        'caption_field': args.caption_field,
        'strengths': args.strengths,
        'steps': args.steps,
        'samples': [
            {'idx': r['idx'],
             'source_image': r.get('source_image'),
             'caption_used': r.get(args.caption_field) or r.get('hand_caption'),
             'hand_caption': r.get('hand_caption'),
             'qwen3vl_caption': r.get('qwen3vl_caption'),
             }
            for r in records
        ]
    }
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'\n[DONE] Results in {out_dir}')


if __name__ == '__main__':
    main()
