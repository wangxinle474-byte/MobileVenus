"""
测试 InstructPix2Pix 用 Venus 提取的 prompt 编辑图像.

用法:
  python tools/data/test_ip2p.py                # 跑 3 个样本
  python tools/data/test_ip2p.py --num 5        # 跑 5 个样本
  python tools/data/test_ip2p.py --indices 0 5 10  # 指定索引

输出: outputs/ip2p_test/<idx>_orig.png, <idx>_edit.png, <idx>_prompt.txt
"""
import os
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_pipeline(model_id='timbrooks/instruct-pix2pix', cache_dir=None):
    print(f'Loading {model_id}...')
    print(f'  HF_ENDPOINT={os.environ.get("HF_ENDPOINT")}')
    print(f'  cache_dir={cache_dir or "default"}')
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=cache_dir,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to('cuda:0')
    pipe.set_progress_bar_config(disable=False)
    return pipe


def edit_image(pipe, image_path, prompt, num_inference_steps=20,
               image_guidance_scale=1.5, guidance_scale=7.5, seed=42):
    """单张图编辑"""
    img = Image.open(image_path).convert('RGB')
    # IP2P 期望短边 ~512, 保持纵横比
    w, h = img.size
    if min(w, h) > 512:
        scale = 512.0 / min(w, h)
        new_w = int(w * scale) // 8 * 8
        new_h = int(h * scale) // 8 * 8
        img = img.resize((new_w, new_h), Image.LANCZOS)

    generator = torch.Generator('cuda:0').manual_seed(seed)
    out = pipe(
        prompt=prompt,
        image=img,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,  # 越大越保留原图
        guidance_scale=guidance_scale,              # 越大越遵循 prompt
        generator=generator,
    )
    return img, out.images[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_json', default='data/venus_edit_prompts.json')
    parser.add_argument('--output_dir', default='outputs/ip2p_test')
    parser.add_argument('--num', type=int, default=3, help='测试图片数 (从 --start 开始)')
    parser.add_argument('--start', type=int, default=0, help='起始索引 (断点续跑)')
    parser.add_argument('--indices', type=int, nargs='+', default=None, help='指定索引')
    parser.add_argument('--skip_existing', action='store_true', help='跳过已存在的编辑图')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--img_guidance', type=float, default=1.5)
    parser.add_argument('--prompt_guidance', type=float, default=7.5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('No CUDA', flush=True)
        return

    free, total = torch.cuda.mem_get_info()
    print(f'GPU: {torch.cuda.get_device_name(0)} '
          f'free={free/1e9:.1f}GB / total={total/1e9:.1f}GB')

    # 载入 prompts
    prompts_data = json.load(open(args.prompts_json, 'r', encoding='utf-8'))
    prompts = prompts_data['prompts']
    print(f'Loaded {len(prompts)} prompts')

    if args.indices is None:
        indices = list(range(args.start,
                             min(args.start + args.num, len(prompts))))
    else:
        indices = args.indices

    # 验证图片存在
    out_dir = PROJECT_ROOT / args.output_dir
    valid = []
    skipped_exist = 0
    for i in indices:
        p = prompts[i]
        if not Path(p['image_path']).exists():
            print(f'  SKIP [{i}] {p["image_path"]} not found')
            continue
        if args.skip_existing and (out_dir / f'{i:04d}_edit.png').exists():
            skipped_exist += 1
            continue
        valid.append((i, p))
    print(f'Valid: {len(valid)} samples'
          + (f' (skipped {skipped_exist} already done)' if skipped_exist else ''))
    if not valid:
        return

    # 加载 IP2P
    pipe = load_pipeline()

    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, p in valid:
        print(f'\n[{idx}] {Path(p["image_path"]).name}')
        print(f'  prompt: "{p["edit_prompt"]}"')
        try:
            orig, edited = edit_image(
                pipe,
                p['image_path'],
                p['edit_prompt'],
                num_inference_steps=args.steps,
                image_guidance_scale=args.img_guidance,
                guidance_scale=args.prompt_guidance,
            )
            orig.save(out_dir / f'{idx:04d}_orig.png')
            edited.save(out_dir / f'{idx:04d}_edit.png')
            with open(out_dir / f'{idx:04d}_prompt.txt', 'w', encoding='utf-8') as f:
                f.write(f'Image: {p["image"]}\n')
                f.write(f'Prompt: {p["edit_prompt"]}\n')
                f.write(f'Full suggestion: {p["suggestion_full"]}\n')
            print(f'  saved: {out_dir}/{idx:04d}_*')
        except Exception as e:
            print(f'  FAIL: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()

    print(f'\nDone. Results: {out_dir}')


if __name__ == '__main__':
    main()
