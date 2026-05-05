"""\u7528\u672c\u5730\u7f13\u5b58\u7684 SDXL-Turbo + img2img \u6a21\u5f0f \u8dd1 5 \u5f20 caption-driven \u7f16\u8f91\u3002

\u7528\u9014: \u5feb\u901f\u9a8c\u8bc1 'a photo of ...' caption \u683c\u5f0f\u7684 aesthetic editing \u6548\u679c.
\u6a21\u578b: stabilityai/sdxl-turbo (\u5df2\u7f13\u5b58 6.5GB)
\u8bbe\u8ba1: \u6bcf\u5f20\u8dd1 3 \u4e2a strength (0.3/0.5/0.7) \u770b\u54ea\u4e2a\u4fdd\u7559/\u7f16\u8f91\u5e73\u8861\u6700\u597d

\u8fd0\u884c:
  python tools/data/local_run_sdxl_turbo.py
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

# \u6e05\u7406\u73af\u5883 (\u4e0d\u8d70\u4e0b\u8f7d, \u7eaf\u672c\u5730\u7f13\u5b58)
os.environ.pop('HF_ENDPOINT', None)
os.environ['HF_HOME'] = r'E:\cache\huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = r'E:\cache\huggingface\hub'
os.environ['HF_HUB_OFFLINE'] = '1'   # \u5f3a\u5236 offline, \u4e0d\u8054\u7f51
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image


def load_pipeline():
    """\u52a0\u8f7d SDXL-Turbo img2img pipeline (8GB VRAM \u53cb\u597d)."""
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


def edit_one(pipe, orig_img, caption, strength, steps=4, guidance=0.0, seed=42):
    """SDXL-Turbo img2img: strength \u8d8a\u5927 \u7f16\u8f91\u8d8a\u591a, \u8d8a\u5c0f \u539f\u56fe\u4fdd\u7559\u8d8a\u591a."""
    generator = torch.Generator('cuda').manual_seed(seed)
    out = pipe(
        prompt=caption,
        image=orig_img,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,  # SDXL-Turbo \u4e0d\u9700 CFG (\u7528 0)
        generator=generator,
    )
    return out.images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', default='data/compare_5_captions.json')
    ap.add_argument('--out_dir', default='outputs/sdxl_turbo_compare')
    ap.add_argument('--strengths', nargs='+', type=float,
                    default=[0.3, 0.5, 0.7],
                    help='3 \u4e2a strength \u503c, \u4f4e\u503c\u4fdd\u7559\u539f\u56fe\u9ad8\u503c\u7f16\u8f91\u52a0\u5927')
    ap.add_argument('--steps', type=int, default=4,
                    help='SDXL-Turbo \u63a8\u8350 1-4 \u6b65')
    ap.add_argument('--short_side', type=int, default=768)
    args = ap.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    cfg_data = json.load(open(PROJECT_ROOT / args.captions, encoding='utf-8'))
    samples = cfg_data['samples']
    print(f'[INFO] {len(samples)} samples, strengths={args.strengths}, steps={args.steps}')

    # GPU \u68c0\u67e5
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f'[GPU] {torch.cuda.get_device_name(0)} '
              f'free={free/1e9:.1f}GB / total={total/1e9:.1f}GB')

    pipe = load_pipeline()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in samples:
        idx = s['idx']
        orig_path = PROJECT_ROOT / s['orig_path']
        if not orig_path.exists():
            print(f'[SKIP] idx={idx}: missing {orig_path}')
            continue
        img = Image.open(orig_path).convert('RGB')
        # Resize: SDXL \u63a8\u8350 \u0142\u7684\u8fb9 >=768
        w, h = img.size
        if min(w, h) != args.short_side:
            scale = args.short_side / min(w, h)
            new_w = int(w * scale) // 8 * 8
            new_h = int(h * scale) // 8 * 8
            img = img.resize((new_w, new_h), Image.LANCZOS)

        print(f'\n[idx={idx}] {s["source_image"]}  ({img.size[0]}x{img.size[1]})')
        print(f'  caption: "{s["new_caption"]}"')
        # \u4fdd\u5b58\u539f\u56fe
        img.save(out_dir / f'{idx:04d}_orig.png')

        for st in args.strengths:
            t0 = time.time()
            try:
                result = edit_one(pipe, img, s['new_caption'],
                                  strength=st, steps=args.steps, seed=42)
            except Exception as e:
                print(f'  FAIL strength={st}: {type(e).__name__}: {e}')
                continue
            dt = time.time() - t0
            out_path = out_dir / f'{idx:04d}_sdxl_s{int(st*10):02d}.png'
            result.save(out_path)
            print(f'  strength={st}  ({dt:.1f}s)  -> {out_path.name}')

    print(f'\n[DONE] Results in {out_dir}')


if __name__ == '__main__':
    main()
