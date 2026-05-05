"""\u5728 AutoDL \u4e0a\u8dd1 InstantX/Qwen-Image-ControlNet-Inpainting (\u5c0f\u7ea2\u4e66 2025/08 \u6700\u65b0)。

\u8f93\u5165: data/compare_5_captions.json + outputs/ip2p_pilot_100/<idx>_orig.png
\u8f93\u51fa: outputs/qwen_cn_compare/<idx>_qwen.png

\u4f9d\u8d56:
  pip install -U diffusers>=0.32.0 transformers accelerate hf_transfer

\u8d44\u6e90:
  Qwen-Image base = 20B params = ~40GB fp16 + Qwen2-VL-7B text encoder ~14GB = ~54GB total
  RTX 5090 32GB \u4e0d\u591f \u2192 \u7528 enable_model_cpu_offload() (peak ~20-24GB)
  \u9884\u8ba1\u63a8\u7406\u901f\u5ea6 ~60-90s/\u5f20 @ 1024 50 steps

\u8fd0\u884c (AutoDL):
  cd /root/autodl-tmp/IntelligenceCamera
  python tools/data/autodl_run_qwen_cn.py
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

# ---- \u73af\u5883\u53d6\u6d4b (important!) ----
# \u79fb\u9664\u5bf9\u5e94\u7684 hf-mirror (\u5bf9\u5927\u6587\u4ef6\u4f1a SSL EOF \u52a0\u7f6e)
os.environ.pop('HF_ENDPOINT', None)
# AutoDL \u56fd\u5185\u5b9a\u4f8b\u6bcd\u4e0d\u9700\u4ee3\u7406; \u8bf7\u7528 hf-mirror, \u8bf7\u6dfb --use_mirror \u53c2\u6570
os.environ.setdefault('HF_HOME', '/root/autodl-tmp/hf_cache')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', '/root/autodl-tmp/hf_cache/hub')
# rust \u4e0b\u8f7d\u5668 (\u6b63\u591a\u9ad8, \u5bf9\u5927\u6587\u4ef6\u6b63\u4e86\u5f88)
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

import torch
from PIL import Image
import numpy as np

QWEN_BASE = 'Qwen/Qwen-Image'
QWEN_CN_INPAINT = 'InstantX/Qwen-Image-ControlNet-Inpainting'


def make_full_mask(img_size):
    """\u751f\u6210 'soft full image' mask: \u8ba9\u6a21\u578b\u5728\u6574\u5e45\u91cd\u753b\u4f46\u4fdd\u7559 ControlNet \u8f6e\u5ed3\u3002

    Qwen-Image-CN-Inpainting \u9700\u8981 mask + image\u3002
    \u5168\u767d mask = \u5168\u91cd\u753b\u3002 \u5168\u9ed1 mask = \u4e0d\u91cd\u753b\u3002
    \u7528\u4e2d\u95f4\u706f mask (\u5168 0.5) \u8ba9\u6574\u4f53\u4f4e\u5f3a\u5ea6\u91cd\u753b\u3002
    """
    w, h = img_size
    return Image.new('L', (w, h), 255)  # \u5168\u767d = \u5168\u91cd\u753b


def make_partial_mask(img_size, ratio=0.5):
    """\u8f6f mask: \u4e2d\u95f4\u533a\u57df\u5168\u91cd\u753b\uff0c\u8fb9\u7f18\u4fdd\u7559\u3002\u9002\u5408\u52a8\u4e2d\u90e9\u3001\u4fdd\u8fb9\u7f18\u3002"""
    w, h = img_size
    arr = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    rx, ry = int(w * ratio / 2), int(h * ratio / 2)
    arr[max(0, cy - ry):min(h, cy + ry),
        max(0, cx - rx):min(w, cx + rx)] = 255
    return Image.fromarray(arr, mode='L')


def load_qwen_pipeline(cache_dir, mode='offload'):
    """\u52a0\u8f7d Qwen-Image-CN-Inpaint \u3002

    mode:
      'offload':    enable_model_cpu_offload  (peak ~20-24GB, RTX 5090 \u63a8\u8350)
      'sequential': enable_sequential_cpu_offload (peak ~6-8GB, \u6b61)
      'cuda':       \u5168\u90e9 GPU (need ~50GB VRAM, only A100 80GB / H100)
    """
    # \u5bfc\u5165 diffusers \u65b0\u7c7b (\u6765\u81ea\u9876\u7a0b namespace, \u517c\u5bb9\u6027\u6b63\u4e86\u597d)
    try:
        from diffusers import QwenImageControlNetInpaintPipeline
        from diffusers import QwenImageControlNetModel
    except ImportError as e:
        print(f'[ERROR] diffusers \u6216\u8005\u5bf9\u5e94\u8fc7\u4f4e: {e}')
        print('  \u8bf7\u8fd0\u884c: pip install -U diffusers>=0.32.0 transformers accelerate hf_transfer')
        raise

    weight_dtype = torch.bfloat16

    print(f'[INFO] Loading ControlNet from {QWEN_CN_INPAINT}...')
    t0 = time.time()
    controlnet = QwenImageControlNetModel.from_pretrained(
        QWEN_CN_INPAINT, torch_dtype=weight_dtype, cache_dir=cache_dir)
    print(f'    loaded in {time.time()-t0:.1f}s')

    print(f'[INFO] Loading Qwen-Image base from {QWEN_BASE}...')
    t0 = time.time()
    pipe = QwenImageControlNetInpaintPipeline.from_pretrained(
        QWEN_BASE,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
        cache_dir=cache_dir,
    )
    print(f'    loaded in {time.time()-t0:.1f}s')

    if mode == 'offload':
        print('[INFO] enable_model_cpu_offload (peak ~20-24GB)')
        pipe.enable_model_cpu_offload()
    elif mode == 'sequential':
        print('[INFO] enable_sequential_cpu_offload (peak ~6-8GB, slow)')
        pipe.enable_sequential_cpu_offload()
    else:
        print('[INFO] Full GPU mode (need ~50GB VRAM)')
        pipe.to('cuda')

    # VAE slicing/tiling \u7f6e\u987a\u72b6\u6001\u70b9\u70b9\u70b9
    try:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    except AttributeError:
        pass

    return pipe


def edit_one(pipe, content_img, caption, mask_img, cfg):
    """\u8dd1\u4e00\u5f20. generator \u7528 CPU \u9006\u51fa offload \u65f6\u8bbe\u5b50\u4e0d\u5339\u914d."""
    generator = torch.Generator('cpu').manual_seed(cfg['seed'])
    out = pipe(
        prompt=caption,
        negative_prompt=cfg.get('negative_prompt', ' '),
        control_image=content_img,
        mask_image=mask_img,
        controlnet_conditioning_scale=cfg['controlnet_conditioning_scale'],
        guidance_scale=cfg.get('guidance_scale', 1.0),       # \u86b9\u6e6f\u5bfc\u63a7, Qwen-Image \u63a8\u8350 1.0
        true_cfg_scale=cfg.get('true_cfg_scale', 4.0),        # \u7cbe\u9009 CFG, Qwen-Image \u63a8\u8350 4.0
        num_inference_steps=cfg.get('num_inference_steps', 50),
        generator=generator,
    )
    return out.images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', default='data/compare_5_captions.json')
    ap.add_argument('--out_dir', default='outputs/qwen_cn_compare')
    ap.add_argument('--cache_dir', default='/root/autodl-tmp/hf_cache')
    ap.add_argument('--mode', default='offload',
                    choices=['offload', 'sequential', 'cuda'])
    ap.add_argument('--mask_strategy', default='full',
                    choices=['full', 'partial'])
    ap.add_argument('--max_long_side', type=int, default=1280,
                    help='长边上限 (防止 OOM). Qwen-Image 原生 1024-1328')
    ap.add_argument('--use_mirror', action='store_true',
                    help='用 hf-mirror (大文件容易 SSL EOF, 不推荐)')
    args = ap.parse_args()

    if args.use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print('[WARN] Using hf-mirror - large files may SSL EOF, be patient')

    cfg_data = json.load(open(args.captions, encoding='utf-8'))
    samples = cfg_data['samples']
    qwen_cfg = cfg_data['qwen_image_cn_inpaint_config']
    print(f'[INFO] {len(samples)} samples')
    print(f'[INFO] config: cn_scale={qwen_cfg["controlnet_conditioning_scale"]}, '
          f'true_cfg={qwen_cfg.get("true_cfg_scale", 4.0)}, '
          f'steps={qwen_cfg.get("num_inference_steps", 50)}')

    # GPU check
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f'[GPU] {torch.cuda.get_device_name(0)} '
              f'free={free/1e9:.1f}GB / total={total/1e9:.1f}GB')

    pipe = load_qwen_pipeline(args.cache_dir, mode=args.mode)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in samples:
        idx = s['idx']
        orig_path = Path(s['orig_path'])
        if not orig_path.exists():
            print(f'[SKIP] idx={idx}: {orig_path} not found')
            continue
        content = Image.open(orig_path).convert('RGB')
        # Resize: 长边上限 max_long_side, 多边 16 的倍数
        w, h = content.size
        if max(w, h) != args.max_long_side:
            scale = args.max_long_side / max(w, h)
            new_w = int(w * scale) // 16 * 16
            new_h = int(h * scale) // 16 * 16
            content = content.resize((new_w, new_h), Image.LANCZOS)

        if args.mask_strategy == 'full':
            mask = make_full_mask(content.size)
        else:
            mask = make_partial_mask(content.size, ratio=0.6)

        print(f'\n[{idx}] {s["source_image"]}  ({content.size[0]}x{content.size[1]})')
        print(f'  caption: "{s["new_caption"]}"')
        t0 = time.time()
        try:
            result = edit_one(pipe, content, s['new_caption'], mask, qwen_cfg)
        except Exception as e:
            print(f'  FAIL: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            continue
        dt = time.time() - t0
        out_path = out_dir / f'{idx:04d}_qwen.png'
        result.save(out_path)
        content.save(out_dir / f'{idx:04d}_orig.png')
        print(f'  -> saved ({dt:.1f}s) -> {out_path}')

    print(f'\n[DONE] Results in {out_dir}')


if __name__ == '__main__':
    main()
