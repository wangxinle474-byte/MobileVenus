"""\u672c\u5730\u8dd1 CSGO \u63a8\u7406 (\u9002\u914d 8GB VRAM RTX 4060 Laptop)\u3002

\u9002\u914d\u70b9:
- HF \u7f13\u5b58: E:\\cache\\huggingface
- CSGO \u4ed3\u5e93: E:\\cache\\CSGO
- \u4f7f\u7528 enable_model_cpu_offload \u8282\u7701 VRAM
- \u8f93\u51fa: outputs/csgo_compare/<idx>_csgo.png

\u9884\u8ba1\u8d44\u6e90:
  \u9996\u6b21\u8fd0\u884c\u4e0b\u8f7d ~17 GB \u5230 E:\\cache\\huggingface
  VRAM peak \u7ea6 6-7 GB (cpu_offload)
  \u63a8\u7406\u7ea6 60-90s/\u5f20

\u8fd0\u884c:
  python tools/data/local_run_csgo.py
"""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

# HF \u76f4\u8fde + 7897 \u4ee3\u7406 (hf-mirror \u5728\u5927\u6587\u4ef6\u4e0a\u4f1a\u5361\u6b7b)
# \u79fb\u9664 HF_ENDPOINT (\u53ef\u80fd\u5df2\u8bbe\u6210 hf-mirror), \u5f3a\u5236\u8d70 huggingface.co
os.environ.pop('HF_ENDPOINT', None)
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ.setdefault('HF_HOME', r'E:\cache\huggingface')
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', r'E:\cache\huggingface\hub')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
# \u91cd\u8bd5\u4f30 + \u8d85\u65f6\u5ef6\u957f
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')

CACHE_DIR = r'E:\cache\huggingface'
CSGO_REPO = r'E:\cache\CSGO'

# \u6a21\u578b ID
SDXL_BASE = 'stabilityai/stable-diffusion-xl-base-1.0'
SDXL_VAE = 'madebyollin/sdxl-vae-fp16-fix'
IPA_ENCODER = 'h94/IP-Adapter'
CONTROLNET_TILE = 'TTPLanet/TTPLanet_SDXL_Controlnet_Tile_Realistic'
CSGO_CKPT_REPO = 'InstantX/CSGO'


def ensure_csgo_repo():
    """\u514b\u9686 CSGO \u4ed3\u5e93\u5e76\u5165\u53e3 sys.path."""
    repo = Path(CSGO_REPO)
    if not repo.exists():
        print(f'[INFO] Cloning CSGO repo to {repo}...')
        cmd = ['git', 'clone', '--depth', '1',
               'https://github.com/instantX-research/CSGO', str(repo)]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode != 0:
            print(f'[WARN] git clone (github) failed: {ret.stderr}')
            print('[INFO] Trying gitcode mirror...')
            cmd = ['git', 'clone', '--depth', '1',
                   'https://gitcode.com/gh_mirrors/csgo/CSGO',
                   str(repo)]
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode != 0:
                raise RuntimeError(f'CSGO clone failed: {ret.stderr}')
    else:
        print(f'[OK] CSGO repo exists at {repo}')

    # \u63d2\u5165 sys.path
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    print(f'[OK] CSGO repo in sys.path')


def download_models():
    """\u4ece HF \u4e0b\u8f7d\u6240\u9700\u6743\u91cd."""
    from huggingface_hub import snapshot_download

    print('\n[1/5] SDXL base...')
    sdxl_path = snapshot_download(
        SDXL_BASE, cache_dir=CACHE_DIR,
        allow_patterns=['*.json', '*.txt',
                        'text_encoder/*.safetensors',
                        'text_encoder_2/*.safetensors',
                        'tokenizer/*', 'tokenizer_2/*',
                        'unet/*.safetensors', 'unet/*.json',
                        'vae/*.safetensors', 'vae/*.json',
                        'scheduler/*'])
    print(f'    -> {sdxl_path}')

    print('\n[2/5] SDXL VAE fp16-fix...')
    vae_path = snapshot_download(SDXL_VAE, cache_dir=CACHE_DIR)
    print(f'    -> {vae_path}')

    print('\n[3/5] IP-Adapter image encoder (sdxl_models/image_encoder)...')
    ipa_path = snapshot_download(
        IPA_ENCODER, cache_dir=CACHE_DIR,
        allow_patterns=['sdxl_models/image_encoder/*'])
    print(f'    -> {ipa_path}')

    print('\n[4/5] ControlNet-Tile (TTPLanet)...')
    cn_path = snapshot_download(
        CONTROLNET_TILE, cache_dir=CACHE_DIR,
        allow_patterns=['*.safetensors', '*.json'])
    print(f'    -> {cn_path}')

    print('\n[5/5] CSGO checkpoint (csgo.bin)...')
    csgo_path = snapshot_download(
        CSGO_CKPT_REPO, cache_dir=CACHE_DIR,
        allow_patterns=['csgo.bin'])
    print(f'    -> {csgo_path}')

    return {
        'sdxl': sdxl_path,
        'vae': vae_path,
        'ipa_encoder': str(Path(ipa_path) / 'sdxl_models' / 'image_encoder'),
        'controlnet_tile': cn_path,
        'csgo_ckpt': str(Path(csgo_path) / 'csgo.bin'),
    }


def load_csgo_pipeline(paths):
    """\u52a0\u8f7d CSGO pipeline \u5e76\u542f\u7528 model CPU offload."""
    import torch
    from diffusers import (AutoencoderKL, ControlNetModel,
                           StableDiffusionXLControlNetPipeline)

    # CSGO \u4ed3\u5e93\u63d0\u4f9b
    from ip_adapter.utils import BLOCKS
    from ip_adapter.utils import controlnet_BLOCKS
    from ip_adapter import CSGO

    weight_dtype = torch.float16
    print('\n[Load] VAE...')
    vae = AutoencoderKL.from_pretrained(paths['vae'], torch_dtype=weight_dtype)

    print('[Load] ControlNet-Tile...')
    controlnet = ControlNetModel.from_pretrained(
        paths['controlnet_tile'], torch_dtype=weight_dtype, use_safetensors=True)

    print('[Load] SDXL pipeline...')
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        paths['sdxl'],
        controlnet=controlnet,
        torch_dtype=weight_dtype,
        add_watermarker=False,
        vae=vae,
    )
    pipe.enable_vae_tiling()
    # \u5173\u952e: \u5728 8GB GPU \u4e0a\u542f\u7528 CPU offload
    print('[Load] enable_model_cpu_offload()...')
    pipe.enable_model_cpu_offload()

    print('[Load] Wrapping with CSGO...')
    csgo = CSGO(
        pipe,
        paths['ipa_encoder'],
        paths['csgo_ckpt'],
        'cuda:0',
        num_content_tokens=4,
        num_style_tokens=32,
        target_content_blocks=BLOCKS['content'],
        target_style_blocks=BLOCKS['style'],
        controlnet=False,
        controlnet_adapter=True,
        controlnet_target_content_blocks=controlnet_BLOCKS['content'],
        controlnet_target_style_blocks=controlnet_BLOCKS['style'],
        content_model_resampler=True,
        style_model_resampler=True,
        load_controlnet=False,
    )
    return csgo


def edit_one(csgo, content_img, caption, cfg):
    """\u8dd1\u4e00\u5f20."""
    import torch
    images = csgo.generate(
        pil_content_image=content_img,
        pil_style_image=content_img,  # self-reference \u4fdd\u6301\u539f\u98ce\u683c
        prompt=caption,
        negative_prompt=cfg['negative_prompt'],
        content_scale=cfg['content_scale'],
        style_scale=cfg['style_scale'],
        guidance_scale=cfg['guidance_scale'],
        num_images_per_prompt=1,
        num_samples=1,
        num_inference_steps=cfg['num_inference_steps'],
        seed=cfg['seed'],
        image=content_img.convert('RGB'),
        controlnet_conditioning_scale=cfg['controlnet_conditioning_scale'],
    )
    return images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', default='data/compare_5_captions.json')
    ap.add_argument('--out_dir', default='outputs/csgo_compare')
    ap.add_argument('--short_side', type=int, default=768,
                    help='\u8f93\u5165\u77ed\u8fb9 (768 \u8282\u7701 VRAM, 1024 \u8d28\u91cf\u66f4\u9ad8)')
    args = ap.parse_args()

    # 1. CSGO \u4ed3\u5e93
    ensure_csgo_repo()

    # 2. \u8bfb\u914d\u7f6e
    cfg_data = json.load(open(args.captions, encoding='utf-8'))
    samples = cfg_data['samples']
    csgo_cfg = cfg_data['csgo_inference_config']
    print(f'\n[INFO] {len(samples)} samples to edit')

    # 3. \u4e0b\u8f7d\u6743\u91cd
    paths = download_models()

    # 4. \u52a0\u8f7d pipeline
    csgo = load_csgo_pipeline(paths)

    # 5. \u9010\u5f20\u8dd1
    from PIL import Image
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    import time
    for s in samples:
        idx = s['idx']
        orig_path = PROJECT_ROOT / s['orig_path']
        if not orig_path.exists():
            print(f'[SKIP] idx={idx}: {orig_path} not found')
            continue
        content = Image.open(orig_path).convert('RGB')
        # Resize
        w, h = content.size
        if min(w, h) != args.short_side:
            scale = args.short_side / min(w, h)
            new_w = int(w * scale) // 8 * 8
            new_h = int(h * scale) // 8 * 8
            content = content.resize((new_w, new_h), Image.LANCZOS)

        print(f'\n[idx={idx}] {s["source_image"]}  ({content.size[0]}x{content.size[1]})')
        print(f'  caption: "{s["new_caption"]}"')
        t0 = time.time()
        try:
            result = edit_one(csgo, content, s['new_caption'], csgo_cfg)
        except Exception as e:
            print(f'  FAIL: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            continue
        dur = time.time() - t0

        result.save(out_dir / f'{idx:04d}_csgo.png')
        content.save(out_dir / f'{idx:04d}_orig.png')
        print(f'  -> saved ({dur:.1f}s) -> {out_dir / f"{idx:04d}_csgo.png"}')

    print(f'\n[DONE] Results in {out_dir.resolve()}')


if __name__ == '__main__':
    main()
