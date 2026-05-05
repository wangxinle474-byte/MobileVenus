"""\u5728 AutoDL \u4e0a\u8dd1 CSGO (\u5c0f\u7ea2\u4e66 InstantX \u5b98\u5408) \u56fe\u50cf\u7f16\u8f91\u3002

\u8f93\u5165: data/compare_5_captions.json + outputs/ip2p_pilot_100/<idx>_orig.png
\u8f93\u51fa: outputs/csgo_compare/<idx>_csgo.png

\u4f9d\u8d56:
  pip install diffusers transformers accelerate safetensors
  CSGO \u4ed3\u5e93: git clone https://github.com/instantX-research/CSGO

\u8fd0\u884c (AutoDL):
  cd /root/autodl-tmp/IntelligenceCamera
  python tools/data/autodl_run_csgo.py
"""
import os
import sys
import json
import argparse
from pathlib import Path

# HF \u955c\u50cf
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HOME', '/root/autodl-tmp/hf_cache')

import torch
from PIL import Image


# CSGO \u4ed3\u5e93\u5728 AutoDL \u4e0a\u7684\u8def\u5f84
CSGO_REPO = '/root/autodl-tmp/CSGO'

# \u6a21\u578b\u4e0b\u8f7d\u8def\u5f84 (\u5728 hf_cache \u91cc)
SDXL_BASE = 'stabilityai/stable-diffusion-xl-base-1.0'
SDXL_VAE = 'madebyollin/sdxl-vae-fp16-fix'
IPA_ENCODER = 'h94/IP-Adapter'  # use sdxl_models/image_encoder
CONTROLNET_TILE = 'TTPLanet/TTPLanet_SDXL_Controlnet_Tile_Realistic'
CSGO_CKPT_REPO = 'InstantX/CSGO'


def setup_csgo_repo():
    """\u514b\u9686 CSGO \u4ed3\u5e93\u5e76\u5165\u53e3 sys.path."""
    if not Path(CSGO_REPO).exists():
        print(f'[INFO] Cloning CSGO repo to {CSGO_REPO}...')
        ret = os.system(f'git clone https://github.com/instantX-research/CSGO {CSGO_REPO}')
        if ret != 0:
            print('[WARN] git clone failed, try gitee mirror...')
            os.system(f'git clone https://gitee.com/mirrors/CSGO {CSGO_REPO}')

    sys.path.insert(0, CSGO_REPO)
    print(f'[INFO] CSGO repo at {CSGO_REPO}')


def download_models(cache_dir):
    """\u4ece HF \u4e0b\u8f7d\u9700\u8981\u7684\u6a21\u578b\u3002"""
    from huggingface_hub import snapshot_download
    print('[INFO] Checking SDXL base...')
    sdxl_path = snapshot_download(SDXL_BASE, cache_dir=cache_dir,
                                  allow_patterns=['*.safetensors', '*.json',
                                                   'tokenizer*/*', 'scheduler/*'])
    print(f'  SDXL: {sdxl_path}')

    print('[INFO] Checking SDXL VAE fp16-fix...')
    vae_path = snapshot_download(SDXL_VAE, cache_dir=cache_dir)
    print(f'  VAE: {vae_path}')

    print('[INFO] Checking IP-Adapter image encoder...')
    ipa_path = snapshot_download(
        IPA_ENCODER, cache_dir=cache_dir,
        allow_patterns=['sdxl_models/image_encoder/*'])
    print(f'  IP-Adapter: {ipa_path}')

    print('[INFO] Checking ControlNet-Tile...')
    cn_path = snapshot_download(CONTROLNET_TILE, cache_dir=cache_dir)
    print(f'  ControlNet-Tile: {cn_path}')

    print('[INFO] Checking CSGO checkpoint...')
    csgo_path = snapshot_download(CSGO_CKPT_REPO, cache_dir=cache_dir,
                                   allow_patterns=['csgo.bin', 'csgo_4_32.bin'])
    print(f'  CSGO: {csgo_path}')

    return {
        'sdxl': sdxl_path,
        'vae': vae_path,
        'ip_adapter_encoder': str(Path(ipa_path) / 'sdxl_models' / 'image_encoder'),
        'controlnet_tile': cn_path,
        'csgo_ckpt': str(Path(csgo_path) / 'csgo.bin'),
    }


def load_csgo_pipeline(paths, device='cuda:0'):
    """\u52a0\u8f7d CSGO pipeline."""
    from diffusers import (AutoencoderKL, ControlNetModel,
                           StableDiffusionXLControlNetPipeline)
    # \u4ece CSGO \u4ed3\u5e93\u5bfc\u5165
    from ip_adapter.utils import BLOCKS as BLOCKS
    from ip_adapter.utils import controlnet_BLOCKS as controlnet_BLOCKS
    from ip_adapter import CSGO

    weight_dtype = torch.float16
    print('[INFO] Loading VAE...')
    vae = AutoencoderKL.from_pretrained(paths['vae'], torch_dtype=weight_dtype)

    print('[INFO] Loading ControlNet-Tile...')
    controlnet = ControlNetModel.from_pretrained(
        paths['controlnet_tile'], torch_dtype=weight_dtype, use_safetensors=True)

    print('[INFO] Loading SDXL pipeline...')
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        paths['sdxl'],
        controlnet=controlnet,
        torch_dtype=weight_dtype,
        add_watermarker=False,
        vae=vae,
    )
    pipe.enable_vae_tiling()

    print('[INFO] Wrapping with CSGO...')
    csgo = CSGO(
        pipe,
        paths['ip_adapter_encoder'],
        paths['csgo_ckpt'],
        device,
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


def edit_one(csgo, content_img, caption, cfg, idx):
    """\u8dd1\u4e00\u5f20."""
    generator = torch.Generator('cuda:0').manual_seed(cfg['seed'])
    images = csgo.generate(
        pil_content_image=content_img,
        pil_style_image=content_img,  # \u81ea\u5f15\u7528: \u4fdd\u6301\u539f\u56fe\u98ce\u683c
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
    ap.add_argument('--captions',
                    default='data/compare_5_captions.json')
    ap.add_argument('--out_dir', default='outputs/csgo_compare')
    ap.add_argument('--cache_dir',
                    default='/root/autodl-tmp/hf_cache')
    args = ap.parse_args()

    # 1. \u51c6\u5907\u73af\u5883
    setup_csgo_repo()

    # 2. \u8bfb\u53d6\u914d\u7f6e
    cfg_data = json.load(open(args.captions, encoding='utf-8'))
    samples = cfg_data['samples']
    csgo_cfg = cfg_data['csgo_inference_config']
    print(f'[INFO] {len(samples)} samples to edit')

    # 3. \u4e0b\u8f7d\u6a21\u578b
    paths = download_models(args.cache_dir)

    # 4. \u52a0\u8f7d pipeline
    csgo = load_csgo_pipeline(paths)

    # 5. \u9010\u5f20\u8dd1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        idx = s['idx']
        orig_path = Path(s['orig_path'])
        if not orig_path.exists():
            print(f'[SKIP] idx={idx}: {orig_path} not found')
            continue
        content = Image.open(orig_path).convert('RGB')
        # Resize to 1024 short side for SDXL
        w, h = content.size
        if min(w, h) > 1024:
            scale = 1024 / min(w, h)
            content = content.resize((int(w * scale) // 8 * 8,
                                       int(h * scale) // 8 * 8), Image.LANCZOS)

        print(f'\n[{idx}] {s["source_image"]}')
        print(f'  caption: "{s["new_caption"]}"')
        result = edit_one(csgo, content, s['new_caption'], csgo_cfg, idx)
        result.save(out_dir / f'{idx:04d}_csgo.png')
        # \u4e5f\u590d\u5236\u539f\u56fe\u4fbf\u4e8e\u5bf9\u6bd4
        content.save(out_dir / f'{idx:04d}_orig.png')
        print(f'  -> saved {out_dir / f"{idx:04d}_csgo.png"}')

    print(f'\n[DONE] Results in {out_dir}')


if __name__ == '__main__':
    main()
