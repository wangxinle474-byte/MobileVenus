"""\u672c\u5730\u8dd1 LongCat-Image-Edit-Turbo (\u7f8e\u56e2 2024/12 \u53d1, \u539f\u751f 8 NFE \u84b8\u998f\u7248)\u3002

\u9002\u914d 8 GB VRAM (RTX 4060 Laptop) \u7684\u73af\u5883:
  - enable_sequential_cpu_offload() \u5ce8 peak VRAM ~6-8 GB
  - \u63a8\u7406 5-10 \u5206\u949f/\u5f20 (5 \u5f20 ~ 30-50 \u5206\u949f)
  - \u4e0b\u8f7d ~30 GB \u6743\u91cd\u5230 E:\\cache\\huggingface (\u9996\u6b21\u8fd0\u884c)

Pipeline (diffusers 0.37.1 \u5df2\u81ea\u5e26):
  LongCatImageEditPipeline - 6B DiT + Qwen2.5-VL text encoder

\u8fd0\u884c:
  python tools/data/editor_models/local_run_longcat_turbo.py
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

# ---- 环境变量 (走 hf-mirror.com 镜像, 不需要代理) ----
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('https_proxy', None)
os.environ.pop('http_proxy', None)
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')   # rust 下载器, 大文件更稳
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
# 减少 8GB VRAM 上的内存碎片
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

# 缓存路径 (Windows 默认; Linux/AutoDL 通过 LONGCAT_MS_CACHE 覆盖)
if os.name == 'nt':
    _DEFAULT_MS_CACHE = r'E:\cache\modelscope'
    _DEFAULT_HF_CACHE = r'E:\cache\huggingface'
else:
    _DEFAULT_MS_CACHE = '/root/autodl-tmp/cache/modelscope'
    _DEFAULT_HF_CACHE = '/root/autodl-tmp/cache/huggingface'
os.environ.setdefault('HF_HOME', _DEFAULT_HF_CACHE)
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(_DEFAULT_HF_CACHE, 'hub'))

import torch
from PIL import Image

MS_CACHE_DIR = os.environ.get('LONGCAT_MS_CACHE', _DEFAULT_MS_CACHE)
MODEL_ID = 'meituan-longcat/LongCat-Image-Edit-Turbo'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def ensure_pkg(pkg):
    """\u68c0\u67e5 pip \u5305\u662f\u5426\u5b89\u88c5, \u7f3a\u5c31\u63d0\u793a."""
    import importlib.util
    if importlib.util.find_spec(pkg) is None:
        print(f'[WARN] missing {pkg}. Install: pip install {pkg}')
        return False
    return True


def download_weights():
    """从 ModelScope 下载权重 (国内镜像, 支持断点续传)."""
    from modelscope import snapshot_download as ms_snapshot_download

    print(f'[DOWNLOAD] {MODEL_ID} from ModelScope')
    print(f'           路径: {MS_CACHE_DIR}')
    print('           首次需 ~30 GB, 国内下载 预计 30-90 分')
    t0 = time.time()
    path = ms_snapshot_download(
        model_id=MODEL_ID,
        cache_dir=MS_CACHE_DIR,
        allow_patterns=['*.json', '*.txt', '*.safetensors',
                        '*.model', 'tokenizer*/*', '*.py'],
    )
    print(f'[OK] weights at {path}  ({time.time()-t0:.0f}s)')
    return path


def load_pipeline(model_path, use_4bit=False, offload='sequential'):
    """加载 LongCatImageEditPipeline.

    官方写法 (8GB+ VRAM, 16GB+ free RAM):
        pipe = LongCatImageEditPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()

    Args:
        model_path: 本地模型路径
        use_4bit: 对 transformer 4-bit nf4 量化.
                  bf16 transformer 需 ~12GB free RAM 上资, 4-bit 仅 ~3GB.
                  RAM 不足 必需开。
        offload: 'sequential' | 'model' | 'none'
                  - sequential: submodule 级 (最省 VRAM, 但 accelerate 有 meta tensor bug)
                  - model: 组件级 (Qwen-VL 7GB 组件 8GB VRAM 可能 OOM)
                  - none: 全 GPU (需 ≥24GB VRAM, RTX 5090 32GB 适用)
    """
    from diffusers import LongCatImageEditPipeline

    if use_4bit:
        from diffusers import LongCatImageTransformer2DModel, BitsAndBytesConfig
        print(f'[LOAD] transformer (4-bit nf4) from {model_path}/transformer')
        t0 = time.time()
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder='transformer',
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
        )
        print(f'    transformer 4-bit loaded ({time.time()-t0:.0f}s)')

        print(f'[LOAD] LongCatImageEditPipeline (住量化 transformer + bf16 其他)')
        t0 = time.time()
        pipe = LongCatImageEditPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        print(f'    pipeline loaded ({time.time()-t0:.0f}s)')
    else:
        print(f'[LOAD] LongCatImageEditPipeline (bf16) from {model_path}')
        t0 = time.time()
        pipe = LongCatImageEditPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16
        )
        print(f'    base loaded in {time.time()-t0:.0f}s')

    if offload == 'sequential':
        print('[LOAD] enable_sequential_cpu_offload (submodule-level, 8GB VRAM)')
        pipe.enable_sequential_cpu_offload()
    elif offload == 'model':
        print('[LOAD] enable_model_cpu_offload (component-level, ~12-16GB VRAM)')
        pipe.enable_model_cpu_offload()
    elif offload == 'none':
        print('[LOAD] to(cuda, bfloat16)  no offload, full GPU (≥24GB VRAM)')
        pipe.to('cuda', torch.bfloat16)
    else:
        raise ValueError(f'offload must be sequential/model/none, got {offload}')

    # VAE tiling/slicing 进一步降 conv 中间激活内存
    try:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    except AttributeError:
        pass

    return pipe


def edit_one(pipe, content_img, caption, cfg):
    """跑一张. 官方示例写法: img/prompt 位置参, negative_prompt=''.

    官方推荐 Turbo 参数:
      guidance_scale=1, num_inference_steps=8
    """
    generator = torch.Generator('cpu').manual_seed(cfg['seed'])
    out = pipe(
        content_img,
        caption,
        negative_prompt='',
        guidance_scale=cfg.get('guidance_scale', 1),
        num_inference_steps=cfg['num_inference_steps'],
        num_images_per_prompt=1,
        generator=generator,
    )
    return out.images[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', default='data/compare_5_captions.json')
    ap.add_argument('--out_dir', default='outputs/longcat_compare')
    ap.add_argument('--max_long_side', type=int, default=1024,
                    help='\u957f\u8fb9\u4e0a\u9650 (\u9632 OOM)')
    ap.add_argument('--steps', type=int, default=4,
                    help='NFE \u6b65\u6570 (\u5b98\u65b9\u63a8\u8350 8, \u63d0\u901f\u53ef\u8bd5 4)')
    ap.add_argument('--guidance_scale', type=float, default=1.0,
                    help='Turbo 蒸馏 1.0 = 不用 CFG (默认 4.5 是非 Turbo)')
    ap.add_argument('--skip_download', action='store_true',
                    help='已下好的跳过下载步骤')
    ap.add_argument('--model_path', default=None,
                    help='手工指定本地模型路径 (跳过自动下载/查找)')
    ap.add_argument('--use_4bit', action='store_true',
                    help='4-bit nf4 量化 transformer (RAM/VRAM 不足时必开)')
    ap.add_argument('--offload', default='sequential',
                    choices=['sequential', 'model', 'none'],
                    help='VRAM 策略: sequential(最省)/model(官方)/none(全开)')
    args = ap.parse_args()

    # GPU \u68c0\u67e5
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f'[GPU] {torch.cuda.get_device_name(0)} '
              f'free={free/1e9:.1f}GB / total={total/1e9:.1f}GB')
    else:
        print('[WARN] CUDA not available, will use CPU (extremely slow)')

    # \u8bfb\u53d6\u914d\u7f6e
    cfg_data = json.load(open(PROJECT_ROOT / args.captions, encoding='utf-8'))
    samples = cfg_data['samples']
    print(f'[INFO] {len(samples)} samples, {args.steps} NFE')

    # 1. 下载 / 定位本地路径
    if args.model_path:
        model_path = args.model_path
        print(f'[USE] model_path={model_path}')
    elif args.skip_download:
        # 查找已下载的 ModelScope 路径
        ms_dir = Path(MS_CACHE_DIR) / MODEL_ID
        if not ms_dir.exists():
            raise FileNotFoundError(f'--skip_download 但未找到 {ms_dir}, 去掉 --skip_download')
        model_path = str(ms_dir)
        print(f'[USE] cached at {model_path}')
    else:
        model_path = download_weights()

    # 2. 加载
    pipe = load_pipeline(model_path, use_4bit=args.use_4bit, offload=args.offload)

    # 3. \u9010\u5f20\u8dd1
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        'num_inference_steps': args.steps,
        'guidance_scale': args.guidance_scale,
        'seed': 43,   # 官方示例用 43
    }

    for s in samples:
        idx = s['idx']
        orig_path = PROJECT_ROOT / s['orig_path']
        if not orig_path.exists():
            print(f'[SKIP] idx={idx}: missing {orig_path}')
            continue
        img = Image.open(orig_path).convert('RGB')
        # Resize: \u957f\u8fb9\u4e0a\u9650 max_long_side
        w, h = img.size
        if max(w, h) != args.max_long_side:
            scale = args.max_long_side / max(w, h)
            new_w = int(w * scale) // 16 * 16
            new_h = int(h * scale) // 16 * 16
            img = img.resize((new_w, new_h), Image.LANCZOS)

        print(f'\n[idx={idx}] {s["source_image"]}  ({img.size[0]}x{img.size[1]})')
        print(f'  caption: "{s["new_caption"]}"')
        # 清理上一轮残留的显存碎片
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t0 = time.time()
        try:
            result = edit_one(pipe, img, s['new_caption'], cfg)
        except Exception as e:
            print(f'  FAIL: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            continue
        dt = time.time() - t0
        # \u65b0\u5e03\u5c40: \u4ec5\u4fdd\u5b58\u7f16\u8f91\u540e\u56fe\u4e3a <idx>.png, \u539f\u56fe\u5728 outputs/compare_5/originals/ \u72ec\u7acb\u4ee3\u7ba1
        out_path = out_dir / f'{idx:04d}.png'
        result.save(out_path)
        print(f'  -> saved ({dt:.0f}s) -> {out_path.name}')

    print(f'\n[DONE] Results in {out_dir}')


if __name__ == '__main__':
    main()
