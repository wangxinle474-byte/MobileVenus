"""
Venus-Q-Stage1 (闭源 MLLM) 重打分

使用我们自己的 Venus (Qwen-VL-Chat 微调) 给增强图打结构化美学分。
Venus 输出 6 维分数 (composition / lighting / color / clarity / subject / overall),
每维 1-10, 比 MUSIQ-AVA 和 LAION-Aes 更接近"专业摄影评审"风格.

Usage:
  python tools/data/rescore_with_venus.py --input outputs/data/aug_high_score.json --max_samples 50
  python tools/data/rescore_with_venus.py --input outputs/data/aug_high_score.json --top_n 50 --sort_by aug_musiq
"""
import os
import re
import sys
import json
import time
import argparse
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.diff_isp import apply_diff_isp
from training.fivek_8param.config import PARAM_NAMES

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


VENUS_DEFAULT_PATH = Path('e:/智能相机/Venus_CVPR2026-main/pretrained_weights/Venus-Q-Stage1')

DIMENSION_NAMES = ["composition", "lighting", "color", "clarity", "subject"]

EVAL_PROMPT = (
    "You are a professional photography critic. "
    "Rate this photo on a scale of 1-10 for each dimension. Be strict and precise.\n\n"
    "Please output ONLY the scores in this exact format:\n"
    "Composition: X\n"
    "Lighting: X\n"
    "Color: X\n"
    "Clarity: X\n"
    "Subject: X\n"
    "Overall: X\n\n"
    "Where X is a number from 1 to 10 (can use decimals like 7.5).\n"
    "Do not add any explanation."
)


def load_venus(venus_path: Path, load_in_4bit: bool = False,
               load_in_8bit: bool = False, cpu_only: bool = False):
    """直接用 transformers 加载 Venus, 不依赖 Django

    8GB GPU 加载策略:
      - load_in_4bit: ~5GB GPU (推荐)
      - load_in_8bit: ~10GB GPU (8GB GPU 装不下, 会自动 CPU offload)
      - cpu_only:     全 CPU 推理 (慢但稳)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f'加载 Venus: {venus_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        str(venus_path), trust_remote_code=True, local_files_only=True
    )

    kwargs = {
        'trust_remote_code': True,
        'local_files_only': True,
        # 流式加载避免 17GB 模型一次性挤爆 RAM (本机仅 5GB free)
        'low_cpu_mem_usage': True,
    }

    if cpu_only:
        kwargs['device_map'] = 'cpu'
        kwargs['torch_dtype'] = torch.float32
        logger.info('  模式: CPU only (慢)')
    elif load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
        )
        # 强制全部放 GPU 0; 'auto' 会误把模块分到 CPU/disk 导致 BNB 报错
        kwargs['device_map'] = {'': 0}
        logger.info('  模式: 4-bit nf4 量化, 全模型放 GPU 0 (~5GB)')
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        kwargs['device_map'] = 'auto'
        logger.info('  模式: 8-bit 量化 (~10GB GPU+CPU offload)')
    else:
        kwargs['device_map'] = 'auto'
        kwargs['bf16'] = True
        logger.info('  模式: bf16 (需 ~16GB GPU)')

    model = AutoModelForCausalLM.from_pretrained(str(venus_path), **kwargs).eval()

    if hasattr(model, 'generation_config'):
        model.generation_config.return_dict_in_generate = False
    logger.info('Venus 加载完成')
    return model, tokenizer


def parse_scores(response: str) -> Dict[str, float]:
    """解析 Venus 输出为 6 维分数"""
    scores: Dict[str, float] = {}
    for dim in DIMENSION_NAMES + ['overall']:
        m = re.search(rf"{dim}\s*[:：]\s*(\d+\.?\d*)", response, re.IGNORECASE)
        if m:
            scores[dim] = min(max(float(m.group(1)), 1.0), 10.0)
    if len(scores) < 6:
        nums = [float(n) for n in re.findall(r"\d+\.?\d*", response)
                if 1.0 <= float(n) <= 10.0]
        for j, dim in enumerate(DIMENSION_NAMES + ['overall']):
            if dim not in scores and j < len(nums):
                scores[dim] = nums[j]
    for dim in DIMENSION_NAMES + ['overall']:
        scores.setdefault(dim, 5.0)
    return scores


@torch.no_grad()
def venus_score(image_path: str, model, tokenizer) -> Optional[Dict]:
    try:
        query = tokenizer.from_list_format([
            {'image': image_path},
            {'text': EVAL_PROMPT},
        ])
        response, _ = model.chat(tokenizer, query=query, history=None)
        parsed = parse_scores(response)
        return {
            'overall': round(parsed['overall'], 2),
            'dimensions': {d: round(parsed[d], 2) for d in DIMENSION_NAMES},
            'raw_response': response,
        }
    except Exception as e:
        warnings.warn(f'venus_score failed for {image_path}: {e}')
        return None


def render_to_path(orig_pil: Image.Image, params: Dict[str, float],
                   image_size: int, device: torch.device, out_path: str):
    """用 diff_ISP 渲染并保存到文件 (Venus 需要文件路径)"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    tensor = transform(orig_pil).unsqueeze(0).to(device)
    params_torch = {p: torch.tensor([float(params.get(p, 0))],
                                    device=device, dtype=torch.float32)
                    for p in PARAM_NAMES}
    rendered = apply_diff_isp(tensor, params_torch).clamp(0, 1)
    arr = (rendered.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    Image.fromarray(arr).save(out_path, quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--output', default='outputs/data/aug_venus_scored.json')
    parser.add_argument('--filtered_output', default='outputs/data/aug_venus_filtered.json')
    parser.add_argument('--venus_path', default=str(VENUS_DEFAULT_PATH))
    parser.add_argument('--max_samples', type=int, default=50,
                        help='最多评分样本数')
    parser.add_argument('--top_n', type=int, default=0,
                        help='按 sort_by 字段降序取 top_n (优先于 max_samples 子采样)')
    parser.add_argument('--sort_by', default='aug_musiq',
                        help='top_n 排序依据字段名')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--filter_min', type=float, default=7.0,
                        help='Venus overall >= 此分保留')
    parser.add_argument('--score_orig', action='store_true',
                        help='同时给原图打分作为 baseline')
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='4-bit nf4 量化加载 (~5GB GPU, 推荐 8GB 显卡)')
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='8-bit 量化加载 (~10GB)')
    parser.add_argument('--cpu_only', action='store_true',
                        help='全 CPU 推理 (慢但不需要 GPU)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # 加载样本
    all_samples = []
    for fp in args.input:
        if not Path(fp).exists():
            logger.warning(f'文件不存在: {fp}')
            continue
        with open(fp, 'r', encoding='utf-8') as f:
            d = json.load(f)
        items = d.get('samples', [])
        for it in items:
            it['_source'] = fp
        all_samples.extend(items)
        logger.info(f'加载 {fp}: {len(items)} 条')
    logger.info(f'总计 {len(all_samples)} 条')

    # 选择评分子集
    if args.top_n > 0 and args.sort_by in (all_samples[0] if all_samples else {}):
        sorted_samples = sorted(all_samples,
                                key=lambda s: float(s.get(args.sort_by, 0)),
                                reverse=True)
        score_samples = sorted_samples[:args.top_n]
        logger.info(f'按 {args.sort_by} 取 top_n={args.top_n}')
    else:
        rng = np.random.RandomState(args.seed)
        if args.max_samples > 0 and len(all_samples) > args.max_samples:
            idx = rng.choice(len(all_samples), args.max_samples, replace=False)
            score_samples = [all_samples[i] for i in idx]
        else:
            score_samples = all_samples
    logger.info(f'评分样本数: {len(score_samples)}')

    # 加载 Venus
    venus_path = Path(args.venus_path)
    if not (venus_path / 'config.json').exists():
        logger.error(f'Venus 路径无效: {venus_path}')
        return
    model, tokenizer = load_venus(
        venus_path,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        cpu_only=args.cpu_only,
    )

    scored = []
    orig_score_cache: Dict[str, Dict] = {}
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix='venus_aug_') as tmpdir:
        for idx, s in enumerate(score_samples):
            img_path = s.get('image_path', '')
            image_name = s.get('image', '')
            if not Path(img_path).exists():
                continue

            try:
                orig_pil = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.warning(f'读图失败 {img_path}: {e}')
                continue

            # 1. 原图分数 (缓存)
            if args.score_orig and image_name not in orig_score_cache:
                orig_tmp = os.path.join(tmpdir, f'orig_{idx}.jpg')
                orig_pil_resized = orig_pil.resize((args.image_size, args.image_size))
                orig_pil_resized.save(orig_tmp, quality=92)
                orig_sc = venus_score(orig_tmp, model, tokenizer)
                if orig_sc is not None:
                    orig_score_cache[image_name] = orig_sc

            # 2. 渲染增强图并打分
            aug_tmp = os.path.join(tmpdir, f'aug_{idx}.jpg')
            render_to_path(orig_pil, s.get('augmented_params', {}),
                           args.image_size, device, aug_tmp)
            aug_sc = venus_score(aug_tmp, model, tokenizer)
            if aug_sc is None:
                continue

            result = dict(s)
            result['venus_overall'] = aug_sc['overall']
            result['venus_dimensions'] = aug_sc['dimensions']
            result['venus_raw'] = aug_sc['raw_response']
            if image_name in orig_score_cache:
                osc = orig_score_cache[image_name]
                result['orig_venus_overall'] = osc['overall']
                result['orig_venus_dimensions'] = osc['dimensions']
                result['delta_venus'] = round(aug_sc['overall'] - osc['overall'], 2)
            scored.append(result)

            if (idx + 1) % 10 == 0 or idx == len(score_samples) - 1:
                elapsed = time.time() - t0
                speed = (idx + 1) / max(elapsed, 1e-3)
                eta = (len(score_samples) - idx - 1) / max(speed, 1e-3) / 60
                logger.info(
                    f'[{idx+1}/{len(score_samples)}] '
                    f'speed={speed:.2f}img/s ETA={eta:.1f}min '
                    f'last venus={aug_sc["overall"]} '
                    f'dims={aug_sc["dimensions"]}'
                )

    # 保存评分
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'count': len(scored), 'samples': scored},
                  f, ensure_ascii=False, indent=2)
    logger.info(f'\n评分结果: {out_path} ({len(scored)} 条)')

    # 统计
    if scored:
        a = np.array([s['venus_overall'] for s in scored])
        logger.info(f'\nVenus overall 统计:')
        logger.info(f'  count={len(a)} mean={a.mean():.2f} std={a.std():.2f}')
        logger.info(f'  min={a.min():.2f} max={a.max():.2f}')
        logger.info(f'  p50={np.percentile(a,50):.2f} p90={np.percentile(a,90):.2f} '
                    f'p95={np.percentile(a,95):.2f}')
        for thr in [5.0, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]:
            n = int((a >= thr).sum())
            logger.info(f'  >= {thr}: {n} ({n/len(a)*100:.1f}%)')

        # 各维度统计
        logger.info('\n各维度均分:')
        for dim in DIMENSION_NAMES:
            d_arr = np.array([s['venus_dimensions'].get(dim, 5.0) for s in scored])
            logger.info(f'  {dim}: mean={d_arr.mean():.2f} max={d_arr.max():.2f}')

    # 过滤
    filtered = [s for s in scored if s['venus_overall'] >= args.filter_min]
    logger.info(f'\n过滤 venus_overall >= {args.filter_min}: '
                f'{len(filtered)}/{len(scored)} '
                f'({len(filtered)/max(len(scored),1)*100:.1f}%)')

    filt_path = Path(args.filtered_output)
    filt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filt_path, 'w', encoding='utf-8') as f:
        json.dump({
            'count': len(filtered),
            'filter_threshold': args.filter_min,
            'samples': filtered,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f'过滤结果: {filt_path}')


if __name__ == '__main__':
    main()
