"""
用 AesExpert (LLaVA-1.5-7B, 4-bit) 给 FiveK JPEG 全量打美学分

加载方式: 手动 bnb 4-bit 量化, 约 4.5GB GPU, 适合 8GB 显卡.
输出: outputs/data/fivek_aesexpert_scores.json

Usage:
  python tools/data/score_fivek_aesexpert.py
  python tools/data/score_fivek_aesexpert.py --jpeg_dir E:/Data/dataset/fivek_jpeg --max_images 100
  python tools/data/score_fivek_aesexpert.py --resume  # 从上次中断处继续
"""
import os
import sys
import time
import json
import re
import gc
import struct
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('PYTHONUNBUFFERED', '1')

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ─── 4-bit 加载工具 (来自 load_aesexpert_4bit.py) ───────────────────

DTYPE_MAP = {
    'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
    'I64': torch.int64, 'I32': torch.int32, 'I16': torch.int16,
    'I8': torch.int8, 'U8': torch.uint8, 'BOOL': torch.bool,
}
NP_DTYPE = {
    torch.int64: np.int64, torch.int32: np.int32,
    torch.int16: np.int16, torch.int8: np.int8,
    torch.uint8: np.uint8, torch.bool: np.bool_,
    torch.float32: np.float32, torch.float16: np.float16,
}


def parse_header(filepath: Path):
    with open(str(filepath), 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len).decode('utf-8'))
        header.pop('__metadata__', None)
    return header_len, header


def read_one_tensor(file_obj, info, data_offset):
    dtype = DTYPE_MAP[info['dtype']]
    shape = info['shape']
    beg, end = info['data_offsets']
    n_bytes = end - beg
    file_obj.seek(data_offset + beg)
    blob = file_obj.read(n_bytes)
    if dtype == torch.bfloat16:
        buf = np.frombuffer(blob, dtype=np.uint16).copy()
        t = torch.from_numpy(buf).view(torch.bfloat16).reshape(shape)
    elif dtype in NP_DTYPE:
        buf = np.frombuffer(blob, dtype=NP_DTYPE[dtype]).copy()
        t = torch.from_numpy(buf).reshape(shape)
    else:
        raise ValueError(f'Unsupported dtype: {dtype}')
    del blob, buf
    return t


def replace_linear_with_4bit(model, skip_paths):
    import bitsandbytes as bnb
    from accelerate import init_empty_weights

    def is_skipped(full_name):
        return any(s in full_name for s in skip_paths)

    def _replace(module, prefix):
        for name, child in list(module.named_children()):
            full = f'{prefix}.{name}' if prefix else name
            if isinstance(child, nn.Linear) and not is_skipped(full):
                with init_empty_weights():
                    new = bnb.nn.Linear4bit(
                        child.in_features, child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.float16,
                        quant_type='nf4',
                        quant_storage=torch.uint8,
                    )
                setattr(module, name, new)
            else:
                _replace(child, full)
    _replace(model, '')


def get_module_param(model, key):
    *path, leaf = key.split('.')
    mod = model
    for p in path:
        if hasattr(mod, p):
            mod = getattr(mod, p)
        else:
            return None, None
    return mod, leaf


def load_to_4bit_gpu(model, shards):
    import bitsandbytes as bnb
    from accelerate.utils import set_module_tensor_to_device

    total_keys = 0
    for shard in shards:
        sz_gb = shard.stat().st_size / 1e9
        t0 = time.time()
        logger.info(f'  Open {shard.name} ({sz_gb:.2f}GB)...')
        header_len, header = parse_header(shard)
        data_offset = 8 + header_len
        n = len(header)

        with open(str(shard), 'rb') as f:
            for i, (key, info) in enumerate(header.items()):
                t = read_one_tensor(f, info, data_offset)
                if t.dtype in (torch.bfloat16, torch.float32):
                    t = t.to(torch.float16)

                module, param_name = get_module_param(model, key)
                if module is None:
                    del t
                    continue

                if isinstance(module, bnb.nn.Linear4bit) and param_name == 'weight':
                    new_w = bnb.nn.Params4bit(
                        t.contiguous(), requires_grad=False,
                        quant_type='nf4',
                    ).to('cuda:0')
                    module.weight = new_w
                    del t, new_w
                else:
                    set_module_tensor_to_device(model, key, 'cuda:0', value=t)
                    del t
                total_keys += 1

                if (i + 1) % 100 == 0:
                    gc.collect()

        gc.collect()
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated() / 1e9
        logger.info(f'  shard done in {time.time()-t0:.1f}s, GPU={used:.2f}GB')

    logger.info(f'  Total keys loaded: {total_keys}')


def load_aesexpert(model_path: Path):
    """加载 AesExpert 模型 (4-bit 量化), 返回 (model, processor)"""
    from transformers import LlavaConfig, LlavaForConditionalGeneration, AutoProcessor
    from accelerate import init_empty_weights
    import bitsandbytes as bnb

    logger.info(f'[1/5] Loading config + processor from {model_path}')
    cfg = LlavaConfig.from_pretrained(str(model_path))
    processor = AutoProcessor.from_pretrained(str(model_path))
    if getattr(processor, 'patch_size', None) is None:
        processor.patch_size = cfg.vision_config.patch_size
    if getattr(processor, 'vision_feature_select_strategy', None) is None:
        processor.vision_feature_select_strategy = cfg.vision_feature_select_strategy
    processor.num_additional_image_tokens = 1

    logger.info('[2/5] Init empty model on meta...')
    with init_empty_weights():
        model = LlavaForConditionalGeneration(cfg)

    logger.info('[3/5] Replace Linear -> bnb.Linear4bit...')
    skip_paths = ['vision_tower', 'lm_head', 'multi_modal_projector']
    replace_linear_with_4bit(model, skip_paths=skip_paths)

    logger.info('[4/5] Stream load weights to GPU...')
    shards = sorted(model_path.glob('model-*.safetensors'))
    if not shards:
        raise FileNotFoundError(f'No safetensors in {model_path}')
    load_to_4bit_gpu(model, shards)

    model.tie_weights()
    model.eval()

    # 修复 embed_tokens 越界
    img_id = cfg.image_token_index
    embed = model.model.language_model.embed_tokens
    if embed.weight.shape[0] <= img_id:
        new_n = max(img_id + 64, 32064)
        logger.info(f'  Padding embed_tokens {embed.weight.shape[0]} -> {new_n}')
        old_w = embed.weight.data
        new_w = torch.zeros(new_n, old_w.shape[1], dtype=old_w.dtype, device=old_w.device)
        new_w[:old_w.shape[0]] = old_w
        embed.weight = nn.Parameter(new_w, requires_grad=False)
        embed.num_embeddings = new_n
        lmh = model.lm_head
        old_lh = lmh.weight.data
        if old_lh.shape[0] != new_n:
            new_lh = torch.zeros(new_n, old_lh.shape[1], dtype=old_lh.dtype, device=old_lh.device)
            new_lh[:old_lh.shape[0]] = old_lh
            lmh.weight = nn.Parameter(new_lh, requires_grad=False)
            lmh.out_features = new_n

    # 强制非 4bit 参数转 fp16
    n_fixed = 0
    for n, p in model.named_parameters():
        if isinstance(p, bnb.nn.Params4bit):
            continue
        if p.dtype != torch.float16:
            with torch.no_grad():
                new_p = nn.Parameter(p.data.to(torch.float16), requires_grad=False)
            *parents, leaf = n.split('.')
            mod = model
            for x in parents:
                mod = getattr(mod, x)
            setattr(mod, leaf, new_p)
            n_fixed += 1
    if n_fixed:
        logger.info(f'  Fixed {n_fixed} non-fp16 params')

    used = torch.cuda.memory_allocated() / 1e9
    logger.info(f'[5/5] Model ready, GPU={used:.2f}GB')
    return model, processor


# ─── 评分逻辑 ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)
USER_MSG = (
    "What is the overall aesthetic quality score of this image on a scale "
    "from 1 to 10? Answer with only a single number."
)


def parse_score(response: str) -> Optional[float]:
    """从模型回复中解析 1-10 分数"""
    # 尝试解析 "overall: X" 格式
    for dim in ['overall', 'score', 'rating', 'quality']:
        m = re.search(rf'{dim}\s*[:\-=]\s*([\d.]+)', response, re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1).rstrip('.'))
                if 0 < v <= 10:
                    return v
            except ValueError:
                pass
    # 回退: 取第一个 1-10 范围的数字
    nums = re.findall(r'(\d+(?:\.\d+)?)', response)
    for n in nums:
        v = float(n)
        if 0 < v <= 10:
            return v
    return None


def parse_dimensions(response: str) -> Dict[str, float]:
    """解析多维分数"""
    dims = {}
    for dim in ['composition', 'lighting', 'color', 'clarity', 'subject', 'overall']:
        m = re.search(rf'{dim}\s*[:\-=]\s*([\d.]+)', response, re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1).rstrip('.'))
                dims[dim] = max(0.0, min(10.0, v))
            except ValueError:
                pass
    return dims


@torch.no_grad()
def score_image(model, processor, pil_img: Image.Image) -> Dict:
    """给单张图片打分"""
    prompt = f"{SYSTEM_PROMPT} USER: <image>\n{USER_MSG} ASSISTANT:"
    inputs = processor(text=prompt, images=pil_img, return_tensors='pt')
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
    inputs = {k: (v.to('cuda:0') if torch.is_tensor(v) else v)
              for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
        num_beams=1,
        pad_token_id=processor.tokenizer.pad_token_id or 0,
    )
    new_tokens = out[0][inputs['input_ids'].shape[-1]:]
    response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    score = parse_score(response)
    dims = parse_dimensions(response)
    return {'response': response, 'score': score, 'dimensions': dims}


def main():
    parser = argparse.ArgumentParser(description='AesExpert FiveK 全量评分')
    parser.add_argument('--model_path', default=r'E:\AesExpert_HF',
                        help='AesExpert HF 模型路径')
    parser.add_argument('--jpeg_dir', default=r'E:\Data\dataset\fivek_jpeg',
                        help='FiveK JPEG 图片目录')
    parser.add_argument('--output', default=str(PROJECT_ROOT / 'outputs' / 'data' / 'fivek_aesexpert_scores.json'),
                        help='输出 JSON 路径')
    parser.add_argument('--max_images', type=int, default=0,
                        help='最多评分图片数 (0=全部)')
    parser.add_argument('--resume', action='store_true',
                        help='从上次中断处继续')
    parser.add_argument('--save_every', type=int, default=50,
                        help='每 N 张保存一次')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error('CUDA 不可用, 无法运行 4-bit 量化')
        return

    free, total = torch.cuda.mem_get_info()
    logger.info(f'GPU: {torch.cuda.get_device_name(0)} '
                f'free={free/1e9:.1f}GB / total={total/1e9:.1f}GB')

    model_path = Path(args.model_path)
    if not (model_path / 'config.json').exists():
        logger.error(f'模型路径无效: {model_path} (缺少 config.json)')
        logger.info('请先下载: huggingface-cli download qyuan/AesMMIT_LLaVA_v1.5_7b_240325 --local-dir E:\\AesExpert_HF')
        logger.info('或运行: scripts/local/local_resume_aesexpert.ps1 -Background  (用 hf-mirror 镜像)')
        return

    # 扫描图片
    jpeg_dir = Path(args.jpeg_dir)
    jpg_files = sorted(jpeg_dir.glob('*.jpg'))
    if not jpg_files:
        logger.error(f'未找到图片: {jpeg_dir}')
        return
    logger.info(f'找到 {len(jpg_files)} 张 FiveK JPEG')

    if args.max_images > 0:
        jpg_files = jpg_files[:args.max_images]
        logger.info(f'限制为前 {args.max_images} 张')

    # Resume: 加载已有结果
    out_path = Path(args.output)
    scored_names = set()
    results = {}
    if args.resume and out_path.exists():
        prev = json.load(open(out_path, 'r', encoding='utf-8'))
        results = prev.get('scores', {})
        scored_names = set(results.keys())
        logger.info(f'Resume: 已有 {len(scored_names)} 条结果')

    # 过滤掉已评分的
    todo_files = [f for f in jpg_files if f.name not in scored_names]
    logger.info(f'待评分: {len(todo_files)} 张')

    if not todo_files:
        logger.info('全部已评分, 跳过')
    else:
        # 加载模型
        model, processor = load_aesexpert(model_path)

        t0 = time.time()
        for idx, img_path in enumerate(todo_files):
            try:
                pil = Image.open(img_path).convert('RGB')
                result = score_image(model, processor, pil)

                results[img_path.name] = {
                    'score': result['score'],
                    'dimensions': result['dimensions'],
                    'response': result['response'][:500],
                }

                elapsed = time.time() - t0
                speed = (idx + 1) / elapsed
                eta_min = (len(todo_files) - idx - 1) / max(speed, 0.001) / 60

                if (idx + 1) % 10 == 0 or idx == 0:
                    logger.info(
                        f'[{idx+1}/{len(todo_files)}] {img_path.name} '
                        f'score={result["score"]} '
                        f'speed={speed:.2f}img/s ETA={eta_min:.0f}min'
                    )

            except Exception as e:
                logger.warning(f'FAIL {img_path.name}: {type(e).__name__}: {e}')
                results[img_path.name] = {'score': None, 'error': str(e)}

            # 定期保存
            if (idx + 1) % args.save_every == 0:
                _save(out_path, results)
                logger.info(f'  checkpoint saved ({len(results)} 条)')

        # 最终保存
        _save(out_path, results)

    # 统计
    all_scores = [v['score'] for v in results.values() if v.get('score') is not None]
    if all_scores:
        a = np.array(all_scores)
        logger.info(f'\n{"="*50}')
        logger.info(f'AesExpert FiveK 评分统计 ({len(a)}/{len(results)} 有效)')
        logger.info(f'  mean={a.mean():.2f}  std={a.std():.2f}')
        logger.info(f'  min={a.min():.2f}  max={a.max():.2f}')
        for q in [25, 50, 75, 90, 95, 99]:
            logger.info(f'  p{q}={np.percentile(a, q):.2f}')
        for thr in [3, 4, 5, 6, 7, 8, 9]:
            n = int((a >= thr).sum())
            logger.info(f'  >= {thr}: {n}/{len(a)} ({n/len(a)*100:.1f}%)')
        logger.info(f'{"="*50}')

    logger.info(f'结果已保存: {out_path}')


def _save(out_path: Path, results: dict):
    """保存结果 JSON"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_scores = [v['score'] for v in results.values() if v.get('score') is not None]
    metadata = {
        'scorer': 'AesExpert_LLaVA-1.5-7B_4bit',
        'num_images': len(results),
        'num_scored': len(all_scores),
        'score_range': [1, 10],
    }
    if all_scores:
        a = np.array(all_scores)
        metadata['stats'] = {
            'mean': round(float(a.mean()), 3),
            'std': round(float(a.std()), 3),
            'min': round(float(a.min()), 3),
            'max': round(float(a.max()), 3),
        }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': metadata,
            'scores': results,
        }, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
