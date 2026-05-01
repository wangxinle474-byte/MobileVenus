"""
用 AesExpert (LLaVA-1.5-7B fine-tune) 给增强数据重打分

加载已转换的 HF 格式 AesExpert (经 convert_aesexpert_to_hf.py),
用 4-bit 量化在 8GB GPU 上推理. 渲染增强图后, 让 AesExpert 输出
1-10 美学评分及描述.

Usage:
  python tools/data/rescore_with_aesexpert.py \
    --input outputs/data/aug_high_score.json \
    --top_n 20 --sort_by aug_musiq \
    --score_orig --filter_min 7.0 \
    --output outputs/data/aug_aesexpert.json \
    --filtered_output outputs/data/aug_aesexpert_filtered.json
"""
import os
import sys
import json
import time
import re
import argparse
import logging
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.diff_isp import apply_diff_isp
from training.fivek_8param.config import PARAM_NAMES
from tools.data.aesexpert_loader import (
    load_aesexpert_4bit, score_image as _score_image_short,
    SCORE_PROMPT as SHORT_SCORE_PROMPT,
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@torch.no_grad()
def score_one_image(model, processor, pil_img: Image.Image,
                    max_new_tokens: int = 10) -> Dict:
    """给一张 PIL 图打分 (短 prompt, 输出纯数字 1-10)."""
    res = _score_image_short(model, processor, pil_img,
                              max_new_tokens=max_new_tokens)
    score = res['score']
    scores = {'overall': score} if score is not None else {}
    return {
        'response': res['response'],
        'scores': scores,
    }


def make_uid(s: Dict) -> str:
    """为一条样本生成唯一 ID (image + augmented_params)."""
    img = s.get('image', s.get('image_path', ''))
    params = s.get('augmented_params', {})
    params_str = json.dumps(params, sort_keys=True, default=str)
    style = s.get('style', '')
    key = f'{img}|{style}|{params_str}'
    return hashlib.md5(key.encode('utf-8')).hexdigest()[:16]


def atomic_save_json(out_path: Path, data: Dict):
    """原子写 JSON: 先写 .tmp 再 rename, 避免崩溃时文件损坏."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def load_existing_scored(out_path: Path) -> tuple:
    """加载已评分结果, 返回 (scored list, set of UIDs)."""
    if not out_path.exists():
        return [], set()
    try:
        with open(out_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        samples = d.get('samples', [])
        uids = {s.get('_uid') for s in samples if s.get('_uid')}
        return samples, uids
    except Exception as e:
        logger.warning(f'加载旧 output 失败 ({e}), 从头开始')
        return [], set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--model_path', default=r'e:/AesExpert_HF',
                        help='ASCII junction path; original at e:/智能相机/Venus_CVPR2026-main/pretrained_weights/AesExpert-HF')
    parser.add_argument('--output', default='outputs/data/aug_aesexpert.json')
    parser.add_argument('--filtered_output', default='outputs/data/aug_aesexpert_filtered.json')
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--top_n', type=int, default=0,
                        help='按 sort_by 字段降序取 top_n')
    parser.add_argument('--sort_by', default='aug_musiq')
    parser.add_argument('--render_size', type=int, default=512,
                        help='diff_ISP 渲染尺寸')
    parser.add_argument('--filter_min', type=float, default=7.0)
    parser.add_argument('--score_orig', action='store_true')
    parser.add_argument('--no_4bit', action='store_true',
                        help='不使用 4-bit 量化 (需要更大显存)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_every', type=int, default=200,
                        help='每多少个样本保存一次检查点')
    parser.add_argument('--no_resume', action='store_true',
                        help='不加载已有结果, 从头开始 (会覆盖老文件)')
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
        method = d.get('method', 'unknown')
        for it in items:
            it['_source'] = fp
            it['_method'] = method
        all_samples.extend(items)
        logger.info(f'加载 {fp}: {len(items)} 条 ({method})')
    logger.info(f'总计: {len(all_samples)} 条')

    rng = np.random.RandomState(args.seed)
    if args.top_n > 0:
        all_samples.sort(key=lambda s: -float(s.get(args.sort_by, 0)))
        score_samples = all_samples[:args.top_n]
        logger.info(f'按 {args.sort_by} 取 top_n={args.top_n}')
    elif args.max_samples > 0 and len(all_samples) > args.max_samples:
        idx = rng.choice(len(all_samples), args.max_samples, replace=False)
        score_samples = [all_samples[i] for i in idx]
    else:
        score_samples = all_samples
    logger.info(f'评分样本数: {len(score_samples)}')

    # 加载模型 (手动 4-bit GPU loader, 绕开 from_pretrained 的 mmap/RAM 问题)
    model_path = Path(args.model_path)
    if not (model_path / 'config.json').exists():
        logger.error(f'AesExpert HF 模型路径无效: {model_path}')
        return
    if args.no_4bit:
        logger.warning('--no_4bit 被忽略: 本脚本只支持 4-bit GPU 加载')
    model, processor = load_aesexpert_4bit(str(model_path), verbose=True)

    # 渲染 transform
    render_tf = transforms.Compose([
        transforms.Resize((args.render_size, args.render_size)),
        transforms.ToTensor(),
    ])

    out_path = Path(args.output)

    # 检查点 / resume
    if args.no_resume:
        scored = []
        existing_uids: Set[str] = set()
    else:
        scored, existing_uids = load_existing_scored(out_path)
        if existing_uids:
            logger.info(f'断点续打: 已有 {len(existing_uids)} 条, 跳过并继续')

    orig_score_cache: Dict[str, Dict] = {}
    t0 = time.time()
    n_processed_this_run = 0  # 仅计本轮新处理的条数 (不计 resume 的)

    def save_checkpoint():
        atomic_save_json(out_path, {
            'count': len(scored),
            'samples': scored,
            'in_progress': True,
        })

    try:
        with tempfile.TemporaryDirectory(prefix='aesexpert_') as tmpdir:
            for idx, s in enumerate(score_samples):
                uid = make_uid(s)
                if uid in existing_uids:
                    continue

                img_path = s.get('image_path', '')
                if not Path(img_path).exists():
                    logger.warning(f'图不存在: {img_path}')
                    continue

                try:
                    pil = Image.open(img_path).convert('RGB')
                    orig_tensor = render_tf(pil).unsqueeze(0).to(device)
                except Exception as e:
                    logger.warning(f'读图失败 {img_path}: {e}')
                    continue

                # 原图 baseline
                if args.score_orig and img_path not in orig_score_cache:
                    try:
                        res = score_one_image(model, processor, pil)
                        orig_score_cache[img_path] = res
                    except Exception as e:
                        logger.warning(f'原图打分失败: {e}')
                        orig_score_cache[img_path] = {'response': '', 'scores': {}}

                # 渲染增强图
                params = s.get('augmented_params', {})
                params_torch = {p: torch.tensor([float(params.get(p, 0))],
                                                device=device, dtype=torch.float32)
                                for p in PARAM_NAMES}
                with torch.no_grad():
                    rendered = apply_diff_isp(orig_tensor, params_torch).clamp(0, 1)
                arr = (rendered[0].cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                aug_pil = Image.fromarray(arr)

                try:
                    res = score_one_image(model, processor, aug_pil)
                except Exception as e:
                    logger.warning(f'增强图打分失败 {img_path}: {e}')
                    continue

                r = dict(s)
                r['_uid'] = uid
                r['aesexpert_response'] = res['response']
                r['aesexpert_scores'] = res['scores']
                r['aesexpert_overall'] = res['scores'].get('overall')
                if args.score_orig and img_path in orig_score_cache:
                    orig = orig_score_cache[img_path]
                    r['orig_aesexpert_response'] = orig['response']
                    r['orig_aesexpert_scores'] = orig['scores']
                    r['orig_aesexpert_overall'] = orig['scores'].get('overall')
                    if r['aesexpert_overall'] is not None and r['orig_aesexpert_overall'] is not None:
                        r['delta_aesexpert'] = r['aesexpert_overall'] - r['orig_aesexpert_overall']
                scored.append(r)
                existing_uids.add(uid)
                n_processed_this_run += 1

                # 进度日志
                if n_processed_this_run % 20 == 0 or idx == len(score_samples) - 1:
                    elapsed = time.time() - t0
                    speed = n_processed_this_run / max(elapsed, 1e-3)
                    n_remaining = len(score_samples) - idx - 1
                    eta_min = n_remaining / max(speed, 1e-3) / 60
                    last_o = r.get('aesexpert_overall', 'N/A')
                    logger.info(
                        f'[{idx+1}/{len(score_samples)}] '
                        f'(本轮 {n_processed_this_run}) '
                        f'speed={speed:.2f}/s ETA={eta_min:.1f}min '
                        f'last={last_o}'
                    )

                # 检查点保存
                if n_processed_this_run % args.save_every == 0:
                    save_checkpoint()
                    logger.info(f'✓ 检查点保存: {out_path} ({len(scored)} 条)')
    except KeyboardInterrupt:
        logger.warning('\n收到中断信号, 保存当前进度...')
        save_checkpoint()
        logger.info(f'已保存 {len(scored)} 条到 {out_path}, 下次会从这里继续')
        raise

    # 最终保存 (标记完成)
    atomic_save_json(out_path, {
        'count': len(scored),
        'samples': scored,
        'in_progress': False,
    })
    logger.info(f'\n评分结果: {out_path} ({len(scored)} 条, 本轮新 {n_processed_this_run})')

    # 统计
    valid_overall = [s['aesexpert_overall'] for s in scored
                     if s.get('aesexpert_overall') is not None]
    if valid_overall:
        a = np.array(valid_overall)
        logger.info(f'\nAesExpert overall 统计:')
        logger.info(f'  count={len(a)} mean={a.mean():.2f} std={a.std():.2f}')
        logger.info(f'  min={a.min():.2f} max={a.max():.2f}')
        for q in [50, 75, 90, 95, 99]:
            logger.info(f'  p{q}={np.percentile(a, q):.2f}')
        for thr in [5.0, 6.0, 7.0, 7.5, 8.0]:
            n = int((a >= thr).sum())
            logger.info(f'  >= {thr}: {n} ({n/len(a)*100:.1f}%)')

    # 过滤
    filtered = [s for s in scored
                if s.get('aesexpert_overall') is not None
                and s['aesexpert_overall'] >= args.filter_min]
    logger.info(f'\n过滤 aesexpert_overall >= {args.filter_min}: '
                f'{len(filtered)}/{len(scored)} '
                f'({len(filtered)/max(len(scored),1)*100:.1f}%)')

    filt_path = Path(args.filtered_output)
    filt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(filt_path, 'w', encoding='utf-8') as f:
        json.dump({
            'count': len(filtered),
            'filter_threshold': args.filter_min,
            'scorer': 'aesexpert_llava15_7b',
            'samples': filtered,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f'过滤结果: {filt_path}')


if __name__ == '__main__':
    main()
