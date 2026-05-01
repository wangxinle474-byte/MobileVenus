"""
AutoDL 端: 用 Venus 大模型评估图片美学质量
对比 原图 / Baseline增强 / Distill增强 三组图片

前置条件:
  1. 已上传 venus_eval/ 目录 (含 original/, baseline/, distill_v2/)
  2. Venus-Q-Stage1 权重在 /root/Venus-Q-Stage1/

用法 (AutoDL):
  python venus_aesthetic_eval.py --eval_dir /root/venus_eval
  python venus_aesthetic_eval.py --eval_dir /root/venus_eval --max_images 20
"""
import os
import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

VENUS_DIR = "/root/Venus-Q-Stage1"
DIMENSIONS = ['composition', 'lighting', 'color', 'clarity', 'subject']

# Venus 评价 prompt — 要求直接打分
EVAL_PROMPT = """You are a professional photography critic. Rate this photo on a scale of 1-10 for each dimension. Be strict and precise.

Please output ONLY the scores in this exact format:
Composition: X
Lighting: X
Color: X
Clarity: X
Subject: X
Overall: X

Where X is a number from 1 to 10 (can use decimals like 7.5).
Do not add any explanation."""

# Venus 诊断 prompt — 判断是否需要进一步调整
DIAGNOSIS_PROMPT = """As a professional photographer, analyze this image briefly:
1. Does this image need any exposure correction? (yes/no, how much)
2. Does the white balance look correct? (yes/no)
3. Is the contrast appropriate? (yes/no)
4. Are the colors well-saturated? (yes/no)
5. Are shadows/highlights well-balanced? (yes/no)
6. Overall, how many adjustments does this image need? (0=perfect, 1-2=minor, 3+=major)

Be concise. Output format:
Exposure: [yes/no] [detail]
WhiteBalance: [yes/no]
Contrast: [yes/no]
Saturation: [yes/no]
ShadowHighlight: [yes/no]
AdjustmentsNeeded: [number]"""


def load_venus():
    """加载 Venus 模型"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 检查模型路径
    venus_dir = VENUS_DIR
    if not os.path.exists(os.path.join(venus_dir, 'config.json')):
        # 尝试备选路径
        for alt in ['/root/autodl-tmp/Venus-Q-Stage1', '/root/Venus-Q-Stage1',
                     '/root/autodl-pub/Venus-Q-Stage1']:
            if os.path.exists(os.path.join(alt, 'config.json')):
                venus_dir = alt
                break
        else:
            raise FileNotFoundError(
                f"Venus 模型未找到! 检查路径: {VENUS_DIR}\n"
                f"请运行: find / -name 'config.json' -path '*/Venus*' 2>/dev/null")

    logger.info(f"加载 Venus: {venus_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        venus_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        venus_dir, device_map="auto", trust_remote_code=True,
        bf16=True, local_files_only=True
    ).eval()
    logger.info("Venus 加载完成")
    return model, tokenizer


def venus_score(model, tokenizer, image_path):
    """用 Venus 给图片打分"""
    import torch

    query = tokenizer.from_list_format([
        {'image': str(image_path)},
        {'text': EVAL_PROMPT},
    ])

    with torch.no_grad():
        response, _ = model.chat(tokenizer, query=query, history=None)

    # 解析分数
    scores = {}
    for dim in DIMENSIONS + ['overall']:
        pattern = rf'{dim}\s*[:：]\s*(\d+\.?\d*)'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            scores[dim] = min(max(float(match.group(1)), 1.0), 10.0)

    # fallback: 提取所有数字
    if len(scores) < 6:
        numbers = re.findall(r'(\d+\.?\d*)', response)
        numbers = [float(n) for n in numbers if 1 <= float(n) <= 10]
        for j, dim in enumerate(DIMENSIONS + ['overall']):
            if dim not in scores and j < len(numbers):
                scores[dim] = numbers[j]

    for dim in DIMENSIONS + ['overall']:
        if dim not in scores:
            scores[dim] = 5.0

    return scores, response


def venus_diagnose(model, tokenizer, image_path):
    """用 Venus 诊断图片是否需要调整"""
    import torch

    query = tokenizer.from_list_format([
        {'image': str(image_path)},
        {'text': DIAGNOSIS_PROMPT},
    ])

    with torch.no_grad():
        response, _ = model.chat(tokenizer, query=query, history=None)

    # 解析需要调整的数量
    adj_match = re.search(r'AdjustmentsNeeded\s*[:：]\s*(\d+)', response, re.IGNORECASE)
    adj_needed = int(adj_match.group(1)) if adj_match else -1

    # 统计 "yes" 的数量
    yes_count = len(re.findall(r'\byes\b', response, re.IGNORECASE))
    no_count = len(re.findall(r'\bno\b', response, re.IGNORECASE))

    return {
        'adjustments_needed': adj_needed,
        'issues_found': yes_count,
        'ok_count': no_count,
        'raw_response': response,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', default='/root/venus_eval',
                        help='包含各模型子目录 (original/, baseline/, distill_v2/, distill_v4/, distill_v5/)')
    parser.add_argument('--groups', nargs='+',
                        default=['original', 'baseline', 'distill_v2', 'distill_v4', 'distill_v5'],
                        help='要评估的分组名列表')
    parser.add_argument('--max_images', type=int, default=50)
    parser.add_argument('--mode', default='score', choices=['score', 'diagnose', 'both'])
    parser.add_argument('--output', default='/root/venus_eval_results.json')
    parser.add_argument('--resume', action='store_true', help='从已有结果断点续传')
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    # 只保留实际存在的 group 目录
    groups = [g for g in args.groups if (eval_dir / g).exists()]
    logger.info(f'有效 groups: {groups}')

    # 检查目录
    for g in groups:
        d = eval_dir / g
        if not d.exists():
            logger.warning(f"目录不存在: {d}")
        else:
            imgs = list(d.glob('*.jpg'))
            logger.info(f"  {g}: {len(imgs)} 张")

    # 找到所有组都有的图片
    all_images = {}
    for g in groups:
        d = eval_dir / g
        if d.exists():
            all_images[g] = {f.stem for f in d.glob('*.jpg')}
    common = set.intersection(*all_images.values()) if all_images else set()
    common = sorted(list(common))[:args.max_images]
    logger.info(f"共同图片: {len(common)} 张")

    if not common:
        logger.error("没有找到共同图片!")
        return

    # 加载 Venus
    model, tokenizer = load_venus()

    # 评估
    results = []
    for i, name in enumerate(common):
        logger.info(f"[{i+1}/{len(common)}] 评估 {name}")
        entry = {'image': name}

        for g in groups:
            img_path = eval_dir / g / f"{name}.jpg"
            if not img_path.exists():
                continue

            if args.mode in ('score', 'both'):
                scores, raw = venus_score(model, tokenizer, str(img_path))
                entry[f'{g}_scores'] = scores
                entry[f'{g}_raw_score'] = raw

            if args.mode in ('diagnose', 'both'):
                diag = venus_diagnose(model, tokenizer, str(img_path))
                entry[f'{g}_diagnosis'] = diag

        results.append(entry)

        # 每10张保存一次 (断点续传)
        if (i + 1) % 10 == 0:
            _save_results(results, args.output, args.mode, groups)

    # 最终保存 + 汇总
    summary = _save_results(results, args.output, args.mode, groups)
    _print_summary(summary, args.mode, groups)


def _save_results(results, output_path, mode, groups):
    """保存结果并计算汇总"""
    import numpy as np

    summary = {}

    if mode in ('score', 'both'):
        for g in groups:
            scores_all = [r[f'{g}_scores'] for r in results if f'{g}_scores' in r]
            if scores_all:
                summary[f'{g}_avg'] = {}
                for dim in DIMENSIONS + ['overall']:
                    vals = [s.get(dim, 5.0) for s in scores_all]
                    summary[f'{g}_avg'][dim] = float(np.mean(vals))
                summary[f'{g}_n'] = len(scores_all)

    if mode in ('diagnose', 'both'):
        for g in groups:
            diags = [r[f'{g}_diagnosis'] for r in results if f'{g}_diagnosis' in r]
            if diags:
                adj = [d['adjustments_needed'] for d in diags if d['adjustments_needed'] >= 0]
                issues = [d['issues_found'] for d in diags]
                summary[f'{g}_avg_adjustments'] = float(np.mean(adj)) if adj else -1
                summary[f'{g}_avg_issues'] = float(np.mean(issues))

    output = {
        'summary': summary,
        'results': results,
        'num_images': len(results),
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"已保存: {output_path} ({len(results)} 张)")

    return summary


def _print_summary(summary, mode, groups):
    """打印汇总"""
    print("\n" + "=" * 70)
    print("  Venus 美学评价汇总")
    print("=" * 70)

    if mode in ('score', 'both'):
        print(f"\n  === 美学评分 (1-10) ===")
        dims = DIMENSIONS + ['overall']
        header = f"  {'维度':<15}"
        for g in groups:
            header += f" {g:>12}"
        print(header)
        print(f"  {'-'*55}")
        for dim in dims:
            row = f"  {dim:<15}"
            for g in groups:
                key = f'{g}_avg'
                if key in summary:
                    row += f" {summary[key].get(dim, 0):>12.2f}"
                else:
                    row += f" {'N/A':>12}"
            print(row)

        # 提升量
        print(f"\n  === 提升量 (vs 原图) ===")
        for g in groups[1:]:
            key_g = f'{g}_avg'
            key_o = 'original_avg'
            if key_g in summary and key_o in summary:
                delta = summary[key_g].get('overall', 0) - summary[key_o].get('overall', 0)
                print(f"  {g}: {delta:+.2f} (overall)")

    if mode in ('diagnose', 'both'):
        print(f"\n  === 调整需求 (越少越好) ===")
        for g in groups:
            adj_key = f'{g}_avg_adjustments'
            iss_key = f'{g}_avg_issues'
            adj = summary.get(adj_key, -1)
            iss = summary.get(iss_key, -1)
            print(f"  {g:<15}: 平均需要 {adj:.1f} 项调整, 发现 {iss:.1f} 个问题")

    print()


if __name__ == '__main__':
    main()
