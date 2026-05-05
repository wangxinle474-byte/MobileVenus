"""
Stage C 训练数据生成: 指令-参数配对

从已有的 FiveK/PPR10K 专家数据中, 生成 (图片, 文本指令, 目标参数) 三元组。

策略:
  1. 基于参数 delta 生成单参数指令 (如 "提高曝光" ← ev_delta > 0)
  2. 基于多参数组合生成复合指令 (如 "日落风格" ← warm + saturated)
  3. 随机采样参数幅度, 配合对应强度的文本 ("稍微调亮" vs "大幅提亮")
  4. LLM 增强: 用大模型改写模板, 增加语言多样性

输出: instruction_data.json
  [
    {
      "image_name": "0001.jpg",
      "instruction": "提高曝光，增加饱和度",
      "base_params": {...},      # Stage B 基准预测 (或 source 参数)
      "target_params": {...},    # 专家目标参数
      "delta_params": {...},     # target - base
    },
    ...
  ]

用法:
  python tools/data/data_prep/generate_instruction_data.py \
    --fivek_params data/fivek_expert_params.json \
    --ppr10k_params data/ppr10k_params.json \
    --output data/instruction_data.json \
    --augment_with_llm   # 可选: 用 LLM 增强多样性
"""
import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
from training.text_condition.config import INSTRUCTION_TEMPLATES, COMPOUND_TEMPLATES


# 参数变化幅度 → 文本强度修饰
INTENSITY_MODIFIERS = {
    'slight': ['稍微', '略微', '一点点', '轻微'],
    'moderate': ['适当', '适度', ''],  # 空字符串=无修饰
    'strong': ['大幅', '明显', '大力', '显著'],
}

# 阈值: delta 多大算哪个强度
INTENSITY_THRESHOLDS = {
    'ev_compensation': (0.3, 1.0),     # slight < 0.3, moderate 0.3~1.0, strong > 1.0
    'white_balance':   (300, 1000),
    'contrast':        (10, 30),
    'shadows':         (10, 30),
    'highlights':      (10, 30),
    'saturation':      (10, 30),
}


def classify_intensity(param_name: str, abs_delta: float) -> str:
    """根据参数变化幅度判断强度等级"""
    low, high = INTENSITY_THRESHOLDS.get(param_name, (10, 30))
    if abs_delta < low:
        return 'slight'
    elif abs_delta < high:
        return 'moderate'
    else:
        return 'strong'


def generate_single_param_instruction(
    param_name: str,
    delta: float,
    intensity: str = None,
) -> str:
    """
    为单个参数变化生成文本指令

    Args:
        param_name: 参数名
        delta: 参数变化量 (正=increase, 负=decrease)
        intensity: 强度等级, None 时自动判断
    """
    templates = INSTRUCTION_TEMPLATES.get(param_name, {})
    direction = 'increase' if delta > 0 else 'decrease'
    candidates = templates.get(direction, [])
    if not candidates:
        return ''

    instruction = random.choice(candidates)

    if intensity is None:
        intensity = classify_intensity(param_name, abs(delta))

    # 添加强度修饰
    if intensity != 'moderate' or random.random() < 0.3:
        modifier = random.choice(INTENSITY_MODIFIERS.get(intensity, ['']))
        if modifier:
            instruction = modifier + instruction

    return instruction


def generate_compound_instruction(deltas: Dict[str, float]) -> str:
    """
    为多参数变化生成复合指令

    策略:
    1. 先尝试匹配预定义的复合模板
    2. 匹配不到则组合单参数指令
    """
    # 找出显著变化的参数
    significant = {}
    for p, d in deltas.items():
        if abs(d) > INTENSITY_THRESHOLDS.get(p, (10, 30))[0]:
            significant[p] = 'increase' if d > 0 else 'decrease'

    if not significant:
        return ''

    # 尝试匹配复合模板
    if random.random() < 0.3:
        for template_text, template_dirs in COMPOUND_TEMPLATES:
            if all(significant.get(p) == d for p, d in template_dirs.items()):
                return template_text

    # 组合单参数指令 (最多选 2-3 个)
    params_to_describe = list(significant.keys())
    random.shuffle(params_to_describe)
    params_to_describe = params_to_describe[:min(3, len(params_to_describe))]

    parts = []
    for p in params_to_describe:
        instr = generate_single_param_instruction(p, deltas[p])
        if instr:
            parts.append(instr)

    if not parts:
        return ''

    # 用逗号或"并且"连接
    connector = random.choice(['，', '，', '，然后', '，并且', '；'])
    return connector.join(parts)


def process_fivek_data(params_file: str) -> List[dict]:
    """从 FiveK 专家参数生成指令数据"""
    with open(params_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    img_params = defaultdict(list)

    # 按图片聚合参数
    for s in data['samples']:
        name = s['image_name']
        params = {p: float(s.get(p, 0)) for p in PARAM_NAMES}
        img_params[name].append(params)

    # 计算均值作为 base, 各专家参数作为 target
    for img_name, param_list in img_params.items():
        if len(param_list) < 1:
            continue

        # base: 所有专家均值
        base = {}
        for p in PARAM_NAMES:
            base[p] = sum(d[p] for d in param_list) / len(param_list)

        # 为每组参数生成指令
        for target in param_list:
            deltas = {p: target[p] - base[p] for p in PARAM_NAMES}

            # 跳过变化太小的
            max_delta = max(abs(d) for d in deltas.values())
            if max_delta < 1.0:
                continue

            instruction = generate_compound_instruction(deltas)
            if not instruction:
                continue

            samples.append({
                'source': 'fivek',
                'image_name': img_name.replace('.dng', '.jpg'),
                'instruction': instruction,
                'base_params': base,
                'target_params': target,
                'delta_params': deltas,
            })

    return samples


def process_ppr10k_data(params_file: str) -> List[dict]:
    """从 PPR10K 专家参数生成指令数据"""
    with open(params_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for s in data['samples']:
        experts = s.get('experts', {})
        source_params = s.get('source_params', {})

        if not experts or not source_params:
            continue

        # base: source 参数
        base = {p: source_params.get(p, 0.0) for p in PARAM_NAMES}

        for expert_name, expert_data in experts.items():
            target = expert_data.get('absolute', {})
            delta = expert_data.get('delta', {})

            if not target or not delta:
                continue

            # 补齐缺失参数
            target_full = {p: target.get(p, 0.0) for p in PARAM_NAMES}
            delta_full = {p: delta.get(p, 0.0) for p in PARAM_NAMES}

            max_delta = max(abs(d) for d in delta_full.values())
            if max_delta < 1.0:
                continue

            instruction = generate_compound_instruction(delta_full)
            if not instruction:
                continue

            samples.append({
                'source': f'ppr10k_{expert_name}',
                'image_name': s['image_name'],
                'instruction': instruction,
                'base_params': base,
                'target_params': target_full,
                'delta_params': delta_full,
            })

    return samples


def generate_synthetic_instructions(
    base_samples: List[dict],
    num_synthetic: int = 5000,
) -> List[dict]:
    """
    合成额外的指令数据

    策略: 从现有样本中随机采样一张图, 随机生成参数变化, 配以指令
    """
    synthetic = []
    for _ in range(num_synthetic):
        s = random.choice(base_samples)
        base = s['base_params'].copy()

        # 随机选 1-3 个参数进行调整
        n_params = random.randint(1, 3)
        params_to_change = random.sample(PARAM_NAMES, n_params)

        deltas = {p: 0.0 for p in PARAM_NAMES}
        target = base.copy()

        for p in params_to_change:
            lo, hi = PARAM_RANGES[p]
            range_size = hi - lo
            # 随机方向和幅度
            direction = random.choice([-1, 1])
            magnitude = random.uniform(0.05, 0.4) * range_size * direction
            deltas[p] = magnitude
            target[p] = max(lo, min(hi, base[p] + magnitude))

        instruction = generate_compound_instruction(deltas)
        if not instruction:
            continue

        synthetic.append({
            'source': 'synthetic',
            'image_name': s['image_name'],
            'instruction': instruction,
            'base_params': base,
            'target_params': target,
            'delta_params': deltas,
        })

    return synthetic


def augment_with_llm(
    samples: List[dict],
    num_augment: int = 2000,
    api_key: str = None,
) -> List[dict]:
    """
    用大模型增强指令多样性

    将模板化的指令改写为更自然的表达
    """
    if not api_key:
        logger.warning('未提供 API key, 跳过 LLM 增强')
        return []

    try:
        import dashscope
        from dashscope import Generation
        dashscope.api_key = api_key
    except ImportError:
        logger.warning('dashscope 未安装, 跳过 LLM 增强')
        return []

    augmented = []
    subset = random.sample(samples, min(num_augment, len(samples)))

    REWRITE_PROMPT = """你是一个摄影调参助手。请将以下调参指令用更自然、口语化的方式改写。
保持语义不变，只改变表达方式。输出一行改写后的指令，不要其他内容。

原始指令: {instruction}
改写:"""

    for i, s in enumerate(subset):
        try:
            response = Generation.call(
                model='qwen-turbo',
                prompt=REWRITE_PROMPT.format(instruction=s['instruction']),
                max_tokens=50,
                temperature=0.8,
            )
            if response and response.output:
                new_instruction = response.output.text.strip()
                if 3 < len(new_instruction) < 50:
                    aug_sample = s.copy()
                    aug_sample['instruction'] = new_instruction
                    aug_sample['source'] = s['source'] + '_llm'
                    augmented.append(aug_sample)
        except Exception as e:
            if i == 0:
                logger.warning(f'LLM 调用失败: {e}')
                break
            continue

        if (i + 1) % 100 == 0:
            logger.info(f'LLM 增强: {i+1}/{len(subset)}')

    return augmented


def main():
    parser = argparse.ArgumentParser(description='Stage C 训练数据生成')
    parser.add_argument('--fivek_params', default='data/fivek_expert_params.json')
    parser.add_argument('--ppr10k_params', default='')
    parser.add_argument('--output', default='data/instruction_data.json')
    parser.add_argument('--num_synthetic', type=int, default=5000)
    parser.add_argument('--augment_with_llm', action='store_true')
    parser.add_argument('--llm_api_key', default='')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    all_samples = []

    # FiveK
    if args.fivek_params and Path(args.fivek_params).exists():
        fivek_samples = process_fivek_data(args.fivek_params)
        logger.info(f'FiveK 指令数据: {len(fivek_samples)}')
        all_samples.extend(fivek_samples)

    # PPR10K
    if args.ppr10k_params and Path(args.ppr10k_params).exists():
        ppr10k_samples = process_ppr10k_data(args.ppr10k_params)
        logger.info(f'PPR10K 指令数据: {len(ppr10k_samples)}')
        all_samples.extend(ppr10k_samples)

    if not all_samples:
        logger.error('没有生成任何数据!')
        return

    # 合成数据
    synthetic = generate_synthetic_instructions(all_samples, args.num_synthetic)
    logger.info(f'合成指令数据: {len(synthetic)}')
    all_samples.extend(synthetic)

    # LLM 增强
    if args.augment_with_llm:
        llm_samples = augment_with_llm(
            all_samples, api_key=args.llm_api_key or '')
        logger.info(f'LLM 增强数据: {len(llm_samples)}')
        all_samples.extend(llm_samples)

    # 统计
    source_counts = defaultdict(int)
    for s in all_samples:
        source_counts[s['source']] += 1
    logger.info(f'\n总数据量: {len(all_samples)}')
    for src, cnt in sorted(source_counts.items()):
        logger.info(f'  {src}: {cnt}')

    # 示例
    logger.info('\n=== 示例数据 ===')
    for s in random.sample(all_samples, min(5, len(all_samples))):
        logger.info(f"  [{s['source']}] \"{s['instruction']}\"")
        sig_deltas = {p: f"{d:+.1f}" for p, d in s['delta_params'].items()
                      if abs(d) > INTENSITY_THRESHOLDS.get(p, (10, 30))[0]}
        logger.info(f"    deltas: {sig_deltas}")

    # 输出
    output = {
        'meta': {
            'total': len(all_samples),
            'sources': dict(source_counts),
            'params': PARAM_NAMES,
        },
        'samples': all_samples,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f'\n输出: {out_path}')


if __name__ == '__main__':
    main()
