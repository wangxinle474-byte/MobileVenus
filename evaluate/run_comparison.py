"""对比实验: 各方法在 FiveK 测试集上的参数预测性能。

生成 comparison_results.json 和 LaTeX 表格。

用法:
    python evaluate/run_comparison.py \
        --jpeg_dir E:/dataset/fivek_jpeg \
        --params_json data/fivek_expert_params.json \
        --checkpoint checkpoints/distill_v8/stage_b/best.pt \
        --output evaluate/results/
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
from training.fivek_8param.dataset import FiveKDataset
from training.semantic_distill.model import DistillParamModel
from .baselines import (RuleBasedPredictor, RandomPredictor,
                        OraclePredictor, StatisticalPredictor)


def denormalize_params(norm_params, param_names=PARAM_NAMES):
    """将归一化参数 [-1,1] 还原到实际范围。"""
    result = {}
    for i, name in enumerate(param_names):
        lo, hi = PARAM_RANGES[name]
        val = (norm_params[i] + 1.0) / 2.0 * (hi - lo) + lo
        result[name] = float(val)
    return result


def compute_metrics(pred_params_list, gt_params_list):
    """计算 MAE, RMSE, Accuracy。"""
    all_mae = {name: [] for name in PARAM_NAMES}
    all_se = {name: [] for name in PARAM_NAMES}

    for pred, gt in zip(pred_params_list, gt_params_list):
        for name in PARAM_NAMES:
            p = pred.get(name, 0.0)
            g = gt.get(name, 0.0)
            all_mae[name].append(abs(p - g))
            all_se[name].append((p - g) ** 2)

    metrics = {}
    for name in PARAM_NAMES:
        metrics[name] = {
            'mae': float(np.mean(all_mae[name])),
            'rmse': float(np.sqrt(np.mean(all_se[name]))),
        }

    # 综合指标
    total_mae = np.mean([metrics[n]['mae'] for n in PARAM_NAMES])
    total_rmse = np.mean([metrics[n]['rmse'] for n in PARAM_NAMES])
    metrics['overall'] = {
        'mae': float(total_mae),
        'rmse': float(total_rmse),
    }

    return metrics


def generate_latex_table(results, output_path):
    """生成 LaTeX 表格。"""
    methods = list(results.keys())
    header = "Method & " + " & ".join(
        [n.replace('_', ' ').title() for n in PARAM_NAMES]
    ) + " & Overall \\\\"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Comparison of parameter prediction methods on FiveK test set (MAE $\\downarrow$).}",
        "\\label{tab:comparison}",
        "\\begin{tabular}{l" + "c" * (len(PARAM_NAMES) + 1) + "}",
        "\\toprule",
        header,
        "\\midrule",
    ]

    for method in methods:
        m = results[method]
        vals = [f"{m[n]['mae']:.2f}" for n in PARAM_NAMES]
        vals.append(f"{m['overall']['mae']:.2f}")
        lines.append(f"{method} & " + " & ".join(vals) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpeg_dir', type=str, required=True)
    parser.add_argument('--params_json', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output', type=str, default='evaluate/results')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 加载测试数据
    test_dataset = FiveKDataset(
        jpeg_dir=args.jpeg_dir,
        params_json=args.params_json,
        split='test',
        image_size=224,
        augment=False,
        max_samples=args.num_samples,
    )

    # 加载 GT 参数
    with open(args.params_json, 'r') as f:
        all_params = json.load(f)

    results = {}

    # --- Baselines ---
    print("Running baselines...")
    rule_pred = RuleBasedPredictor()
    random_pred = RandomPredictor()
    stat_pred = StatisticalPredictor(all_params)

    gt_list = []
    rule_list = []
    random_list = []
    stat_list = []

    for i in range(len(test_dataset)):
        item = test_dataset[i]
        gt = denormalize_params(item['params'].numpy())
        gt_list.append(gt)
        rule_list.append(rule_pred.predict(item['image']))
        random_list.append(random_pred.predict())
        stat_list.append(stat_pred.predict())

    results['Rule-Based'] = compute_metrics(rule_list, gt_list)
    results['Random'] = compute_metrics(random_list, gt_list)
    results['Statistical'] = compute_metrics(stat_list, gt_list)

    # --- Our Model ---
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading model: {args.checkpoint}")
        model = DistillParamModel()
        state = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(state['model_state_dict'], strict=False)
        model = model.to(args.device).eval()

        model_list = []
        loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(args.device)
                out = model(images)
                for j in range(images.shape[0]):
                    pred = {}
                    for name in PARAM_NAMES:
                        pred[name] = out['raw_params'][name][j].item()
                    model_list.append(pred)

        results['Ours'] = compute_metrics(model_list, gt_list)

    # 保存结果
    output_json = os.path.join(args.output, 'comparison_results.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_json}")

    # LaTeX 表格
    output_tex = os.path.join(args.output, 'table1_latex.tex')
    generate_latex_table(results, output_tex)
    print(f"Saved: {output_tex}")


if __name__ == '__main__':
    main()
