"""
NIMA-Aesthetic 评估脚本
对比 original / baseline / distill_v2 三组图片的美学评分

用法:
  python tools/eval_nima.py
  python tools/eval_nima.py --eval_dir outputs/venus_eval --output outputs/nima_results.json
"""
import sys
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_images(group_dir: Path):
    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    return sorted([p for p in group_dir.iterdir() if p.suffix.lower() in exts])


def run_nima(eval_dir: Path, output_path: Path):
    import torch
    import pyiqa

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    print('加载 NIMA-Aesthetic 模型 (首次运行需下载 ~25MB)...')
    metric = pyiqa.create_metric('nima', device=device)
    print('NIMA 加载完成\n')

    groups = ['original', 'baseline', 'distill_v2']
    results = {}
    summary = {}

    for group in groups:
        group_dir = eval_dir / group
        if not group_dir.exists():
            print(f'[跳过] {group_dir} 不存在')
            continue

        images = load_images(group_dir)
        if not images:
            print(f'[跳过] {group} 目录为空')
            continue

        scores = []
        print(f'评估 {group} ({len(images)} 张)...')
        for i, img_path in enumerate(images, 1):
            try:
                score = metric(str(img_path)).item()
                scores.append({'image': img_path.stem, 'score': round(score, 4)})
                if i % 10 == 0:
                    avg = sum(s['score'] for s in scores) / len(scores)
                    print(f'  [{i}/{len(images)}] 当前均值: {avg:.4f}')
            except Exception as e:
                print(f'  [错误] {img_path.name}: {e}')

        if scores:
            avg = sum(s['score'] for s in scores) / len(scores)
            std = (sum((s['score'] - avg) ** 2 for s in scores) / len(scores)) ** 0.5
            results[group] = scores
            summary[group] = {
                'mean': round(avg, 4),
                'std': round(std, 4),
                'n': len(scores),
                'min': round(min(s['score'] for s in scores), 4),
                'max': round(max(s['score'] for s in scores), 4),
            }
            print(f'  {group}: mean={avg:.4f} ± {std:.4f}\n')

    # 计算 delta
    if 'original' in summary:
        orig_mean = summary['original']['mean']
        for g in ['baseline', 'distill_v2']:
            if g in summary:
                summary[g]['delta_vs_original'] = round(summary[g]['mean'] - orig_mean, 4)
        summary['original']['delta_vs_original'] = 0.0

    output = {'summary': summary, 'results': results}
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # 打印汇总
    print('=' * 60)
    print('  NIMA-Aesthetic 评估汇总')
    print('=' * 60)
    print(f'  {"组别":<15} {"均值":>8} {"std":>8} {"Delta":>8}')
    print(f'  {"-"*45}')
    for g in groups:
        if g in summary:
            s = summary[g]
            delta = s.get('delta_vs_original', 0)
            sign = '+' if delta >= 0 else ''
            print(f'  {g:<15} {s["mean"]:>8.4f} {s["std"]:>8.4f} {sign}{delta:>7.4f}')
    print('=' * 60)
    print(f'\n结果已保存: {output_path}')

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dir', default='outputs/data/venus_eval',
                        help='评估图片目录 (含 original/baseline/distill_v2)')
    parser.add_argument('--output', default='outputs/data/nima_results.json')
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    output_path = Path(args.output)

    if not eval_dir.exists():
        print(f'[错误] 评估目录不存在: {eval_dir}')
        print('请先解压 outputs/venus_eval.zip 到 outputs/venus_eval/')
        sys.exit(1)

    run_nima(eval_dir, output_path)


if __name__ == '__main__':
    main()
