"""
用我们之前的评分标准给 IP2P 编辑前后的图打分.

指标 (与之前 eval 一致):
  - MUSIQ-AVA: 美学评分 (主用), 1-10
  - NIMA-VGG16-AVA: AVA 上的 NIMA, 1-10
  - CLIPIQA+: 通用图像质量, 0-1
  - LAION-Aes: LAION 美学预测器, 0-10

输出:
  - 控制台对比表
  - outputs/ip2p_test/scores.json
"""
import os
import sys
import json
import argparse
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('HF_HOME', r'E:\cache\huggingface')
os.environ.setdefault('HF_HUB_CACHE', r'E:\cache\huggingface\hub')

import torch
import pyiqa
from PIL import Image
from torchvision import transforms

ALL_METRICS = [
    ('musiq-ava',         '美学(主)',   1, 10),
    ('nima-vgg16-ava',    'NIMA-AVA',   1, 10),
    ('clipiqa+',          'CLIPIQA+',   0,  1),
    ('laion_aes',         'LAION-Aes',  0, 10),
    ('qualiclip',         'QualiCLIP',  0,  1),
    ('qualiclip+',        'QualiCLIP+', 0,  1),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='outputs/ip2p_test',
                        help='目录, 含 NNNN_orig.png / NNNN_edit.png / NNNN_prompt.txt')
    parser.add_argument('--metrics', nargs='+', default=None,
                        help='指标名 (pyiqa 的名字), 默认全部 4 个 (不含 qualiclip)')
    args = parser.parse_args()

    OUT_DIR = Path(args.dir)
    metric_map = {m[0]: m for m in ALL_METRICS}
    if args.metrics:
        # 按指定名称查找, 未知的用默认 label / 范围
        METRICS = [metric_map.get(n, (n, n, 0, 10)) for n in args.metrics]
    else:
        METRICS = ALL_METRICS[:4]  # 默认 4 个 (不含 qualiclip, 因 qualiclip 需要额外下 CLIP backbone)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Scoring dir: {OUT_DIR}')
    print(f'Metrics: {[m[0] for m in METRICS]}')

    # 加载所有评分器
    scorers = {}
    for name, _, _, _ in METRICS:
        print(f'  loading {name}...')
        try:
            scorers[name] = pyiqa.create_metric(name, device=device, as_loss=False)
        except Exception as e:
            print(f'    FAIL: {e}')

    # 找出所有样本
    import re
    samples = sorted(set(int(re.match(r'(\d+)_', f.name).group(1))
                         for f in OUT_DIR.glob('*_orig.png')))
    print(f'\nFound {len(samples)} samples in {OUT_DIR}')

    # 评分
    to_tensor = transforms.ToTensor()
    results = []
    for idx in samples:
        orig_p = OUT_DIR / f'{idx:04d}_orig.png'
        edit_p = OUT_DIR / f'{idx:04d}_edit.png'

        orig = Image.open(orig_p).convert('RGB')
        edit = Image.open(edit_p).convert('RGB')

        ot = to_tensor(orig).unsqueeze(0).to(device)
        et = to_tensor(edit).unsqueeze(0).to(device)

        scores_orig = {}
        scores_edit = {}
        for name in scorers:
            try:
                with torch.no_grad():
                    so = float(scorers[name](ot).cpu().item())
                    se = float(scorers[name](et).cpu().item())
                scores_orig[name] = so
                scores_edit[name] = se
            except Exception as e:
                print(f'    [{idx}] {name} fail: {e}')

        results.append({
            'idx': idx,
            'image': f'{idx:04d}',
            'scores_orig': scores_orig,
            'scores_edit': scores_edit,
        })

    # 输出对比表
    print()
    print('=' * 100)
    print(f'{"Sample":<10}{"Metric":<18}{"Original":>12}{"Edited":>12}{"Δ (edit-orig)":>18}{"判定":>10}')
    print('-' * 100)
    for r in results:
        for name, label, lo, hi in METRICS:
            if name not in r['scores_orig']:
                continue
            so = r['scores_orig'][name]
            se = r['scores_edit'][name]
            delta = se - so
            symbol = '↑提升' if delta > 0.05 * (hi - lo) else ('↓下降' if delta < -0.05 * (hi - lo) else '~持平')
            print(f'{r["image"]:<10}{label:<18}{so:>12.3f}{se:>12.3f}{delta:>+18.3f}{symbol:>12}')
        print('-' * 100)

    # 各指标平均 Δ
    print()
    print('=== 平均提升 ===')
    for name, label, lo, hi in METRICS:
        if name not in results[0]['scores_orig']:
            continue
        deltas = [r['scores_edit'][name] - r['scores_orig'][name] for r in results
                  if name in r['scores_orig'] and name in r['scores_edit']]
        if deltas:
            mean_delta = sum(deltas) / len(deltas)
            n_up = sum(1 for d in deltas if d > 0)
            print(f'  {label:<18} mean Δ={mean_delta:+.3f}  '
                  f'({n_up}/{len(deltas)} 张提升)')

    # 保存 JSON
    out_json = OUT_DIR / 'scores.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': [m[0] for m in METRICS],
            'results': results,
        }, f, ensure_ascii=False, indent=2)
    print(f'\nSaved: {out_json}')


if __name__ == '__main__':
    main()
