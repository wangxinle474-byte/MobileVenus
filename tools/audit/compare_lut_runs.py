"""横向对比所有 lut_v* 实验的 best PSNR + 训练曲线 + per-action 性能.

用法:
  python tools/compare_lut_runs.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.firered_baseline.train_lut import (  # noqa: E402
    LUTPredictor, PerActionLUTPredictor, HybridLUTPredictor,
    LUTDataset, build_data, ACTIONS,
)
from torch.utils.data import DataLoader

CKPT_ROOT = PROJECT_ROOT / 'checkpoints'

# 实验描述 (顺序 = 表格列出顺序)
EXPS = [
    ('lut_v1', 'shared LUT, dim=33, 3 basis'),
    ('lut_v2', 'shared LUT, dim=17, 强正则'),
    ('lut_v3', 'shared LUT + color_aug + crop'),
    ('lut_v4', 'shared LUT, 5 basis'),
    ('lut_v5', 'per-action LUT, dim=33, 3 basis'),
    ('lut_v6', 'per-action LUT, dim=17, dropout=0.5'),
    ('lut_v7', 'hybrid: wb=3-gain + others=per-action LUT'),
]

JSONL = PROJECT_ROOT / 'outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl'


def load_history(d):
    p = d / 'history.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


@torch.no_grad()
def eval_per_action(ckpt_path, val_loader, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt['args']
    if a.get('hybrid_wb'):
        m = HybridLUTPredictor(
            n_luts=a['n_luts'], lut_dim=a['lut_dim'],
            image_size=a['image_size'], dropout=a['dropout'],
        )
    elif a.get('per_action_lut'):
        m = PerActionLUTPredictor(
            n_luts=a['n_luts'], lut_dim=a['lut_dim'],
            image_size=a['image_size'], dropout=a['dropout'],
        )
    else:
        m = LUTPredictor(
            n_luts=a['n_luts'], lut_dim=a['lut_dim'],
            image_size=a['image_size'], dropout=a['dropout'],
        )
    m.load_state_dict(ckpt['model_state_dict'])
    m.to(device).eval()

    pa = {a: [] for a in ACTIONS}
    for batch in val_loader:
        ei = batch['enc_input'].to(device)
        og = batch['orig'].to(device)
        tg = batch['target'].to(device)
        oh = batch['action_onehot'].to(device)
        out, _ = m(ei, og, oh)
        for b in range(out.shape[0]):
            mse = ((out[b] - tg[b]) ** 2).mean()
            psnr = (-10 * torch.log10(mse.clamp(min=1e-10))).item()
            act = batch['action'][b]
            pa[act].append(psnr)
    return {k: np.mean(v) if v else 0.0 for k, v in pa.items()}, ckpt.get('val_psnr', 0.0)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}\n')

    # Build val set (same seed across all)
    train_s, val_s = build_data(
        JSONL, val_ratio=0.2,
        tier_filter=('A excellent', 'B good', 'C acceptable'),
        seed=42)
    val_ds = LUTDataset(val_s, image_size=256, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    rows = []
    for name, desc in EXPS:
        ckpt_dir = CKPT_ROOT / name
        ckpt_path = ckpt_dir / 'best.pt'
        if not ckpt_path.exists():
            print(f'[SKIP] {name}: no checkpoint')
            continue

        h = load_history(ckpt_dir)
        if h:
            best = max(h, key=lambda r: r['val']['psnr'])
            best_ep = best['epoch']
            n_ep = len(h)
            train_psnr_at_best = best['train']['psnr']
            val_psnr_at_best = best['val']['psnr']
            gap = train_psnr_at_best - val_psnr_at_best
        else:
            best_ep = -1
            n_ep = -1
            gap = 0.0
            val_psnr_at_best = 0.0

        try:
            pa, ckpt_psnr = eval_per_action(ckpt_path, val_loader, device)
        except Exception as e:
            print(f'[ERR] {name}: {e}')
            continue

        overall = np.mean(list(pa.values()))
        rows.append({
            'name': name, 'desc': desc,
            'best_ep': best_ep, 'n_ep': n_ep,
            'val_psnr_history': val_psnr_at_best,
            'val_psnr_eval': overall,
            'train_val_gap': gap,
            **{f'pa_{k}': pa[k] for k in ACTIONS},
        })

    # 排序
    rows.sort(key=lambda r: -r['val_psnr_eval'])

    # 主表
    print('\n' + '=' * 108)
    print('LUT 实验横向对比 (val PSNR @ best epoch, 按整体PSNR降序)')
    print('=' * 108)
    print(f'{"name":<8} | {"desc":<40} | {"PSNR":>6} | {"train-val gap":>12} | {"best_ep":>7}')
    print('-' * 108)
    for r in rows:
        marker = ' ⭐ NEW BEST' if r == rows[0] else ''
        print(f'{r["name"]:<8} | {r["desc"]:<40} | '
              f'{r["val_psnr_eval"]:>6.2f} | {r["train_val_gap"]:>12.2f} | '
              f'{r["best_ep"]:>4}/{r["n_ep"]}{marker}')

    # Per-action
    print('\n' + '=' * 108)
    print('Per-action PSNR 详细对比 (7D ISP ceiling: 23.94 dB)')
    print('=' * 108)
    header = f'{"name":<8} | ' + ' | '.join(f'{a:>10}' for a in ACTIONS) + f' | {"overall":>7}'
    print(header)
    print('-' * 108)
    for r in rows:
        line = f'{r["name"]:<8} | '
        for a in ACTIONS:
            v = r[f'pa_{a}']
            mark = '✓' if v > 23.94 else ' '
            line += f'{v:>9.2f}{mark} | '
        line += f'{r["val_psnr_eval"]:>7.2f}'
        print(line)

    # 总结
    print('\n' + '=' * 108)
    print('SUMMARY')
    print('=' * 108)
    print(f'7D ISP ceiling reference: 23.94 dB')
    if rows:
        best = rows[0]
        diff = best['val_psnr_eval'] - 23.94
        status = '✅ 突破 ceiling' if diff > 0 else '⚠️ 未破 ceiling'
        print(f'最佳实验: {best["name"]} ({best["desc"]})')
        print(f'  整体 PSNR: {best["val_psnr_eval"]:.2f} dB  ({status}, diff={diff:+.2f} dB)')
        print(f'  train-val gap: {best["train_val_gap"]:.2f} dB')
        n_break = sum(1 for a in ACTIONS if best[f'pa_{a}'] > 23.94)
        print(f'  突破 ceiling 的 action 数: {n_break}/{len(ACTIONS)}')

    # 保存 JSON
    out_path = PROJECT_ROOT / 'outputs/lut_comparison.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out_path, 'w'), indent=2, default=float)
    print(f'\n详细结果已保存: {out_path}')


if __name__ == '__main__':
    main()
