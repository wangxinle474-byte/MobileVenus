"""
训练曲线可视化工具
用法:
  python tools/plot_training.py                                    # 自动扫描 checkpoints/
  python tools/plot_training.py --files a/history.json b/history.json --labels baseline distill
"""
import json
import argparse
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11, 'figure.dpi': 150, 'savefig.dpi': 150,
    'figure.facecolor': 'white', 'axes.grid': True,
    'grid.alpha': 0.3, 'lines.linewidth': 1.8,
})

PARAM_NAMES = [
    'ev_compensation', 'white_balance', 'contrast', 'brightness',
    'shadows', 'highlights', 'saturation', 'vibrance',
]
PARAM_DISPLAY = {
    'ev_compensation': 'EV Comp', 'white_balance': 'White Balance',
    'contrast': 'Contrast', 'brightness': 'Brightness',
    'shadows': 'Shadows', 'highlights': 'Highlights',
    'saturation': 'Saturation', 'vibrance': 'Vibrance',
}
COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#FF9800',
          '#9C27B0', '#00BCD4', '#795548', '#E91E63']


def load_history(path: str) -> List[dict]:
    with open(path, 'r') as f:
        return json.load(f)


def detect_type(history: List[dict]) -> str:
    rec = history[0]
    if isinstance(rec.get('train'), dict) and 'cos_sim_mean' in rec['train']:
        return 'stage_a'
    if 'mae' in rec:
        return 'param'
    if isinstance(rec.get('train'), dict) and 'total' in rec['train']:
        return 'loss_only'  # nested train/val without MAE (e.g. fivek_semantic)
    if 'train_loss' in rec:
        return 'flat_loss'  # flat format (e.g. distill_v5 stage_b)
    return 'unknown'


def plot_stage_a(history: List[dict], save_path: Path):
    epochs = [r['epoch'] for r in history]
    train_loss = [r['train']['total'] for r in history]
    val_loss = [r['val']['total'] for r in history]
    train_cos = [r['train']['cos_sim_mean'] for r in history]
    val_cos = [r['val']['cos_sim_mean'] for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(epochs, train_loss, 'o-', ms=3, label='Train')
    ax.plot(epochs, val_loss, 's-', ms=3, label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Stage A — Semantic Alignment Loss')
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, train_cos, 'o-', ms=3, label='Train')
    ax.plot(epochs, val_cos, 's-', ms=3, label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Cosine Similarity')
    ax.set_title('Stage A — Cosine Similarity')
    ax.set_ylim(min(min(train_cos), min(val_cos)) - 0.02, 1.01)
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"保存: {save_path}")


def plot_param_loss(histories: List[List[dict]], labels: List[str], save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = [('o-', '#2196F3'), ('s-', '#FF5722'), ('^-', '#4CAF50'), ('D-', '#FF9800')]

    for i, (hist, label) in enumerate(zip(histories, labels)):
        epochs = [r['epoch'] for r in hist]
        if 'train_loss' in hist[0]:
            train_l = [r['train_loss'] for r in hist]
        else:
            train_l = [r['train']['total'] for r in hist]
        val_l = [r['val']['total'] for r in hist]
        st, c = styles[i % len(styles)]
        ax.plot(epochs, train_l, st, ms=2, color=c, alpha=0.5, label=f'{label} train')
        ax.plot(epochs, val_l, st, ms=3, color=c, label=f'{label} val')

    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Parameter Prediction — Training Loss')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"保存: {save_path}")


def plot_param_mae(histories: List[List[dict]], labels: List[str], save_path: Path):
    mae_keys = [k for k in PARAM_NAMES if k in histories[0][0].get('mae', {})]
    if not mae_keys:
        print("无 MAE 数据")
        return

    n = len(mae_keys)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    styles = ['-', '--', '-.', ':']

    for idx, key in enumerate(mae_keys):
        ax = axes[idx]
        for i, (hist, label) in enumerate(zip(histories, labels)):
            epochs = [r['epoch'] for r in hist]
            vals = [r['mae'][key] for r in hist]
            ax.plot(epochs, vals, styles[i % len(styles)],
                    color=COLORS[idx], linewidth=2, label=label)
            best_v = min(vals)
            best_ep = epochs[vals.index(best_v)]
            ax.annotate(f'{best_v:.1f}', xy=(best_ep, best_v),
                        fontsize=8, color=COLORS[idx], fontweight='bold',
                        xytext=(5, 5), textcoords='offset points')

        ax.set_title(PARAM_DISPLAY.get(key, key), fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('MAE', fontsize=9)
        if len(labels) > 1:
            ax.legend(fontsize=8)

    for idx in range(len(mae_keys), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Per-Parameter MAE Curves', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"保存: {save_path}")


def plot_param_mae_summary(histories: List[List[dict]], labels: List[str], save_path: Path):
    """柱状图对比各模型最终/最优 MAE"""
    mae_keys = [k for k in PARAM_NAMES if k in histories[0][0].get('mae', {})]
    if not mae_keys:
        return

    import numpy as np
    x = np.arange(len(mae_keys))
    width = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (hist, label) in enumerate(zip(histories, labels)):
        best_mae = {}
        best_epoch = min(hist, key=lambda r: r['val']['total'])
        for k in mae_keys:
            best_mae[k] = best_epoch['mae'][k]
        values = [best_mae[k] for k in mae_keys]
        bars = ax.bar(x + i * width, values, width, label=label,
                      color=COLORS[i % len(COLORS)], alpha=0.85)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels([PARAM_DISPLAY.get(k, k) for k in mae_keys], rotation=20, ha='right')
    ax.set_ylabel('MAE (Best Epoch)')
    ax.set_title('Model Comparison — Best MAE per Parameter')
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"保存: {save_path}")


def plot_loss_only(history: List[dict], label: str, save_path: Path):
    """fivek_semantic 等只有嵌套 train/val.total 的格式"""
    epochs = [r['epoch'] for r in history]
    train_l = [r['train']['total'] for r in history]
    val_l = [r['val']['total'] for r in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_l, 'o-', ms=3, label='Train')
    ax.plot(epochs, val_l, 's-', ms=3, label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'{label} — Training Loss')
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"保存: {save_path}")


def plot_flat_loss(history: List[dict], label: str, save_path: Path):
    """distill_v5 stage_b 等 flat 键格式，过滤 NaN"""
    import math
    valid = [r for r in history
             if r.get('train_loss') is not None
             and not math.isnan(float(r['train_loss']))]
    if not valid:
        print(f"  [跳过] {label}: history.json 全部为 NaN，无法绘制训练曲线")
        return

    epochs = [r['epoch'] for r in valid]
    train_l = [r['train_loss'] for r in valid]
    val_l = [r.get('val_img_loss', float('nan')) for r in valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_l, 'o-', ms=3, label='Train')
    ax.plot(epochs, val_l, 's-', ms=3, label='Val (img)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'{label} — Training Loss')
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"保存: {save_path}")


def auto_scan(root: str = 'checkpoints') -> List[tuple]:
    """自动扫描 checkpoints 下的 history.json"""
    results = []
    for p in sorted(Path(root).rglob('history.json')):
        label = str(p.parent.relative_to(root)).replace('\\', '/')
        results.append((str(p), label))
    return results


def main():
    parser = argparse.ArgumentParser(description='训练曲线可视化')
    parser.add_argument('--files', nargs='+', help='history.json 路径列表')
    parser.add_argument('--labels', nargs='+', help='对应的标签名')
    parser.add_argument('--scan_dir', default='checkpoints', help='自动扫描目录')
    parser.add_argument('--output_dir', default='images/training_curves', help='图片输出目录')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.files:
        pairs = list(zip(args.files, args.labels or [Path(f).parent.name for f in args.files]))
    else:
        pairs = auto_scan(args.scan_dir)
        if not pairs:
            print(f"在 {args.scan_dir}/ 下未找到 history.json")
            return

    print(f"找到 {len(pairs)} 个训练历史:")
    for f, l in pairs:
        print(f"  [{l}] {f}")

    stage_a_list, param_list = [], []
    for fpath, label in pairs:
        hist = load_history(fpath)
        t = detect_type(hist)
        # 每个训练单独一个子目录
        safe_name = label.replace("/", "_").replace("\\", "_")
        sub_dir = out_dir / safe_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        if t == 'stage_a':
            plot_stage_a(hist, sub_dir / 'loss_and_cosine.png')
            stage_a_list.append((hist, label))
        elif t == 'param':
            plot_param_loss([hist], [label], sub_dir / 'loss.png')
            plot_param_mae([hist], [label], sub_dir / 'mae_curves.png')
            plot_param_mae_summary([hist], [label], sub_dir / 'mae_summary.png')
            param_list.append((hist, label))
        elif t == 'loss_only':
            plot_loss_only(hist, label, sub_dir / 'loss.png')
        elif t == 'flat_loss':
            plot_flat_loss(hist, label, sub_dir / 'loss.png')

    # 多模型对比图放在 comparison/ 子目录
    if len(param_list) > 1:
        cmp_dir = out_dir / 'comparison'
        cmp_dir.mkdir(parents=True, exist_ok=True)
        hists = [h for h, _ in param_list]
        lbls = [l for _, l in param_list]
        plot_param_loss(hists, lbls, cmp_dir / 'loss_compare.png')
        plot_param_mae(hists, lbls, cmp_dir / 'mae_curves_compare.png')
        plot_param_mae_summary(hists, lbls, cmp_dir / 'mae_summary_compare.png')

    print(f"\n所有图表保存到: {out_dir}/")


if __name__ == '__main__':
    main()
