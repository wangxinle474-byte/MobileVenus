"""
Venus 美学评分对比图
读取 outputs/data/venus_eval_results_all.json，生成三张图：
  1. Overall 分数柱状图（含 delta）
  2. 各维度分组柱状图
  3. Delta vs 原图雷达图
"""
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DATA_FILE = Path('outputs/data/venus_eval_results_all.json')
OUT_DIR   = Path('images/results')

MODELS = ['original', 'baseline', 'distill_v2', 'distill_v4', 'distill_v5']
LABELS = ['Original', 'Baseline', 'Distill v2', 'Distill v4', 'Distill v5']
COLORS = ['#94A3B8', '#5B9BD5', '#70AD47', '#E11D48', '#F59E0B']
DIMS   = ['composition', 'lighting', 'color', 'clarity', 'subject', 'overall']
DIM_LABELS = ['Comp.', 'Lighting', 'Color', 'Clarity', 'Subject', 'Overall']


def load(path=DATA_FILE):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d['summary']


def plot_overall(s, out_dir):
    overall   = [s[f'{m}_avg']['overall'] for m in MODELS]
    orig_val  = overall[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(LABELS, overall, color=COLORS, width=0.55,
                  edgecolor='white', linewidth=1.2)
    ax.axhline(orig_val, color='gray', linestyle='--', alpha=0.6,
               label=f'Original baseline = {orig_val:.2f}')

    for bar, v in zip(bars, overall):
        delta = v - orig_val
        sign  = '+' if delta >= 0 else ''
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f'{v:.2f}\n({sign}{delta:.2f})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(4.7, 5.65)
    ax.set_ylabel('Venus Overall Score (1–10)', fontsize=12)
    ax.set_title('Venus Aesthetic Evaluation — Overall Score', fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', labelsize=11)
    ax.legend(fontsize=10)
    fig.tight_layout()
    path = out_dir / 'fig_venus_overall.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_dimensions(s, out_dir):
    fig, ax = plt.subplots(figsize=(13, 6))
    x     = np.arange(len(DIMS))
    total = len(MODELS)
    width = 0.15

    for i, (m, label, color) in enumerate(zip(MODELS, LABELS, COLORS)):
        vals = [s[f'{m}_avg'][d] for d in DIMS]
        offset = (i - total / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=color, alpha=0.88, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(DIM_LABELS, fontsize=11)
    ax.set_ylim(4.8, 5.85)
    ax.set_ylabel('Venus Score (1–10)', fontsize=12)
    ax.set_title('Venus Aesthetic Evaluation — Per-Dimension Scores',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    fig.tight_layout()
    path = out_dir / 'fig_venus_dimensions.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_radar(s, out_dir):
    radar_dims  = DIMS[:5]  # exclude overall
    radar_labels = DIM_LABELS[:5]
    N      = len(radar_dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))

    for m, label, color in zip(MODELS[1:], LABELS[1:], COLORS[1:]):
        delta = [s[f'{m}_avg'][d] - s['original_avg'][d] for d in radar_dims]
        delta += delta[:1]
        ax.plot(angles, delta, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, delta, alpha=0.08, color=color)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=12)
    ax.set_title('Delta vs Original (per dimension)',
                 fontsize=13, fontweight='bold', pad=25)
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.35, 1.15))
    fig.tight_layout()
    path = out_dir / 'fig_venus_radar.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def print_table(s):
    orig_overall = s['original_avg']['overall']
    print()
    print('=' * 65)
    print('  Venus 美学评分汇总')
    print('=' * 65)
    header = f"  {'Model':<14} {'Overall':>8} {'Delta':>8}"
    for dl in DIM_LABELS[:5]:
        header += f' {dl[:5]:>7}'
    print(header)
    print(f"  {'-'*62}")
    for m, label in zip(MODELS, LABELS):
        ov    = s[f'{m}_avg']['overall']
        delta = ov - orig_overall
        row   = f"  {label:<14} {ov:>8.2f} {delta:>+8.2f}"
        for d in DIMS[:5]:
            row += f" {s[f'{m}_avg'][d]:>7.2f}"
        print(row)
    print()


def main():
    if not DATA_FILE.exists():
        print(f'[ERROR] 找不到 {DATA_FILE}')
        return
    s = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_overall(s, OUT_DIR)
    plot_dimensions(s, OUT_DIR)
    plot_radar(s, OUT_DIR)
    print_table(s)


if __name__ == '__main__':
    main()
