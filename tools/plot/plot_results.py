"""
实验结果可视化
生成论文级图表: PSNR/SSIM 对比、参数MAE 雷达图、训练曲线
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path
import json

# ============ 配置 ============
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
})

# 尝试中文字体
for font_name in ['SimHei', 'Microsoft YaHei', 'STSong', 'Arial Unicode MS']:
    if any(font_name in f.name for f in fm.fontManager.ttflist):
        plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

IMG_DIR = Path(__file__).parent.parent.parent / 'images'
RESULTS_DIR = IMG_DIR / 'results'
CURVES_DIR = IMG_DIR / 'training_curves' / 'semantic_distill_v3'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CURVES_DIR.mkdir(parents=True, exist_ok=True)

# ============ 数据 ============
MODELS = ['Baseline', 'Distill v1', 'Distill v2', 'Distill v3', 'Distill v4', 'Distill v5']
COLORS = ['#5B9BD5', '#70AD47', '#ED7D31', '#A855F7', '#E11D48', '#F59E0B']
HATCHES = ['/', '\\', 'x', '.', '+', 'o']

PSNR    = [32.05, 33.09, 33.15, 32.85, 33.11, 34.11]
SSIM    = [0.9269, 0.9257, 0.9311, 0.9278, 0.9326, 0.9449]
MS_SSIM = [0.9805, 0.9802, 0.9813, 0.9811, 0.9813, 0.9849]

PARAM_NAMES = ['EV', 'WB', 'Contrast', 'Brightness', 'Shadows', 'Highlights', 'Saturation', 'Vibrance']
MAE = {
    'Baseline':   [0.09, 501.98, 7.47, 3.77, 6.38, 7.53, 1.35, 6.06],
    'Distill v1': [0.08, 627.18, 7.20, 4.16, 5.53, 6.58, 1.21, 6.46],
    'Distill v2': [0.06, 620.45, 7.44, 3.81, 5.83, 6.99, 1.46, 6.50],
    'Distill v3': [0.07, 630.90, 7.28, 3.80, 5.81, 7.19, 1.36, 6.76],
    'Distill v4': [0.06, 618.73, 7.31, 3.77, 5.85, 7.10, 1.52, 6.81],
    'Distill v5': [0.03, 620.98, 2.08, 3.00, 0.13, 0.00, 1.00, 0.94],
}

# WB 归一化到百分比 (range=8000)
MAE_NORMALIZED = {}
PARAM_RANGES = [6.0, 8000, 200, 200, 200, 200, 200, 200]
for model, values in MAE.items():
    MAE_NORMALIZED[model] = [v / r * 100 for v, r in zip(values, PARAM_RANGES)]


def plot_psnr_ssim():
    """图1: PSNR/SSIM/MS-SSIM 柱状图对比"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))

    x = np.arange(len(MODELS))
    width = 0.6

    # PSNR
    bars1 = ax1.bar(x, PSNR, width, color=COLORS, edgecolor='white', linewidth=1.5)
    ax1.set_ylabel('PSNR (dB)', fontsize=13)
    ax1.set_title('PSNR Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODELS, fontsize=11)
    ax1.set_ylim(31.5, 34.5)
    for bar, val in zip(bars1, PSNR):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    best_psnr_idx = PSNR.index(max(PSNR))
    ax1.annotate('Best ★', xy=(best_psnr_idx, max(PSNR)), xytext=(best_psnr_idx + 0.5, 33.38),
                fontsize=10, color=COLORS[best_psnr_idx], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS[best_psnr_idx]))
    ax1.axhline(y=32.05, color='#5B9BD5', linestyle='--', alpha=0.5, label='Baseline')
    ax1.legend(fontsize=9)

    # SSIM
    bars2 = ax2.bar(x, SSIM, width, color=COLORS, edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('SSIM', fontsize=13)
    ax2.set_title('SSIM Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODELS, fontsize=11)
    ax2.set_ylim(0.922, 0.950)
    for bar, val in zip(bars2, SSIM):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    best_ssim_idx = SSIM.index(max(SSIM))
    ax2.annotate('Best ★', xy=(best_ssim_idx, max(SSIM)), xytext=(best_ssim_idx - 1.2, 0.935),
                fontsize=10, color=COLORS[best_ssim_idx], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS[best_ssim_idx]))
    ax2.axhline(y=0.9269, color='#5B9BD5', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend(fontsize=9)

    # MS-SSIM
    bars3 = ax3.bar(x, MS_SSIM, width, color=COLORS, edgecolor='white', linewidth=1.5)
    ax3.set_ylabel('MS-SSIM', fontsize=13)
    ax3.set_title('MS-SSIM Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(MODELS, fontsize=11)
    ax3.set_ylim(0.978, 0.987)
    for bar, val in zip(bars3, MS_SSIM):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00008,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    best_ms = max(MS_SSIM)
    best_idx = MS_SSIM.index(best_ms)
    ax3.annotate('Best ★', xy=(best_idx, best_ms), xytext=(best_idx - 0.8, best_ms + 0.0005),
                fontsize=10, color='#A855F7', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#A855F7'))
    ax3.axhline(y=MS_SSIM[0], color='#5B9BD5', linestyle='--', alpha=0.5, label='Baseline')
    ax3.legend(fontsize=9)

    plt.suptitle('Image Quality Metrics: v2 Best PSNR, v4 Best SSIM (FiveK-domain Stage A)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = RESULTS_DIR / 'fig_psnr_ssim_comparison.png'
    fig.savefig(path)
    plt.close()
    print(f'[1/5] PSNR/SSIM/MS-SSIM 对比图 → {path}')


def plot_psnr_gain():
    """图2: PSNR 提升量 (vs Baseline) + 域偏移标注"""
    fig, ax = plt.subplots(figsize=(8, 5))

    gains = [p - PSNR[0] for p in PSNR]
    x = np.arange(len(MODELS))

    bars = ax.bar(x, gains, 0.6, color=COLORS, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, gains):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.06
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'+{val:.2f} dB' if val > 0 else f'{val:.2f} dB',
                ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=11, fontweight='bold')

    # 域偏移标注
    ax.annotate('Domain Shift!\n5.3K COCO data\nhurts generalization',
                xy=(3, gains[3]), xytext=(1.5, 0.25),
                fontsize=9, color='#A855F7',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F3E8FF', edgecolor='#A855F7'),
                arrowprops=dict(arrowstyle='->', color='#A855F7', lw=1.5))

    # v4 FiveK域标注
    ax.annotate('FiveK-domain\nStage A',
                xy=(4, gains[4]), xytext=(2.8, 1.2),
                fontsize=9, color='#E11D48',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE4E6', edgecolor='#E11D48'),
                arrowprops=dict(arrowstyle='->', color='#E11D48', lw=1.5))

    # v5 最优标注
    ax.annotate('Expert C\nSingle-Expert ★ Best',
                xy=(5, gains[5]), xytext=(3.8, 1.9),
                fontsize=9, color='#F59E0B',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF3C7', edgecolor='#F59E0B'),
                arrowprops=dict(arrowstyle='->', color='#F59E0B', lw=1.5))

    ax.set_ylabel('PSNR Gain vs Baseline (dB)', fontsize=13)
    ax.set_title('Semantic Distillation: PSNR Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylim(-0.2, 2.3)

    # Stage A 数据量标注
    stage_a_labels = ['No distill', '3K COCO', '3K COCO\n(tuned B)', '5.3K COCO', '2K FiveK', '2K FiveK\n+Expert C']
    for i, (lbl, color) in enumerate(zip(stage_a_labels, COLORS)):
        ax.text(i, -0.15, lbl, ha='center', fontsize=7.5, color=color)

    plt.tight_layout()
    path = RESULTS_DIR / 'fig_psnr_gain.png'
    fig.savefig(path)
    plt.close()
    print(f'[2/5] PSNR 提升量图 → {path}')


def plot_param_mae_radar():
    """图3: 参数MAE 雷达图 (归一化)"""
    # 去掉 WB (范围太大影响可视化)，用归一化百分比
    params_display = ['EV', 'Contrast', 'Brightness', 'Shadows', 'Highlights', 'Saturation', 'Vibrance']
    idx = [0, 2, 3, 4, 5, 6, 7]  # 跳过 WB

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(params_display), endpoint=False).tolist()
    angles += angles[:1]

    for i, (model, color) in enumerate(zip(['Baseline', 'Distill v2', 'Distill v3'], ['#5B9BD5', '#ED7D31', '#A855F7'])):
        values = [MAE_NORMALIZED[model][j] for j in idx]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color, markersize=6)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(params_display, fontsize=11)
    ax.set_title('Parameter MAE (% of range)\nLower = Better', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.set_ylim(0, 5)

    plt.tight_layout()
    path = RESULTS_DIR / 'fig_param_mae_radar.png'
    fig.savefig(path)
    plt.close()
    print(f'[3/5] 参数MAE 雷达图 → {path}')


def plot_training_curves():
    """图4: v3 Stage A + Stage B 训练曲线"""
    ckpt_dir = Path(__file__).parent.parent.parent / 'checkpoints' / 'semantic_distill_v3'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stage A
    sa_path = ckpt_dir / 'stage_a' / 'history.json'
    if sa_path.exists():
        with open(sa_path) as f:
            sa = json.load(f)
        epochs_a = [e['epoch'] for e in sa]
        cos_train = [e['train']['cos_sim_mean'] for e in sa]
        cos_val = [e['val']['cos_sim_mean'] for e in sa]
        loss_train = [e['train']['total'] for e in sa]
        loss_val = [e['val']['total'] for e in sa]

        ax = axes[0]
        ax2 = ax.twinx()
        l1, = ax.plot(epochs_a, loss_train, 'b-', linewidth=2, label='Train Loss')
        l2, = ax.plot(epochs_a, loss_val, 'b--', linewidth=2, label='Val Loss')
        l3, = ax2.plot(epochs_a, cos_val, 'r-', linewidth=2, label='Val Cos Sim')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12, color='blue')
        ax2.set_ylabel('Cosine Similarity', fontsize=12, color='red')
        ax2.set_ylim(0.99, 1.001)
        ax.set_title('Stage A: Semantic Alignment (5.3K)', fontsize=13, fontweight='bold')
        lines = [l1, l2, l3]
        ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc='center right')

    # Stage B
    sb_path = ckpt_dir / 'stage_b' / 'history.json'
    if sb_path.exists():
        with open(sb_path) as f:
            sb = json.load(f)
        epochs_b = [e['epoch'] for e in sb]
        train_loss = [e['train_loss'] for e in sb]
        val_loss = [e['val']['total'] for e in sb]
        wb_mae = [e['mae']['white_balance'] for e in sb]

        ax = axes[1]
        ax2 = ax.twinx()
        l1, = ax.plot(epochs_b, train_loss, 'b-', linewidth=2, label='Train Loss')
        l2, = ax.plot(epochs_b, val_loss, 'b--', linewidth=2, label='Val Loss')
        l3, = ax2.plot(epochs_b, wb_mae, 'r-', linewidth=2, alpha=0.7, label='WB MAE')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12, color='blue')
        ax2.set_ylabel('WB MAE (K)', fontsize=12, color='red')
        ax.set_title('Stage B: Parameter Fine-tuning', fontsize=13, fontweight='bold')
        # 标注 LR 下降点
        ax.axvline(x=11, color='gray', linestyle=':', alpha=0.5)
        ax.text(11.5, max(train_loss)*0.95, 'LR drop', fontsize=8, color='gray')
        lines = [l1, l2, l3]
        ax.legend(lines, [l.get_label() for l in lines], fontsize=9)

    plt.tight_layout()
    path = CURVES_DIR / 'fig_v3_training_curves.png'
    fig.savefig(path)
    plt.close()
    print(f'[4/5] 训练曲线图 → {path}')


def plot_summary_table():
    """图5: 综合结果汇总表"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    col_labels = ['Model', 'Stage A\nData', 'PSNR (dB)', 'SSIM', 'Δ PSNR', 'EV MAE', 'WB MAE', 'Status']
    row_data = [
        ['Baseline',   'N/A',       '32.05', '0.9269', '—',      '0.09', '502.0', '—'],
        ['Distill v1', '3K COCO',   '33.09', '0.9257', '+1.04',  '0.08', '627.2', '✓'],
        ['Distill v2', '3K COCO',   '33.15', '0.9311', '+1.10',  '0.06', '620.5', '★ PSNR Best'],
        ['Distill v3', '5.3K COCO', '32.85', '0.9278', '+0.80',  '0.07', '630.9', '↓ Domain Shift'],
        ['Distill v4', '2K FiveK',  '33.11', '0.9326', '+1.06',  '0.06', '618.7', '★ SSIM Best'],
    ]

    colors = [
        ['#E8F0FE'] * 8,
        ['#E8F8E8'] * 8,
        ['#FFF3E0'] * 8,
        ['#FCE4EC'] * 8,
        ['#FFE4E6'] * 8,
    ]

    table = ax.table(cellText=row_data, colLabels=col_labels,
                     cellColours=colors, colColours=['#D6E4F0']*8,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Bold header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
        cell.set_edgecolor('#CCCCCC')

    ax.set_title('MobileVenus: Semantic Distillation Results Summary',
                fontsize=14, fontweight='bold', pad=20)

    path = RESULTS_DIR / 'fig_results_summary.png'
    fig.savefig(path)
    plt.close()
    print(f'[5/5] 结果汇总表 → {path}')


def plot_venus_scores():
    """图6: Venus 美学评分雷达图 + 柱状图"""
    venus_json = Path(__file__).parent.parent.parent / 'outputs' / 'data' / 'venus_eval_results.json'
    if not venus_json.exists():
        print(f'[跳过] Venus 结果文件不存在: {venus_json}')
        return

    with open(venus_json) as f:
        data = json.load(f)
    s = data['summary']

    dims = ['composition', 'lighting', 'color', 'clarity', 'subject', 'overall']
    dims_label = ['Composition', 'Lighting', 'Color', 'Clarity', 'Subject', 'Overall']
    groups = ['original', 'baseline', 'distill_v2']
    labels = ['Original', 'Baseline', 'Distill v2']
    colors = ['#5B9BD5', '#70AD47', '#ED7D31']

    scores = {g: [s[f'{g}_avg'][d] for d in dims] for g in groups}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── 左图: 柱状图 (Overall + 各维度) ──
    x = np.arange(len(dims_label))
    w = 0.25
    for i, (g, label, color) in enumerate(zip(groups, labels, colors)):
        vals = scores[g]
        bars = ax1.bar(x + (i - 1) * w, vals, w, label=label, color=color,
                       edgecolor='white', linewidth=1.2)
        # 标注 overall 列的数值
        bar_overall = bars[5]
        ax1.text(bar_overall.get_x() + bar_overall.get_width()/2,
                 bar_overall.get_height() + 0.02,
                 f'{vals[5]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Venus Score (1-10)', fontsize=12)
    ax1.set_title('Venus Aesthetic Scoring\n(Teacher Model Evaluation)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(dims_label, fontsize=10)
    ax1.set_ylim(4.5, 6.2)
    ax1.legend(fontsize=11)
    ax1.axhline(y=scores['original'][5], color='#5B9BD5', linestyle='--', alpha=0.4)

    # 标注关键提升
    ax1.annotate('Distill v2\nbeats Original\non Lighting (+0.19)',
                xy=(1 + 0.25, scores['distill_v2'][1]),
                xytext=(2.5, 5.9),
                fontsize=8, color='#ED7D31',
                arrowprops=dict(arrowstyle='->', color='#ED7D31', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E0', edgecolor='#ED7D31'))

    # ── 右图: Overall 分数 + Delta ──
    overall_scores = [scores[g][5] for g in groups]
    deltas = [v - overall_scores[0] for v in overall_scores]

    ax2_twin = ax2.twinx()
    bars2 = ax2.bar(labels, overall_scores, 0.5, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val, delta in zip(bars2, overall_scores, deltas):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}\n({delta:+.2f})', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    ax2.set_ylabel('Overall Venus Score', fontsize=12)
    ax2.set_title('Overall Score & Delta vs Original', fontsize=13, fontweight='bold')
    ax2.set_ylim(4.5, 6.0)
    ax2.axhline(y=overall_scores[0], color='#5B9BD5', linestyle='--', alpha=0.5, label='Original baseline')

    # 关键结论文本框
    conclusion = (
        "Key Finding:\n"
        "• Distill v2 > Original (+0.14)\n"
        "• Baseline < Original (−0.16)\n"
        "• Semantic distillation\n"
        "  is the critical factor"
    )
    ax2.text(0.97, 0.05, conclusion, transform=ax2.transAxes,
             fontsize=9, va='bottom', ha='right',
             bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50', alpha=0.9))

    plt.tight_layout()
    path = RESULTS_DIR / 'fig_venus_aesthetic_scores.png'
    fig.savefig(path)
    plt.close()
    print(f'[6/6] Venus 美学评分图 → {path}')


def plot_venus_radar():
    """图7: Venus 评分雷达图"""
    venus_json = Path(__file__).parent.parent.parent / 'outputs' / 'venus_eval_results.json'
    if not venus_json.exists():
        return

    with open(venus_json) as f:
        data = json.load(f)
    s = data['summary']

    dims = ['composition', 'lighting', 'color', 'clarity', 'subject']
    dims_label = ['Composition', 'Lighting', 'Color', 'Clarity', 'Subject']
    groups = ['original', 'baseline', 'distill_v2']
    labels = ['Original', 'Baseline', 'Distill v2']
    colors = ['#5B9BD5', '#70AD47', '#ED7D31']

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    for g, label, color in zip(groups, labels, colors):
        vals = [s[f'{g}_avg'][d] for d in dims]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=label, color=color, markersize=7)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims_label, fontsize=12)
    ax.set_ylim(4.5, 6.5)
    ax.set_title('Venus Aesthetic Scores\n(Radar View)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=11)

    plt.tight_layout()
    path = RESULTS_DIR / 'fig_venus_radar.png'
    fig.savefig(path)
    plt.close()
    print(f'[7/7] Venus 雷达图 → {path}')


def plot_nima_vs_venus():
    """图8: NIMA + Venus 综合对比图"""
    nima_json = Path(__file__).parent.parent.parent / 'outputs' / 'data' / 'nima_results.json'
    venus_json = Path(__file__).parent.parent.parent / 'outputs' / 'data' / 'venus_eval_results.json'

    if not nima_json.exists() or not venus_json.exists():
        print(f'[跳过] NIMA 或 Venus 结果文件不存在')
        return

    with open(nima_json) as f:
        nima_data = json.load(f)
    with open(venus_json) as f:
        venus_data = json.load(f)

    groups = ['original', 'baseline', 'distill_v2']
    labels = ['Original', 'Baseline', 'Distill v2']
    colors = ['#5B9BD5', '#70AD47', '#ED7D31']

    nima_means = [nima_data['summary'][g]['mean'] for g in groups]
    venus_overall = [venus_data['summary'][f'{g}_avg']['overall'] for g in groups]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # ── 左图: NIMA 分数 ──
    ax1 = axes[0]
    bars = ax1.bar(labels, nima_means, 0.5, color=colors, edgecolor='white', linewidth=1.5)
    nima_deltas = [v - nima_means[0] for v in nima_means]
    for bar, val, delta in zip(bars, nima_means, nima_deltas):
        sign = '+' if delta >= 0 else ''
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}\n({sign}{delta:.4f})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('NIMA Score (AVA)', fontsize=12)
    ax1.set_title('NIMA-Aesthetic Score', fontsize=13, fontweight='bold')
    ax1.set_ylim(4.2, 4.8)
    ax1.axhline(y=nima_means[0], color='#5B9BD5', linestyle='--', alpha=0.5)

    # ── 中图: Venus Overall ──
    ax2 = axes[1]
    bars2 = ax2.bar(labels, venus_overall, 0.5, color=colors, edgecolor='white', linewidth=1.5)
    venus_deltas = [v - venus_overall[0] for v in venus_overall]
    for bar, val, delta in zip(bars2, venus_overall, venus_deltas):
        sign = '+' if delta >= 0 else ''
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}\n({sign}{delta:.2f})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Venus Overall Score (1-10)', fontsize=12)
    ax2.set_title('Venus Aesthetic Score', fontsize=13, fontweight='bold')
    ax2.set_ylim(4.6, 5.8)
    ax2.axhline(y=venus_overall[0], color='#5B9BD5', linestyle='--', alpha=0.5)

    # ── 右图: 综合排名 (归一化 Delta) ──
    ax3 = axes[2]
    nima_norm = [d / abs(nima_deltas[1]) if nima_deltas[1] != 0 else 0 for d in nima_deltas]
    venus_norm = [d / abs(venus_deltas[1]) if venus_deltas[1] != 0 else 0 for d in venus_deltas]

    x = np.arange(len(labels))
    w = 0.35
    ax3.bar(x - w/2, nima_norm, w, label='NIMA', color='#7B68EE', edgecolor='white', linewidth=1.2)
    ax3.bar(x + w/2, venus_norm, w, label='Venus', color='#FF6B6B', edgecolor='white', linewidth=1.2)
    ax3.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=11)
    ax3.set_ylabel('Normalized Delta vs Original', fontsize=11)
    ax3.set_title('Relative Ranking\n(Both Metrics Agree: Distill v2 > Baseline)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=11)

    # 结论框
    conclusion = (
        "Consistent Finding:\n"
        "Distill v2 > Baseline\n"
        "on BOTH NIMA & Venus\n\n"
        "NIMA note: both models\n"
        "below original (expected\n"
        "for professional photos)"
    )
    ax3.text(0.97, 0.97, conclusion, transform=ax3.transAxes,
             fontsize=8.5, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#FBC02D', alpha=0.95))

    plt.suptitle('Multi-Metric Aesthetic Evaluation: NIMA vs Venus', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / 'fig_nima_vs_venus.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'[8/8] NIMA vs Venus 对比图 → {path}')


if __name__ == '__main__':
    print('=' * 50)
    print('  MobileVenus 实验结果可视化')
    print('=' * 50)
    plot_psnr_ssim()
    plot_psnr_gain()
    plot_param_mae_radar()
    plot_training_curves()
    plot_summary_table()
    plot_venus_scores()
    plot_venus_radar()
    plot_nima_vs_venus()
    print(f'\n所有图表已保存到 {RESULTS_DIR}/ 和 {CURVES_DIR}/')
