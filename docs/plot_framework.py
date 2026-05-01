"""
训练框架图 v4 — 顶刊风格 (CVPR / NeurIPS)
设计原则: 白底、3色、大字、留白、无交叉箭头
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib
matplotlib.rcParams['font.family'] = ['Arial', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

W, H = 16, 14
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── 配色 (3色 + 灰) ──
BLU   = '#3B82F6';  BLU_L = '#DBEAFE'   # 蓝 - 编码器
PUR   = '#8B5CF6';  PUR_L = '#EDE9FE'   # 紫 - Stage B
COR   = '#EF4444';  COR_L = '#FEE2E2'   # 红 - Stage C
TEA   = '#10B981';  TEA_L = '#D1FAE5'   # 绿 - 损失/部署
AMB   = '#F59E0B';  AMB_L = '#FEF3C7'   # 琥珀 - 数据/Venus
GR    = '#6B7280';  GR_D  = '#1F2937'   # 灰
WH    = '#FFFFFF'

# ── 绘图工具 ──
def mod(x, y, w, h, title, sub='', fc=WH, ec=BLU, lw=1.5, fs=10.5):
    ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle='round,pad=0.12',
                 fc=fc, ec=ec, lw=lw, zorder=3))
    if sub:
        ax.text(x+w/2, y+h*0.62, title, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=GR_D, zorder=4)
        ax.text(x+w/2, y+h*0.3, sub, ha='center', va='center',
                fontsize=fs-2, color=GR, zorder=4)
    else:
        ax.text(x+w/2, y+h/2, title, ha='center', va='center',
                fontsize=fs, fontweight='bold', color=GR_D, zorder=4)

def harr(x1, x2, y, c=GR, lw=1.8):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=lw), zorder=2)

def varr(x1, y1, x2=None, y2=None, c=GR, lw=1.5):
    """垂直/斜线箭头: varr(x1,y1,x2,y2) 或 varr(x,y1,y2=y2)"""
    if x2 is None:
        x2 = x1
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=lw), zorder=2)

def curved(x1, y1, x2, y2, c=BLU, lw=1.8, rad=0.25):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=c, lw=lw,
                               connectionstyle=f'arc3,rad={rad}',
                               linestyle='--'), zorder=2)

def sec_bg(y, h, c, alpha=0.08):
    ax.add_patch(FancyBboxPatch((0.4, y), W-0.8, h, boxstyle='round,pad=0.2',
                 fc=c, ec='none', alpha=alpha, zorder=0))

def sec_label(x, y, num, text, c):
    """Section label with circled number - placed ABOVE boxes"""
    circle = plt.Circle((x, y), 0.24, fc=c, ec=WH, lw=2, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y, str(num), ha='center', va='center', fontsize=10,
            fontweight='bold', color=WH, zorder=6)
    ax.text(x + 0.45, y, text, va='center', fontsize=10.5,
            fontweight='bold', color=c, zorder=5)

def lbl(x, y, text, c=GR, fs=8.5):
    ax.text(x, y, text, ha='center', va='center', fontsize=fs,
            color=c, style='italic', zorder=5,
            bbox=dict(boxstyle='round,pad=0.1', fc=WH, ec='none', alpha=0.9))

# ── 布局常量 ──
BW = 2.4   # box width
BH = 1.05  # box height
G  = 0.35  # gap between boxes

Y4 = 12.2  # Data row
Y3 = 9.2   # Stage A
Y2 = 6.2   # Stage B
Y1 = 3.2   # Stage C
Y0 = 0.8   # Deploy

# ══════════════════════════════════════
# Title
# ══════════════════════════════════════
ax.text(W/2, 13.65, 'Semantic-to-Parameter Framework',
        ha='center', fontsize=18, fontweight='bold', color=GR_D)

# ══════════════════════════════════════
# [1] Data Preparation
# ══════════════════════════════════════
sec_bg(Y4-0.15, BH+0.9, AMB)
sec_label(1.6, Y4+BH+0.35, 1, 'Data Preparation', AMB)

xs = [1.6, 1.6+BW+G, 1.6+2*(BW+G), 1.6+3*(BW+G)+0.2, 1.6+4*(BW+G)+0.4]
mod(xs[0], Y4, BW, BH, 'FiveK Dataset', '5120 images  |  5 experts', ec=AMB, fc=WH)
mod(xs[1], Y4, BW+0.3, BH, 'Venus Stage-1', 'Qwen-VL 9.6B + SFT', ec=AMB, fc=AMB_L)
mod(xs[2], Y4, BW, BH, 'Aesthetic Text', '5120 guidance texts', ec=AMB, fc=WH)
mod(xs[3], Y4, BW+0.3, BH, 'Quality Filter', 'AesExpert + 14 IQA metrics', ec=TEA, fc=WH)
mod(xs[4], Y4, BW, BH, 'Expert GT', '8 ISP params [-1, 1]', ec=BLU, fc=WH)

harr(xs[0]+BW, xs[1], Y4+BH/2, c=AMB)
harr(xs[1]+BW+0.3, xs[2], Y4+BH/2, c=AMB)
harr(xs[2]+BW, xs[3], Y4+BH/2, c=TEA)

# ══════════════════════════════════════
# [2] Stage A: Visual-Semantic Alignment
# ══════════════════════════════════════
sec_bg(Y3-0.15, BH+0.9, BLU)
sec_label(1.6, Y3+BH+0.35, 2, 'Stage A: Visual-Semantic Alignment', BLU)

xa_pos = [1.6, 1.6+BW+G+0.4, 1.6+2*(BW+G)+0.8, 1.6+3*(BW+G)+1.2]
mod(xa_pos[0], Y3, BW, BH, 'Text Encoder', 'MiniLM-L6  (frozen)', ec=GR, fc=WH)
mod(xa_pos[1], Y3, BW+0.2, BH, 'Visual Encoder', 'MobileViT-S  (trainable)', ec=BLU, fc=BLU_L, lw=2)
mod(xa_pos[2], Y3, BW, BH, 'Semantic Head', 'Linear + GELU  384d->256d', ec=BLU, fc=WH)
mod(xa_pos[3], Y3, BW+0.5, BH, 'Aligned Encoder', 'Cosine+MSE loss -> output', ec=BLU, fc=BLU_L, lw=2.5)

harr(xa_pos[0]+BW, xa_pos[1], Y3+BH/2, c=BLU)
harr(xa_pos[1]+BW+0.2, xa_pos[2], Y3+BH/2, c=BLU)
harr(xa_pos[2]+BW, xa_pos[3], Y3+BH/2, c=TEA)

# Data -> Stage A (语义正确: FiveK→Visual, Venus→Text)
# FiveK images → Visual Encoder (右侧垂直)
vis_cx = xa_pos[1] + (BW+0.2)/2   # Visual Encoder center
varr(xs[0]+BW*0.7, Y4, vis_cx, Y3+BH, c=BLU, lw=1.3)
lbl((xs[0]+BW*0.7 + vis_cx)/2 + 0.5, (Y4+Y3+BH)/2, 'images', BLU, 8)
# Venus text → Text Encoder (左侧垂直)
txt_cx = xa_pos[0] + BW/2          # Text Encoder center
varr(xs[1]+BW*0.3, Y4, txt_cx, Y3+BH, c=AMB, lw=1.3)
lbl((xs[1]+BW*0.3 + txt_cx)/2 - 0.5, (Y4+Y3+BH)/2, 'text', AMB, 8)

# ══════════════════════════════════════
# [3] Stage B: Visual -> ISP Parameters
# ══════════════════════════════════════
sec_bg(Y2-0.15, BH+0.9, PUR)
sec_label(1.6, Y2+BH+0.35, 3, 'Stage B: Visual-to-Parameter', PUR)

xb_pos = [1.6, 1.6+BW+G+0.4, 1.6+2*(BW+G)+0.8, 1.6+3*(BW+G)+1.2]
mod(xb_pos[0], Y2, BW+0.2, BH, 'Frozen Encoder', 'Stage A  (locked)', ec=PUR, fc=PUR_L)
mod(xb_pos[1], Y2, BW+0.2, BH, 'Param Decoder', 'MLP  256->256->8', ec=PUR, fc=WH)
mod(xb_pos[2], Y2, BW, BH, 'Diff-ISP', 'Differentiable rendering', ec=PUR, fc=WH)
mod(xb_pos[3], Y2, BW+0.5, BH, 'ISP Predictor', 'L1+SSIM+MSE -> output', ec=PUR, fc=PUR_L, lw=2.5)

harr(xb_pos[0]+BW+0.2, xb_pos[1], Y2+BH/2, c=PUR)
harr(xb_pos[1]+BW+0.2, xb_pos[2], Y2+BH/2, c=PUR)
harr(xb_pos[2]+BW, xb_pos[3], Y2+BH/2, c=TEA)

# Stage A -> Stage B weight transfer (右侧产出 -> 左侧冻结)
right_a = xa_pos[3] + (BW+0.5)/2  # Aligned Encoder 中心 x
left_b  = xb_pos[0] + (BW+0.2)/2  # Frozen Encoder 中心 x
curved(right_a, Y3, left_b, Y2+BH, c=BLU, lw=2, rad=0.35)
lbl((right_a+left_b)/2 + 2.0, (Y3+Y2+BH)/2, 'weight transfer', BLU)

# Expert GT -> Stage B
gt_cx = xs[4] + BW/2
varr(gt_cx, Y4, gt_cx, Y2+BH, c=GR, lw=1.2)
lbl(gt_cx + 0.35, (Y4+Y2+BH)/2, 'GT', GR, 7.5)

# ══════════════════════════════════════
# [4] Stage C: Text + Visual -> ISP
# ══════════════════════════════════════
sec_bg(Y1-0.15, BH+0.9, COR)
sec_label(1.6, Y1+BH+0.35, 4, 'Stage C: Text-Guided Prediction (FiLM)', COR)

xc_pos = [1.6, 1.6+BW+G+0.4, 1.6+2*(BW+G)+0.8, 1.6+3*(BW+G)+1.2]
mod(xc_pos[0], Y1, BW+0.2, BH, 'Stage B Backbone', 'Encoder + Decoder', ec=COR, fc=COR_L)
mod(xc_pos[1], Y1, BW, BH, 'Text Encoder', 'MiniLM-L6', ec=COR, fc=WH)
mod(xc_pos[2], Y1, BW+0.2, BH, 'FiLM Fusion', u'\u03b3(text)\u00b7vis + \u03b2(text)', ec=COR, fc=COR_L)
mod(xc_pos[3], Y1, BW+0.5, BH, 'Text-Guided ISP', '"too dark" -> exp +0.3', ec=COR, fc=COR_L, lw=2.5)

harr(xc_pos[0]+BW+0.2, xc_pos[1], Y1+BH/2, c=COR)
harr(xc_pos[1]+BW, xc_pos[2], Y1+BH/2, c=COR)
harr(xc_pos[2]+BW+0.2, xc_pos[3], Y1+BH/2, c=COR)

# Stage B -> Stage C weight transfer
right_b = xb_pos[3] + (BW+0.5)/2
left_c  = xc_pos[0] + (BW+0.2)/2
curved(right_b, Y2, left_c, Y1+BH, c=PUR, lw=2, rad=0.35)
lbl((right_b+left_c)/2 + 2.0, (Y2+Y1+BH)/2, 'weight transfer', PUR)

# ══════════════════════════════════════
# [5] Bottom: Text Sources + Deploy
# ══════════════════════════════════════
# Text source box
ax.add_patch(FancyBboxPatch((1.6, Y0), 7.5, 0.85, boxstyle='round,pad=0.12',
             fc='#F9FAFB', ec=GR, lw=0.8, ls='--', zorder=2))
ax.text(5.35, Y0+0.55, 'Text Sources', fontsize=9, fontweight='bold', color=GR_D,
        ha='center', zorder=3)
ax.text(5.35, Y0+0.2, 'Venus guidance (5120)  |  Colloquial: "too dark"  |  Param: "exp: +0.3"',
        fontsize=8, color=GR, ha='center', zorder=3)

# Deploy
mod(10.5, Y0, 3.5, 0.85, 'Mobile Deploy', 'Stage C  ~1M params  |  real-time',
    ec=TEA, fc=TEA_L, lw=2.5, fs=10)

# Arrows to bottom
varr(xc_pos[1]+BW/2, Y1, 5.35, Y0+0.85, c=GR, lw=1)
output_cx = xc_pos[3] + (BW+0.5)/2
varr(output_cx, Y1, 12.25, Y0+0.85, c=TEA, lw=1.8)

plt.tight_layout(pad=0.3)
out = r'e:\智能相机\Venus_CVPR2026-main\IntelligenceCamera\docs\training_framework.png'
plt.savefig(out, dpi=250, bbox_inches='tight', facecolor='white', pad_inches=0.2)
print(f'Saved: {out}')
plt.show()
