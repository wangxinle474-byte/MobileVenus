"""
Figure 1 Teaser — compact paper-style layout.
Left: (a)(b)(c) method comparison and our architecture.
Right: (d) qualitative examples using local images.
"""
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Arial", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG_DIR = r"e:\智能相机\效果对比"
IMG = {
    "raw_device": IMG_DIR + r"\原图\1.jpg",
    "enh_device": IMG_DIR + r"\生成图像\1.jpeg",
    "raw_road": IMG_DIR + r"\原图\2.jpg",
    "enh_road": IMG_DIR + r"\生成图像\2.1.jpeg",
    "raw_night": IMG_DIR + r"\原图\3.jpg",
    "enh_night": IMG_DIR + r"\生成图像\3.1.jpeg",
    "raw_room": IMG_DIR + r"\原图\4.png",
}

C_BLUE = "#4A90D9"
C_BLUE_L = "#D6E6F5"
C_BLUE_BG = "#EBF3FB"
C_ORG = "#E8913A"
C_ORG_L = "#FDF0E0"
C_GRN = "#5CB85C"
C_GRN_L = "#DFF0DF"
C_RED = "#D9534F"
C_RED_L = "#F8E0DF"
C_YEL = "#F0AD4E"
C_YEL_L = "#FEF3D4"
C_PUR = "#8B5CF6"
C_GR = "#6B7280"
C_GR_D = "#1F2937"
C_GR_L = "#E5E7EB"
WH = "#FFFFFF"
BLK = "#000000"

W, H = 18, 10
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(WH)


def box(x, y, w, h, fc=WH, ec=C_BLUE, lw=1.4, ls="-", rad=0.06, z=2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={rad}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def text(x, y, s, fs=8, color=C_GR_D, ha="center", va="center", fw="normal", **kwargs):
    ax.text(
        x,
        y,
        s,
        fontsize=fs,
        color=color,
        ha=ha,
        va=va,
        fontweight=fw,
        zorder=8,
        **kwargs,
    )


def arrow(x1, y1, x2, y2, color=C_GR, lw=1.5, style="->"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw),
        zorder=5,
    )


def image(x, y, w, h, key, ec=C_GR, label="Image"):
    path = IMG.get(key)
    if path and Path(path).exists():
        img = mpimg.imread(path)
        ax.imshow(img, extent=[x, x + w, y, y + h], aspect="auto", interpolation="bilinear", zorder=3)
        box(x, y, w, h, fc="none", ec=ec, lw=1.8, rad=0.015, z=6)
    else:
        box(x, y, w, h, fc="#F3F4F6", ec=ec, lw=1.2, rad=0.02)
        ax.plot([x, x + w], [y, y + h], color=ec, lw=0.5, alpha=0.35, zorder=4)
        ax.plot([x + w, x], [y, y + h], color=ec, lw=0.5, alpha=0.35, zorder=4)
        text(x + w / 2, y + h / 2, label, fs=7, color=C_GR, style="italic")


def module(x, y, w, h, title, subtitle="", fc=WH, ec=C_BLUE, fs=8, lw=1.4):
    box(x, y, w, h, fc=fc, ec=ec, lw=lw, rad=0.055)
    if "\n" in title:
        text(x + w / 2, y + h * 0.58, title, fs=fs, color=ec, fw="bold")
    else:
        text(x + w / 2, y + h * 0.64, title, fs=fs, color=ec, fw="bold")
    if subtitle:
        text(x + w / 2, y + h * 0.30, subtitle, fs=fs - 1.5, color=C_GR)


def panel_label(x, y, label, title, color=C_GR_D):
    text(x, y, f"{label} {title}", fs=10.5, color=color, fw="bold", ha="left")


text(W / 2, 9.65, "MobileVenus: Semantic-to-Parameter ISP Control", fs=16, color=BLK, fw="bold")
text(W / 2, 9.35, "From fixed enhancement to language-guided ISP parameter prediction", fs=9, color=C_GR)

# Left top: compact comparison
box(0.35, 6.42, 8.35, 2.62, fc="#FAFAFA", ec=C_GR_L, lw=1.1, rad=0.10, z=0)
box(0.60, 6.74, 3.82, 1.90, fc=WH, ec=C_GR_L, lw=1.0, rad=0.08, z=1)
box(4.62, 6.74, 3.82, 1.90, fc=C_BLUE_BG, ec=C_BLUE_L, lw=1.0, rad=0.08, z=1)
panel_label(0.55, 8.80, "(a)", "Manual ISP", C_GR_D)
panel_label(4.82, 8.80, "(b)", "Learning-based", C_BLUE)

image(0.83, 7.34, 0.82, 0.76, "raw_road", ec=C_ORG)
arrow(1.72, 7.72, 2.03, 7.72, C_GR)
module(2.08, 7.31, 1.05, 0.82, "Manual", "tuning", fc=C_GR_L, ec=C_GR, fs=7.6)
arrow(3.18, 7.72, 3.48, 7.72, C_GR)
image(3.55, 7.34, 0.82, 0.76, "enh_road", ec=C_GRN)
text(2.60, 7.02, "slow / expert-dependent", fs=7.0, color=C_RED, fw="bold")

image(4.90, 7.34, 0.82, 0.76, "raw_device", ec=C_ORG)
arrow(5.80, 7.72, 6.10, 7.72, C_BLUE)
module(6.15, 7.31, 1.00, 0.82, "CNN", "visual", fc=C_BLUE_L, ec=C_BLUE, fs=7.6)
arrow(7.22, 7.72, 7.52, 7.72, C_BLUE)
image(7.58, 7.34, 0.82, 0.76, "enh_device", ec=C_GRN)
text(6.68, 7.02, "fixed style / no text control", fs=7.0, color=C_ORG, fw="bold")

# Left bottom: ours architecture
box(0.35, 1.75, 8.35, 4.35, fc="#FFF5F5", ec=C_RED, lw=2.0, rad=0.12, z=0)
panel_label(0.55, 5.82, "(c)", "Semantic-to-Parameter (Ours)", C_RED)
text(0.70, 5.50, "RGB & Text", fs=7.5, color=C_GR_D, fw="bold", ha="left")

image(0.75, 4.25, 1.15, 0.92, "raw_room", ec=C_ORG)
box(0.75, 3.10, 1.15, 0.78, fc=C_ORG_L, ec=C_ORG, lw=1.4, rad=0.05)
text(1.325, 3.55, "\"make it\nbrighter\"", fs=7.3, color=C_ORG, fw="bold")
arrow(2.00, 4.72, 2.50, 4.72, C_BLUE)
arrow(2.00, 3.50, 2.50, 3.85, C_ORG)

box(2.45, 3.05, 2.10, 2.25, fc=C_BLUE_BG, ec=C_BLUE, lw=1.2, rad=0.08, z=1)
text(3.50, 5.10, "Feature Extraction", fs=7.4, color=C_BLUE, fw="bold")
module(2.70, 4.30, 1.60, 0.65, "Visual Encoder", "MobileViT-S", fc=C_BLUE_L, ec=C_BLUE, fs=7.5, lw=1.7)
module(2.70, 3.35, 1.60, 0.65, "Text Encoder", "MiniLM-L6", fc=WH, ec=C_GR, fs=7.5, lw=1.4)

arrow(4.62, 4.58, 5.05, 4.58, C_BLUE)
arrow(4.62, 3.68, 5.05, 3.88, C_ORG)
module(5.10, 3.35, 1.55, 1.60, "FiLM Fusion", "γ(t)·v + β(t)", fc=C_RED_L, ec=C_RED, fs=8.8, lw=2.2)
box(5.82, 3.45, 0.60, 0.28, fc=C_RED, ec=C_RED, lw=1, rad=0.035, z=5)
text(6.12, 3.59, "~1M", fs=7, color=WH, fw="bold")

arrow(6.72, 4.15, 7.10, 4.15, C_RED)
module(7.15, 3.58, 1.22, 1.12, "8 ISP\nParams", "EV/WB/HDR...", fc=C_YEL_L, ec=C_YEL, fs=7.8, lw=1.7)

text(1.35, 2.55, "Text-guided", fs=7.5, color=C_RED, fw="bold")
text(3.45, 2.55, "Lightweight", fs=7.5, color=C_PUR, fw="bold")
text(5.85, 2.55, "Real-time", fs=7.5, color=C_GRN, fw="bold")
text(1.35, 2.28, "natural language control", fs=6.5, color=C_GR)
text(3.45, 2.28, "~1M trainable params", fs=6.5, color=C_GR)
text(5.85, 2.28, "~15 ms on device", fs=6.5, color=C_GR)

# Right: qualitative examples
box(9.05, 1.75, 8.60, 7.30, fc="#FAFAFA", ec=C_GR_D, lw=1.3, ls="--", rad=0.12, z=0)
panel_label(9.25, 8.78, "(d)", "Qualitative Examples", C_GR_D)
text(10.20, 8.35, "Input", fs=8.5, color=C_GR_D, fw="bold")
text(12.65, 8.35, "Text Guidance", fs=8.5, color=C_ORG, fw="bold")
text(15.60, 8.35, "Enhanced", fs=8.5, color=C_GRN, fw="bold")

rows = [
    ("raw_device", "enh_device", "\"remove clutter,\nmake object white\""),
    ("raw_road", "enh_road", "\"brighten night scene,\nrecover road details\""),
    ("raw_night", "enh_night", "\"increase exposure,\nreduce deep shadows\""),
]
ys = [6.70, 4.55, 2.40]
for i, ((raw, enh, prompt), y) in enumerate(zip(rows, ys), start=1):
    text(
        9.35,
        y + 0.65,
        f"{i}",
        fs=8,
        color=WH,
        fw="bold",
        bbox=dict(boxstyle="circle,pad=0.25", fc=C_RED, ec=WH, lw=1),
    )
    image(9.75, y, 1.75, 1.25, raw, ec=C_ORG)
    arrow(11.62, y + 0.62, 12.00, y + 0.62, C_GR)
    box(12.08, y + 0.08, 1.55, 1.10, fc=C_ORG_L, ec=C_ORG, lw=1.3, rad=0.05)
    text(12.855, y + 0.63, prompt, fs=6.6, color=C_GR_D)
    arrow(13.75, y + 0.62, 14.18, y + 0.62, C_GRN)
    image(14.30, y, 2.00, 1.25, enh, ec=C_GRN)

# Bottom summary
box(0.35, 0.50, 5.35, 0.78, fc=WH, ec=C_BLUE, lw=1.1, rad=0.06)
text(3.025, 1.00, "Knowledge Distillation", fs=8, color=C_BLUE, fw="bold")
text(3.025, 0.75, "Venus (7B) → MobileVenus (~1M)", fs=7.2, color=C_GR_D)

box(6.15, 0.50, 5.35, 0.78, fc=WH, ec=C_GRN, lw=1.1, rad=0.06)
text(8.825, 1.00, "Latency Budget", fs=8, color=C_GRN, fw="bold")
text(8.825, 0.75, "Stage C predictor ~15 ms  |  on-device", fs=7.2, color=C_GR_D)

box(11.95, 0.50, 5.35, 0.78, fc=WH, ec=C_ORG, lw=1.1, rad=0.06)
text(14.625, 1.00, "Output", fs=8, color=C_ORG, fw="bold")
text(14.625, 0.75, "8 ISP params  |  text-guided camera control", fs=7.2, color=C_GR_D)

text(
    W / 2,
    0.16,
    "Figure 1. Existing methods either require manual tuning or predict a fixed enhancement style. "
    "MobileVenus maps image and language semantics to lightweight ISP parameters.",
    fs=7.2,
    color=C_GR,
    va="bottom",
)

out = r"e:\智能相机\Venus_CVPR2026-main\IntelligenceCamera\images\architecture\figure1_teaser.png"
plt.savefig(out, dpi=260, bbox_inches="tight", facecolor=WH, pad_inches=0.12)
print(f"Saved: {out}")
plt.show()
