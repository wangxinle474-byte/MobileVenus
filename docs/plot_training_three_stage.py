"""
Three-stage training pipeline figure for MobileVenus.
Output: images/architecture/training_pipeline_three_stage.png
"""
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Arial", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG_DIR = r"e:\智能相机\效果对比"
IMG = {
    "raw_device": IMG_DIR + r"\原图\1.jpg",
    "raw_road": IMG_DIR + r"\原图\2.jpg",
    "raw_room": IMG_DIR + r"\原图\4.png",
    "enh_device": IMG_DIR + r"\生成图像\1.jpeg",
}

C_BLUE = "#4A90D9"
C_BLUE_L = "#D6E6F5"
C_BLUE_BG = "#EBF3FB"
C_ORG = "#E8913A"
C_ORG_L = "#FDF0E0"
C_ORG_BG = "#FEF6ED"
C_GRN = "#5CB85C"
C_GRN_L = "#DFF0DF"
C_GRN_BG = "#EDF7ED"
C_RED = "#D9534F"
C_RED_L = "#F8E0DF"
C_PUR = "#8B5CF6"
C_PUR_L = "#EDE9FE"
C_YEL = "#F0AD4E"
C_YEL_L = "#FEF3D4"
C_GR = "#6B7280"
C_GR_D = "#1F2937"
C_GR_L = "#E5E7EB"
WH = "#FFFFFF"
BLK = "#000000"

W, H = 18, 10.6
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(WH)


def box(x, y, w, h, fc=WH, ec=C_BLUE, lw=1.4, ls="-", rad=0.06, z=2):
    p = FancyBboxPatch(
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
    ax.add_patch(p)
    return p


def text(x, y, s, fs=8, color=C_GR_D, ha="center", va="center", fw="normal", **kwargs):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va=va, fontweight=fw, zorder=8, **kwargs)


def arrow(x1, y1, x2, y2, color=C_GR, lw=1.5, style="->", ls="-"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls),
        zorder=5,
    )


def image(x, y, w, h, key, ec=C_ORG, label="Image"):
    path = IMG.get(key)
    if path and Path(path).exists():
        img = mpimg.imread(path)
        ax.imshow(img, extent=[x, x + w, y, y + h], aspect="auto", interpolation="bilinear", zorder=3)
        box(x, y, w, h, fc="none", ec=ec, lw=1.6, rad=0.015, z=6)
    else:
        box(x, y, w, h, fc="#F3F4F6", ec=ec, lw=1.2, rad=0.02)
        text(x + w / 2, y + h / 2, label, fs=7, color=C_GR, style="italic")


def module(x, y, w, h, title, subtitle="", fc=WH, ec=C_BLUE, fs=8, lw=1.4):
    box(x, y, w, h, fc=fc, ec=ec, lw=lw, rad=0.055)
    text(x + w / 2, y + h * 0.63, title, fs=fs, color=ec, fw="bold")
    if subtitle:
        text(x + w / 2, y + h * 0.31, subtitle, fs=fs - 1.4, color=C_GR)


def stage_badge(x, y, n, color):
    ax.add_patch(Circle((x, y), 0.23, facecolor=color, edgecolor=WH, linewidth=1.6, zorder=9))
    text(x, y, str(n), fs=10, color=WH, fw="bold")


def stage_title(x, y, n, title, color):
    stage_badge(x, y, n, color)
    text(x + 0.35, y, title, fs=10.5, color=color, ha="left", fw="bold")


# Title
text(W / 2, 10.15, "MobileVenus Training Pipeline", fs=16, color=BLK, fw="bold")
text(W / 2, 9.82, "Three-stage training: visual-semantic alignment → ISP parameter distillation → text-guided prediction", fs=9, color=C_GR)

# Data preparation band
box(0.35, 8.35, 17.30, 1.15, fc=C_ORG_BG, ec=C_ORG, lw=1.1, rad=0.10, z=0)
text(0.55, 9.20, "Data Preparation", fs=10, color=C_ORG, ha="left", fw="bold")
image(0.75, 8.55, 0.90, 0.62, "raw_device", ec=C_ORG)
text(2.25, 8.86, "FiveK Dataset\n5120 images + Expert GT", fs=7.5, color=C_GR_D)
module(3.45, 8.52, 1.70, 0.65, "Venus Stage-1", "Qwen-VL + SFT", fc=C_ORG_L, ec=C_ORG, fs=7.5)
module(5.75, 8.52, 1.70, 0.65, "Aesthetic Text", "5120 guidance", fc=WH, ec=C_ORG, fs=7.5)
module(8.05, 8.52, 1.70, 0.65, "Text Augment", "colloquial + param", fc=WH, ec=C_PUR, fs=7.5)
module(10.35, 8.52, 1.70, 0.65, "Expert GT", "8 ISP params", fc=WH, ec=C_BLUE, fs=7.5)
module(12.65, 8.52, 1.80, 0.65, "Quality Filter", "AesExpert + IQA", fc=WH, ec=C_GRN, fs=7.5)
module(15.05, 8.52, 1.80, 0.65, "Training Set", "image, text, params", fc=C_YEL_L, ec=C_YEL, fs=7.5)
arrow(1.75, 8.86, 3.25, 8.86, C_ORG)
arrow(5.20, 8.86, 5.55, 8.86, C_ORG)
arrow(7.50, 8.86, 7.85, 8.86, C_PUR)
arrow(9.80, 8.86, 10.15, 8.86, C_BLUE)
arrow(12.10, 8.86, 12.45, 8.86, C_GRN)
arrow(14.50, 8.86, 14.85, 8.86, C_YEL)

# Stage A
YA = 6.45
box(0.35, YA - 0.15, 17.30, 1.45, fc=C_BLUE_BG, ec=C_BLUE_L, lw=1.1, rad=0.10, z=0)
stage_title(0.70, YA + 1.03, 1, "Stage A: Visual-Semantic Alignment", C_BLUE)
image(0.80, YA + 0.12, 0.92, 0.70, "raw_road", ec=C_ORG)
module(2.15, YA + 0.10, 1.70, 0.72, "Visual Encoder", "MobileViT-S", fc=C_BLUE_L, ec=C_BLUE, fs=7.8, lw=1.7)
module(4.35, YA + 0.10, 1.70, 0.72, "Semantic Head", "384→256", fc=WH, ec=C_BLUE, fs=7.8)
module(6.55, YA + 0.10, 1.70, 0.72, "Image Embed", "z_img", fc=WH, ec=C_BLUE, fs=7.8)
module(9.05, YA + 0.10, 1.70, 0.72, "Text Encoder", "MiniLM-L6 frozen", fc=WH, ec=C_GR, fs=7.8)
module(11.25, YA + 0.10, 1.70, 0.72, "Text Embed", "z_txt", fc=WH, ec=C_GR, fs=7.8)
module(13.55, YA + 0.10, 1.90, 0.72, "Align Loss", "cosine + MSE", fc=C_GRN_L, ec=C_GRN, fs=7.8)
module(15.90, YA + 0.10, 1.30, 0.72, "Aligned\nEncoder", "output", fc=C_BLUE_L, ec=C_BLUE, fs=7.3, lw=1.8)
arrow(1.82, YA + 0.46, 2.05, YA + 0.46, C_BLUE)
arrow(3.95, YA + 0.46, 4.25, YA + 0.46, C_BLUE)
arrow(6.15, YA + 0.46, 6.45, YA + 0.46, C_BLUE)
arrow(10.85, YA + 0.46, 11.15, YA + 0.46, C_GR)
arrow(8.30, YA + 0.46, 8.90, YA + 0.46, C_ORG, ls="--")
arrow(8.25, YA + 0.46, 13.40, YA + 0.46, C_GRN, ls="--")
arrow(13.05, YA + 0.46, 13.45, YA + 0.46, C_GRN)
arrow(15.55, YA + 0.46, 15.78, YA + 0.46, C_BLUE)
text(8.60, YA + 0.88, "Venus text guidance", fs=6.8, color=C_ORG, fw="bold")

# Stage B
YB = 4.25
box(0.35, YB - 0.15, 17.30, 1.45, fc=C_GRN_BG, ec=C_GRN, lw=1.1, rad=0.10, z=0)
stage_title(0.70, YB + 1.03, 2, "Stage B: Visual-to-Parameter Distillation", C_GRN)
module(1.05, YB + 0.10, 1.90, 0.72, "Frozen Encoder", "Stage A weights", fc=C_BLUE_L, ec=C_BLUE, fs=7.8)
module(3.55, YB + 0.10, 1.80, 0.72, "Param Decoder", "MLP 256→8", fc=WH, ec=C_GRN, fs=7.8)
module(5.95, YB + 0.10, 1.70, 0.72, "8 ISP Params", "θ_pred", fc=C_YEL_L, ec=C_YEL, fs=7.8)
module(8.25, YB + 0.10, 1.70, 0.72, "Diff-ISP", "render image", fc=WH, ec=C_YEL, fs=7.8)
image(10.55, YB + 0.08, 0.92, 0.72, "enh_device", ec=C_GRN)
module(12.10, YB + 0.10, 1.80, 0.72, "Image Loss", "L1+SSIM+MSE", fc=C_GRN_L, ec=C_GRN, fs=7.8)
module(14.70, YB + 0.10, 2.00, 0.72, "ISP Predictor", "image → params", fc=C_GRN_L, ec=C_GRN, fs=7.8, lw=1.9)
arrow(3.05, YB + 0.46, 3.45, YB + 0.46, C_GRN)
arrow(5.45, YB + 0.46, 5.85, YB + 0.46, C_GRN)
arrow(7.75, YB + 0.46, 8.15, YB + 0.46, C_YEL)
arrow(10.05, YB + 0.46, 10.45, YB + 0.46, C_YEL)
arrow(11.60, YB + 0.46, 12.00, YB + 0.46, C_GRN)
arrow(14.00, YB + 0.46, 14.55, YB + 0.46, C_GRN)
text(9.10, YB + 1.05, "Expert GT supervises parameters and rendered image", fs=7.1, color=C_GRN, fw="bold")

# Stage C
YC = 2.05
box(0.35, YC - 0.15, 17.30, 1.45, fc="#FFF5F5", ec=C_RED, lw=1.8, rad=0.10, z=0)
stage_title(0.70, YC + 1.03, 3, "Stage C: Text-Guided Parameter Prediction", C_RED)
module(1.05, YC + 0.10, 1.80, 0.72, "Stage B\nBackbone", "init weights", fc=C_RED_L, ec=C_RED, fs=7.3)
module(3.45, YC + 0.10, 1.70, 0.72, "Text Encoder", "MiniLM-L6", fc=WH, ec=C_GR, fs=7.8)
module(5.75, YC + 0.10, 1.80, 0.72, "FiLM Fusion", "γ(t)·v+β(t)", fc=C_RED_L, ec=C_RED, fs=8.0, lw=2.1)
module(8.15, YC + 0.10, 1.80, 0.72, "Fused Decoder", "256→8", fc=WH, ec=C_RED, fs=7.8)
module(10.55, YC + 0.10, 1.70, 0.72, "Text Loss", "params + image", fc=C_GRN_L, ec=C_GRN, fs=7.8)
module(12.85, YC + 0.10, 1.90, 0.72, "Text-Guided ISP", "image + text → θ", fc=C_RED_L, ec=C_RED, fs=7.8, lw=1.9)
module(15.35, YC + 0.10, 1.70, 0.72, "Mobile\nDeploy", "~1M params", fc=C_YEL_L, ec=C_YEL, fs=7.3, lw=1.8)
arrow(2.95, YC + 0.46, 3.35, YC + 0.46, C_RED)
arrow(5.25, YC + 0.46, 5.65, YC + 0.46, C_RED)
arrow(7.65, YC + 0.46, 8.05, YC + 0.46, C_RED)
arrow(10.05, YC + 0.46, 10.45, YC + 0.46, C_GRN)
arrow(12.35, YC + 0.46, 12.75, YC + 0.46, C_RED)
arrow(14.85, YC + 0.46, 15.25, YC + 0.46, C_YEL)
text(10.20, YC + 1.05, "Augmented texts: Venus guidance + colloquial instructions + parameter-oriented prompts", fs=7.1, color=C_RED, fw="bold")

# Bottom summary
box(0.55, 0.55, 4.65, 0.78, fc=WH, ec=C_BLUE, lw=1.1, rad=0.06)
text(2.875, 1.05, "Stage A Objective", fs=8, color=C_BLUE, fw="bold")
text(2.875, 0.78, "align image and language semantics", fs=7.2, color=C_GR_D)
box(5.55, 0.55, 4.65, 0.78, fc=WH, ec=C_GRN, lw=1.1, rad=0.06)
text(7.875, 1.05, "Stage B Objective", fs=8, color=C_GRN, fw="bold")
text(7.875, 0.78, "distill visual encoder into ISP params", fs=7.2, color=C_GR_D)
box(10.55, 0.55, 4.65, 0.78, fc=WH, ec=C_RED, lw=1.1, rad=0.06)
text(12.875, 1.05, "Stage C Objective", fs=8, color=C_RED, fw="bold")
text(12.875, 0.78, "control ISP prediction with text", fs=7.2, color=C_GR_D)
box(15.55, 0.55, 1.85, 0.78, fc=WH, ec=C_YEL, lw=1.1, rad=0.06)
text(16.475, 1.05, "Output", fs=8, color=C_YEL, fw="bold")
text(16.475, 0.78, "8 params", fs=7.2, color=C_GR_D)

text(
    W / 2,
    0.17,
    "Figure. Three-stage MobileVenus training pipeline. Stage A learns image-text alignment, Stage B learns visual-to-ISP parameters, and Stage C injects natural-language control through FiLM fusion.",
    fs=7.2,
    color=C_GR,
    va="bottom",
)

out = r"e:\智能相机\Venus_CVPR2026-main\IntelligenceCamera\images\architecture\training_pipeline_three_stage.png"
plt.savefig(out, dpi=260, bbox_inches="tight", facecolor=WH, pad_inches=0.12)
print(f"Saved: {out}")
plt.show()
