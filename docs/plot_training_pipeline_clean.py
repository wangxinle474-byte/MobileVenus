from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Arial", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

IMG_DIR = r"e:\智能相机\效果对比"
IMG = {
    "raw_device": IMG_DIR + r"\原图\1.jpg",
    "raw_road": IMG_DIR + r"\原图\2.jpg",
    "raw_room": IMG_DIR + r"\原图\4.png",
    "enh_device": IMG_DIR + r"\生成图像\1.jpeg",
}

BLUE = "#3B82F6"
BLUE_L = "#DBEAFE"
BLUE_BG = "#EFF6FF"
GREEN = "#22A55E"
GREEN_L = "#DCFCE7"
GREEN_BG = "#F0FDF4"
RED = "#EF4444"
RED_L = "#FEE2E2"
RED_BG = "#FFF1F2"
AMBER = "#F59E0B"
AMBER_L = "#FEF3C7"
PURPLE = "#8B5CF6"
PURPLE_L = "#EDE9FE"
GRAY = "#6B7280"
DARK = "#111827"
LIGHT = "#F8FAFC"
C_GR_L = "#E5E7EB"
WHITE = "#FFFFFF"

W, H = 18, 10.4
fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor(WHITE)


def box(x, y, w, h, fc=WHITE, ec=GRAY, lw=1.2, rad=0.06, ls="-", z=2):
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


def text(x, y, s, fs=8, color=DARK, fw="normal", ha="center", va="center", **kwargs):
    ax.text(x, y, s, fontsize=fs, color=color, fontweight=fw, ha=ha, va=va, zorder=9, **kwargs)


def arrow(x1, y1, x2, y2, color=GRAY, lw=1.4, style="->", ls="-"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls),
        zorder=6,
    )


def img(x, y, w, h, key, ec=GRAY):
    path = IMG.get(key)
    if path and Path(path).exists():
        image = mpimg.imread(path)
        ax.imshow(image, extent=[x, x + w, y, y + h], aspect="auto", interpolation="bilinear", zorder=3)
        box(x, y, w, h, fc="none", ec=ec, lw=1.5, rad=0.015, z=7)
    else:
        box(x, y, w, h, fc=LIGHT, ec=ec, lw=1.2, rad=0.025, z=3)
        text(x + w / 2, y + h / 2, "image", fs=7, color=GRAY, style="italic")


def module(x, y, w, h, title, sub="", fc=WHITE, ec=BLUE, fs=8, lw=1.3):
    box(x, y, w, h, fc=fc, ec=ec, lw=lw, rad=0.045, z=4)
    text(x + w / 2, y + h * 0.62, title, fs=fs, color=ec, fw="bold")
    if sub:
        text(x + w / 2, y + h * 0.30, sub, fs=fs - 1.5, color=GRAY)


def badge(x, y, n, color):
    ax.add_patch(Circle((x, y), 0.22, facecolor=color, edgecolor=WHITE, linewidth=1.5, zorder=10))
    text(x, y, str(n), fs=10, color=WHITE, fw="bold")


def stage_card(x, y, w, h, n, title, color, bg):
    box(x, y, w, h, fc=bg, ec=color, lw=1.8, rad=0.12, z=1)
    badge(x + 0.35, y + h - 0.35, n, color)
    text(x + 0.70, y + h - 0.35, title, fs=10, color=color, fw="bold", ha="left")


text(W / 2, 10.05, "MobileVenus Three-Stage Training Pipeline", fs=16, color=DARK, fw="bold")
text(W / 2, 9.72, "Data construction → Stage A semantic alignment → Stage B ISP distillation → Stage C text-guided parameter prediction", fs=9, color=GRAY)

box(0.40, 8.35, 17.20, 1.05, fc="#FFFBEB", ec=AMBER, lw=1.2, rad=0.10, z=1)
text(0.70, 9.08, "Training Data & Supervision", fs=9.5, color=AMBER, fw="bold", ha="left")
img(0.85, 8.55, 0.75, 0.50, "raw_device", ec=AMBER)
module(1.90, 8.52, 1.55, 0.55, "FiveK", "5120 raw images", fc=WHITE, ec=AMBER, fs=7.5)
module(3.85, 8.52, 1.65, 0.55, "Venus Stage-1", "aesthetic text", fc=AMBER_L, ec=AMBER, fs=7.5)
module(5.90, 8.52, 1.65, 0.55, "Text Augment", "colloquial/param", fc=PURPLE_L, ec=PURPLE, fs=7.5)
module(7.95, 8.52, 1.65, 0.55, "Expert GT", "8 ISP params", fc=WHITE, ec=BLUE, fs=7.5)
module(10.00, 8.52, 1.65, 0.55, "Diff-ISP", "rendered image", fc=WHITE, ec=GREEN, fs=7.5)
module(12.05, 8.52, 1.75, 0.55, "AesExpert/IQA", "quality filter", fc=WHITE, ec=GREEN, fs=7.5)
module(14.25, 8.52, 1.90, 0.55, "Final Triples", "image, text, params", fc=AMBER_L, ec=AMBER, fs=7.5)
arrow(3.50, 8.80, 3.75, 8.80, AMBER)
arrow(5.55, 8.80, 5.80, 8.80, PURPLE)
arrow(7.60, 8.80, 7.85, 8.80, BLUE)
arrow(9.60, 8.80, 9.90, 8.80, GREEN)
arrow(11.70, 8.80, 11.95, 8.80, GREEN)
arrow(13.90, 8.80, 14.15, 8.80, AMBER)

xA, xB, xC = 0.55, 6.25, 11.95
yS, wS, hS = 3.15, 5.10, 4.55
stage_card(xA, yS, wS, hS, 1, "Stage A: Visual-Semantic Alignment", BLUE, BLUE_BG)
stage_card(xB, yS, wS, hS, 2, "Stage B: Visual → ISP Parameters", GREEN, GREEN_BG)
stage_card(xC, yS, wS, hS, 3, "Stage C: Text-Guided ISP", RED, RED_BG)

img(xA + 0.40, yS + 2.85, 0.95, 0.70, "raw_road", ec=AMBER)
module(xA + 1.75, yS + 3.00, 1.25, 0.62, "Visual Enc", "MobileViT-S", fc=BLUE_L, ec=BLUE, fs=7.4)
module(xA + 3.35, yS + 3.00, 1.25, 0.62, "Proj Head", "384→256", fc=WHITE, ec=BLUE, fs=7.4)
module(xA + 0.55, yS + 1.65, 1.25, 0.62, "Text Enc", "MiniLM frozen", fc=WHITE, ec=GRAY, fs=7.4)
module(xA + 2.15, yS + 1.65, 1.25, 0.62, "Text Emb", "z_txt", fc=WHITE, ec=GRAY, fs=7.4)
module(xA + 3.55, yS + 1.65, 1.20, 0.62, "Align Loss", "cos+MSE", fc=GREEN_L, ec=GREEN, fs=7.4)
module(xA + 1.70, yS + 0.45, 1.85, 0.70, "Output", "aligned visual encoder", fc=BLUE_L, ec=BLUE, fs=7.6, lw=1.7)
arrow(xA + 1.40, yS + 3.20, xA + 1.65, yS + 3.20, BLUE)
arrow(xA + 3.05, yS + 3.31, xA + 3.25, yS + 3.31, BLUE)
arrow(xA + 4.00, yS + 2.95, xA + 4.05, yS + 2.30, GREEN)
arrow(xA + 1.85, yS + 1.96, xA + 2.05, yS + 1.96, GRAY)
arrow(xA + 3.45, yS + 1.96, xA + 3.50, yS + 1.96, GREEN)
arrow(xA + 4.10, yS + 1.60, xA + 2.65, yS + 1.20, BLUE, ls="--")
text(xA + 2.60, yS + 2.55, "learn image-text semantic space", fs=7, color=BLUE, fw="bold")

module(xB + 0.35, yS + 3.00, 1.35, 0.62, "Frozen Enc", "Stage A", fc=BLUE_L, ec=BLUE, fs=7.4)
module(xB + 2.05, yS + 3.00, 1.25, 0.62, "Param MLP", "256→8", fc=WHITE, ec=GREEN, fs=7.4)
module(xB + 3.65, yS + 3.00, 1.10, 0.62, "θ_pred", "8 params", fc=C_YEL_L if False else AMBER_L, ec=AMBER, fs=7.4)
module(xB + 0.35, yS + 1.80, 1.30, 0.62, "Diff-ISP", "render", fc=WHITE, ec=AMBER, fs=7.4)
img(xB + 2.00, yS + 1.78, 0.95, 0.66, "enh_device", ec=GREEN)
module(xB + 3.30, yS + 1.80, 1.35, 0.62, "Image Loss", "L1+SSIM+MSE", fc=GREEN_L, ec=GREEN, fs=7.2)
module(xB + 1.55, yS + 0.45, 2.00, 0.70, "Output", "visual ISP predictor", fc=GREEN_L, ec=GREEN, fs=7.6, lw=1.7)
arrow(xB + 1.75, yS + 3.31, xB + 1.95, yS + 3.31, GREEN)
arrow(xB + 3.35, yS + 3.31, xB + 3.55, yS + 3.31, GREEN)
arrow(xB + 4.20, yS + 2.95, xB + 1.00, yS + 2.45, AMBER)
arrow(xB + 1.70, yS + 2.11, xB + 1.90, yS + 2.11, AMBER)
arrow(xB + 3.00, yS + 2.11, xB + 3.20, yS + 2.11, GREEN)
arrow(xB + 4.05, yS + 1.78, xB + 2.55, yS + 1.18, GREEN, ls="--")
text(xB + 2.55, yS + 2.55, "supervised by Expert GT + rendered image", fs=7, color=GREEN, fw="bold")

module(xC + 0.35, yS + 3.00, 1.35, 0.62, "Backbone", "Stage B init", fc=RED_L, ec=RED, fs=7.4)
module(xC + 0.35, yS + 1.85, 1.35, 0.62, "Text Enc", "MiniLM-L6", fc=WHITE, ec=GRAY, fs=7.4)
module(xC + 2.10, yS + 2.42, 1.35, 0.82, "FiLM", "γ(t)·v+β(t)", fc=RED_L, ec=RED, fs=8.0, lw=2.0)
module(xC + 3.80, yS + 2.52, 1.00, 0.62, "Decoder", "256→8", fc=WHITE, ec=RED, fs=7.4)
module(xC + 3.80, yS + 1.55, 1.00, 0.62, "Loss", "θ + image", fc=GREEN_L, ec=GREEN, fs=7.4)
module(xC + 1.55, yS + 0.45, 2.00, 0.70, "Output", "image+text → ISP params", fc=RED_L, ec=RED, fs=7.6, lw=1.7)
arrow(xC + 1.75, yS + 3.31, xC + 2.00, yS + 2.90, RED)
arrow(xC + 1.75, yS + 2.16, xC + 2.00, yS + 2.70, RED)
arrow(xC + 3.50, yS + 2.84, xC + 3.70, yS + 2.84, RED)
arrow(xC + 4.30, yS + 2.48, xC + 4.30, yS + 2.22, GREEN)
arrow(xC + 4.00, yS + 1.52, xC + 2.55, yS + 1.18, RED, ls="--")
text(xC + 2.60, yS + 3.75, "Venus + colloquial + param prompts", fs=7, color=RED, fw="bold")

arrow(xA + wS, yS + 2.25, xB, yS + 2.25, BLUE, lw=1.7)
text(5.95, yS + 2.55, "freeze/init", fs=7, color=BLUE, fw="bold")
arrow(xB + wS, yS + 2.25, xC, yS + 2.25, GREEN, lw=1.7)
text(11.65, yS + 2.55, "init backbone", fs=7, color=GREEN, fw="bold")

box(xA + 0.55, yS + hS - 0.78, 3.85, 0.34, fc=WHITE, ec=BLUE, lw=1.0, rad=0.035, z=7)
text(xA + 2.48, yS + hS - 0.61, "Inputs: FiveK image + Venus aesthetic text", fs=6.9, color=BLUE, fw="bold")
box(xB + 0.55, yS + hS - 0.78, 3.85, 0.34, fc=WHITE, ec=GREEN, lw=1.0, rad=0.035, z=7)
text(xB + 2.48, yS + hS - 0.61, "Inputs: Stage A encoder + Expert GT params", fs=6.9, color=GREEN, fw="bold")
box(xC + 0.55, yS + hS - 0.78, 3.85, 0.34, fc=WHITE, ec=RED, lw=1.0, rad=0.035, z=7)
text(xC + 2.48, yS + hS - 0.61, "Inputs: Stage B backbone + augmented text", fs=6.9, color=RED, fw="bold")

box(0.75, 1.25, 16.50, 0.92, fc=LIGHT, ec=C_GR_L, lw=1.1, rad=0.08, z=1)
text(2.65, 1.82, "Stage A", fs=8.3, color=BLUE, fw="bold")
text(2.65, 1.55, "semantic alignment", fs=7.4, color=GRAY)
text(6.50, 1.82, "Stage B", fs=8.3, color=GREEN, fw="bold")
text(6.50, 1.55, "visual → ISP distillation", fs=7.4, color=GRAY)
text(10.50, 1.82, "Stage C", fs=8.3, color=RED, fw="bold")
text(10.50, 1.55, "text-guided prediction", fs=7.4, color=GRAY)
text(14.70, 1.82, "Final model", fs=8.3, color=AMBER, fw="bold")
text(14.70, 1.55, "~1M params, 8 ISP outputs", fs=7.4, color=GRAY)

text(
    W / 2,
    0.45,
    "Figure. The proposed training pipeline uses Venus-generated aesthetic guidance and Expert GT to progressively train semantic alignment, visual ISP prediction, and direct natural-language control.",
    fs=7.5,
    color=GRAY,
)

out = r"e:\智能相机\Venus_CVPR2026-main\IntelligenceCamera\images\architecture\training_pipeline_three_stage.png"
plt.savefig(out, dpi=260, bbox_inches="tight", facecolor=WHITE, pad_inches=0.12)
print(f"Saved: {out}")
plt.show()
