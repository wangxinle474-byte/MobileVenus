# MobileVenus Overview 图生成提示词 (5参数精简版)

> **当前版本**: v3 — 5 参数精简版核心方法图
> 
> v1（单面板）和 v2（双面板）已归档至 `docs/history.md`。
> 本文件仅保留 v3（5参数版），完全围绕 **EV / 白平衡 / 对焦点 / HDR / 拍摄模式** 设计。
> 
> 适合作为论文方法图 (Figure 3) 或 Supplementary 中的详细方法说明。

---

## 📝 提示词（中文版）

```
创建一张高质量的学术论文方法图，展示 MobileVenus 的 5 参数预测核心架构。
本图聚焦于 "从视觉特征到 5 个相机参数" 的完整预测管线。

=====================================
【整体布局】三段式，自上而下
=====================================

宽度 3600px，高度 2400px，白色背景。
分为三大横向条带：
  ① 顶部 (20%): 输入与视觉编码
  ② 中部 (55%): 核心预测模块 (⭐ 重点)
  ③ 底部 (25%): 输出与应用

=====================================
【① 顶部：输入 → 视觉特征】
=====================================

左侧：一张手机拍摄的低质量照片示例（偏暗、色偏）
右侧流程：

  Image I ∈ ℝ^{3×224×224}
      ↓
  ┌──────────────────────────────────────┐
  │  MobileViT-Small Vision Encoder     │  浅蓝色 #E3F2FD
  │  + SE Channel Attention             │
  │  + FPN Multi-Scale Fusion           │
  │  → Global Pool → X_v ∈ ℝ^{384}     │
  │  (~5.6M params)                     │
  └──────────────┬───────────────────────┘
                 ↓
  ┌──────────────────────────────────────┐
  │  Vision Projection                   │  浅蓝色
  │  Linear(384→512) + GELU + LN        │
  │  → X_v ∈ ℝ^{B×512}                 │
  └──────────────┬───────────────────────┘
                 ↓
           ┌─────┴─────┐
           ↓           ↓
     Aesthetic      Problem
     Scorer         Classifier
     (→ S)          (→ P, H)

=====================================
【② 中部：5 参数预测核心 (⭐ 本图重点)】
=====================================

用一个大的浅橙色 (#FFF3E0) 圆角矩形包裹整个中部区域，
标题："Semantic-to-Parameter Translation (5 Parameters)"

内部分为三个子模块，纵向排列：

━━━ 子模块 A：Problem Classifier（浅绿色 #E8F5E9）━━━

  输入: X_v ∈ ℝ^{B×512} + S ∈ ℝ^{B×5}
  
  [X_v ‖ S] → MLP(517→256) → LayerNorm → GELU
            → MLP(256→256) → LayerNorm → GELU
            ├─ Problem Head → σ → P ∈ ℝ^{B×9}（9 类问题概率）
            └─ Severity Heads ×9 → Softmax → Ŝ ∈ ℝ^{B×9×3}

  右侧用 9 个小图标/色块表示 9 类问题：
    曝光不足 | 曝光过度 | 色偏 | 构图差 | 模糊
    噪声 | 逆光 | 低对比度 | 背景杂乱

  输出: P (问题概率), H ∈ ℝ^{B×256} (隐藏特征)

━━━ 子模块 B：Cross-Attention Parameter Mapping（浅橙色 #FFF3E0，⭐核心创新）━━━

  【最重要的部分，用更大的面积和更醒目的边框】

  顶部：5 个可学习参数 Token，用 5 个带标签的彩色圆圈表示：
    ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
    │ T_EV│  │T_WB │  │T_Foc│  │T_HDR│  │T_Mod│
    │  ●  │  │  ●  │  │  ●  │  │  ●  │  │  ●  │
    └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
       └────────┴────────┴────┬───┴────────┘
                              ↓
    T ∈ ℝ^{1×5×256}   ← 标注 "⭐ Learnable Parameter Tokens"

  中间：2 层 Cross-Attention Block
    ┌─────────────────────────────────────────────┐
    │  Cross-Attention Block ×2                    │
    │                                              │
    │  Q = T (5 param tokens)                      │
    │  K = V = [X_v', H, S']                       │
    │                                              │
    │  MultiHead-Attention(Q, K, V)                │
    │      ↓ + Residual → LayerNorm                │
    │  FFN (D → 2D → D)                            │
    │      ↓ + Residual → LayerNorm                │
    │                                              │
    │  输出: T' ∈ ℝ^{B×5×256}                     │
    └─────────────────────────────────────────────┘

  底部：5 个独立解码头，用 5 列并排显示，每列一个参数：

    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  EV 解码 │ │  WB 解码 │ │ Focus解码│ │ HDR 解码 │ │ Mode解码 │
    │          │ │          │ │          │ │          │ │          │
    │ T'_0     │ │ T'_1     │ │ T'_2     │ │ T'_3     │ │ T'_4     │
    │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │
    │ MLP      │ │ MLP      │ │ MLP      │ │ MLP      │ │ MLP      │
    │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │
    │ tanh     │ │ sigmoid  │ │ sigmoid  │ │ softmax  │ │ softmax  │
    │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │
    │ EV       │ │ WB (K)   │ │ (x, y)   │ │ {off,on} │ │ {5 类}  │
    │[-3, +3]  │ │[2K, 10K] │ │[0,1]²    │ │          │ │          │
    └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
         ↑ 连续值       ↑ 连续值     ↑ 连续值     ↑ 分类        ↑ 分类

    颜色编码：连续值参数用蓝色调，分类参数用橙色调

  右侧分支：置信度估计
    [T'_0...T'_4] → Flatten → MLP → σ → Confidence ∈ ℝ^{B×5}
    用 5 段进度条可视化置信度

━━━ 子模块 C：Validation & Smoothing（浅灰色 #F5F5F5）━━━

  Raw Params → Range Clipping → Conflict Check → EMA Smoothing → Final

  安全范围表：
    EV: [-3, +3]  |  WB: [2K, 10K]  |  Focus: [0,1]²

=====================================
【③ 底部：输出与应用】
=====================================

左侧：最终参数输出卡片
  ┌────────────────────────┐
  │  📷 Camera Parameters  │
  │                        │
  │  EV:    +1.5    ■■■■□  │  (置信度进度条)
  │  WB:    5500K   ■■■■■  │
  │  Focus: (0.6,   ■■■□□  │
  │          0.4)          │
  │  HDR:   ON      ■■■■□  │
  │  Mode:  Night   ■■■□□  │
  └────────────────────────┘

中间：Before → After 对比图
  左图 (暗、色偏) → 右图 (明亮、色彩正确)
  评分: 5.8 → 8.5

右侧：应用平台
  iOS (AVFoundation) / Android (Camera2)
  实时自动应用参数

=====================================
【底部标题】
=====================================

Figure X. **The proposed 5-Parameter Prediction Pipeline of MobileVenus.**
The system extracts visual features via MobileViT-Small, detects 9 aesthetic 
problems via Problem Classifier, and predicts 5 camera parameters (exposure 
compensation, white balance, focus point, HDR, shooting mode) through a 
Cross-Attention mechanism with learnable parameter tokens. Each token is decoded
by a parameter-specific head, producing both the adjustment value and a 
per-parameter confidence score.

=====================================
【视觉风格规范】
=====================================

配色：
  - 浅绿 #E8F5E9 / 深绿 #4CAF50：Problem Classifier
  - 浅橙 #FFF3E0 / 深橙 #FF9800：Parameter Mapping (创新核心)
  - 浅蓝 #E3F2FD / 深蓝 #2196F3：Vision Encoder
  - 浅灰 #F5F5F5 / 灰色 #9E9E9E：Validation / I/O
  - 参数 Token 圆圈：5 种不同颜色
    T_EV=#FF5722, T_WB=#2196F3, T_Focus=#4CAF50, T_HDR=#FF9800, T_Mode=#9C27B0

箭头：
  - 实线黑色 1.5pt：主数据流
  - 虚线灰色 1pt：辅助/跨模块引用

字体：
  - 标题：Times New Roman Bold, 14pt
  - 模块名：Times New Roman, 12pt
  - 数学符号：Times New Roman Italic, 11pt
  - 维度标注：Courier, 9pt

输出：
  - PNG 3600×2400 像素
  - 300 DPI
  - 白色背景
```

---

## 📝 提示词（英文版 — 推荐用于 AI 绘图工具）

```
Create a high-quality academic method figure for MobileVenus, showing the 
complete 5-Parameter Prediction Pipeline. This figure focuses on predicting
exactly 5 camera parameters from visual features.

=== LAYOUT: Three Horizontal Bands (Top-to-Bottom) ===

Canvas: 3600×2400px, white background, 300 DPI.

Band ① Top (20%): Input & Visual Encoding
Band ② Middle (55%): Core 5-Parameter Prediction (⭐ KEY FOCUS)
Band ③ Bottom (25%): Output & Application

=== BAND ① TOP: Input → Visual Features ===

Left: Sample low-quality photo (dark, color cast)
Right pipeline:

  Image I ∈ ℝ^{3×224×224}
    ↓
  MobileViT-Small Encoder [light blue #E3F2FD]
    + SE Attention + FPN Fusion
    → Global Pool → 384-dim (~5.6M params)
    ↓
  Vision Projection [light blue]
    Linear(384→512) + GELU + LayerNorm → X_v ∈ ℝ^{B×512}
    ↓
  Split → Aesthetic Scorer (→ S ∈ ℝ^{B×5})
        → Problem Classifier (→ P, H)

=== BAND ② MIDDLE: 5-Parameter Prediction Core (⭐) ===

Wrap entire band in a large rounded rectangle with light orange border.
Title: "Semantic-to-Parameter Translation (5 Parameters)"

--- Sub-module A: Problem Classifier (light green #E8F5E9) ---
  [X_v ‖ S] → MLP(517→256) → LN → GELU → MLP(256→256) → LN → GELU
    ├─ Problem Head → σ → P ∈ ℝ^{B×9} (9 problem types)
    └─ Severity Heads ×9 → Softmax → Ŝ ∈ ℝ^{B×9×3}
  Show 9 problem icons: Under/Over/Color/Comp/Blur/Noise/Back/Contr/Messy
  Output: P, H ∈ ℝ^{B×256}

--- Sub-module B: Cross-Attention Parameter Mapping (light orange #FFF3E0, ⭐CORE) ---

  Top: 5 Learnable Parameter Tokens shown as 5 colored circles:
    [T_EV ● red] [T_WB ● blue] [T_Focus ● green] [T_HDR ● orange] [T_Mode ● purple]
    → T ∈ ℝ^{1×5×256}
    Label: "⭐ Learnable Parameter Tokens"

  Middle: Cross-Attention Block ×2
    Q = T (5 param tokens)
    K = V = [X_v', H, S']
    → MultiHead Attention → +Residual → LayerNorm
    → FFN(D→2D→D) → +Residual → LayerNorm
    Output: T' ∈ ℝ^{B×5×256}

  Bottom: 5 PARALLEL decode columns (key visual element):
    ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
    │  EV  │ │  WB  │ │Focus │ │ HDR  │ │ Mode │
    │T'_0  │ │T'_1  │ │T'_2  │ │T'_3  │ │T'_4  │
    │ MLP  │ │ MLP  │ │ MLP  │ │ MLP  │ │ MLP  │
    │ tanh │ │  σ   │ │  σ   │ │softmx│ │softmx│
    │[-3,3]│ │[2K,  │ │[0,1]²│ │{0,1} │ │{5cls}│
    │      │ │ 10K] │ │      │ │      │ │      │
    └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
    Blue tones for continuous params, orange tones for classification params.

  Side branch: Confidence Estimation
    [T'_0..T'_4] → Flatten → MLP → σ → Confidence ∈ ℝ^{B×5}
    Visualize as 5 progress bars.

--- Sub-module C: Validation & Smoothing (light gray #F5F5F5) ---
  Raw → Range Clip → Conflict Check → EMA Smooth → Final
  Range table: EV:[-3,+3] | WB:[2K,10K] | Focus:[0,1]²

=== BAND ③ BOTTOM: Output & Application ===

Left: Final parameter card with confidence bars:
  EV: +1.5 ████░ (87%)
  WB: 5500K █████ (95%)
  Focus: (0.6, 0.4) ███░░ (72%)
  HDR: ON ████░ (85%)
  Mode: Night ███░░ (68%)

Center: Before → After comparison photos
  Score: 5.8 → 8.5/10

Right: Platform icons (iOS / Android)
  Real-time automatic parameter application

=== BOTTOM CAPTION ===

Figure X. **The proposed 5-Parameter Prediction Pipeline of MobileVenus.**
The system extracts visual features via MobileViT-Small, detects 9 types of 
aesthetic problems via Problem Classifier, and predicts 5 essential camera 
parameters (exposure compensation, white balance, focus point, HDR, shooting 
mode) through a Cross-Attention mechanism with 5 learnable parameter tokens. 
Each token is decoded by a parameter-specific head, producing both the 
adjustment value and a per-parameter confidence score. This minimal yet 
effective 5-parameter set covers the most impactful camera adjustments for 
mobile photography.

=== VISUAL STYLE ===

Colors:
  Light green  #E8F5E9 / border #4CAF50 → Problem Classifier
  Light orange #FFF3E0 / border #FF9800 → Parameter Mapping (core)
  Light blue   #E3F2FD / border #2196F3 → Vision Encoder
  Light gray   #F5F5F5 / border #9E9E9E → Validation / I/O
  Token colors: T_EV=#FF5722, T_WB=#2196F3, T_Focus=#4CAF50, T_HDR=#FF9800, T_Mode=#9C27B0

Arrows: Solid black 1.5pt (main), dashed gray 1pt (auxiliary)
Math: Italic variables, upright operators
Fonts: Times New Roman all, Courier for dimensions
Output: PNG 3600×2400px, 300 DPI, white background
```

---

## 🎨 推荐绘图工具

1. **Figma / FigJam** — 专业矢量设计，推荐
2. **Draw.io / diagrams.net** — 免费在线，适合架构图
3. **PowerPoint / Keynote** — 最易上手，适合快速原型
4. **DALL-E 3 / Midjourney** — AI 生成初稿，需手动调整

---

## 💡 配色参考

| 模块 | 背景色 | 边框色 |
|------|--------|--------|
| Problem Classifier | #E8F5E9 | #4CAF50 |
| Parameter Mapping | #FFF3E0 | #FF9800 |
| Vision Encoder | #E3F2FD | #2196F3 |
| Validation / I/O | #F5F5F5 | #9E9E9E |

**Token 颜色**: T_EV=#FF5722, T_WB=#2196F3, T_Focus=#4CAF50, T_HDR=#FF9800, T_Mode=#9C27B0

---

**更新日期**: 2026-03-20
**版本**: v3（5参数精简版）
**参考**: GM-MOE (CVPR) Figure 3 布局风格

---

# Figure 1: Overview 总览图提示词 (v4)

> **用途**: 论文 Figure 1，一眼看懂系统全貌，强调动机和整体流程
> **生成工具**: 推荐 PPT/Figma 手工绘制，或 ChatGPT 生成 SVG 代码（不要用 DALL-E）

## 📝 提示词（英文版）

```
Create a professional academic overview figure for a CVPR/ICCV-level computer vision paper titled "MobileVenus: Real-time Aesthetic Guidance for Mobile Photography".

=== CONCEPT ===

This is Figure 1 (Overview). It should tell the story in ONE glance:
"A phone takes a bad photo → our lightweight model analyzes it in real-time → outputs 5 camera parameter adjustments → the photo becomes much better."

=== LAYOUT: Left-to-Right Flow, Two Rows ===

Canvas: 3600×1600px, white background, 300 DPI, clean academic style.
Use a clean, minimalist design similar to top-tier CV papers (DETR, SAM, LLaVA style).

--- ROW 1 (Top): The Problem & Motivation ---

Left panel: "Existing MLLMs" (e.g., GPT-4V, LLaVA)
  - Show a camera icon → image → speech bubble saying vague praise:
    "The photo has nice colors..." (❌ No actionable guidance)
  - Label: "Complimentary but NOT actionable"
  - Gray/muted color tone

Right panel: "MobileVenus (Ours)"
  - Same camera icon → image → concrete output card:
    "EV: +1.5 | WB: 5500K | Focus: (0.6, 0.4) | HDR: ON | Mode: Night"
  - Label: "Precise & Executable Camera Parameters"
  - Vibrant color tone, highlighted with orange border

A VS divider or arrow between the two panels.

--- ROW 2 (Bottom): System Architecture Pipeline ---

A clean left-to-right pipeline with 5 connected blocks:

[1] INPUT
  - Phone camera icon + raw preview image (dark, blurry)
  - Label: "Live Camera Preview"

  → arrow →

[2] VISION ENCODER (light blue #E3F2FD rounded box)
  - "MobileViT-Small"
  - "SE + FPN"
  - "5.6M params"
  - Small icon: feature map grid

  → arrow →

[3] AESTHETIC SCORER (light purple #F3E5F5 rounded box)
  - "5-Dimension Scoring"
  - Show 5 mini bar charts:
    Composition: 7.2
    Lighting: 4.1 (highlighted red = problem)
    Color: 6.8
    Clarity: 5.5
    Subject: 7.0
  - Overall: 6.1/10

  → arrow →

[4] PARAMETER PREDICTOR (light orange #FFF3E0 rounded box, ⭐ largest block)
  - "Problem Detection → Cross-Attention → 5 Parameters"
  - Show: 5 learnable tokens (colored circles: 🔴🔵🟢🟠🟣)
  - Show: Cross-Attention symbol (Q/K/V)
  - Output: 5 parameter cards with confidence bars
  - Label: "⭐ Core Innovation: Semantic-to-Parameter Translation"

  → arrow →

[5] OUTPUT
  - "Real-time Adjustment" 
  - Before/After comparison:
    Left mini photo: dark, poor → Right mini photo: bright, vivid
    Score: 6.1 → 8.5
  - Phone mockup with improved photo
  - "<80ms on mobile device"

--- BOTTOM BANNER ---

Key statistics in a horizontal bar:
  "7B → 9M params | 14GB → 18MB model | 2.5s → 80ms latency | 92% accuracy preserved"

=== CAPTION ===

Figure 1. Overview of MobileVenus. Unlike existing multimodal LLMs that provide 
only complimentary text feedback, MobileVenus delivers precise, executable camera 
parameter adjustments in real-time (<80ms). Our system (1) encodes the live camera 
preview via a lightweight MobileViT encoder, (2) scores 5 aesthetic dimensions, 
(3) detects 9 types of photographic problems, and (4) predicts 5 camera parameters 
(exposure compensation, white balance, focus point, HDR, shooting mode) with 
per-parameter confidence through a novel Cross-Attention mechanism with learnable 
parameter tokens.

=== STYLE GUIDELINES ===

- Academic paper quality, clean and professional
- Minimalist design, NO decorative elements
- Color palette:
  - Vision Encoder: #E3F2FD (light blue)
  - Aesthetic Scorer: #F3E5F5 (light purple)
  - Parameter Predictor: #FFF3E0 (light orange) — most prominent
  - Arrows: solid black 1.5pt
  - Background: pure white
- Font: Times New Roman or similar serif font
- All text must be legible at printed paper size
- Math notation in italic where applicable
- Rounded rectangles with subtle shadows for module blocks
- NO emojis in the actual figure (only used here for description)
- Style reference: DETR (ECCV 2020) Fig.1, SAM (ICCV 2023) Fig.1, LLaVA (NeurIPS 2023) Fig.1
```

---

# Figure 2: Detailed Architecture 详细架构图提示词 (v5)

> **用途**: 论文 Figure 2/3，展示模块内部结构、数据流和 tensor 维度
> **生成工具**: 推荐 PPT/Figma 手工绘制，或 ChatGPT 生成 SVG 代码（不要用 DALL-E）

## 📝 提示词（英文版）

```
Create a detailed academic architecture figure for a CVPR paper, showing the 
INTERNAL structure of MobileVenus's 5-Parameter Prediction system. This is NOT 
an overview — it must show every module's internals, tensor shapes, data flow 
arrows, and mathematical operations.

Style: DETR (ECCV 2020) Figure 2, Mask2Former (CVPR 2022) Figure 2, DINO 
(ICLR 2023) Figure 2 — these all show detailed internal architecture with 
tensor dimensions.

=== LAYOUT ===
Canvas: 4000×2800px, white background, 300 DPI.
Flow: Top-to-Bottom, with the middle section expanded to show internals.

=== SECTION 1 (Top 15%): Visual Feature Extraction ===

Show as a horizontal pipeline with tensor shapes annotated:

  Input Image (3×224×224)
    ↓
  ┌────────────────────────────────────────────────────┐
  │ MobileViT-Small Vision Encoder [light blue #E3F2FD]│
  │                                                     │
  │ Conv Stem → MV2 Block → MobileViT Block ×3         │
  │     ↓           ↓              ↓                    │
  │   C2(64)     C3(128)       C4(256)    ← Show as    │
  │     ↓           ↓              ↓         3 feature  │
  │   ┌─── SE Attention (squeeze→excite) ───┐  maps     │
  │   │                                      │          │
  │   └──→ FPN Fusion (C2+C3+C4→384) ──────┘          │
  │              ↓                                      │
  │        Global Average Pool → x_v ∈ ℝ^{B×384}      │
  └────────────────────┬───────────────────────────────┘
                       ↓
  ┌──────────────────────────────────────┐
  │ Vision Projection [light blue]       │
  │ Linear(384→512) → GELU → LN         │
  │ → x_v ∈ ℝ^{B×512}                  │
  └────────────┬─────────────────────────┘
               ↓
         ┌─────┴─────┐
         ↓           ↓

=== SECTION 2 (Left Branch, 20%): Aesthetic Scorer ===

  [Light purple #F3E5F5 rounded box]

  x_v ∈ ℝ^{B×512}
    ↓
  ┌──────────────────────────────┐
  │ Shared Encoder               │
  │ Linear(512→256) → GELU      │
  │ Linear(256→256) → GELU      │
  │ → h ∈ ℝ^{B×256}            │
  └────────┬─────────────────────┘
           ↓
  ┌──────────────────────────────────────────┐
  │ 5× Dimension Attention (parallel)        │
  │                                          │
  │  h → Linear(256→64) → Softmax → α_i     │
  │  h ⊙ α_i → score_head_i → s_i ∈ [0,10] │
  │                                          │
  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐    │
  │  │Comp│ │Ligh│ │Colo│ │Clar│ │Subj│    │
  │  │ s₁ │ │ s₂ │ │ s₃ │ │ s₄ │ │ s₅ │    │
  │  └────┘ └────┘ └────┘ └────┘ └────┘    │
  └────────┬─────────────────────────────────┘
           ↓
  S ∈ ℝ^{B×5} (5 dimension scores)
  weighted_score = Σ(w_i · s_i)

=== SECTION 3 (Right Branch, 55%): Parameter Predictor ⭐ MAIN FOCUS ===

  Wrap in large orange border (#FF9800), label: 
  "⭐ Semantic-to-Parameter Translation"
  This section must be the LARGEST and most detailed.

  --- Stage 1: Problem Classifier [light green #E8F5E9] ---

  Inputs: x_v ∈ ℝ^{B×512}, S ∈ ℝ^{B×5}
    ↓
  [x_v ∥ S] → concat → z ∈ ℝ^{B×517}
    ↓
  ┌──────────────────────────────────────────┐
  │ MLP Fusion                               │
  │ Linear(517→256) → LayerNorm → GELU      │
  │ Linear(256→256) → LayerNorm → GELU      │
  │ → h_fused ∈ ℝ^{B×256}                  │
  └────────┬─────────────────────────────────┘
           ↓
     ┌─────┴─────┐
     ↓           ↓
  ┌────────┐  ┌──────────────────────┐
  │Problem │  │Severity Heads ×9     │
  │Head    │  │                      │
  │Lin→σ   │  │Each: Lin(256→3)→SM  │
  │        │  │→ mild/moderate/severe│
  │P∈ℝ^{9}│  │→ Ŝ∈ℝ^{B×9×3}       │
  └────────┘  └──────────────────────┘

  Show 9 problem type labels in small colored tags:
  [Underexposure] [Overexposure] [Color Cast] [Poor Composition] 
  [Blur] [Noise] [Backlight] [Low Contrast] [Cluttered BG]

  --- Stage 2: Cross-Attention Parameter Mapping ⭐⭐ ---
  [light orange #FFF3E0]

  THIS IS THE CORE. Show full Cross-Attention detail:

  Top: 5 Learnable Parameter Tokens (show as colored embedding vectors):
    T_EV ∈ ℝ^{256} [red circle]
    T_WB ∈ ℝ^{256} [blue circle]
    T_Focus ∈ ℝ^{256} [green circle]
    T_HDR ∈ ℝ^{256} [orange circle]
    T_Mode ∈ ℝ^{256} [purple circle]
    → T ∈ ℝ^{5×256}
    Label: "Learnable Parameter Tokens (nn.Parameter)"

  Cross-Attention Block (show ×2 stacked, with internal detail for one):
  ┌─────────────────────────────────────────────────────────┐
  │ Cross-Attention Block (×2 layers)                       │
  │                                                         │
  │     Q = W_q · T        (from Parameter Tokens)          │
  │     K = W_k · [x_v'; h_fused; S']   (from features)    │
  │     V = W_v · [x_v'; h_fused; S']                      │
  │                                                         │
  │     Attention = Softmax(QK^T / √d) · V                 │
  │         ↓                                               │
  │     + Residual → LayerNorm                              │
  │         ↓                                               │
  │     FFN: Linear(256→512) → GELU → Linear(512→256)      │
  │         ↓                                               │
  │     + Residual → LayerNorm                              │
  │                                                         │
  │     → T' ∈ ℝ^{B×5×256}  (enriched parameter tokens)   │
  └────────────────────┬────────────────────────────────────┘
                       ↓

  --- Stage 3: Parameter-Specific Decoding Heads ---

  Show 5 PARALLEL columns, each decoding one token:

  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ EV Head  │ │ WB Head  │ │Focus Head│ │ HDR Head │ │Mode Head │
  │          │ │          │ │          │ │          │ │          │
  │ T'₀      │ │ T'₁      │ │ T'₂      │ │ T'₃      │ │ T'₄      │
  │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │
  │ Lin(256  │ │ Lin(256  │ │ Lin(256  │ │ Lin(256  │ │ Lin(256  │
  │  →128)   │ │  →128)   │ │  →128)   │ │  →64)    │ │  →64)    │
  │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │
  │ Lin(128  │ │ Lin(128  │ │ Lin(128  │ │ Lin(64   │ │ Lin(64   │
  │   →1)    │ │   →1)    │ │   →2)    │ │   →2)    │ │   →5)    │
  │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │
  │  tanh    │ │ sigmoid  │ │ sigmoid  │ │ softmax  │ │ softmax  │
  │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │ │   ↓      │
  │  ×3.0    │ │ ×8K+2K   │ │ (x,y)   │ │{off,on}  │ │{5 modes} │
  │          │ │          │ │          │ │          │ │          │
  │ EV∈[-3,3]│ │WB∈[2K,   │ │∈[0,1]²  │ │binary    │ │5-class   │
  │          │ │   10K]   │ │          │ │          │ │          │
  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
    continuous    continuous   continuous   discrete    discrete

  Blue background for continuous params, orange for discrete params.

  Side branch from T':
    [T'₀∥T'₁∥...∥T'₄] → Flatten(5×256=1280) 
    → Lin(1280→256) → Lin(256→5) → σ
    → Confidence ∈ ℝ^{B×5} ∈ [0,1]
    Show as 5 horizontal confidence bars

=== SECTION 4 (Bottom 10%): Output Summary ===

  Final output card:
    EV: +1.5 (conf: 0.87) | WB: 5500K (conf: 0.95) 
    | Focus: (0.6, 0.4) (conf: 0.72)
    HDR: ON (conf: 0.85) | Mode: Night (conf: 0.68)

=== VISUAL STYLE (CRITICAL) ===

- Academic paper quality, CVPR/ICCV standard
- Clean lines, no decorative elements, no gradients
- EVERY tensor shape must be annotated (e.g., ℝ^{B×512})
- EVERY arrow must show what data flows through it
- Module blocks: rounded rectangles with LIGHT fill + COLORED border
  - Vision Encoder: fill #E3F2FD, border #2196F3
  - Aesthetic Scorer: fill #F3E5F5, border #9C27B0
  - Problem Classifier: fill #E8F5E9, border #4CAF50
  - Cross-Attention: fill #FFF3E0, border #FF9800
  - Decode Heads: fill #FFF8E1, border #FFC107
- Arrows: solid black 1.5pt for main flow, dashed gray for auxiliary
- Font: Times New Roman, sizes 10-14pt
- All mathematical notation in italic
- Parameter tokens shown as colored filled circles with labels
- Cross-Attention internals clearly showing Q/K/V computation
- The figure should be SELF-EXPLANATORY even without reading the caption

=== CAPTION ===

Figure 2. Detailed architecture of MobileVenus. Visual features extracted by 
MobileViT-Small (with SE attention and FPN fusion) are fed into two branches: 
(1) Aesthetic Scorer with dimension-specific attention produces 5 aesthetic 
scores, and (2) Parameter Predictor with a Problem Classifier detecting 9 
problem types, followed by a Cross-Attention mechanism where 5 learnable 
parameter tokens attend to visual and problem features to decode 5 camera 
parameters with per-parameter confidence scores.
```

---

**更新日期**: 2026-03-24
**版本**: v3 (方法细节图) + v4 (Figure 1 总览) + v5 (Figure 2 详细架构)
**参考**: DETR Fig.1/2, SAM Fig.1, Mask2Former Fig.2, LLaVA Fig.1
