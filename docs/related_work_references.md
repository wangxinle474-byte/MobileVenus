# Related Work — 文献引用与实验映射

> 本文档收录在 v1-v10 实验中**实际参考过**的文献, 标注每篇论文 在哪个实验/组件中被引用, 便于论文写作和回溯.
>
> Last updated: 2026-05-15

---

## 1. 论文索引 (按引入时间倒序)

### [R15] NamedCurves: Learned Image Enhancement via Color Naming ⭐ 直接祖先
- **Venue**: ECCV 2024 (arXiv 2407.09892)
- **作者**: David Serrano-Lozano, Luis Herranz, Michael S. Brown, Javier Vazquez-Corral
- **核心方法**: 四段式管线 — (1) UNet-like backbone (LPIENet-inspired, MobileNet+CBAM, 3 encoder + 2 decoder) 标准化输入; (2) Color Naming (Van de Weijer 11 色→6 组: red/blue/green/orange-brown-yellow/pink-purple/achromatic) 分解为概率图; (3) BCPE (Bezier Control Point Estimator, 64-d contextual features + 6 分支, 每分支输出 3 通道×M 控制点, M=11 best, 正增量+归一化保证单调) 全局曲线; (4) Attention-based fusion (Q=backbone output, K/V=globally-adjusted, 1/8 分辨率, 6 组) + 概率加权平均 (阈值 0.2) 模拟局部编辑.
- **Loss**: α·L2(backbone_out, GT) + L2(final, GT) + SSIM(final, GT), α=0.5
- **训练设定**: Adam lr=1e-4, decay 50% per 50 epochs, horizontal flip aug; MIT5K 200 epochs, PPR10K 100 epochs
- **数据集**: MIT-Adobe FiveK (Expert C only, 3 版本: DPE 2250/2250/500, UPE 4500/500, UEGAN 4500/500), PPR10K (8875/2286, 360p augmented ×5)
- **PSNR 结果** (MIT5K):
  - DPE: **24.91 dB** (6 grouped channels; 24.72 dB with 11 channels)
  - UEGAN: **25.59 dB** (from pretrained model)
  - 对比: DeepLPF=23.90, 3DLUT ~23.6, CURL ~24.0, AdaInt ~24.3
- **Ablation 关键发现**: Color Naming 是最大贡献 (+0.32 dB CN concat, +0.31 dB CN-weighted avg); 11 控制点最优; backbone 选择次要 (换任何 backbone 仍超 SOTA)
- **Limitation**: 少颜色区域图像 (如单色场景) 时优势减小, 退化为近似全局调整
- **借鉴在**: **v9a-v11d 全系列** — 我们的 `NamedCurvesPredictor` 直接继承 BCPE + Color Naming 架构; v9 系列加入 per-action curves, v11a 加入 action-gated context, v11d 加入 action-conditioned context. 我们的关键创新是将 NamedCurves 从"单一 Expert C 目标"扩展到"多 action 条件化参数预测", 并用可微 ISP 替代直接图像输出.
- **与我们的差异**: (1) 他们的任务是 input→Expert C 图像, 我们的任务是 input+action→ISP 参数; (2) 他们没有 action conditioning, 我们新增 action-gated/action-conditioned context; (3) 他们的 backbone 输出直接监督, 我们的 coarse output 由 diff_isp 渲染
- **关键引文段**: §3.2 "We used the color naming model from Van de Weijer et al. to obtain the probability maps... We reduce the set of 11 probability maps to just 6 by grouping orange-brown-yellow, pink-purple, and white-grey-black."
- **Code**: https://github.com/davidserra9/namedcurves

### [R16] PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching
- **Venue**: AAAI 2026 (arXiv 2511.12998)
- **作者**: Chang, Zewei; Duan, Zheng-Peng; Zhang, Jianxing et al. (Nankai University + Samsung)
- **核心方法**: Diffusion-based (Stable Diffusion + ControlNet) 语义级 retouching.
  - 4 属性: colorfulness, contrast, color temperature, brightness ([-1,1])
  - SAM panoptic segmentation → multi-channel parameter maps (每通道一个属性, 每像素=所属区域得分)
  - Semantic Replacement: 训练时随机替换区域 (从其他样本取最不同属性区域), 增强区域边界感知
  - Parameter Perturbation: 对 parameter map 加 channel shift + blurring, 防止过拟合分割边界
  - VLM Agent: weak instructions (midpoint 默认值 + 历史偏好) / strong instructions (用户指定区域+属性+强度)
  - Feedback-driven Rethinking: 迭代闭环 — VLM 评估当前结果是否满足指令, 不满足则修正参数
  - Scene-aware Memory: 存储用户每次编辑的场景偏好
- **数据集**: MIT-Adobe FiveK (all 5 experts A/B/C/D/E), 训练/测试分割未明确 (同 FiveK 标准)
- **对比方法**: DiffRetouch, StarEnhancer, TSFlow, PIE-Net, JarvisArt
- **PSNR**: 论文 Table 1 含 PSNR/SSIM/LPIPS 全 5 expert 对比 (HTML 表格不可提取具体数字; 声称 maintains or surpasses SOTA)
- **User Study**: 50 volunteers × 30 images, majority preference over all baselines
- **与我们的关系**:
  - **相同**: 都用 MIT-Adobe FiveK + 属性级控制 (他们的 colorfulness/contrast/color_temp/brightness ≈ 我们的 saturation/contrast/wb/shadows+highlights)
  - **相同**: 都有 VLM/Agent 与用户交互的 pipeline (对应我们 MobileVenus Stage C)
  - **核心差异**: 他们是 **diffusion-based (重, 服务端)**, 我们是 **lightweight NamedCurves (轻, 端上)**; 他们的参数是 spatial per-region, 我们的是 global ISP 参数; 他们的 "参数→图像" 由 diffusion 完成, 我们由显式 diff_isp 完成
  - **论文定位**: 在 related work 中引用, 对比强调我们的 **端侧部署优势** 和 **显式可解释 ISP 参数** vs 他们的隐式 diffusion 生成
- **Code**: https://github.com/Auroral703/PerTouch

### [R1] Off the Planckian Locus: Using 2D Chromaticity to Improve In-Camera Color
- **Venue**: arXiv 2511.17133v2 (Nov 2025)
- **核心方法**: CST-MLP, 用 2D 色度坐标 (xy chromaticity) 代替 1D CCT 预测全局 3×3 颜色变换, 突破 Planckian locus 约束.
- **解决我们的痛点**: `models/diff_isp.py:color_temp_to_rgb_gains` 把 wb 锁死在 1D 黑体曲线 → off-Planckian 色偏 (品红/绿调) 不可表达.
- **借鉴在**: **v10a (Off-Planckian WB head)** — 重写 `color_temp_to_rgb_gains` 接受 2D (u, v) chromaticity offset 替代 1D K, gains 不再受黑体曲线约束.
- **关键引文段**: §3 Method "predicts a global CST in a single forward pass, enforces spatial consistency, requires no spectral sensitivity data, and generalizes better across diverse light sources by operating in 2D chromaticity space."

### [R2] JarvisArt: Liberating Human Artistic Creativity via an Intelligent Photo Retouching Agent
- **Venue**: NeurIPS 2025 (arXiv 2506.17612)
- **核心方法**: MLLM-driven agent, 两阶段训练 (CoT-SFT + GRPO-R RL), 调用 200+ Lightroom 工具, 通过 Agent-to-Lightroom Protocol 直接操作 Lightroom.
- **解决我们的痛点**: 完全跳过"反推 7D 参数 + 渲染"路线, 让 LLM 直接产 Lightroom XMP/preset, 绕过 ISP ceiling.
- **借鉴在**: **未直接采用** (代价过大, 需要 MLLM + RL 训练栈), 但 *论文 related work 节中对照引用* — 我们的 v10c 简化版与之对比, 强调 v10c 不需要 MLLM 仍可实现可学习参数空间.
- **关键引文段**: Abstract "outperforms GPT-4o with a 60% improvement in average pixel-level metrics on MMArt-Bench for content fidelity".

### [R3] VeraRetouch: A Lightweight Fully Differentiable Framework for Multi-Task Reasoning Photo Retouching
- **Venue**: arXiv 2604.27375 (2026)
- **核心方法**: Retouch Encoder (ResNet) + Retouch Renderer (MLP per-pixel mapping); 把编辑空间显式分成 3 个 disentangled latent (lighting / global color / specific color) + binary mask 训练; 配套 AetherRetouch-1M+ 数据集.
- **解决我们的痛点**: 7D ISP 容量不够, 无法表达教师编辑中的 hue rotation / per-channel curve / split toning.
- **借鉴在**: **v10c (VeraRetouch-style Encoder-Renderer)** —
  1. 把固定 `apply_diff_isp` 替换为可学习的 MLP Renderer (per-pixel color mapping), 自由度 >> 7D
  2. 用 Encoder 从 (orig, target) 对学反推 (替代 LBFGS inverse_fit)
  3. **不直接使用 disentanglement masks** (我们数据规模不够), 但保留架构思路供后续扩展
- **实测结果**: v10c 简化版在 372 pseudo-label 上 best_val_psnr=21.90 dB, 明显低于 v9a/v10b. 结论: VeraRetouch/INR 类 full renderer 需要更大且更干净的真实 retouch 数据, 不适合当前小规模生成式伪标签.
- **关键引文段**: §3.1 "Retouch Renderer ... a lightweight pure MLP for per-pixel color mapping ... synthesizes the output image ... by additively injecting the latent z into its hidden layers."

### [R4] INRetouch: Context Aware Implicit Neural Representation for Photography Retouching
- **Venue**: arXiv 2412.03848 (Dec 2024)
- **核心方法**: Context-aware Implicit Neural Representation, 从 before-after 对学习编辑变换, 支持 single-example 学专家 preset; 配套 100K 张 × 170 Lightroom presets 数据集 (PRD).
- **解决我们的痛点**: 完全绕开"参数化"假设, 直接用 INR 隐式建模"教师编辑".
- **借鉴在**: **未直接采用** (需要更换整个数据 pipeline), 但 *论文 future work 节作为重要对比基线* — 我们的 v9 系列论证了"显式参数化 + 7D ISP" 路线的失败, INRetouch 的隐式表示是天然的下一步.
- **关键引文段**: Abstract "context-aware Implicit Neural Representation that learns to apply edits adaptively based on image content and context, and is capable of learning from a single example."

### [R5] Revisiting Image Fusion for Multi-Illuminant White-Balance Correction
- **Venue**: ICCV 2025 (arXiv 2503.14774)
- **核心方法**: 用 5 种不同 WB 渲染同一 sRGB 场景, 学预测像素级 fusion weights; 配套 16k 张 sRGB 多光源数据集.
- **解决我们的痛点**: WB 单一全局 gain 无法处理多光源场景.
- **借鉴在**: **未直接采用** (我们任务是 single-illuminant retouch, 不是 multi-illuminant correction), 但 *论文 related work 节引用* 以表明我们 wb action 的范围限制.
- **关键引文段**: Abstract "introduce a large-scale multi-illuminant dataset comprising over 16,000 sRGB images rendered with five different WB settings".

### [R6] NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement
- **Venue**: AAAI 2024 (arXiv 2306.11920)
- **核心方法**: 用一个小 MLP 隐式表示 3D LUT, 输入 (R, G, B) 输出修改后 (R', G', B'); 内存 <1MB, 支持多 style 融合.
- **解决我们的痛点**: 现有 33×33×33×3 LUT 容量大但插值不灵活, 训练时空开销大.
- **借鉴在**: **v10b (NILUT residual)** —
  1. 在 7D ISP 输出 (coarse) 上加 NILUT MLP 学残差 color transform
  2. 保留可解释的 7D 作为先验, 用 NILUT 修复 7D 表达不了的部分
  3. 引用 mv-lab 官方 PyTorch 实现作为基础
- **实测结果**: v10b baseline=24.42 dB, Plan C resume+EMA=24.47 dB, 但 per-action eval 显示 NILUT gate=0.0000; 强行解锁 gate=1 后仅 23.36 dB. 结论: 当前数据下 NILUT 残差不是有效提升来源, Plan C 的小幅收益来自 EMA/微调.
- **关键引文段**: Abstract "Neural Implicit LUT (NILUT), an implicitly defined continuous 3D color transformation parameterized by a neural network."

### [R7] AdaInt: Learning Adaptive Intervals for 3D Lookup Tables
- **Venue**: CVPR 2022
- **核心方法**: 自适应非均匀 3D LUT 采样, 同时预测 sampling coordinates 和 output values.
- **借鉴在**: 现有 v8/v9 系列 LUT head 设计参考 (33³ 均匀 LUT), AdaInt 风格的非均匀采样可作为 v10b 的备选.

### [R8] SepLUT: Separable Image-adaptive Lookup Tables for Real-time Image Enhancement
- **Venue**: ECCV 2022
- **核心方法**: 1D + 3D LUT 级联, 分离 component-independent 和 component-correlated sub-transforms.
- **借鉴在**: **v10b 备选方案** — 如果 NILUT 训练不稳定, fallback 到 SepLUT 的 1D+3D 结构.

### [R9] CURL: Neural Curve Layers for Global Image Enhancement
- **Venue**: BMVC 2020 (arXiv 1911.13175)
- **核心方法**: 多色空间 (HSV/Lab/RGB) 神经曲线层, 模拟 Photoshop curves tool.
- **借鉴在**: 现有 v9 NamedCurvesPredictor 的 Bezier curve 设计的早期参考.

### [R10] Neural Color Operators for Sequential Image Retouching
- **Venue**: ECCV 2022
- **核心方法**: 模拟传统色彩算子的可训练 sequential 操作, 每个 operator 是一个 pixelwise color transformation 网络.
- **借鉴在**: 现有 `apply_diff_isp` 的 sequential 设计的早期参考; v10c MLP Renderer 思路与之相通.

### [R11] Image-Adaptive 3D Lookup Tables for Real-Time Image Enhancement
- **Venue**: CVPR 2020 (Zeng et al., 后续 ECCV 2024 扩展版)
- **核心方法**: 多个 3D LUT + 图像自适应权重融合.
- **借鉴在**: 现有 v8 baseline 的 ParamResidualLUTPredictor 直接继承此设计; n_luts=3 + 自适应 weights 即来自此架构.

### [R12] FC4: Fully Convolutional Color Constancy with Confidence-weighted Pooling
- **Venue**: CVPR 2017
- **核心方法**: FCN 预测 patch-level illumination + confidence-weighted pooling.
- **借鉴在**: v9f WB head 设计参考 (`_WBHead` class), 但简化为单一全局 3-gain 输出.

### [R13] Deep White-Balance Editing
- **Venue**: CVPR 2020 (Afifi & Brown)
- **核心方法**: 学三个 WB 设置 (auto/incandescent/shade) 的 sRGB-domain editing.
- **借鉴在**: v9g WB supervision 设计的概念参考 — 用 log(target_mean / orig_mean) 当 wb gains 监督信号; 但我们没用其完整 multi-WB 框架.

### [R14] Diffusion-Distilled InstructPix2Pix (相关 baseline)
- **Venue**: 我们 baseline 的两个 teacher: **FireRed-Image-Edit-1.0** (HuggingFace) 和 **Qwen-Image-Edit-2509** (Alibaba DashScope)
- **借鉴在**: data generation 阶段 — `tools/data/editor_models/run_firered_*.py`, `run_qwen_image_edit_dashscope.py`. 双教师对照实验 (v9a_cleanwb + v9a_qwen) 论证生成式 editor 作为 ISP 监督的失败.

---

## 2. 实验 → 论文映射表

| 实验 | 主要参考 | 次要参考 | 借鉴部分 |
|------|----------|----------|----------|
| v1-v7 (early) | [R11] Image-Adaptive 3D LUT | [R7] AdaInt | LUT-based enhancement framework |
| v8 ParamResidualLUT | [R11] | [R7][R8] | 3-LUT 自适应权重 + ISP residual |
| v9a NamedCurves | **[R15] NamedCurves** | [R9] CURL, [R10] Neural Color Op | BCPE + Color Naming 直接继承 |
| v9b/v9c attention/per-action | [R15] | [R9] | Per-action curves + attention |
| v9d/v9d_fix context map | [R15][R12] | — | 局部上下文调制 |
| v9e learnable CN | [R15] | — | Color naming 可学习化 |
| v9f WB head | [R15][R12] FC4 | [R13] Deep WB | 专用 WB 分支 |
| v9g WB supervision | [R15][R13] | — | log_gains 显式监督 |
| v9g_aug 合成 wb | [R5] (loosely) | — | Planckian 物理渲染数据扩充 |
| v9a_cleanwb (FireRed) | [R14] FireRed | — | 严格 prompt 重生数据 |
| v9a_qwen (Qwen) | [R14] Qwen-Image-Edit | — | 双教师对照实验 |
| **v10a Off-Planckian WB** | **[R1] Off Planckian Locus** | — | 2D chromaticity 替代 1D CCT |
| **v10b NILUT residual** | **[R6] NILUT** | [R7][R8] | Implicit 3D LUT MLP residual; 实测 gate=0 锁死, gate=1 负迁移 |
| **v10c VeraRetouch-like** | **[R3] VeraRetouch** | [R10] | Encoder + MLP per-pixel Renderer; 实测 21.90 dB, 小数据失败 |
| MobileVenus overall | [R15][R16] | [R2] JarvisArt | 端侧 NamedCurves vs 服务端 diffusion/agent 方案定位 |

---

## 3. 论文 related work 节建议结构 (供后续写作)

```
Section 2.1 Color-Name-Guided Curve Enhancement
  [R15] NamedCurves (ECCV24)  ← v9-v11 直接祖先, BCPE + CN + Attention fusion
  [R9] CURL (curve layers)  ← 早期曲线参考
  [R10] Neural Color Operators (sequential)

Section 2.1b Differentiable ISP / LUT Methods
  [R11] Image-Adaptive 3D LUT  ← v8 baseline 继承
  [R7][R8] AdaInt / SepLUT (LUT efficiency)

Section 2.2 White Balance Beyond Planckian Locus
  [R12] FC4 (CNN-based color constancy)
  [R13] Deep WB Editing (sRGB-domain)
  [R5] Multi-Illuminant Fusion (multi-light)
  [R1] Off-Planckian 2D Chromaticity  ← v10a 直接采用

Section 2.3 Implicit / Neural LUT Representations
  [R6] NILUT  ← v10b 直接采用
  [R4] INRetouch (context-aware INR)

Section 2.4 Personalized / Agent-based Retouching
  [R16] PerTouch (AAAI26, diffusion + VLM agent, 语义级)  ← 服务端方案对比
  [R2] JarvisArt (NeurIPS25, MLLM agent + Lightroom)  ← 我们对比的 alternative

Section 2.5 Generative Image Editor Distillation (我们的负面结果定位)
  [R14] FireRed / Qwen-Image-Edit (teachers we tested)
  [R3] VeraRetouch (parallel approach with disentangled latents)  ← v10c 直接采用
```

---

## 4. 与现有实验的差异化定位

我们的工作与 [R3] VeraRetouch / [R4] INRetouch 等"端到端"路线**互补**:

- **他们**: 用 Lightroom 真实参数 (Param-Retouch 子集) 当 GT 训练;
- **我们**: 探索"用生成式编辑器 (FireRed/Qwen) + inverse_fit 反推 GT" 这条**捷径**, **诚实记录其失败** (双教师 ablation), 并在 v10 系列尝试架构升级 (Off-Planckian WB / NILUT residual / Encoder-Renderer).

这个负面发现可以独立成节: **"Why Generative Image Editors Are Not Free ISP Labels"** — 是当前文献中没有的对照实验. v10b/v10c 进一步说明, 即使引入 NILUT/INR/VeraRenderer 这类更强表达模型, 也不能从低质量 pseudo-label 中创造稳定 WB 监督信号.

---

## 5. 数据集引用

| 数据集 | 来源 | 用途 |
|--------|------|------|
| MIT-Adobe FiveK | Bychkovsky et al., CVPR 2011 | 主训练集 (Expert C) |
| LSMI Multi-Illuminant | Kim et al., ICCV 2021 | [R1] / [R5] 引用为 multi-light benchmark |
| AetherRetouch-1M+ | [R3] VeraRetouch 配套 | v10c 未来扩展数据源 |
| PRD (100K × 170 presets) | [R4] INRetouch 配套 | v10c 未来扩展数据源 |

---

## 6. 2026 年新增引用 (paper §2 后续段落)

### [R17] JarvisEvo: Towards a Self-Evolving Photo Editing Agent with Synergistic Editor-Evaluator Optimization
- **Venue**: CVPR 2026 (arXiv 2511.23002, accepted 2026-02-21)
- **作者**: Lin, Yunlong; Lin, Zixu; Lin, Kunjie et al. (Xiamen University + CUHK + Bytedance + NUS + Tsinghua) — JarvisArt 同团队
- **核心方法**: 在 [R2] JarvisArt 基础上加 *interleaved multimodal Chain-of-Thought (iMCoT)*, 编辑→评估→反思闭环, 用 editor + evaluator 协同 RL.
- **解决 [R2] 痛点**: text-only CoT 仍有 instruction hallucination; 加入 visual feedback loop 能更好地自我修正.
- **与我们的关系**: 仍属 server-side MLLM agent + Lightroom tool, 与 PerTouch/JarvisArt 同 family; 我们论文 §2.4 引用以表明 reasoning agent 路线最新进展, 强化我们 lightweight on-device 的差异化.
- **Code**: https://github.com/LYL1015/JarvisEvo

### [R18] MonetGPT: Solving Puzzles Enhances MLLMs' Image Retouching Skills
- **Venue**: ACM TOG 2025 (arXiv 2505.06176)
- **核心方法**: 用专门设计的 *visual puzzles* (operation 识别 + 参数估计 + sequence planning) LoRA-finetune MLLM, 让 MLLM 学会输出可执行的 Lightroom 操作序列.
- **解决我们的痛点**: 不需要 GRPO RL 也能让 MLLM 产 Lightroom 参数, 简化训练栈.
- **与我们的关系**: §2.4 中作为 [R2]/[R17] 之外的 alternative VLM-based retouching path; 仍是 server-side, 不冲击我们 on-device 定位.
- **关键引文段**: Abstract "MLLMs learn retouching operations by solving specially designed visual puzzles that teach operation recognition, parameter understanding, and sequence planning."

### [R19] DreamLite: A Lightweight On-Device Unified Model for Image Generation and Editing
- **Venue**: arXiv 2603.28713 (2026)
- **核心方法**: 389 M 参数的 unified diffusion (T2I + 编辑共享 backbone), 4-step distillation, in-context conditioning (target/source 在 latent 空间左右拼接, prefix [Generate] / [Edit] token 路由), Qwen3-VL-2B 文本编码, TinyVAE.
- **关键性能**: Xiaomi 14 上单图 <1s 生成或编辑.
- **与我们的关系**: §2.5 / §5.3 Path 3 引用 — 这是当前最接近 "deployable diffusion editing" 的 baseline, 但仍是 *generic instruction editing* 而非专门的 *parametric photo retouching*. 蒸馏 FireRed/Qwen-Image-Edit 到 ~400M 范畴是我们 future work 的具体落点.
- **关键引文段**: Abstract "DreamLite, a unified and compact diffusion model capable of performing both image generation and editing within a single network ... generate or edit a 1024×1024 image in less than 1s on a Xiaomi 14 smartphone."
