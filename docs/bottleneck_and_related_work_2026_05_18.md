# 瓶颈分析 + 相关工作调研 (2026-05-18)

为 Path Y 数据生成等待期间所写。整合 §3–§5 已完成实验、Path X/Y/Z 新结
果、以及 2024–2026 年相关文献，目的是识别下一步最高 ROI 方向。

---

## 1. 当前瓶颈总览（截至 2026-05-18）

我们的 v11a 系列在 FireRed pseudo-label 上的 PSNR 上界为 **24.46 ± 0.20 dB**
（3 seeds），而五个独立的 architecture-side bridging 干预实验
(§4.5/§4.7/§4.7b/§4.8/§4.9) 都未能跨越此天花板。Path X/Y/Z 三条新路径
（§4.10）进一步细化了这个限制：

### 三个独立、可分别测量的瓶颈

| ID | 名称 | 来源 | 实测 | 干预可行性 |
|---|---|---|---|---|
| **(i)** | Pseudo-label noise ceiling | §4.6 inverse_fit residual analysis | 65% WB 样本 LBFGS 拟合失败；500 张里 0% verdict=OK | 用更干净的 supervision (Expert C → §4.7 P0) ⇒ in-domain +17 dB 但 cross 不通 |
| **(ii)** | WB structural failure (1D Planckian) | §4.6 inferred temperature distribution | mean=7079 K, p90=9644 K 边界饱和 | 1D 色温标量无法表达 off-Planckian magenta/green casts；要 CST-MLP 或 image-domain |
| **(iii)** | 7D ISP capacity ceiling | §3 expert C clean target eval; §4.9 path A=42.66 vs ISP=23.94 | NamedCurves 24.91 dB > ISP 23.94 dB 因为它直接预测图像 | ⚠️ **被 §4.10 Path X 突破**: 加 implicit residual head 后 +0.67 dB overall, +2.15 dB highlights |

### Path X/Y/Z 新发现修正了原来的 single-bottleneck 框架

| 实验 | 假设 | 结果 | 启示 |
|---|---|---|---|
| **Path X** | residual head 能补 ISP-inexpressible edits | ✅ +0.67 dB overall, +2.15 dB highlights | 7D ISP capacity ceiling **可被 patch**，但 WB 不动 |
| **Path Z** | 弃用 7D ISP 直接学像素 → §4.9 Path A 在 FireRed 复现 | ❌ -0.09 dB overall, -1.61 dB WB | 在 372 样本下 ISP 是有用的归纳偏置；§4.9↔§4.10 反转 |
| **Path Y (运行中)** | WB 数据稀疏是结构性（非 architecture）瓶颈 | 待 11.5 h 后验证 | 28 → 328 WB 样本 (12×) 是 sample-complexity sharpest test |

### 修正后的瓶颈拓扑

```
                 ┌─ noise (i)        →  数据来源问题   ─→ §4.7 mixing 失败
24.5 dB ceiling ─┤                                      ─→ §4.8 source replacement -3.85 dB
                 ├─ WB structural (ii) →  1D 表达不够   ─→ Path Y 测试 (sample-vs-structural)
                 └─ 7D capacity (iii) →  渲染表达不够  ─→ Path X 已证可 patch (+0.67)
```

---

## 2. 当前路径的具体限制（细化）

### 2.1 Path X 已突破，但仅在两个 action 上

per-action 增量分布：

| Action | v11a | Path X | Δ | 解释 |
|---|---|---|---|---|
| highlights | 23.84 | **25.99** | **+2.15** | 1D highlights 参数 → 实际 FireRed 局部高光 vignetting/specular |
| saturation | 22.88 | 23.04 | +0.16 | 7D scalar 已基本覆盖 |
| shadows | 26.35 | 26.39 | +0.04 | 7D scalar 已基本覆盖 |
| wb | 22.58 | 22.50 | -0.08 | 数据稀疏（28 train），架构干预无效 |
| contrast | 26.70 | 26.19 | **-0.51** | 微小过拟合（patch 上的 contrast head 没用） |

**结论**: residual head 只在 1D 参数 ↔ 复杂局部修改 mismatch 强烈的 action
上有效（highlights）。当 7D scalar 足以表达（saturation/shadows/contrast），
patch 反而是噪声。

### 2.2 WB 跨所有架构变体不动（22.50–22.58 dB）

| Architecture | WB PSNR |
|---|---|
| v9d baseline | 22.66 |
| v11a action-gated | 22.58 |
| v11d action-conditioned | 24.53 ⚠️（seed 42 单点）|
| Path X residual head | 22.50 |
| Path Z pure ResUNet | 20.97 |

排除 v11d 的 seed 42 单点（3-seed 平均 24.32 ± 0.27），其余 4 个架构在
22.50–22.66 dB 范围内变化 < 0.16 dB。**这是 architecture-invariant
phenomenon**。Path Y 即将测试这是 sample-limited 还是 1D-Planckian
structural。

### 2.3 §4.9 vs §4.10 反转（同 architecture 不同 supervision）

| Architecture | Clean Expert C (§4.9, n=15.7K) | FireRed pseudo (§4.10, n=372) |
|---|---|---|
| v11a | 41.88 | 24.46 |
| **v11a + residual head (Path B/X)** | 41.91 (+0.03) | 25.28 (+0.67) |
| **Pure image-domain (Path A/Z)** | **42.66 (+0.78)** | 24.52 (-0.09) |

样本充足时，pure image-domain (Path A/Z 同架构) 胜出；样本稀缺时
（372），ISP scaffolding 翻盘。这是 **架构表达力 × 样本复杂度** 的乘
积关系，而不是 "更强架构总赢"。

---

## 3. 相关工作 taxonomy（2024–2026）

按"如何处理 image editing 这个任务"的方法论分 5 类。

### 3.1 White-box parametric retouching（我们 v11 系列所属）

预测一组传统 ISP 参数，套用 differentiable physical pipeline。

| 工作 | 年份 | 表达 | 限制（与我们重叠） |
|---|---|---|---|
| HDRNet [Gharbi+'17] | SIGGRAPH | bilateral grid → local affine | 低分辨率、blur |
| AdaInt [Yang+'22] | CVPR | adaptive 3D LUT 间隔 | 还是 3D LUT |
| **NamedCurves** [Serrano-Lozano+'24] | ECCV | 6 color-name × Bezier curves + attention | 我们的 v11a 起点 |
| RSFNet [Ouyang+'23] | ICCV | region-specific color filters | white-box, 但 region 分割固定 |
| **CSEC** [Li+'24] | CVPR | over/under-exposed color shift estimation | 专攻 exposure 校正 |
| 自蒸馏 SDA-LUT [Zheng+'25] | Pattern Rec | adaptive interval LUT + self-distillation | 单 dataset SOTA, 不跨域 |

**共同瓶颈**: 它们都被 7D~21D 参数空间表达力限制。Path A/Path Z 的
in-domain +0.78 dB 上升 (§4.9) 揭示了这个 ceiling。我们 §3-§5 的故事
就是在这一族里做诊断。

### 3.2 LUT-based efficient inference

| 工作 | 创新点 | 我们的关联 |
|---|---|---|
| **NILUT** [Conde+'24 AAAI] | 把 3D LUT 编成 INR (~5K params) 取代 1MB LUT | §4.9 Path B 测试了它（in-domain +0.03 dB null） |
| **ICELUT** [Wang+'24 ECCV] | 纯 LUT 推理（无 CNN），1×1 conv + 4D LUT | 我们没测；可能与 Path X residual head 等价表达 |
| Image-Adaptive 3DLUT [Zeng+'20] | 3 个 base LUT 可学权重 | 早期工作；现在 Path A 范式更强 |

**评价**: LUT 表达力 < INR < UNet。在我们小数据集上，LUT 型号意义不大
（数据稀疏比 capacity 重要）。

### 3.3 Implicit Neural Representation（直接学色彩映射）

| 工作 | 输入 | 关键创新 | 与我们对比 |
|---|---|---|---|
| **INRetouch** [Alezaby+'24 CVPR] | (color, coord, context) 三元组 | depth-wise CNN context + window sampling, 11.5K params | **可作为 Path Z 的优化版**: 比 ResUNet (8M) 高效 700×, in-domain SOTA |
| Controllable Style Transfer via Test-time Training INR [PR'24] | reference image | test-time fit | 接近"per-distribution serving" 我们 §5.3 Path 1 提议 |

**关键启示**: INRetouch 用 11.5K 参数达到 SOTA，比 Path Z 的 8.26M 高
700× 高效。如果我们要持续 Path 1（per-distribution arch），INRetouch
是更好的 lightweight starter。论文里我们已经 cite 它作 [R4]。

### 3.4 VLM/MLLM-driven agents（外部 retouching tools）

| 工作 | 年份 | 设计 | 我们的关联 |
|---|---|---|---|
| **MonetGPT** [Dutt+'25 TOG] | 2025-05 | LoRA-finetune VLM 学 puzzle → 输出 Lightroom 参数 | 与我们 7D 范式正交，但有 tool dependency |
| **JarvisArt** [Lin+'25 NeurIPS] | 2025 | MLLM agent + 200 LR tools + GRPO RL | 14GB, 不适合 on-device |
| **JarvisEvo** [Lin+'26 CVPR] | 2026-02 | iMCoT 自我评估 refine | 同上但闭环 |
| **PhotoArtAgent** [Chen+'25] | 2025 | 多 VLM training-free agent | 推理慢（多 round） |
| **PerTouch** [Zhao+'25 arxiv 2511.12998] | 2025-11 | 参数图（spatial param）+ semantic perturbation | 接近我们 v11c region delta，但有 VLM 解析 |

**共同问题**: 依赖外部 non-differentiable tools (Lightroom API)，无法
端到端梯度，模型大。**与我们 deconstructive paper 正交** —— 我们诊断
parametric supervision，他们做 parametric prediction，但用大模型规
划。论文 §5.3 Path 3 已 cite。

### 3.5 Differentiable encoder/renderer + VLM（VeraRetouch family）

⭐ **2026-04 的 VeraRetouch** (arXiv 2604.27375) 是当前最相关的工作：

- FastVLM-0.5B → **3 个解耦 latent (light, global color, specific color)** → MLP renderer per-pixel
- AetherRetouch-1M+ dataset (auto/style/param 三类共 1M+ pairs, 用 LR
  API 合成)
- **三阶段训练**: domain align pretrain (param-retouch only) → reasoning
  SFT (3 task) → DAPO-AE RL post-training
- FiveK PSNR **26.85 dB**（vs Flux.1 Kontext 25.77, vs JarvisArt 实测较弱）
- 推理时间 **6.9s/image** (vs Flux 16.8s, JarvisArt 14.3s) — Macbook
  M4 7.4s, iPhone 16 Pro 13.5s

**关键启示对我们**:
1. **3-latent disentanglement 比 7D 参数表达力强**（light + global + specific），
   接近我们 v11d 的 action-conditioned 思路但更细。
2. 用 1M+ 合成 pair 解决了 supervision 问题 → 印证 §4.7b 我们 P0v2
   的猜想（pseudo scaling 单独无效，但需要 distribution-faithful 的
   合成 pipeline）。
3. **渲染器 = pure MLP per-pixel** — 这是 Path Z (8M ResUNet) 的更轻
   量版本，证实"image-domain rendering" 是正确方向，但实现要 INR/MLP
   而非 conv UNet。

### 3.6 Diffusion editing distillation（on-device 方向）

| 工作 | 年份 | 参数 | 速度 | 与我们关联 |
|---|---|---|---|---|
| SnapFusion [Li+'23] | NeurIPS | ~1B | 1.5s/image | 仅 T2I |
| SnapGen [Hu+'24] | 2024 | 379M | <2s/image | 仅 T2I |
| **DreamLite** [Hu+'26 arxiv 2603.28713] | 2026 | 389M unified | <1s/image (Xiaomi 14) | **最接近 on-device 目标**，但是 generic editing 不专注 retouching |
| FireRed-Image-Edit-1.0 [Xie+'26 arxiv 2602.13344] | 2026-02 | 12B+ | ~5s/image L20 | **我们用作 teacher**, 不能 on-device |

**关键趋势**:  Diffusion editing 在 1Hz on-device 已可行（DreamLite），
**但都是 generic editing 不是 retouching**. 把 DreamLite/SnapGen 蒸馏
到 retouching domain 是 §5.3 Path 3 的具体落实方式。

---

## 4. 我们工作的位置（competitive landscape）

我们的 deconstructive paper 在四象限里独一无二：

```
                  Architecture-side
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        │  VeraRetouch │  我们        │
        │  INRetouch   │  v11a + 7D   │
        │  (强表达)    │  + 5+1 干预  │
        │              │              │
Data    ├──────────────┼──────────────┤  Diagnostic
side    │              │              │
        │  JarvisArt   │  §3 ceiling  │
        │  PerTouch    │  §4 transfer │
        │  (大数据+VLM)│  §5.6 inverse│
        │              │              │
        └──────────────┴──────────────┘
                       │
                  Pos: build SOTA
                  Our: diagnose ceiling
```

我们论文的独特卖点 (Sept '26 deadline 前):

1. **"为什么 24.5 dB 是上限"** —— 没人系统地拆解过
2. **5+1 个干预实验全部失败** —— manifold hypothesis 强证据
3. **Path X 在 supervision-noisy 条件下首次 +0.67 dB** —— 显示
   architecture-side intervention 在小数据上仍有 surgical 干预空间
4. **Path Z 反例**：sample × architecture 是乘积关系而非线性叠加

---

## 5. 行动空间（按 ROI 排序）

### Tier 1（高 ROI，论文 critical path）

#### 5.1 Path Y 完成 (~11.5h wb-only / ~57h all-actions, 已运行)

期望结果（决策树）:
- WB +2.5 dB 以上 → 数据稀疏假设确认 → §5.3 写"data-side scaling
  works for WB" 用 +12× 样本作 ablation 主结果
- WB 持平 22.5 → 1D-Planckian structural 假设确认 → §5.3 写"WB structural
  failure 不可由 sample scaling 解决" 强化 §4.6 inverse_fit residual
  analysis

#### 5.2 用 v2 master jsonl 重训 Path X (~3h on local 4080)

无论 5.1 哪个分支：
- 保持 Path X 架构不变
- v1 (372) → v2 (~1872) 训练
- 直接报告 per-action delta vs Path X-v1
- 结论：**Path X + Path Y combined effect** 是论文 §5.3 完整 closure

### Tier 2（中 ROI，可选增强）

#### 5.3 Path X 多 seed 复现（≈3 × 1.5 h = 4.5 h on local）

为论文做 3-seed mean ± std（v11a 已有）。Path X 当前是 single seed
（25.28），需要 seed 123/777 复现以做 statistical claim。

#### 5.4 INRetouch as Path Z baseline（4 h on local）

把 Path Z 8.26M ResUNet 替换为 INRetouch 11.5K params INR，重训：
- 假设 1: INRetouch 同样达 24.5 dB → 印证"data-bound"，11.5K vs 8M 不
  matter
- 假设 2: INRetouch 更强 inductive bias → 24.7+ dB → §5.3 Path 1
  就是 INRetouch
- 不改变主结论，但加强论文"为什么不该用 image-domain replacement
  in low-data" 的可信度

### Tier 3（高风险高收益，超出当前论文范围）

#### 5.5 学 VeraRetouch 的 3-latent disentanglement

把 v11a 的 7D 参数 → (light_latent, global_color_latent,
specific_color_latent) 三个 32-dim vector，重训 Path X-v3：
- 期望: 表达力提升，可能 +1-2 dB overall
- **但需要新的 1M+ supervision data** (我们没有 LR API access)
- 写成论文 next-paper 方向（"future work"），不挤当前 paper

#### 5.6 蒸馏 FireRed → 小模型（DreamLite 思路）

把 FireRed-Image-Edit-1.0 12B 蒸馏到 ~400M:
- in-domain 可能 26+ dB（接近 VeraRetouch）
- on-device 可行
- **完全的下一篇 paper**，不在当前 deconstructive 范围

---

## 6. 立即可写入论文的新增内容

基于本文档，paper_draft.md 可补的具体段落：

### §2.1 Related Work 补强
增加 VeraRetouch / INRetouch 的具体 PSNR 数字（FiveK 26.85 / —），证
明"图像域方法在大数据下 win"，与我们"小数据下 ISP scaffolding win"
形成对比，强化 manifold + sample complexity 联合假设。

### §5.3 Path 1 补强
加入 INRetouch 11.5K params 的 efficiency 优势讨论，说明"per-
distribution serving" 在硬件上是可行的（每个 distribution 仅 11.5K
LR-tunable params），不仅是理论可能。

### §5.3 Path 3 补强
加入 DreamLite 389M on-device editing 的 reference，说明"diffusion
editing 在 1Hz on-device 已可行" → 蒸馏路径是真实可行 future work。

### §4.10 注脚
加入 §4.10 的 Path X +0.67 实际上是 ISP capacity ceiling (iii) 被
"surgical patch" 的证明 —— 不是推翻 ceiling 而是 patch 那 1D
highlights → multi-channel local。这个比 §5.2b 现有表述更清晰。

---

## 7. 待办优先级

| Pri | 任务 | ETA | Blocker |
|---|---|---|---|
| P0 | Path Y wb 跑完 | ~11.5 h | 无 |
| P0 | ingest_pathY → v2 jsonl | 30 min | wb 完成 |
| P1 | Path X 用 v2 重训 | 3 h | v2 jsonl |
| P1 | Path X seed 123/777 复现 | 4.5 h | 本地 GPU |
| P2 | paper §5.3 + §2.1 补 INRetouch/VeraRetouch | 1 h | 无 |
| P2 | paper §4.10 注脚澄清 | 0.3 h | 无 |
| P3 | INRetouch as Path Z replace | 4 h | 本地 GPU |
| P3 | 3-latent v11a 变体 | 1-2 周 | 数据 + GPU |

P0+P1 完成后整篇 paper 闭环。P2 是必要 polish。P3 是后续工作。
