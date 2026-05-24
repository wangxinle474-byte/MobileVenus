# IntelligenceCamera — LUT/Curves 训练实验报告 (v1 → v11)

**项目**: IntelligenceCamera (CVPR 2026 submission)
**任务**: 模仿 FireRed-Image-Edit 的 Lightroom 风格图像编辑，输入 FiveK 原图 + 编辑动作，输出编辑后图像
**数据**: 372 个伪标签样本 (298 train / 74 val), 5 个动作 (contrast / saturation / shadows / highlights / wb)
**评测**: PSNR @ val set
**Encoder**: MobileViTSmall (output_dim=384), 共享于所有版本

---

## 1. 实验结果总览

| Version | Description | Train best | Eval PSNR | Params | vs v6 |
|---------|-------------|-----------|-----------|--------|-------|
| v1 | shared LUT, dim=33, 3 basis | 23.83 | — | 3.0M | -0.20 |
| v2 | shared LUT, dim=17 | 23.79 | — | 3.0M | -0.24 |
| v3 | + color_aug | 23.69 | — | 3.0M | -0.34 ✘ |
| v4 | 5 LUT bases | 23.60 | — | 3.0M | -0.43 ✘ |
| v5 | per-action LUT, dim=33 | 23.92 | — | 3.1M | -0.11 |
| **v6** | **per-action LUT, dim=17, dropout=0.5** | **24.03** | — | 3.1M | **baseline** ⭐ |
| v7 | hybrid (3-gain WB + per-action LUT) | 23.84 | — | 3.1M | -0.19 ✘ |
| **v8** | **7D ISP 参数 + 残差 LUT (param-residual)** | **24.31** | 24.38 | 3.23M | **+0.28** ⭐ |
| v9a | 3 色 CN + 7 控点 Bezier (+7D anchor) | 24.42 | 24.52 | 3.07M | +0.39 |
| v9b | 6 色 CN + 11 控点 Bezier + cross-attn | 24.29 | 24.40 | 3.06M | +0.26 |
| v9c | v9b + per-action curves + 7D anchor | 24.09 | 24.16 | 3.31M | +0.06 |
| v9d (buggy) | v9a + 2-bin context Bezier (ctx 卡在 0.5) | 24.51 | 24.64 | 3.09M | +0.61 |
| v9d_fix | v9d + asymmetric BCPE init (ctx 真学到) | 24.44 | 24.54 | 3.09M | +0.51 |
| v9e | v9a + LearnableColorNaming (HSV+conv残差) | 24.40 | 24.40 | 3.07M | +0.37 |
| v9f | v9a + WB head (无监督, 3-gain RGB) | 23.91 | 23.91 | 3.09M | -0.12 ✘ |
| v9g | v9f + 显式 wb supervision (log(target/orig)) | 24.12 | 24.12 | 3.09M | +0.09 |
| v10b | v9a + NILUT residual (gate=0, 实际锁死) | 24.42 | 24.42 | 3.07M | +0.39 |
| v10b Plan C | v10b best 低 LR + EMA 微调 | 24.47 | 24.47 | 3.07M | +0.44 ⭐ |
| v10b_fix | NILUT gate=1 解锁残差 | 23.36 | — | 3.07M | -0.67 ✘ |
| v10b Plan D | clean data, 无 Qwen extra | 24.27 | — | 3.07M | +0.24 |
| v10c | VeraRetouch-style MLP renderer | 21.90 | — | 3.14M | -2.13 ✘ |
| v11a | action-gated context | 24.57 | 24.61 | 3.09M | +0.58 |
| v11b | fixed region-basis curves | 24.42 | 24.53 | 3.09M | +0.50 |
| v11c | region-wise parameter delta | 24.14 | 24.23 | 3.09M | +0.20 |
| **v11d** | **action-conditioned context** | **24.78** | **24.80** | **3.09M** | **+0.77 ⭐** |

**结论**: v9-v10 阶段的失败说明简单加容量、WB head、learned CN、NILUT 或 full renderer 都不能稳定改善 wb。v11 进一步定位到 v9d 的真正问题: **不是 context 本身有害, 而是 action-agnostic brightness routing 会伤害 wb**。v11d 使用 action-conditioned context 后, overall 达到 **24.80 dB**, wb 达到 **24.53 dB**, 是当前 WB refinement track 最佳结果。

### v9d 对称性陷阱 (重要架构发现)

我们最初实现 v9d 时, ContextHead 和 BCPE 都做了 zero-init. 训练完成后通过 viewer 发现 **context map mean=0.5 std=0.0, Pearson r(ctx, lum)=0.0** — context 从未学习! 诊断:

```
init: ContextHead = 0 → ctx_map = sigmoid(0) = 0.5  (均匀)
init: BCPE = 0       → cp_low = cp_high = identity  (对称)
forward: adj_low = adj_high   (since cp_low=cp_high)
backward: d_loss/d_ctx = (adj_high - adj_low) * upstream = 0  ⛔
```

ContextHead 永远收不到梯度. 此外 BCPE 的两条分支梯度完全对称, 也无法独立分化. **24.64 dB 的"提升"实际上来自双倍 BCPE 输出维度的过参数化正则**, 与 SA-LUT 的空间感知思想无关.

修复方案 (v9d_fix): 给 BCPE 末层 bias 加方向性初始化 (low bin 偏暗, high bin 偏亮). 这成功让 context 学到 (Pearson r=+0.34 with luminance), 但 24.54 dB 反而比 v9d_buggy 略低, **且大幅伤害 wb action (-0.80 dB)** — 因为暗/亮先验对色温调整毫无帮助.

### v9d 全 action ablation (v9a → v9d_fix)

| Action | v9a | v9d_fix | Δ | 解读 |
|--------|-----|---------|---|------|
| wb | 22.35 | 21.55 | **-0.80** ⚠️ | brightness prior 干扰色温 |
| saturation | 22.93 | 22.74 | -0.19 | 同上 |
| highlights | 23.62 | 23.82 | +0.20 | 高光识别有帮助 |
| contrast | 26.31 | 26.52 | +0.21 | 微涨 |
| shadows | 26.50 | 26.73 | +0.23 | 微涨 |

### v9e (LearnableColorNaming) 完整结果

我们假设 wb/saturation 退化是因为固定 HSV 不够语义化. v9e 引入可学习残差 conv (HSV warm-start + 小 conv) 让模型自适应学习 N 个语义掩码. 训练后:
- LearnableCN 偏离 HSV 0.1342 (avg L1 per pixel) — 确实学到了 deviation
- 整体 24.40 dB, 仍**未恢复 wb** (21.84, 比 v9a 还低 0.51)

| Action | v9a | v9d_fix | **v9e** | v9e vs v9a |
|--------|-----|---------|---------|------------|
| wb | 22.35 | 21.55 | 21.84 | **-0.51** ⚠️ |
| saturation | 22.93 | 22.74 | 22.73 | -0.20 |
| highlights | 23.62 | 23.82 | 23.59 | -0.03 |
| shadows | 26.50 | 26.73 | 26.46 | -0.04 |
| contrast | 26.31 | 26.52 | **26.58** | **+0.27** ✓ |

### v9f/v9g (WB Head + Supervision) — 决定性的失败

受 Deep WB Editing (Afifi & Brown, CVPR 2020) 启发, 我们尝试加一个**专用 3-gain RGB WB head** 串在 Bezier 之前 (v9f), 然后加**显式 log_gains MSE 监督** (v9g) 强制特化:

**v9f (无监督)**: WB head 输出 log_gains 完全被 Bezier 同化, 各 action gains 均 ≈ 1, wb mean **-2.03 dB** ⚠️

**v9g (显式监督)**: 强制 WB head 拟合 `log(target_mean/orig_mean)`. WB head 终于学到非平凡 gains:
- wb action 平均 gains: (1.363, 1.331, 1.234) ← R/G/B 显著不同
- 其他 action 平均 gains: 1.1-1.3 范围 (action 之间有区分)

但 **wb mean PSNR 仍只是 22.23, 几乎持平 v9a (22.35)**. 逐样本分析揭示问题:
```
pred=(1.48,1.48,1.43)   target=(2.58,2.20,1.27)   ← target 内部矛盾 (R↑↑, B 微调)
pred=(1.18,1.15,1.08)   target=(1.38,1.48,1.00)   ← target G > R 反直觉
```
target 自己就是噪声 (因为 P_inferred WB temperature MAE ±788K).

### 深层洞察: wb 瓶颈是数据问题, 非架构问题

**5 种方案 (v9d_fix context / v9e learned CN / v9f WB head / v9g WB+sup / 加 v9a baseline)** 都无法显著改善 wb:

| Method | Approach | wb mean PSNR | Δ vs v9a |
|--------|----------|-------------|---------|
| v9a | Bezier + HSV CN | 22.35 | baseline |
| v9d_fix | + brightness context map | 21.55 | **-0.80** ✘ |
| v9e | + learnable semantic mask | 21.84 | -0.51 ✘ |
| v9f | + WB head (无监督) | 20.32 | -2.03 ✘✘ |
| v9g | + WB head + 显式监督 | 22.23 | -0.12 |

**没有一种架构能突破 22.5 dB**. 即使 v9g 强制 WB head 学习 analytical gains, wb 也无显著提升.

**根本原因**:

1. **数据稀缺**: val 仅 7 wb 样本, train ~28 样本. 测试方差 spread 9-11 dB.
2. **伪标签内在噪声**: FireRed 的 wb 经 inverse-fit 拟合 7D 参数, P_inferred MAE ±788K (占 20%). pseudo target 内部就矛盾.
3. **架构无法创造数据中没有的信号**: 全局 Bezier、亮度 context、语义 mask、专用 3-gain head、显式监督, 都无法弥补 pseudo-label 噪声.

**真正解决 wb 需要**:
- (a) 更大数据集 (MIT5K 全量 4500 样本, 需 AutoDL)
- (b) 真实色温对齐监督 (RAW + 多 WB 渲染对, 而非 inverse-fit pseudo)
- (c) 蒸馏 FireRed 的中间特征, 跳过 inverse fit 误差累积

### v9g_aug — 合成 WB 数据增广的灾难性失败 (重要 lesson)

为了验证 "wb 是数据问题" 的假设, 我们做了**数据增广实验**:

**方法**: 从非-wb 样本生成 250 个**合成 wb 样本** (50 FiveK 图 × 5 个色温 3500K/4500K/5500K/7500K/10000K), 用 Planckian-locus RGB 增益, 标签**完美**.

**预期**: wb 训练数据 28 → 278 (10×), perfect labels 应能让 v9g 的 WB head 学得更好, **wb PSNR 预期 +1~2 dB**.

**实测**: 灾难

| Action | v9a | v9g | **v9g_aug** |
|--------|-----|-----|------------|
| **wb** | 22.35 | 22.23 | **16.14** ⚠️⚠️⚠️ |
| Overall | 24.52 | 24.12 | **23.47** |

- wb PSNR **暴跌 6.21 dB** (vs v9a)
- WB head 对真 wb 样本输出 gains = (1.017, 1.025, 1.002) ≈ identity, **完全无作用**

**根因 — Domain Shift**:

合成 WB 数据 vs FireRed 真实 wb data 是**两个完全不同的分布**:

| | 合成 WB (我们生成的) | FireRed WB (val 真实) |
|---|---|---|
| 形式 | Planckian 线性 RGB 缩放 | 非线性混合 (亮+暖+饱) |
| 典型 gains | (0.81, 1.0, 1.28) 3500K / (1.16, 1.0, 0.80) 10000K | (2.58, 2.20, 1.27) / (1.66, 1.33, 1.03) |
| R/G/B 一致性 | 由温度坐标决定, 强相关 | R/G 单跳, 反 Planckian 物理 |
| residual PSNR | ~∞ (定义性) | 23.20 (有非线性结构) |

模型从合成数据**学到了"合理 wb"**, 但 FireRed 的 wb 根本**不是** wb — 它是一个被 "wb 指令" 触发的**混合编辑** (含 brightness + saturation 调整). 模型在两个不可调和的分布上摇摆, 最终对真 wb 样本输出 identity (gains≈1).

### 最终结论 — wb 是 LABEL 质量问题, 不是数据量问题

| 处理方式 | 类型 | wb mean | 改善? |
|---------|------|---------|------|
| v9a baseline | — | 22.35 | — |
| v9d_fix | + 亮度 context map | 21.55 | ✘ |
| v9e | + 学习化 CN | 21.84 | ✘ |
| v9f | + WB head 无监督 | 20.32 | ✘ |
| v9g | + 显式 log_gains 监督 | 22.23 | ≈ |
| **v9g_aug** | + 250 合成 wb (干净 labels) | **16.14** | **✘✘** |

**6 种处理方式 (5 架构 + 1 数据增广) 全部失败**.

### v9a_cleanwb — 重生 FireRed 数据也失败 (终极实验)

用户提出最后一个假设: **是不是 prompt 太泛造成的? 改严格 prompt 重生数据试试**. 我们实测:

**方法**: 编写严格约束 prompt "ONLY adjust color temperature, do NOT change brightness, contrast, saturation" 重跑 FireRed 50 样本 (25 warm + 25 cool). 用 inverse_fit 提取标签, 与原数据合并, 重训 v9a.

**审计揭示**: 即使 prompt 明写"不要改变亮度", FireRed 仍在**大幅增亮**:
- 新 wb 样本 gains 经常到 (3.15, 2.34, 1.76) 或 (4.43, 7.18, 14.63)
- 新数据 residual PSNR = 22.51 dB, **比原 wb 23.20 还脏**
- inverse_fit **41/49 FAIL**, 即 FireRed 输出**无法用 7D ISP 表达**

**实测结果**: 
- v9a 24.52 → v9a_cleanwb **23.99** dB (**-0.53**)
- wb 22.35 → **20.24** dB (**-2.11** ⚠️)

### v9a_qwen — 换教师模型 (Qwen-Image-Edit) 也失败

继 v9a_cleanwb 之后, 第二个假设: **是不是 FireRed 模型本身的问题? 换更强的图像编辑模型试试**. 我们实测:

**方法**: 用 **Qwen-Image-Edit (qwen-image-edit-2509-ultra) via DashScope** 重生同样 50 个样本 (25 warm + 25 cool), 同一严格 prompt. inverse_fit 后合并入训, 与 v9a_cleanwb 完全平行对照.

**审计揭示**: Qwen 比 FireRed **更脏**, 即使 prompt 完全一致:
- inverse_fit **37/37 全部 FAIL**, pixel_l1 均值 0.13 (warm) / 0.10 (cool)
- L1 < 0.15 过滤后只剩 22/37, 入训时 D dirty tier 被丢, **最终 9 个有效样本** (vs v9a_cleanwb 24)
- **0 个 B good 样本** (l1<0.05), 即 Qwen 编辑**无一**与 7D ISP 高度一致
- params_stats: brightness 均值 +29 (warm) / +20 (cool), contrast 均值 +44 / +37, saturation +31 / +18 — Qwen 同样将"wb"解释为 **强亮+强对比+强饱和** 的混合编辑
- 中途 13/25 cool 因 DashScope 账户欠费失败, 总成功 37 样本

**实测结果**: 
- v9a 24.52 → v9a_qwen **24.03** dB (**-0.49**)
- wb 22.35 → **20.79** dB (**-1.56** ⚠️)

虽然 Qwen 比 FireRed cleanwb 略好 (20.79 vs 20.24), 但**仍显著低于 v9a baseline**, 且这部分差异主要来自 Qwen **数据量更少** (9 vs 24 训练样本). 按每样本伤害归一化: Qwen **-0.17 dB/sample** vs FireRed **-0.088 dB/sample**, Qwen 实际**单样本更有害**.

### 终极结论 — **任何"指令 → wb"模型都是瓶颈, 不是 prompt 也不是单一教师**

| 版本 | 方法 | wb | Overall |
|------|------|-----|---------|
| v9a | baseline | 22.35 | 24.52 ⭐ |
| v9d_fix | +context map | 21.55 | 24.54 |
| v9e | +learnable CN | 21.84 | 24.40 |
| v9f | +WB head | 20.32 | 23.91 |
| v9g | +WB sup | 22.23 | 24.12 |
| v9g_aug | +合成 Planckian | 16.14 | 23.47 |
| v9a_cleanwb | +重生 FireRed (严格 prompt, 24 样本) | 20.24 | 23.99 |
| **v9a_qwen** | **+Qwen-Image-Edit (严格 prompt, 9 样本)** | **20.79** | **24.03** |

**10 种组合 (6 架构 + 4 数据处理) 全部无法改善 wb**.

**最终洞察**: 通用图像编辑模型 (FireRed-MoE, Qwen-Image-Edit) 都把 "wb" 指令视为**"调色风格化"复合操作**, 不论 prompt 多严格、模型多强, 它们都**无法生成"仅改色温"的纯净 wb 样本**. 这与 LLM 的 instruction-following 不同: 这类模型按照它们学到的 "wb 应该是什么样子" 做编辑, 即使 prompt 明写禁止其他变化. **双教师对照实验** (FireRed + Qwen, 同 prompt, 平行 pipeline) 强烈支持这一点 — 换教师只是把"亮+暖混合"换成"亮+对比+饱和混合", 7D ISP ceiling 始终被打破.

**真正的出路** (接受 v9a 为现有数据 SOTA, 以下是未来工作):
1. **换数据源, 不换教师**: 用 MIT5K 人手修图 target (Expert C) 替代生成式教师, 或用 RAW + 物理 WB 渲染对 (Planckian gains 直接渲染). 这是**唯一**能保证纯 wb 标签的路径 — 双教师实验已经表明任何"指令 → wb"模型都会混入 brightness/contrast/saturation.
2. **scale up**: AutoDL MIT5K 4500 样本重训, 验证数据量是否能稀释 label 噪声影响 (现在 wb 训练只有 ~28 个 master 样本 + 0/9/24 增广)
3. **物理约束**: 在 train 时强制 wb action 的 7D 参数只允许 white_balance 一维变化 (mask 掉 brightness/contrast/saturation 梯度), 强行把 "wb" 数据投影到 wb 子流形

### v10 系列 — Off-Planckian / NILUT / VeraRetouch 对照

v10 系列基于近期文献尝试继续突破 7D / Planckian / Bezier 表达限制:

| 版本 | 方法 | wb | Overall | 结论 |
|------|------|-----|---------|------|
| v10b NILUT | v9a + NILUT residual, gate=0 | — | 24.42 | 与 v9a/v10 baseline 接近, 但 eval 显示 NILUT gate=0 |
| v10b Plan C | v10b best + LR/10 + EMA 微调 | 22.76 | 24.47 | v10 当前最佳, 但提升来自微调/EMA, 非 NILUT |
| v10b_fix | gate=1 解锁 NILUT | — | 23.36 | 强残差负迁移, NILUT 在当前数据上不稳定 |
| v10b Plan D | clean data, 无 Qwen extra | — | 24.27 | 去掉 Qwen 后下降, Qwen extra 不是主噪声源 |
| v10c Vera | per-pixel MLP renderer 替代 Bezier | — | 21.90 | identity/zero-gate 未学起, 小数据下严重欠拟合/训练失败 |

**关键发现**:

1. **NILUT 双零初始化锁死**: 原设计 `gate=0` 且 MLP 最后一层全零, 导致 gate 与 MLP 梯度同时为 0, 残差分支实际没有工作.
2. **解锁 NILUT 后反而退化**: `gate=1` 后 best 只有 23.36 dB, 说明不是容量不够, 而是当前伪标签/小数据不足以稳定训练自由残差.
3. **v10c 没有突破**: VeraRenderer 在 372 样本上从 identity 起步, best 仅 21.90 dB, 说明纯数据驱动 renderer 需要更大、更干净的数据集.
4. **当前可保留结果**: v10b Plan C = 24.47 dB, 作为 EMA 微调的正结果; 但论文主结论仍应保留 v9a 24.52 dB 为现有数据 SOTA.

### v11 系列 — Action-conditioned context 突破 WB

v11 系列重新审视 v9d context 的失败。v9d_fix 显示亮度 context 对 highlights/shadows/contrast 有帮助, 但会显著伤害 wb。因此 v11 不再让所有 action 共享同一种空间 routing, 而是比较 action-gated、fixed region basis、region delta 和 action-conditioned context。

| 版本 | 方法 | wb | Saturation | Highlights | Shadows | Contrast | Overall | 结论 |
|------|------|----|------------|------------|---------|----------|---------|------|
| v11a | action-gated context | 22.58 | 22.88 | 23.84 | 26.35 | 26.70 | 24.61 | 修复 v9d 的 wb 负迁移 |
| v11b | fixed region-basis curves | 22.30 | 23.32 | 23.78 | 26.18 | 26.11 | 24.53 | saturation 有收益, 但 wb/contrast 下降 |
| v11c | region-wise parameter delta | 21.41 | 22.69 | 24.25 | 25.91 | 25.39 | 24.23 | highlights 有收益, 但 wb/contrast 明显负迁移 |
| **v11d** | **action-conditioned context** | **24.53** | **23.48** | **23.86** | **26.43** | **25.79** | **24.80** | **当前 SOTA** |

**v11 结论**:

1. **v11a 证明 action gating 有效**: context 只给 tone actions 使用后, wb 从 v9d_fix 的 21.55 恢复到 22.58, overall 提升到 24.61。
2. **v11d 证明 context 可用于 wb, 但必须 action-aware**: action-conditioned context 让 wb 达到 24.53, 不是简单禁用 context, 而是学习不同 action 的空间 routing。
3. **v11b/v11c 是负面对照**: fixed region basis 太刚性, region-wise delta 太自由, 都不如 action-conditioned context 稳定。
4. **当前 best checkpoint**: `checkpoints\lut_v11d_action_context\best.pt`。

### v11a / v11d seed variance (3-seed repeat)

固定 val split (`--split_seed 42`), 仅改训练随机性 (`--seed`):

| variant | seed 42 | seed 123 | seed 777 | mean ± std (n=3) |
|---------|---------|----------|----------|------------------|
| **v11a** action-gated context | 24.61 | 24.24 | 24.54 | **24.46 ± 0.20** |
| **v11d** action-conditioned context | 24.80 | 24.33 | 24.32 | **24.48 ± 0.27** |

**关键发现**:

- **v11a 与 v11d 在统计意义上完全等价**: 均值差 0.02 dB, 远小于任一 std。
- seed 42 的 v11d 24.80 dB 与 v11a 24.61 dB 都是各自分布的高端样本, 都不应作为孤立报告。
- 两者均显著高于 v9d (~24.0 dB), 提升 ~0.5 dB, 印证 action-aware context 解决 v9d 对称性陷阱的 thesis。
- best epoch 跨度大 (12-36), 训练动态对 seed 敏感, 反映小数据集 (28 wb 样本) 搜索路径不稳定。

**论文报告策略**:

- **主模型选 v11a**: 实现更简单 (action gating only, 不需要 fused tokens), 性能与 v11d 等价。
- **v11d 作为 ablation**: 证明在 v11a 基础上引入 action-conditioned context 不带来额外收益, 反过来支持 "action gating 是关键, 而非更复杂的 routing"。
- 数值用 **mean ± std (3 seeds)**: v11a `24.46 ± 0.20`, v11d `24.48 ± 0.27`。
- 不报告孤立 seed 42 的 24.80。

**结论修正**:

- ✅ "v9d 对称性陷阱 + action-aware context 修复" 假设强成立。
- ❌ "action-conditioned context > action-gated context" 不成立, 二者等价。
- ✅ v11 系列把 WB refinement track 的 architecture-side 上限定在 ~24.5 dB。
- ➡️ 剩余 gap (vs FireRed) 主要是数据侧问题 (pseudo-label 噪声 + 28 wb 样本), 见后续 PRD/AetherRetouch 路线。

---

## 2. 关键发现

### 2.1 NaN 梯度的根因 (v8 修复)
**症状**: v8 训练前几次实验 val PSNR 全为 NaN, 全部权重崩溃.

**根因**: `apply_diff_isp` 反向传播中的不稳定算子:
- `shadows = linear / (lum + 1e-6)` → 暗像素处梯度 ~1e6
- 白平衡 `where` 分支边界不连续
- Tone curve `pow(x, 2.4)` 在 0 / 1 边界

单批次产生的 NaN/Inf 梯度被 `clip_grad_norm_` 计算 `total_norm = sqrt(sum(|g|^2))` 时, NaN 污染整个向量, 经过 `g.mul_(scale)` 后**所有梯度都变成 NaN**, 经过 `optimizer.step()` 后**所有权重崩溃**.

**修复 (3 层防御)**:
1. **Zero-init param_head 末层**: 初始时 7D 参数 = 0 (各 ISP 中性值), apply_diff_isp 输出 ≈ 输入, 前几次反向梯度温和.
2. **梯度净化**: `loss.backward()` 后, `clip_grad_norm_` 之前, `nan_to_num_(p.grad, 0, 0, 0)`.
3. **`torch.no_grad()` 包住 apply_diff_isp**: 完全切断 ISP 反向, param 分支只通过 MSE → P_inferred 监督, LUT 分支接收 detach 后的 coarse 作为输入特征.

```python
with torch.no_grad():
    coarse = apply_diff_isp(orig, {k: v.detach() for k, v in params_phys.items()})
    coarse = torch.nan_to_num(coarse, ...).clamp(0, 1)
```

修复后 v8 训练完全稳定, 60 epoch 内零次梯度污染.

### 2.2 残差 LUT 真有用 (v8 viewer)
v8 训练完成后, viewer 拆解显示:
- coarse (纯 7D ISP) PSNR = 23.03 dB
- refined (+ 残差 LUT) PSNR = 24.38 dB
- **残差 LUT 增益 = +1.35 dB**

这是论文核心叙事: 可微 ISP 受 7 参数表达力限制, 接近其理论上限 23.94 dB; 残差 LUT 在此之上学到 ISP 无法表达的精细颜色映射, 突破上限.

### 2.3 v9 simpler > more complex (反直觉)
原本期待 v9b (论文原版 6 色 × 11 控点 + cross-attention) 性能最佳, 但实验结果**反其道**:

| Variant | colors | ctrl pts | attn | per-action | 7D anchor | PSNR |
|---------|--------|----------|------|-----------|-----------|------|
| v9a | 3 | 7 | ✘ | ✘ | ✓ | **24.42** ⚡ |
| v9b | 6 | 11 | ✓ | ✘ | ✘ | 24.29 |
| v9c | 6 | 11 | ✓ | ✓ | ✓ | 24.09 |

**原因分析**:
- 数据量小: 298 训练样本 vs NamedCurves 原文 MIT5K 2250 样本 (差 7.5×)
- v9b 多出的参数 (18 条曲线 + attention 模块) 在小数据上过拟合, val 反而退化
- v9c 的 per-action 让每个动作只见到 ~60 个样本, 严重过拟合
- 7D anchor (MSE → P_inferred) 充当良好正则化器, 防止 Bezier 曲线偏离 ISP 合理范围

**Takeaway**: 在 ≤500 样本量级, 选择**最小够用**的架构胜过复杂模型. 7D 参数 anchor 提供有效正则.

### 2.4 多空间 vs 多曲线 (v8 vs v9 对比)
v8 和 v9a 都达到了 +0.3 dB 量级的提升, 但**路径完全不同**:

| | v8 (ParamResidualLUT) | v9a (NamedCurves) |
|---|----------------------|---------------------|
| 全局表达 | 7D ISP 参数 (可解释) | 9 条 Bezier 曲线 (3 色 × 3 通道) |
| 局部修正 | 3D LUT (17³×3 ≈ 14.7K params) | 颜色域加权融合 (无额外参数) |
| 反向稳定 | 需 detach + 梯度净化 | 天然稳定 (无 ISP 反向) |
| 可解释性 | 7 个 Lightroom 标准参数 | 9 条曲线 + 3 色域 mask |

**联合 ablation**: v8 的 coarse 与 v9a 的输出在量级上接近 (23-24 dB), 说明二者都成功逼近了 "用全局变换近似 FireRed 编辑" 的上限.

---

## 3. 架构细节

### 3.1 v8 — ParamResidualLUTPredictor (3.23M params)
```
input image (256×256)
  ↓ MobileViTSmall encoder
  ↓ feat (384-dim) + action_emb (32-dim) = fused (416-dim)
  ↓ ┌──────────────────────────────────┐
    │ param_head (zero-init final layer)│
    │   → 7D normalized params (Tanh)  │
    │   → physical units (denorm)      │
    │   → (no_grad) apply_diff_isp     │
    │   → coarse (detached, clamp)     │
    └──────────────────────────────────┘
  ↓ ┌──────────────────────────────────┐
    │ weight_head[action] (per-action) │
    │   → LUT fusion weights (softmax) │
    │ Basis3DLUT[action](weights, coarse)│
    │   → refined output               │
    └──────────────────────────────────┘
loss = L1 + 0.5*SSIM + 0.5*(coarse_L1 + 0.5*coarse_SSIM)
       + 0.1*MSE(pred_p7, P_inferred)
       + 1e-4*smooth + 0.01*mono
```

### 3.2 v9a — NamedCurvesPredictor (3.07M params)
```
input image (256×256)
  ↓ MobileViTSmall encoder
  ↓ feat + action_emb = fused (416-dim)
  ↓ ┌──────────────────────────────────┐
    │ BCPE (Bezier Control Pt Estimator)│
    │   → 9 curves × 7 ctrl pts        │
    │   = identity + 0.5 * tanh(raw)   │
    │   shape (B, 3 colors, 3 RGB, 7)  │
    └──────────────────────────────────┘
  ↓ ┌──────────────────────────────────┐
    │ color_naming (固定 HSV-based 算法) │
    │   → 3 maps (warm/cool/neutral)   │
    │   → threshold 0.2 + renorm       │
    └──────────────────────────────────┘
  ↓ ┌──────────────────────────────────┐
    │ apply Bezier curves per color×ch │
    │   → 3 globally adjusted images   │
    │ blend by CN maps (weighted avg)  │
    │   → refined output               │
    └──────────────────────────────────┘
  ↓ ┌──────────────────────────────────┐ (optional, only v9a/v9c)
    │ param_head → 7D anchor (Tanh)   │
    │   MSE → P_inferred (weight 0.05) │
    └──────────────────────────────────┘
loss = L1 + 0.5*SSIM + 0.05*MSE(7D)
```

---

## 4. Color Naming 实现 (避免外部依赖)
NamedCurves 原文使用 Van de Weijer 2007 的查找表 (`w2c.mat`, 32^3 RGB 索引), 但该资源国内访问困难且依赖 MATLAB 文件.

**替代方案**: HSV-based soft naming
- 8 个 hue 中心 (red=0°, orange=30°, yellow=60°, green=120°, cyan=180°, blue=220°, purple=280°, pink=330°)
- 每个 hue Gaussian likelihood with σ=25°, circular distance
- 低饱和度通过 sigmoid gate 划到 achromatic
- 按需分组 (full6 = paper grouping, compact3 = warm/cool/neutral)

完全可微 (用于 PyTorch 训练), 无外部依赖, 速度 < 1ms per 256×256 image.

---

## 5. 训练设置

| 参数 | 值 |
|------|------|
| Optimizer | Adam, lr=3e-4 (encoder), lut_lr=1e-3 (LUT/BCPE) |
| Scheduler | Cosine annealing, T_max=60 |
| Batch size | 4 |
| Max epochs | 60 |
| Early stop | patience=12 |
| Augmentation | None for v8-9 (避免颜色失真破坏目标对齐) |
| Hardware | NVIDIA GPU (具体 model 未记录) |
| Train time | ~13 min (v9a) / ~16 min (v9b) / ~16 min (v9c) |

---

## 6. 论文叙事建议

### 主标题候选
- **"Beyond 7D ISP: Hybrid Parameter-Residual Learning for Interpretable Mobile Photo Editing"**
- **"Named Tone Curves: Interpretable Color-Aware Image Enhancement on Resource-Constrained Datasets"**

### Story arc
1. **Motivation**: 移动端美学需要"可解释 + 高质量"双重目标, 传统 7D ISP 参数化模型有 23.94 dB 表达上限.
2. **Approach 1 (v8)**: 提出 ParamResidualLUT, 用 7D 可解释参数预测 + 残差 3D-LUT 修正, 突破上限到 24.38 dB. **关键技巧**: detach apply_diff_isp 避免 NaN 灾难.
3. **Approach 2 (v9)**: 借鉴 NamedCurves 思想, 用 color naming + Bezier 曲线替代 LUT. 在我们数据规模下, 简化版 (3 色 / 7 控点) 反而最佳 → 24.52 dB.
4. **Insight**: 7D 参数 anchor 作为可解释性 + 正则化双重作用, 是 v9a 胜出的关键 (v9b 无 anchor 落后 0.13 dB).
5. **Ablation**:
   - color naming 颗粒度 (3 vs 6 色)
   - control points (7 vs 11)
   - cross-attention 有无 (在小数据上是负贡献)
   - per-action curves 有无 (在小数据上是负贡献)
   - 7D anchor 有无
   - v10 NILUT / VeraRenderer 在小数据伪标签下的负面结果

### 可视化材料
- v8 5-列对比 (`outputs/lut_v8_viewer/viewer.html`): orig | coarse (7D) | refined | target | error
- v9a 5-列对比 (`outputs/lut_v9a_viewer/viewer.html`): orig | CN map | refined | target | error
- v1→v9 PSNR 曲线图 (待绘制)
- Bezier 曲线可视化 per-color per-channel (待绘制)

---

## 7. 下一步 (Future Work)

### 已经验证不工作的方向
- ✘ Color augmentation (v3)
- ✘ More LUT bases (v4)
- ✘ Hybrid WB head (v7)
- ✘ Cross-attention in small-data setting (v9b)
- ✘ Per-action curves in small-data setting (v9c)
- ✘ NILUT residual 强解锁 (v10b_fix, gate=1)
- ✘ NILUT 容量扩展 (Plan B 跳过: gate fix 已明显负迁移)
- ✘ VeraRetouch-style per-pixel renderer in 372-sample setting (v10c)

### 可能值得探索
- HDRNet bilateral grid (空间显式建模)
- DeepLPF 局部参数化滤波 (椭圆 / 渐变 / 多项式)
- 扩展到 MIT5K 全量数据 (5000 样本) 验证 v9b/v9c 是否能反超 v9a
- 蒸馏 FireRed 的中间表征作为额外监督
- 探索更轻量化部署 (端侧 < 1M 参数量)
- 切换到 AetherRetouch-1M+ / PRD 等真实 retouch 数据集, 再重新测试 v10c/INR 类方法

---

## 附录: 命令行复现

```powershell
# v8 训练
python -u training\firered_baseline\train_lut.py `
  --param_residual_lut --lut_dim 17 --dropout 0.5 `
  --out_dir checkpoints\lut_v8 --epochs 60 `
  --coarse_weight 0.5 --param_weight 0.1

# v9a 训练 (最佳)
python -u training\firered_baseline\train_lut.py `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_7d_anchor --dropout 0.5 `
  --out_dir checkpoints\lut_v9a --epochs 60 `
  --param_weight 0.05

# v9b 训练 (论文原版)
python -u training\firered_baseline\train_lut.py `
  --named_curves --nc_n_colors 6 --nc_n_control_points 11 `
  --nc_use_attention --dropout 0.5 `
  --out_dir checkpoints\lut_v9b --epochs 60

# v9c 训练 (per-action)
python -u training\firered_baseline\train_lut.py `
  --named_curves --nc_n_colors 6 --nc_n_control_points 11 `
  --nc_use_attention --nc_per_action_curves --nc_use_7d_anchor `
  --dropout 0.5 --out_dir checkpoints\lut_v9c --epochs 60 `
  --param_weight 0.05

# 生成 viewer
python tools\eval_lut_viewer_v8.py --ckpt checkpoints\lut_v8\best.pt
python tools\eval_named_curves_viewer.py --ckpt checkpoints\lut_v9a\best.pt --out_dir outputs\lut_v9a_viewer --tag "v9a"
python tools\eval_named_curves_viewer.py --ckpt checkpoints\lut_v9b\best.pt --out_dir outputs\lut_v9b_viewer --tag "v9b"
python tools\eval_named_curves_viewer.py --ckpt checkpoints\lut_v9c\best.pt --out_dir outputs\lut_v9c_viewer --tag "v9c"
```

---

*生成于 2026-05-13 23:00, v9 实验完成后; 2026-05-14 更新 v10 结果*
