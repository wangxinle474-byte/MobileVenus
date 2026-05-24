# IntelligenceCamera 全版本实验记录

> 更新时间: 2026-05-05 (补记 v12-v14 + compare_5 编辑对比 baseline)
> 项目目标: 基于语义理解的智能 ISP 参数预测，最终目标 MUSIQ-AVA 8.5+
> 评估数据: MIT-Adobe FiveK 验证集 (50张) + compare_5 (编辑模型对比)
> 训练平台: AutoDL (RTX 5090)

---

## 版本演进总览

```
Baseline → v1-v3(语义蒸馏探索) → v4-v5(精度优化) → v6(Venus NL) → v7(退化增强)
    → v8(Stage C文本条件) → v9(美学感知) → v10(端到端E2E) → v11(RefineNet V1)
    → v12(RefineNet V4 512+MUSIQ) → v13(多尺度+EMA+WarmRestarts)
    → v14(AesExpert 高质量数据混训)
```

| 版本 | 核心改进 | PSNR | MUSIQ-AVA (512) | 状态 |
|------|---------|------|-----------------|------|
| Baseline | MobileViT + 8参数 | 32.05 | — | ✅ |
| v5 | Expert C 单专家精准监督 | **34.11** | — | ✅ |
| v6 | Venus NL 语义对齐 + 6参数 | 32.30 | 4.413 | ✅ |
| v7 | 退化增强 + 对比学习 | 25.99 | 4.425 | ✅ |
| v8 | Stage C 文本条件融合 | — | **4.446** | ✅ |
| v9 | 美学感知微调 (MUSIQ 监控) | — | 4.418 | ✅ |
| v10 | 端到端 image_loss 通过 diff_isp | — | 完成 | ✅ |
| v11 | RefinementNet V1 + MUSIQ loss | — | ~3.71 (baseline) | ✅ |
| v12 | RefineNet V4 + 512训练 + MUSIQ主导 (16M) | — | **4.15** | ✅ |
| v13 | 多尺度 MUSIQ + EMA + WarmRestarts | — | **4.20** | ✅ |
| v14 | AesExpert 高质量数据混训 ParamModel | — | 计划中 | 🔄 |

### 天花板参考

| 对象 | MUSIQ-AVA (512) | 说明 |
|------|-----------------|------|
| 原图 (未修图) | 4.515 | 基线 |
| **Expert C (Lightroom 专家修图)** | **4.754** | 当前天花板 |
| 目标 | **8.5+** | 超越专家 |

---

## Baseline: FiveK 8参数训练

### 改进内容
- MobileViT-Small 视觉编码器 (SE + FPN)
- 8 参数预测: EV, WB, 对比度, 阴影, 高光, 饱和度, 亮度, 鲜艳度
- diff_isp 可微渲染管线 (Lightroom 风格)
- 5 专家共识加权损失

### 改进原因
- 建立端到端可微的 ISP 参数预测基线
- 利用 FiveK 数据集的多专家标注

### 效果
- **PSNR = 32.05**, SSIM = 0.9269
- 权重: `checkpoints/fivek_8param/best.pt`

---

## Distill v1-v3: 语义蒸馏探索

### 改进内容
- v1: COCO MiniLM embedding → Stage A 语义对齐
- v2: Stage B 参数微调，使用语义空间指导
- v3: 架构微调

### 改进原因
- 引入 Venus 美学模型的语义知识，指导参数预测
- 探索"语义理解 → ISP 参数"这条路线

### 效果
- v2: PSNR = 33.15 (+1.10 dB), SSIM = 0.9311
- v3: PSNR = 32.85

---

## Distill v4: FiveK Stage A

### 改进内容
- Stage A 改用 FiveK 域内图片训练语义对齐（之前用 COCO）
- 同域分布使语义空间更贴合 ISP 任务

### 改进原因
- COCO 图片和 FiveK 的分布差异大，域内对齐应该更有效

### 效果
- PSNR = 33.11 (+1.06 dB), SSIM = **0.9326**

---

## Distill v5: Expert C 单专家精准监督

### 改进内容
- 放弃多专家共识，只用 Expert C 作为 GT
- 精准单目标优化

### 改进原因
- 多专家标注存在分歧，模型学到的是"平均值"
- 单专家提供一致、明确的优化方向

### 效果
- **PSNR = 34.11 (+2.06 dB)**, **SSIM = 0.9449** ★ 技术还原最优
- 证明减少标注噪声能显著提升技术指标

---

## Distill v6: Venus 自然语言语义对齐

### 改进内容
- **Stage A**: 用 Venus 美学模型生成的自然语言描述做语义对齐（替代简单 embedding）
- **Stage B**: 简化为 6 参数（去掉亮度/鲜艳度，它们与其他参数冗余）
- 数据: 5K COCO (Venus美学文本) + 4500 FiveK 图片

### 改进原因
- Venus 自然语言描述包含更丰富的美学语义信息
- 8→6 参数简化减少模型自由度，降低过拟合风险
- 为后续 Stage C (文本→参数) 奠定语义基础

### 训练细节
- Stage A: 30 epochs, lr=5e-4, batch=32, cos_sim≈0.91-0.95
- Stage B: 50 epochs, lr=1e-4, batch=16, 先冻结 backbone 再解冻
- 损失: param_loss 为唯一梯度来源，image_loss 仅监控 (diff_isp 梯度不稳定)

### 效果
- PSNR = 32.30, best_val = 0.3524
- Venus overall = 5.00 (-0.10 vs 原图)
- MUSIQ-AVA (512) = **4.413**
- **结论**: 技术指标略优于 Baseline，但美学未超过原图

### 关键代码
- `training/legacy/train_v6_stage_a.py`: Stage A 训练
- `training/legacy/train_v6_stage_b.py`: Stage B 训练
- Checkpoint: `/root/autodl-tmp/checkpoints/distill_v6/`

---

## Distill v7: 退化增强 + 对比学习

### 改进内容
1. **合成退化增强 (方案1)**: 训练时随机退化输入图 (EV偏移, WB偏移, 对比度降低, 阴影压暗, 高光过曝, 饱和度降低)，同时调整 GT 参数补偿
2. **双路对比学习 (方案4)**: 原图+退化图同时训练，语义 embedding 一致性约束
3. GPU 批量退化 `batch_degrade()`: 全参数覆盖，效率高

### 改进原因
- v6 对退化输入不鲁棒——手机实拍常有曝光/白平衡偏差
- 对比学习让模型学到"同一场景无论质量如何，语义理解应一致"
- 增强泛化能力，为实际部署做准备

### 训练细节
- 复用 v6 Stage A 权重
- 50 epochs, lr=1e-4, batch=16
- consistency_weight=0.1, degrade_prob=0.5
- 损失: p_loss_orig + p_loss_deg + 0.1 * consistency_loss

### 效果
- PSNR = 25.99 (暴跌 6dB, 因参数偏离 Expert C)
- Venus overall = **5.38 (+0.28 vs 原图)** ★ 美学最优
- MUSIQ-AVA (512) = **4.425**
- **退化修复**: 9 个 NR-IQA 指标中 8 个优于 v6

### 核心发现
- **PSNR ≠ 美学**: PSNR 暴跌但 Venus 大幅上升
- **退化增强有效**: v7 退化修复能力远超 v6
- Venus/TANet 不适合评估 ISP 级退化

### 关键代码
- `training/legacy/train_v7_stage_b.py`: 含 `batch_degrade()` 和对比学习
- Checkpoint: `/root/autodl-tmp/checkpoints/distill_v7/`

---

## Distill v8: Stage C 文本条件融合

### 改进内容
1. **Stage C 架构**: TextConditionedModel — 接受文本+图像联合输入
   - TextEncoder: 轻量 Transformer (vocab=5000, embed=128, hidden=256, 2层4头)
   - FiLM 融合: 文本 embedding 通过 FiLM 层调制视觉特征
   - 条件化 Decoder: 基于融合特征预测 ISP 参数
2. **口语化指令数据**: 生成训练数据支持口语输入 (如"有点黄，清爽一点")
3. 冻结 Stage B backbone，只训练 Stage C 新增模块

### 改进原因
- 项目核心创新: 用户用自然语言控制 ISP 参数
- Stage C 是连接"语义理解"和"参数预测"的关键
- FiLM 融合轻量高效，适合移动端部署

### 训练细节
- 基于 v7 Stage B (美学最佳) 构建
- 指令数据: FiveK + PPR10K 生成的口语化指令
- 冻结 vision_encoder + semantic_projector
- 训练 text_encoder + fusion + cond_decoder

### 效果
- MUSIQ-AVA (512) = **4.446** ★ 当前参数预测最佳
- 支持口语化文本条件输入

### 关键代码
- `training/legacy/train_stage_c.py`: Stage C 训练
- `tools/data/data_prep/generate_instruction_data.py`: 指令数据生成
- `training/text_condition/`: TextEncoder, FiLM, TextConditionedModel
- Checkpoint: `/root/autodl-tmp/checkpoints/distill_v8/`

---

## Distill v9: 美学感知微调

### 改进内容
1. **MUSIQ 美学监控**: 训练过程中用 MUSIQ-AVA 实时评估渲染图美学分
2. **增强版 diff_isp**: 
   - 参数化三次 S 曲线 (tone curve) 替代简单对比度
   - Clarity (局部对比度增强, unsharp mask on luminance)
   - 改进阴影/高光: 平滑 sigmoid mask + 色彩比例保持
3. 从 v7 checkpoint 微调

### 改进原因
- v6/v7 训练完全没有美学信号参与
- diff_isp 渲染质量是瓶颈: 简单全局调整无法产生高美学分
- 增强 diff_isp 能力以缩小与 Expert C 的差距

### 训练细节
- 30 epochs, lr=5e-5, batch=16
- param_loss 仍为唯一梯度来源 (MUSIQ 只监控, no_grad)
- 原因: diff_isp 梯度链不稳定 (power 操作导致 NaN)
- aesthetic_every=5 (每5 batch 计算一次 MUSIQ)

### 效果
- best_param = 0.0045
- MUSIQ-AVA (512) = **4.418** (略低于 v8)
- **结论**: 仅监控不参与梯度，美学分未显著提升

### 关键发现
- MUSIQ 只做监控不做优化 → 效果有限
- diff_isp 增强 (tone curve, clarity) 提升了渲染质量但不足以超越专家
- 必须让美学信号真正参与梯度才能突破

### 关键代码
- `training/main/train_v9_aesthetic.py`
- `training/aesthetic_loss.py`: AestheticLoss 封装
- `models/diff_isp.py`: 增强版 (tone curve, clarity, shadows/highlights)

---

## 评估分辨率问题 (重要发现)

### 问题
- 早期评估在 224×224 渲染后上采样到 512×512 给 MUSIQ
- 上采样引入模糊伪影，人为压低 MUSIQ 分数

### 修复
- 改为 224×224 预测参数 → 512×512 渲染 → MUSIQ 评估
- ISP 参数与分辨率无关，可以在任意分辨率渲染

### 效果对比

| 版本 | 224 渲染 MUSIQ | 512 渲染 MUSIQ | 提升 |
|------|---------------|---------------|------|
| v6 | ~3.7 | 4.413 | +0.7 |
| v7 | ~3.7 | 4.425 | +0.7 |
| v8 | ~3.8 | **4.446** | +0.6 |
| v9 | ~3.7 | 4.418 | +0.7 |

### 关键代码
- `evaluate/eval_hires.py`: 高分辨率评估脚本

---

## Expert C 天花板评估

### 目的
确定 MUSIQ-AVA 的理论上限——Lightroom 专家修图能到多少分？

### 方法
- 50 张评估图，取 Expert C 专家修图版本
- Resize 到 512×512，用 MUSIQ-AVA 评分

### 结果

| 对象 | MUSIQ-AVA | vs 原图 |
|------|-----------|---------|
| 原图 | 4.515 | — |
| **Expert C** | **4.754** | **+0.239** |
| v8 (最佳模型) | 4.446 | -0.069 |

### 关键发现
- Expert C 天花板 = **4.754**
- 我们最佳模型 v8 = 4.446，**比原图还低 0.069**
- diff_isp 渲染反而降低了图像质量
- 距天花板 = **-0.308**
- **结论**: 6 个全局 ISP 参数的上限就在 ~5 分，要到 8.5+ 需要根本性变化

### 关键代码
- `evaluate/eval_ceiling.py`: 天花板评估脚本

---

## Distill v10: 端到端图像损失训练 (进行中)

### 改进内容
1. **image_loss 真正参与反向传播**: L1 + SSIM (rendered vs Expert C) 梯度流过 diff_isp
2. **可微 denorm**: 参数反归一化保持梯度链完整
3. **NaN 保护**: 检测渲染/梯度 NaN 并跳过
4. **Warmup**: 前 5 epoch 只用 param_loss，稳定后加入 image_loss
5. 从 v8 (MUSIQ 最佳) 开始

### 改进原因
- v9 证明: 美学信号只监控不参与梯度 → 效果有限
- v6/v7/v8 的 image_loss 都是 no_grad → 模型从未"看到"渲染结果
- 端到端训练让模型学习"什么参数能产生接近 Expert C 的渲染结果"

### 训练细节
- 50 epochs, lr=3e-5, batch=16
- 损失: param_weight=1.0, img_weight=5.0, ssim_weight=1.0
- grad_clip=0.5 (比之前更激进)
- warmup_epochs=5 (前5 epoch 只有 param_loss)
- 保存指标: **val_img** (图像质量优先, 而非 val_param)

### 当前状态 (Epoch 4/50, WARMUP 阶段)
- val_param=0.0054, val_img=0.1060, val_aes=3.79
- **NaN = 0** (梯度完全稳定)
- Epoch 6 将进入 E2E 阶段

### 关键代码
- `training/main/train_v10_e2e.py`

---

## Distill v11: RefinementNet 精修 (进行中)

### 改进内容
1. **RefinementNet**: 轻量图像精修网络 (~130K 参数)
   - MultiScaleFeature: 1×1, 3×3, 5×5, 7×7 并行卷积
   - EnhanceBlock: Conv + InstanceNorm + ChannelAttention + SpatialAttention + 残差
   - 全局上下文调制 (global context modulation)
   - 残差输出 (初始 = identity, 逐步学习增强)
2. **两阶段训练**:
   - Phase 1 (前20 epoch): L1 + SSIM vs Expert C (学基础增强)
   - Phase 2 (后40 epoch): + MUSIQ loss (直接最大化美学分, 目标 8.5+)
3. **冻结参数模型**: 用 v8 参数模型 + diff_isp 粗渲染，RefinementNet 学精修

### 改进原因
- **根本问题**: 6 个全局 ISP 参数无法达到 8.5+，即使 Expert C 也只有 4.754
- RefinementNet 能学到全局参数无法做的事:
  - 局部锐化 & 细节增强
  - 色彩分级 (近似 learned 3D LUT)
  - 自适应对比度
  - 去噪 / 去伪影
- Phase 2 直接用 MUSIQ 梯度优化 → 生成MUSIQ认为"美"的图像

### 架构
```
原图 → param_model(冻结v8) → diff_isp(粗渲染) → RefinementNet(训练) → 高质量输出
                                                       ↑
                                               Phase2: MUSIQ 梯度
```

### 训练细节
- 60 epochs, lr=3e-4, batch=16
- Phase 1: l1_weight=1.0, ssim_weight=0.5
- Phase 2: + musiq_weight=0.05, target=8.5
- MUSIQ 模型参数冻结, 但允许梯度流过 (用于优化 RefinementNet)

### 当前状态 (Phase 1, early batches)
- total=0.1686, l1=0.0795, ssim=0.1782, MUSIQ=3.71
- RefinementNet: 129.6K 参数 (极轻量)

### 关键代码
- `models/refinement_net_v4.py`: RefinementNet 最新架构 (V1-V3 已归档删除)
- `training/main/train_v11_refine.py`: 训练脚本

---

## Distill v12: RefinementNet 大模型 + 512 训练 + MUSIQ 主导

### 改进内容
1. **RefinementNetV2**: U-Net 编解码结构, ~1.9M 参数 (v11=130K, 15x)
   - 64 base channels, 编码器下采样 → 大感受野 → 解码器上采样
   - Skip connections + CBAM 双路注意力 + 全局色彩调制
2. **512×512 训练** (v11=224, MUSIQ 在 512 信号更强)
3. **Phase 2 MUSIQ 直接最大化**: 损失 = `-musiq_score.mean()` (v11 用 hinge loss)
4. **CLIPIQA+ 辅助**: 自然度约束 (weight=0.1)
5. **梯度累积**: accum_steps=8, 等效 batch=16

### 架构演进
```
原图(224) → param_model(冻结v8) → diff_isp(512渲染) → RefinementNet(训练) → 高质量输出
                                                              ↑
                                                     Phase2: -MUSIQ + CLIPIQA 梯度
```

### 超参调优阶段 (V2, base_ch=32, 1.91M)

| 轮次 | 配置变化 | val_MUSIQ | 分析 |
|------|----------|-----------|------|
| 1 | musiq_w=1.0, hinge loss (target=5.5) | 4.10 | Hinge 梯度弱 |
| 2 | 改为 -score loss, lr=5e-4, grad_clip=5.0 | 4.12 | 直接最大化有效 |
| 3 | musiq_w=2.0, 去掉 L1/SSIM, accum=8 | 4.15 | CLIPIQA 未加载 |
| 4 | +CLIPIQA 0.1 (手动上传权重) | 4.15 | CLIPIQA 生效但只跑3ep |
| 5 | musiq_w=5.0, lr=1e-4, 精细调整 | 4.15 | 10ep 无变化, **确认为 V2 天花板** |

**结论**: 超参调优对 V2 无效, 4.15 为架构容量上限

### 架构升级阶段

#### V3: 3级编解码 (8.93M) ✅ 已完成
- **架构**: RefinementNetV3, base_ch=48, 3级编解码器 (/8 下采样)
- **改进**: 瓶颈层 3×ResBlock, 仿射色彩调制 (scale+shift)
- **配置**: musiq_w=5.0, clipiqa_w=0.1, lr=3e-4, phase1=5ep, epochs=120
- **最终结果**: val_MUSIQ = **4.21**, val_CLIP = 0.467, best_l1 = 0.0713
- **关键发现**: **架构升级 (+0.06) 远比超参调优 (+0.00) 有效**

#### V4: 双分支架构 (待训练, ~16M)
- **架构**: RefinementNetV4, base_ch=64
  - **全局色彩分支**: 自适应 3×3 色彩矩阵 + per-channel gamma 曲线
  - **局部细节分支**: 3级 U-Net 解码器产生空间细节残差
- **设计动机**: MUSIQ-AVA 评价全局美感 + 局部质量, 分支专精化优化
- **预期**: val_MUSIQ ≥ 4.30

### 关键经验
1. **架构升级 >> 超参调优**: V2 调5轮卡在 4.15, V3 换架构直接 4.20
2. **MUSIQ 直接最大化**: `-score.mean()` 比 hinge loss 更有效
3. **去掉 L1/SSIM 约束**: Phase2 纯 MUSIQ 让梯度更集中
4. **模型容量决定天花板**: 1.91M→4.15, 8.93M→4.20, 16M→?

### 关键代码
- `models/refinement_net_v4.py`: V4 双分支架构 (16M, 保留最新版; V2/V3 已归档删除)
- `training/main/train_v12_refine_hd.py`: 统一训练脚本

---

## Distill v13: 多尺度 MUSIQ + EMA + Warm Restarts

### 改进内容
1. **多尺度 MUSIQ Loss**: 512 全图 + 384 随机裁剪，梯度信号更丰富
2. **纯 MUSIQ 优化**: Phase2 去掉 L1/SSIM/CLIP，全部梯度集中在 MUSIQ
3. **EMA** (Exponential Moving Average): decay=0.999，验证时用 EMA 模型
4. **CosineAnnealingWarmRestarts**: T_0=50，多次逃出局部最优
5. **musiq_weight=8.0** (V3 用 5.0，更激进)
6. 200 epochs (V3 用 120)
7. 随机水平翻转增强

### 改进原因
- V12-V3 在 Ep111 后 MUSIQ 停滞 (4.21)
- 单尺度 MUSIQ 梯度信号不够丰富，容易陷入局部最优
- EMA 平滑权重抖动，WarmRestarts 提供多次重启机会

### 训练细节
- 基于 V12-V3 架构 (RefinementNetV3, 8.93M, base_ch=48)
- 冻结 V8 ParamModel + diff_isp 粗渲染
- batch=2, accum=8, eff_batch=16
- lr=3e-4, CosineAnnealingWarmRestarts T_0=50

### 当前状态
- AutoDL 训练就绪 (`training/legacy/train_v13_multiscale.py`)
- 目标: val_MUSIQ ≥ 4.30

### 关键代码
- `training/legacy/train_v13_multiscale.py`: 训练脚本 (含 EMA, 多尺度, WarmRestarts)
- `models/refinement_net_v3.py`: V3 架构 (复用)

---

## Distill v14: AesExpert 高质量增强数据混训 ParamModel (计划中)

### 改进内容
1. **AesExpert MLLM 美学打分**: 用 AesExpert (LLaVA-1.5-7B 微调) 对 22394 增强数据重打分
2. **双门槛过滤**: AesExpert ≥ 7.0 AND AADB ≥ 4.5 → 约 15% 保留 (~3400 高质量样本)
3. **混合训练**: FiveK Expert C (4500 条) + AesExpert 高质量增强 (~3400 条)
4. **DistillParamModel**: 复用 V6 Stage A 权重，V13 RefinementNet 兼容
5. **增强样本权重 0.8**: 高于默认 0.6，因 MLLM 已把关

### 改进原因
- V8 ParamModel 只在 FiveK Expert C 上训练 → 参数分布窄
- AesExpert ≥ 7 的增强数据提供了 "产生美学高分渲染的 ISP 参数" 信号
- 丰富 ParamModel 的参数搜索空间 → 更好的 V13 RefinementNet 基座
- 最终目标: V15 = V13 RefinementNet on V14 → MUSIQ ≥ 4.30

### 数据流
```
aug_venus_guided.json + aug_lut.json (22394 条)
  → rescore_with_aesexpert.py (AesExpert 4-bit 全量重打分)
  → merge_aesexpert_aadb.py (双门槛过滤)
  → aug_high_quality.json (~3400 条, AesExpert≥7 + AADB≥4.5)
  → V14 训练: FiveK Expert + aug_high_quality 混合
```

### Dry-run 验证
| 覆盖 | AesExpert 有效 | 保留 | 保留率 | AesExpert mean |
|------|--------------|------|--------|---------------|
| 1800/22394 | 1800 | 296 | 16.4% | 7.35 |
| 4400/22394 | 4400 | 664 | 15.1% | 7.30 |

- CPU smoke test: DistillParamModel (1.90M params), forward+loss 通过
- V13 ← V14 state_dict key 兼容性: strict load OK (324 keys)

### 训练命令
```bash
# V14 ParamModel (Windows smoke):
python tools/train/train_v14_aesexpert_param.py --epochs 2 --batch_size 4

# V14 ParamModel (AutoDL 正式):
python tools/train/train_v14_aesexpert_param.py --root /root/autodl-tmp --epochs 50 --batch_size 32

# V15 = V13 RefinementNet on V14:
python training/legacy/train_v13_multiscale.py --param_version v14
```

### 关键代码
- `tools/data/scoring/rescore_with_aesexpert.py`: AesExpert 全量重打分 (checkpoint/resume)
- `tools/data/merge_aesexpert_aadb.py`: 合并 + 双门槛过滤
- `tools/train/train_v14_aesexpert_param.py`: V14 训练脚本 (DistillParamModel)
- 输出: `checkpoints/distill_v14/stage_b/best.pt` (V13 兼容)

---

## Neural ISP 探索 (已搁置)

### 改进内容
- FiLM-conditioned U-Net (~300K 参数)
- 学习 (raw_image, params) → expert_image 的映射
- 替代 diff_isp 的手工渲染管线

### 改进原因
- diff_isp 渲染质量有限，考虑用神经网络学习更好的渲染

### 效果
- 训练完成但 MUSIQ 分数未超过 diff_isp
- 主要受限于评估分辨率问题 (后来修复)
- **搁置原因**: 修复分辨率后 diff_isp 表现合理，且 RefinementNet 方案更优

### 关键代码
- `models/neural_isp.py`: NeuralISP 模型
- `training/main/train_neural_isp.py`: 训练脚本

---

## diff_isp 增强记录

### 原始版本
- sRGB ↔ linear 转换
- 白平衡 (色温 → RGB gains)
- EV 补偿 (曝光调整)
- 对比度 (简单 `0.5 + (x - 0.5) * factor`)
- 阴影/高光 (基于亮度 mask 的简单调整)
- 饱和度 (灰度混合)

### 增强版本 (v9+)
1. **Tone Curve**: 参数化三次 S 曲线替代简单对比度
   - `t_curve = t + strength * t * (1 - t) * (2*t - 1)`
   - 保持端点不变，中间调增强
2. **Clarity**: 局部对比度增强
   - Unsharp mask on luminance channel
   - `clarity = lum + strength * (lum - lum_blur)`
3. **Shadows/Highlights 改进**:
   - 平滑 sigmoid mask (替代硬阈值)
   - 色彩比例保持 (避免色彩偏移)
4. **NaN 保护**: sRGB/linear 转换添加 clamping

### 关键代码
- `models/diff_isp.py`

---

## 评估指标说明

### MUSIQ-AVA (主要指标)
- Multi-scale Image Quality Transformer, 在 AVA 数据集训练
- 范围: ~1-10, 越高越好
- 评估: 整体美学质量 (构图、色彩、曝光、清晰度)
- **注意**: 需要 512×512+ 输入才准确

### 其他指标
| 指标 | 类型 | 范围 | 说明 |
|------|------|------|------|
| PSNR | 有参考 | dB ↑ | 像素级保真度 |
| SSIM | 有参考 | 0-1 ↑ | 结构相似性 |
| NIQE | 无参考 | ↓ | 自然度 |
| BRISQUE | 无参考 | ↓ | 失真检测 |
| Venus | 无参考 | 1-10 ↑ | 高层美学 (不适合退化评估) |

---

## 核心经验总结

1. **PSNR ≠ 美学**: v7 PSNR 暴跌 6dB 但 Venus +0.28, 偏离 GT 不等于变差
2. **评估分辨率很重要**: 224→512 上采样造成 MUSIQ 人为压低 ~0.7 分
3. **只监控不优化 = 无效**: v9 MUSIQ 只做监控, 分数未提升
4. **全局参数有上限**: 6 个 ISP 参数的天花板 ≈ Expert C (4.754)
5. **退化增强有效**: v7 在 8/9 个 NR-IQA 指标优于 v6
6. **要到 8.5+ 必须超越参数**: RefinementNet + MUSIQ loss 是当前最有希望的路线

---

## 文件清单

### 训练脚本
| 文件 | 版本 | 说明 |
|------|------|------|
| `training/legacy/train_v6_stage_a.py` | v6 | Stage A 语义对齐 |
| `training/legacy/train_v6_stage_b.py` | v6 | Stage B 参数预测 |
| `training/legacy/train_v7_stage_b.py` | v7 | 退化增强 + 对比学习 |
| `training/legacy/train_stage_c.py` | v8 | Stage C 文本条件 |
| `training/main/train_v9_aesthetic.py` | v9 | 美学感知微调 |
| `training/main/train_v10_e2e.py` | v10 | 端到端 image_loss |
| `training/main/train_v11_refine.py` | v11 | RefineNet V1 精修 (130K) |
| `training/main/train_v12_refine_hd.py` | v12 | RefineNet V4 512+MUSIQ 主导 (16M) |
| `training/legacy/train_v13_multiscale.py` | v13 | 多尺度 + EMA + WarmRestarts |
| `tools/train/train_v14_aesexpert_param.py` | v14 | AesExpert 高质量数据混训 ParamModel |
| `training/main/train_neural_isp.py` | — | Neural ISP (已搁置) |

### 模型文件
| 文件 | 说明 |
|------|------|
| `models/diff_isp.py` | 可微 ISP 管线 (增强版) |
| `models/refinement_net_v4.py` | 图像精修网络 V4 双分支 (~16M) |
| `models/neural_isp.py` | Neural ISP (已搁置) |
| `models/vision_encoder.py` | MobileViT-Small (SE + FPN) |
| `models/semantic_bridge.py` | SemanticProjector + ParameterDecoder |
| `models/parameter_predictor.py` | 参数预测器 |
| `models/aesthetic_scorer.py` | 美学评分器 |
| `training/semantic_distill/model.py` | SemanticDistillModel + DistillParamModel |
| `training/fivek_8param/model.py` | FiveK8ParamModel + LightroomDecoder |
| `training/text_condition/model.py` | TextConditionedModel + FiLM + LightTextEncoder |
| `training/aesthetic_loss.py` | MUSIQ 美学损失封装 |

### 评估脚本
| 文件 | 说明 |
|------|------|
| `evaluate/eval_hires.py` | 512×512 高分辨率评估 |
| `evaluate/eval_ceiling.py` | Expert C 天花板评估 |
| `evaluate/eval_neural_isp.py` | Neural ISP 对比评估 |
| `tools/eval/eval_v8_autodl.py` | v6-v9 统一评估 |
| `tools/eval/eval_psnr_ssim.py` | PSNR/SSIM 评估 |
| `tools/eval/eval_nr_iqa_full.py` | 9 指标 NR-IQA |

### Checkpoints (AutoDL)
```
/root/autodl-tmp/checkpoints/
├── distill_v6/stage_a/best.pt     # Stage A 语义对齐
├── distill_v6/stage_b/best.pt     # v6 参数预测
├── distill_v7/stage_b/best.pt     # v7 退化增强
├── distill_v8/stage_b/best.pt     # v8 Stage C (MUSIQ 最佳参数模型)
├── distill_v8/stage_c/best.pt     # v8 文本条件
├── distill_v9/stage_b/best.pt     # v9 美学微调
├── distill_v10/stage_b/            # v10 端到端 (训练中)
├── refinement/                     # v11 RefinementNet
├── refinement_v2/                  # v12 V2 (1.91M, best_MUSIQ=4.15)
├── refinement_v3/                  # v12 V3 (8.93M, best_MUSIQ=4.21)
└── refinement_v4/                  # v12 V4 (16M, 训练中)
```

---
---

# 附录 A: 早期实验详细数据 (v1-v4)

> 合并自: training_log.md

## Baseline vs Distill v1-v2 逐参数 MAE 对比

### 完整验证集结果 (500 images)

| 指标 | Baseline | Distill v1 | Distill v2 | v2 vs Baseline |
|------|----------|------------|------------|----------------|
| **PSNR (dB)** | 32.05 | 33.09 | **33.15** | **+1.10 dB** |
| **SSIM** | 0.9269 | 0.9257 | **0.9311** | **+0.0042** |
| **MS-SSIM** | 0.9812 | 0.9804 | **0.9820** | **+0.0008** |
| ev_compensation MAE | 0.09 | 0.08 | **0.06** | ↓ 33% |
| white_balance MAE | 501.98 | 627.18 | 620.45 | ↑ 24% |
| contrast MAE | 7.47 | 7.20 | 7.44 | ≈ |
| brightness MAE | 3.77 | 4.16 | 3.81 | ≈ |
| shadows MAE | 6.38 | 5.53 | 5.83 | ↓ 9% |
| highlights MAE | 7.53 | 6.58 | 6.99 | ↓ 7% |
| saturation MAE | 1.35 | 1.21 | 1.46 | ↑ 8% |
| vibrance MAE | 6.06 | 6.46 | 6.50 | ↑ 7% |

> 参数 MAE ≠ 视觉质量: WB MAE 更高但 PSNR 反而更好，蒸馏模型学到了更好的参数间协调

## Distill v1 训练策略

- Stage A: 3K图, MiniLM embedding, cos_sim=0.999
- Stage B: 30 epochs, 前50%冻结backbone, 解冻后 lr=0.05x
- Val Loss: 0.0037

## Distill v2 改进

| 项目 | v1 | v2 |
|------|-----|-----|
| 总 epochs | 30 | **50** |
| 解冻时机 | 50% (ep16) | **33% (ep17)** |
| Backbone LR | 0.05x | **0.1x** |
| 解冻后 scheduler | 继续旧 cosine | **重建 cosine** |

## Distill v3 域偏移分析

- Stage A: 5,377 张 (完整 Stage1), val_loss 更低
- **但 PSNR 从 33.15 降到 32.85** — 额外数据把语义空间拉向了与 FiveK 无关的方向
- **结论**: 数据量 ≠ 质量，Stage A 数据需要与目标域分布匹配

## Distill v4 FiveK 域验证

- Stage A: 2000 张 FiveK JPEG + Venus 分析
- SSIM = **0.9326** (全局最优), PSNR = 33.11
- 验证了域匹配假设

## Venus 美学评价 (教师验证学生)

| 维度 | Original | Baseline | Distill v2 | Delta (v2 vs orig) |
|------|---------|---------|-----------|-------------------|
| Lighting | 5.32 | 5.14 | **5.51** | **+0.19** |
| Color | 5.20 | 5.16 | **5.36** | **+0.16** |
| Clarity | 5.20 | 5.14 | **5.36** | **+0.16** |
| Subject | 5.18 | 5.14 | **5.38** | **+0.20** |
| **Overall** | 5.22 | 5.06 | **5.36** | **+0.14** |

> 语义蒸馏是决定性因素：有/无语义对齐，Venus 评分相差 +0.30

## NIMA 评估

| 组别 | 均值 | Delta vs orig |
|------|------|---------------|
| Original | 4.5265 | — |
| Baseline | 4.4289 | -0.0976 |
| Distill v2 | 4.4360 | -0.0905 |

> 三个指标一致: Distill v2 > Baseline (PSNR +1.10dB, Venus +0.30, NIMA +0.007)

## 多专家一致性验证

| 模型 | Expert-Default PSNR | Expert-A PSNR | Expert-B PSNR |
|------|---------------------|--------------|---------------|
| Baseline | 32.17 | 32.47 | 17.43 |
| Distill v2 | **33.10** | **33.82** | 17.54 |
| Distill v4 | 33.11 | 33.73 | 17.49 |

> Expert-A: +1.35 dB，跨标注者泛化验证通过

---

# 附录 B: v6/v7 详细评估数据

> 合并自: experiment_results_v6v7.md

## Venus 美学评分 (50张干净图)

| 维度 | 原图 | v6 | v7 |
|------|------|-----|-----|
| composition | 5.23 | 4.98 | **5.34** |
| lighting | 5.08 | 5.08 | **5.52** |
| color | 5.10 | 5.08 | **5.34** |
| clarity | 5.12 | 5.08 | **5.32** |
| subject | 5.14 | 5.12 | **5.38** |
| **overall** | 5.10 | 5.00 (-0.10) | **5.38 (+0.28)** |

## NR-IQA 9 指标 v6 vs v7

| Group | NIQE↓ | BRISQUE↓ | MUSIQ↑ | CLIPIQA+↑ | MANIQA↑ | DBCNN↑ | NIMA↑ | TOPIQ↑ | HyperIQA↑ |
|-------|-------|----------|--------|-----------|---------|--------|-------|--------|-----------|
| 原图 | 4.52 | 18.85 | 63.24 | 0.61 | 0.35 | 0.48 | 4.45 | 0.45 | 0.47 |
| v6 干净 | 4.83 | 28.37 | 55.72 | 0.55 | 0.32 | 0.43 | 4.14 | 0.41 | 0.47 |
| v7 干净 | 4.88 | 29.36 | **57.09** | 0.53 | 0.32 | 0.43 | 4.14 | 0.41 | 0.47 |
| v7 退化修复 | **5.04** | **27.77** | **58.68** | 0.52 | **0.32** | **0.45** | **4.10** | **0.42** | **0.49** |

> v7 退化修复在 8/9 个指标上优于 v6，退化增强训练显著提升修复泛化能力

## 评估工具适用性总结

| 工具 | 检测ISP退化？ | 区分v6/v7？ | 适合本项目？ |
|------|-------------|-----------|-----------|
| PSNR/SSIM | ✓ (vs GT) | ✓ | ✓ 技术还原 |
| Venus | ✗ (反涨分) | ✓ (美学) | ✓ 美学评估 |
| NIQE/BRISQUE | ✗ | ✓ | ✓ 技术质量 |
| MUSIQ/CLIPIQA+ | ✗ | ✓ | ✓ 感知质量 |
| TANet | ✗ | ✗ | ✗ |

## 核心发现

1. **PSNR ≠ 美学**: v7 PSNR 暴跌 6dB 但 Venus 美学评分 +0.28
2. **退化增强有效**: v7 退化修复在 9 个 NR-IQA 中 8 个优于 v6
3. **推荐评估方案**: PSNR(技术还原) + Venus(美学) + MUSIQ(感知质量) 三维评估

---

# 附录 C: IQA 指标系统性评估 (18 指标)

> 合并自: metric_evaluation_summary.md

## 指标全景表

| # | 指标名称 | 类别 | orig | v7 | 排序 |
|---|---------|------|------|-----|------|
| 1 | NIQE | 传统统计 | — | — | orig >> v7 |
| 2 | BRISQUE | 传统统计 | — | — | orig >> v7 |
| 3-9 | NIMA/DBCNN/HyperIQA/MUSIQ/CLIPIQA+/LIQE/UNIQUE | CNN/CLIP | — | — | orig > v7 |
| 10-12 | Q-Align/Compare2Score/AesExpert | 开源MLLM(7B) | — | — | orig ≈ v7 |
| 13 | **MUSIQ-AVA** | 美学Transformer | 4.359 | **4.362** | **v7 ≈ orig** |
| 14-17 | LAION_AES/Qwen-VL/GPT-4o | 闭源 | — | — | orig ≥ v7 |
| 18 | **Venus** | 专业美学VLM | 7.2 | **7.5** | **v7 > orig ★** |

## 关键发现

### 语义深度与偏见递减梯度
```
传统统计         CNN            CLIP           开源MLLM(7B)      闭源MLLM        Venus
(NIQE/BRISQUE)  (MUSIQ/DBCNN)  (LIQE/CLIPIQA) (Q-Align/C2S)    (Qwen/GPT-4o)   (专业美学)
     │               │              │               │                │              │
     ▼               ▼              ▼               ▼                ▼              ▼
 orig >> v7      orig >> v7     orig > v7       orig ≈ v7        orig ≥ v7      v7 > orig ★
```

### 结论
1. **现有 IQA 指标存在系统性技术偏见** — 所有自动化指标均将 original 评为最优
2. **语义理解深度与偏见缩小正相关** — 模型越强，差距越小
3. **Venus 是唯一判定 v7 > original 的评估方式** — 专业美学评估需要针对性训练
4. **论文贡献点**: 首次横向对比 18 种指标，揭示 IQA 的技术偏见连续谱

---

# 附录 D: SOTA 对比实验方案

> 合并自: experiment_plan.md

## 评估协议

- 数据集: MIT-Adobe FiveK 500 test, Expert C 作为 GT
- 图片统一 480p, PSNR/SSIM 在 sRGB 空间计算
- 效率指标: #Params, FLOPs, Latency

## SOTA 方法数据 (论文报告值, FiveK 480p Expert C)

| 方法 | 年份 | PSNR ↑ | SSIM ↑ | #Params | 可解释 |
|------|------|--------|--------|---------|--------|
| HDRNet | SIGGRAPH'17 | 23.93 | 0.897 | 482K | ❌ |
| White-Box | CVPR'18 | 24.55 | 0.920 | - | ✅ 部分 |
| 3D LUT | TPAMI'20 | 25.29 | 0.927 | 593K | ❌ |
| CSRNet | ECCV'22 | 25.56 | 0.929 | 37K | ✅ 部分 |
| SepLUT | ECCV'22 | 25.70 | 0.933 | 58K | ❌ |
| eLIR-Net | WACV'25 | ~26.0 | ~0.935 | 120K | ✅ 部分 |

## 消融实验设计

### 语义蒸馏消融

| 实验 | Stage A 数据量 | 方法 |
|------|---------------|------|
| w/o distill | - | 直接训练 |
| + 3K | 3,000 | MiniLM align |
| + 5.3K | 5,377 | MiniLM align |
| + FiveK域 | 2,000 | FiveK + Venus |

### 骨干网络消融

| Backbone | #Params | FLOPs |
|----------|---------|-------|
| MobileViT-XS | 2.3M | 0.7G |
| MobileViT-S | 5.6M | 1.8G |
| EfficientNet-B0 | 5.3M | 0.4G |

## 论文 Table 1 模板

| Method | PSNR ↑ | SSIM ↑ | #Params ↓ | Interpretable |
|--------|--------|--------|-----------|---------------|
| HDRNet | 23.93 | 0.897 | 482K | ❌ |
| 3D LUT | 25.29 | 0.927 | 593K | ❌ |
| CSRNet | 25.56 | 0.929 | 37K | Partial |
| SepLUT | 25.70 | 0.933 | 58K | ❌ |
| **Ours (Baseline)** | TBD | TBD | ~6M | ✅ Full |
| **Ours (+ Semantic)** | TBD | TBD | ~6M | ✅ Full |

## 关键风险

| 风险 | 应对方案 |
|------|----------|
| ISP 渲染精度不足，PSNR 偏低 | 改进 ISP 或用 Lightroom SDK |
| 8参数自由度太低 | 强调可解释性+效率优势 |
| PSNR 不如 SOTA | 效率维度 Pareto 曲线 |

---

# 附录 C: FiveK Teacher Pseudo-labels 数据增强 (2026-05-11)

> 目标: 用 FireRed 1.1 (online API) 作为 teacher editor 生成"编辑后图像", 然后通过 `inverse_fit` 反推 7-D ISP 参数, 构造 (orig, caption, teacher_edit, P_ISP) 四元组用于 LoRA SFT 训练 (text-conditioned param prediction).

## 实验时间线

### Step 1: micro-pilot synthetic round-trip (上午)
- 20 张 FiveK + synthetic teacher (用 GT P 自渲) + 6 个 ssim_w sweep
- 全部 PASS (L1<0.01), 验证 inverse_fit 算法本身正确
- **结论**: solver 对合成 target 完美工作

### Step 2: a0006 real-pilot (FireRed 真实输出)
- caption: "Increase contrast and saturation to bring out more vivid colors and textures. Apply slightly warmer white balance. Keep the original composition and subject unchanged."
- 单图 FireRed 编辑 → inverse_fit
- L1=0.0438, delta=0.103 → WARN (近 PASS), 验证 e2e pipeline 可行

### Step 3: 5-action top20 batch (失败)
- caption: "Increase contrast, enhance saturation, lift shadows, soften highlights, and balance white balance. ..."
- 20 张, LBFGS + 默认 init
- **结果**: 18/20 FAIL, L1 median=0.078, 多张 0.1s 提前退出
- 初步诊断: solver 问题 → 加 heuristic init + 换 Adam 优化器
- **结果**: 同 LBFGS, 17/20 FAIL → 证明**不是 solver 问题, 是 expressibility 问题**
- 关键证据: 不同图像产生**完全相同 P_inferred** ⇒ Adam 跑满 300 iter 后 L1=LBFGS 早退结果, FireRed 5-action 输出**不在 7D ISP 表达流形上**

### Step 4: 1-action 对比实验 (重大改善)
- 同 20 张图, caption: "Increase contrast. Keep the original composition and subject unchanged."
- 5-action median L1=0.072 → **1-action median L1=0.044**  (-39%)
- 可用率 (L1<0.10) 65% → **85%**
- B-tier good (L1<0.05) 2 → 12
- **结论**: caption 复杂度 ↑ → FireRed 编辑越复杂 → 非 ISP 表达占比越大 → fit 越差

### Step 5: per-action × 5 × 20 = 100 张主实验
- 5 个单 action: contrast / saturation / shadows / highlights / wb
- 各选 20 张 unique image (从 score≥4 候选池 deterministic shuffle)
- 100 张 FireRed teacher edit + 100 张 inverse_fit (Adam)

#### 整体结果

| 指标 | 值 |
|---|---|
| n_total | 100 |
| L1 mean | 0.0764 |
| L1 median | 0.0655 |
| delta mean | 0.1135 |
| **可用率 (L1<0.10)** | **74/100 (74%)** |
| good (L1<0.05) | 38 |
| Tier A (L1<0.02) | 1 |
| Tier B (L1<0.05) | 37 |
| Tier C (L1<0.10) | 36 |
| Tier D (L1≥0.10) | 26 |

#### 按 action 分化 (核心发现)

| action | n | L1 median | delta mean | usable | good | tier counts |
|---|---|---|---|---|---|---|
| **shadows** | 20 | 0.0414 | 0.0506 | **19/20** | 14 | A=1 B=13 C=5 D=1 |
| **highlights** | 20 | 0.0667 | 0.1074 | 17/20 | 7 | B=7 C=10 D=3 |
| contrast | 20 | 0.0513 | 0.0798 | 16/20 | 9 | B=9 C=7 D=4 |
| saturation | 20 | 0.0867 | 0.1398 | 12/20 | 4 | B=4 C=8 D=8 |
| **wb** | 20 | 0.1003 | 0.1899 | **10/20** | 4 | B=4 C=6 D=10 |

**核心机理**:
- shadows/highlights/contrast 是**全局 tone-curve** 操作 → diff_isp 7D 能很好表达 → 高 fit 率
- saturation/wb 涉及 **HSV 空间和 per-channel 色度调整** → FireRed 倾向加入空间局部色彩重映射 (e.g., 仅天空更暖) → 7D 全局参数无法表达 → 低 fit 率
- wb 'warmer' caption 解释最歧义 (FireRed 可能做局部色温调整)

## 交付物

```
outputs/inverse_fit_pilot/fivek_per_action_master/
├── pseudo_labels.jsonl        # 100 records, 含 P_inferred + quality_tier
├── summary.json               # overall + per-action 统计
└── viewer.html                # 单页 HTML 浏览器 (action × tier filter, sort by L1)
```

每条 record 字段:
```json
{
  "rank", "idx", "source_image", "orig_path", "target_path",
  "caption", "tone_target", "action",
  "P_inferred": {"white_balance", "brightness", "contrast", "shadows",
                 "highlights", "saturation", "clarity"},
  "P_init_heuristic": {...},
  "pixel_l1", "pixel_l2", "final_loss",
  "delta_target_orig", "fit_size", "runtime_sec",
  "verdict", "quality_tier"
}
```

## 关键脚本

| 文件 | 用途 |
|---|---|
| `tools/data/data_prep/select_fivek_per_action.py` | 选 100 张 × 5 group action |
| `tools/data/editor_models/run_firered_online.py` | FireRed 1.1 API 批量编辑 |
| `tools/data/data_prep/inverse_fit_batch.py` | inverse_fit 批量 + Adam solver + heuristic init |
| `tools/data/data_prep/combine_per_action_results.py` | 合并 5 路径为 master + 报表 |
| `tools/data/data_prep/build_per_action_viewer.py` | 生成 viewer.html |
| `tools/data/data_prep/compare_fit_runs.py` | 对比两个 inverse_fit batch (例: 5-action vs 1-action) |

## inverse_fit 算法改进

- **保留** `tools/data/data_prep/inverse_fit.py` 原版 (LBFGS) 不动 (训练管线在用)
- **`inverse_fit_batch.py` 内部新增** `inverse_fit_adam()`: 同 sigmoid 参数化 + 多 restart, 替换 LBFGS+strong_wolfe 为 Adam (lr=0.05, grad_clip=1.0)
- 同时新增 `estimate_init_params(orig, target)`: 基于像素均值/方差/HSV 估计初值 (brightness/contrast/saturation/wb), 让 Adam 从更优起点出发
- CLI: `--optim adam|lbfgs` (默认 adam), `--no_heuristic_init` 可禁用

## 下一步候选

| 方向 | 描述 | 工作量 |
|---|---|---|
| **A. scale 到 N=500** | 5 action × 100 张, ~3h FireRed + 30min fit, 交付 500 伪标签 | 中 |
| **B. 训 LoRA prove out** | 用当前 100 伪标签训 LoRA, 看是否能学到 ISP 参数预测 | 中 |
| **C. 提升 wb/saturation 质量** | 调研为何 wb/sat fit 率低: (1) 改 caption 措辞 (2) 加局部色彩参数到 diff_isp (3) 在 selector 中过滤难场景 | 大 |
| **D. 多 ssim_weight ablation** | 当前 ssim_w=0.5, 试 0.0/0.3/0.7 看是否影响 fit 质量 | 小 |

### Step 6: per-action × 5 × 100 = 499 张扩充实验 (方向 A 落地)

**目标**: 在 Step 5 (N=100) 基础上, 把单 action × 80 张追加进每组, 验证 N 增加后 Tier 分布是否稳定, 同时把 master 扩大到 499 (1 张 highlights 多次重试也 timeout, 实际落盘 499).

#### 流程

1. `select_fivek_per_action.py --per_action 80 --exclude_from_teacher_jsons <5 原 JSON> --out_suffix extend80 --seed 43` → 生成 5 个 `teacher_edits_fivek_<action>_extend80.json` (各 80 张, 全新, 无重叠)
2. `run_firered_online.py` 跑 5×80=400 张 (按 action 顺序 sequential): 主体 ~50min, 12 张 timeout
3. `check_missing_per_action.py --suffix extend80` 生成 retry JSON, FireRed retry ~2min, 补上 11/12 (1 张 highlights 多轮重试仍失败)
4. `run_inverse_fit_extend.py` (调度 `inverse_fit_batch.py --optim adam --heuristic_init --maxiter 200 --n_restarts 3`) 跑 5×80, 共 ~12min
5. `combine_per_action_results.py --sources per_action per_action_ext --out_dir fivek_500_master` 合并 100+399=499
6. `build_per_action_viewer.py --master_dir fivek_500_master --per_action_roots per_action per_action_ext` 生成 viewer (N=499)

整套用 `watch_and_finalize.py` 一键串联 (FireRed 完成 → stagnation 检测 4min → retry → fit → combine → viewer).

#### 整体结果

| 指标 | N=100 (Step 5) | **N=499 (Step 6)** | 变化 |
|---|---|---|---|
| L1 mean | 0.0764 | **0.0765** | -0.0% |
| L1 median | 0.0655 | **0.0609** | -7% |
| L1 max | 0.3245 | **0.3339** | +3% |
| delta mean | 0.1135 | **0.1097** | -3% |
| **可用率 (L1<0.10)** | **74.0%** (74/100) | **74.5%** (372/499) | +0.5pp |
| good (L1<0.05) | 38.0% (38/100) | **41.5%** (207/499) | **+3.5pp** |
| Tier A (L1<0.02) | 1 | **16** | +15 |
| Tier B (L1<0.05) | 37 | **191** | +154 |
| Tier C (L1<0.10) | 36 | **165** | +129 |
| Tier D (L1≥0.10) | 26 | **127** | +101 |

**核心: Step 5 的 Tier 分布在 N=499 上完全复现**, 可用率 74% 极稳定, Tier B 占比反而略升 (37%→38.3%), Tier A 从 1% 提升到 3.2% (heuristic init + Adam 在更大数据上找到更多极优解).

#### 按 action 分化 (N=499)

| action | n | L1 mean | L1 med | usable | good | tier counts | vs Step 5 usable |
|---|---|---|---|---|---|---|---|
| **shadows** | 100 | **0.0496** | 0.0420 | **93/100** | 66 | A=6 B=60 C=27 D=7 | 95% → **93%** |
| contrast | 100 | 0.0581 | 0.0446 | 86/100 | 56 | A=5 B=51 C=30 D=14 | 80% → **86%** ↑ |
| highlights | 99 | 0.0681 | 0.0624 | 83/99 | 41 | A=2 B=39 C=42 D=16 | 85% → **84%** |
| saturation | 100 | 0.0831 | 0.0787 | 75/100 | 27 | A=2 B=25 C=48 D=25 | 60% → **75%** ↑ |
| **wb** | 100 | **0.1232** | 0.1150 | **35/100** | 17 | A=1 B=16 C=18 D=65 | 50% → **35%** ↓ |

**核心发现复现 + 加强**:
- **shadows 仍是最佳** (93% usable, A+B = 66%); contrast/highlights 紧随
- **wb 仍是最差**, 且 N=100→500 fit 率反而下降 (50%→35%), D-tier 占比 65% — FireRed 在更多 wb 样本上引入了空间局部色温调整 (非全局 7D 可表达)
- saturation 在更大样本下回升 (60%→75%), 提示原 N=20 是偏差较大的子样本

#### 失败原因再分析 (wb 65 D-tier)

延续 Step 5 D-tier 分析: 主要失败模式仍是 **FireRed 对 "warmer wb" 解读**:
1. 仅天空区色温平移 (其它区域不变)
2. 加暖色滤镜叠加高光晕染
3. 整体加暖且增加柔焦/HDR — 等价于多 ISP 操作

对于这 65 张 wb D-tier, diff_isp 7D 完全表达不出 FireRed 的实际编辑路径, 应在 selector 阶段对 wb caption 加更强约束 (例如要求 "Apply warmer white balance globally, no other adjustments") 或考虑放弃 wb action 单独做.

#### 交付物 (N=499)

```
outputs/inverse_fit_pilot/fivek_500_master/
├── pseudo_labels.jsonl        # 499 records (60% 来自 extend80, 40% 来自 Step 5)
├── summary.json               # overall + per-action 统计
└── viewer.html                # 单页 HTML, 240KB, 4-panel × 499
outputs/inverse_fit_pilot/per_action_ext/<action>/
├── pseudo_labels.jsonl        # 80 records (extend80 组)
└── comparison/<idx>.png       # 4-panel 缩略图
```

#### 关键脚本 (新增/升级)

| 文件 | 用途 |
|---|---|
| `select_fivek_per_action.py` | **+ `--exclude_from_teacher_jsons` + `--out_suffix`** |
| `combine_per_action_results.py` | **+ `--sources` 多源 + `--out_dir`** |
| `build_per_action_viewer.py` | **+ CLI `--master_dir` `--per_action_roots` `--title`**, 修正 "Expert C" 标注为 "默认 raw 渲染" |
| `check_missing_per_action.py` | **+ `--suffix` `--retry_suffix`** |
| `run_inverse_fit_extend.py` | **新**: 调度 5 个 action 的 inverse_fit batch |
| `watch_and_finalize.py` | **新**: 一键 watcher (FireRed → retry → fit → combine → viewer) |
| `check_candidate_pool.py` | **新**: 验证 candidate pool 规模 (确认 1073 张 score≥4 余量) |
| `print_master_summary.py` | **新**: 终端打印 master summary 简报 |

#### 数据说明 (关于 GT 参数)

- 当前 master 中所有 `P_inferred` 都是 inverse_fit 从 (orig=默认 raw 渲染, target=FireRed 编辑) 反推, **不是 Expert C 真参数**
- Expert C 的 Lightroom 参数虽然在 `fivek_expert_c/` JPEG 和原始 FiveK catalog 中存在, 但当前 `data/fivek_expert_params.json` 只导出了 A/B 参数, **C/D/E 未提取**
- 若需训练时同时用 Expert C 真参数对照, 需追加 `extract_expert_c_params.py` 解析 Lightroom XMP/catalog

#### 下一步候选 (N=499 后)

| 方向 | 描述 | 工作量 |
|---|---|---|
| **A. 训 LoRA prove out** | 用 372 张 usable (排除 D-tier) 训 LoRA, 验证 ISP 参数学习 | 中 |
| **B. wb action 重做** | 改 caption 措辞 + 加 selector 黑名单 (天空/局部色彩场景) | 中 |
| **C. 扩展 ISP 模型** | diff_isp 加局部色温/区域 tone-curve 参数, 提高 wb/sat 表达力 | 大 |
| **D. 引入 Expert C GT** | 抽 Expert C 参数 → 训 (image, ISP) → (P_C_gt) 监督模型 | 中 |
| **E. scale 到 N=2000+** | 候选池有 1073 张 score≥4 余量, 可再扩 2-3 倍, 但 wb 收益边际递减 | 中 |

### Step 7: lrcat 提取 5 expert 真 GT (方向 D 落地)

**目标**: 用户指出 dataset 中存在 Lightroom catalog 原文件, 直接从中抽 ABCDE 真实调参参数, 替代/对照 N=499 伪标签.

#### 关键文件定位

| 路径 | 内容 | 大小 |
|---|---|---|
| `E:\Data\dataset\fivek_dataset\raw_photos\fivek.lrcat` | **Lightroom catalog (SQLite)** | **1.79 GB** |
| `E:\Data\dataset\fivek_dataset\raw_photos\HQa{1to700,...,4201to5000}/photos/*.dng` | 5000 原 DNG, 7 个分卷目录 | ~50 GB |
| `E:\Data\dataset\fivek_dataset\raw_photos\fivek Previews.lrdata/` | Lightroom 预览缓存 | - |

之前的 `fivek_expert_settings.json` (60K records) 只是该 lrcat 的一个子集 (只 dump 了 default + UUID-A + UUID-B), 缺 C/D/E.

#### lrcat 结构发现 (SQLite schema, 57 tables)

| Table | rows | 用途 |
|---|---|---|
| `Adobe_images` | 60000 | 5000 master + 55000 virtual copies (11 版本/张) |
| `Adobe_imageDevelopSettings` | 96458 | develop settings, `text` 字段是 Lua 风格 KV 表 |
| `Adobe_libraryImageDevelopHistoryStep` | 414263 | 每个 vcopy 的 history step (含真专家步骤名) |
| `Adobe_libraryImageDevelopSnapshot` | 14610 | 用户保存的 snapshot ("Import N" 是导入快照, **非 expert 标识**) |
| `AgLibraryFile` | 5000 | 原 DNG 文件名 (baseName + extension) |
| **`AgLibraryCollection`** | 23 | **关键**: 含 `name='A'`, `'B'`, `'C'`, `'D'`, `'E'` 各 5000 张 |
| `AgLibraryCollectionImage` | 60000 | collection ↔ image 关联 |

**核心 join 链** (一个 expert 的真 GT):
```
AgLibraryCollection(name=C, id=930899)
  → AgLibraryCollectionImage      (5000 rows)
  → Adobe_images (virtual copy)   → masterImage
  → Adobe_images (master)         → rootFile
  → AgLibraryFile (baseName.ext)  ← 原 DNG 名
+ Adobe_imageDevelopSettings.text ← Lua KV 表, 解析得 7D 参数
```

#### 提取结果 (`tools/data/data_prep/parse_fivek_lrcat.py`)

输出 `data/fivek_expert_abcde_params.json` (15.5 MB):

| Expert | unique images | full-join records | 备注 |
|---|---|---|---|
| **A** | 5000 (in collection) | **2800** | 部分 vcopy join AgLibraryFile 失败 (待修) |
| **B** | 5000 | **5000** | ✓ 完整 |
| **C** | 5000 | **5000** | ✓ **完整, 主目标** |
| **D** | 5000 | **2200** | 部分 vcopy join 失败 (同 A) |
| **E** | 5000 | **5000** | ✓ 完整 |
| 合计 | - | **20000** | 远超 N=499 |

每条 record schema:
```json
{
  "image_name": "a0001-jmac_DSC1459.dng",
  "expert": "C",
  "lr_copy_name": "Copy 3",
  "master_id": 8912,
  "params": {  // 映射 7D
    "white_balance": 4750.0,   // Temperature (K)
    "brightness": 0.0,         // LR Brightness 或 PV2012 Exposure
    "contrast": 22.0,
    "shadows": 3.0,            // PV2010 FillLight / PV2012 Shadows
    "highlights": -37.0,       // -HighlightRecovery / PV2012 Highlights
    "saturation": 12.0,
    "clarity": 0.0
  },
  "raw_lr": {  // 保留 LR 原始 Lua KV 字段, 用于下游精细映射
    "Temperature": 4750, "Tint": 4, "Exposure": 0,
    "Brightness": 0, "Contrast": 22, "Shadows": 12,
    "HighlightRecovery": 37, "FillLight": 3,
    "Saturation": 12, "Vibrance": 49, "Version": "4.5", ...
  }
}
```

#### Expert C 真 GT 风格 (5000 张平均, 物理含义清晰)

| param | C 均值 | 含义 |
|---|---|---|
| white_balance | **4873 K** | 偏暖 (比 D65 5500K 暖 627K) |
| brightness | 6.74 | 微提亮 (PV2010 字段, 多数为 0) |
| contrast | **16.84** | 适度对比度提升 |
| shadows | 4.68 | 微提阴影 |
| highlights | **-44.58** | **强力压低高光 (Expert C 标志性手法)** |
| saturation | 5.27 | 弱饱和度提升 |
| clarity | 0.01 | 几乎不用 |

vs 其它 expert 风格 (供参考):

| expert | wb_mean | bright | contr | shad | high | sat | clar | 风格关键词 |
|---|---|---|---|---|---|---|---|---|
| A | 4882 | 34.7 | 29.7 | 4.5 | -2.9 | 2.6 | 0.0 | 高 brightness + 高 contrast, 不压高光 |
| B | 5274 | 6.9 | 19.0 | 0.7 | -19.4 | 1.5 | 0.0 | 中性, 轻压高光 |
| **C** | **4873** | 6.7 | 16.8 | 4.7 | **-44.6** | 5.3 | 0.0 | **偏暖 + 重压高光 + 适度饱和度** |
| D | 5173 | 19.3 | 0.2 | 8.0 | -21.2 | 0.0 | 0.0 | 中性 brightness, 不调对比 |
| E | 4927 | 19.8 | 10.4 | 19.1 | -28.6 | 5.0 | 0.0 | 偏暖 + 提阴影 + 压高光 |

#### N=499 伪标签 vs Expert C 真 GT 对比 (核心发现)

**100% 重叠**: N=499 所有 image 都在 Expert C 5000 张中, 完美对比子集.

**伪标签与 GT 偏差极大** (contrast action 5 张 sample):

| param | 伪标签 P_inferred 范围 | GT_C 范围 | 差距诊断 |
|---|---|---|---|
| **highlights** | **-2.5 ~ +2.5** | **-14 ~ -37** | GT 必压, 伪标签未体现 |
| white_balance | 5145-6066 | 4124-6538 | 差距常 800K+ |
| contrast | 19-52 | 0-46 | 接近但变异大 |
| brightness | -4 ~ +29 | 全 0 | GT 不动, 伪标签乱调 |
| saturation | 3-27 | 0-4 | 伪标签过度估计 |

**根本原因**: 两者**学习目标完全不同**
- **伪标签**: 从 FireRed "Increase contrast" 编辑反推的等效 7D, **只局部修 contrast 维度, 其它维度由 init 决定**
- **GT_C**: Expert C 在 Lightroom 给原图做的**全套统一风格 7D 调整** (含强力压高光等手法)

→ N=499 伪标签**不能直接用作 Expert 风格学习的训练目标**, 它学的是 "5 种 FireRed 单 action 编辑各自的 7D 投影", 而非"专家的统一调参风格".

#### 关键脚本

| 文件 | 用途 |
|---|---|
| `tools/data/data_prep/probe_fivek_lrcat.py` | 列 lrcat 表 + 行数 + 候选字段 |
| `tools/data/data_prep/probe_lrcat_samples.py` | 看 snapshot/develop settings 样本 |
| `tools/data/data_prep/probe_lrcat_fields.py` | 穷举 develop text 所有字段 |
| `tools/data/data_prep/probe_lrcat_v2.py` | 找 collection ABCDE, copyName 分析 |
| `tools/data/data_prep/probe_lrcat_join.py` | 验证 collection→file→develop join 链 |
| `tools/data/data_prep/probe_lrcat_ad_missing.py` | 调查 A/D 缺失原因 (待修) |
| `tools/data/data_prep/parse_fivek_lrcat.py` | **正式提取 5 expert 7D 参数** |
| `tools/data/data_prep/spot_check_expert_c.py` | 随机 sample + 伪 vs 真对比 |

#### 数据 schema 说明

`data/fivek_expert_abcde_params.json` 与原 `data/fivek_expert_params.json` 的关系:

- 原 JSON: 60000 records, 来自 `fivek_expert_settings.json` 子集 (只 default + UUID-A + UUID-B)
- **新 JSON: 20000 records, 来自 lrcat 直接 join, 5 expert 真 GT, 含 raw_lr 完整字段**
- 两者**字段命名兼容** (params.{white_balance, brightness, contrast, shadows, highlights, saturation, clarity}), 可在训练脚本中直接替换

#### 已知问题与待办

1. **A=2800/D=2200 (不足 5000)**: full-join 时 `m.rootFile → AgLibraryFile.id_local` 对部分 vcopy 失败. C/B/E 不受影响. 修复优先级低 (C 已完整够用).
2. **PV2010 vs PV2012 字段语义差**: 当前简单映射, 后续若做监督训练应按 ProcessVersion 分组归一化.

#### 后续路径 (基于 lrcat GT 重新规划)

| 方向 | 描述 | 优势 |
|---|---|---|
| **D1. C-GT 监督 LoRA** | 用 5000 张 Expert C 真 GT 训 (orig_jpg, "make it look like Expert C") → P_C_pred 监督模型 | 数据干净, 风格统一 |
| **D2. 多 expert 风格嵌入** | 训 (orig_jpg, expert_id ∈ {A,B,C,D,E}) → P_pred, 学风格条件预测 | 一次拿 17200 records (B+C+E 全 + A/D 部分) |
| **D3. 伪+真混合训练** | 伪标签做"action-driven 编辑" task, GT 做"expert-style" task, 联合多任务 | 各取所长 |
| **D4. 渲染验证 ISP 模型** | 用真 GT 跑 diff_isp render, 比对 Expert C 渲染 JPG (需另下 expert_c JPG) | 闭环验证 ISP 可表达性 |

### Step 8: D1 落地 — Expert C 监督 baseline

**目标**: 用 Step 7 提取的 5000 张 Expert C 真 GT 训一个 baseline 监督回归模型, 验证 (orig_jpg) → P_C_7D 任务可行性.

#### 实验设定

| 项 | 值 |
|---|---|
| 数据 | 5000 张 Expert C, 过滤 WB=None 后 4997 → train 4498 / val 499 |
| 原图 | `E:\Data\dataset\fivek_jpeg\*.jpg` (默认 raw 渲染 JPEG) |
| 目标 | 7D 参数归一化 ∈ [-1, 1] (wb/bri/con/shad/hi/sat/clarity) |
| backbone | MobileViTSmall (2.89M) + LN + Linear(384→192→7) + Tanh |
| 总参数量 | **2.96M** |
| 输入 size | 256×256 (改自 224 — MobileViT 7×7 feature map 不能整除 patch=2) |
| loss | 加权 MSE, clarity 权重 0.1 (因 GT 近乎全 0), 其它 1.0 |
| optim | AdamW lr=1e-4, weight_decay=1e-4 |
| schedule | CosineAnnealingLR, T_max=30, eta_min=1e-6 |
| batch | 16 |
| epochs | 30 |
| device | RTX 4060 Laptop 8GB, num_workers=0 (Windows) |
| seed | 42 |
| aug | 50% 水平翻转 |

整套训练耗时 **~17 min** (35s/epoch).

#### "predict mean" baseline

直接对每个参数预测训练集均值, val MAE 作为无学习上限:

```
white_balance  mean=+4878.72  val_mae=783.73
brightness     mean=   +6.54  val_mae= 13.41
contrast       mean=  +16.85  val_mae= 14.45
shadows        mean=   +4.68  val_mae=  6.58
highlights     mean=  -44.37  val_mae= 15.92
saturation     mean=   +5.29  val_mae=  9.03
clarity        mean=   +0.01  val_mae=  0.01
```

#### 训练最佳 (Ep 7, val_loss=0.0248)

train 0.036→0.015 (↓); val 0.036→0.0248@Ep7 → 0.028@Ep30. Ep 8+ 轻度过拟合, 靠 best.pt 保底.

| param | baseline MAE | trained MAE | dMAE | Pearson R | 信号 |
|---|---|---|---|---|---|
| **white_balance** | 783.73 | **615.82** | **-21%** | **+0.66** | **强** |
| **shadows** | 6.58 | **5.72** | -13% | **+0.40** | 中 |
| **highlights** | 15.92 | **15.49** | -3% | **+0.32** | 中 |
| brightness | 13.41 | 12.80 | -5% | +0.02 | **无** |
| contrast | 14.45 | 14.46 | 0% | +0.07 | 无 |
| saturation | 9.03 | 9.74 | +8% | +0.06 | 无 |
| clarity | 0.01 | 1.93 | (N/A) | +0.00 | 无 |

#### 核心发现

**能学到的维度** (3D):
- **white_balance** (R=0.66): 色温有强视觉信号 (天空蓝调/室内黄调/夕阳暖调), MobileViT 能从整图色彩分布学出
- **shadows** (R=0.40): Expert C 对暗区提亮有统一风格, 模型能识别"画面 dark tone 占比高" → 预测提阴影
- **highlights** (R=0.32): Expert C 强压高光 (mean -44), 模型能识别"画面有 blown-out 高光" → 预测压高光

**学不到的维度** (4D):
- **brightness** (R=0.02): Expert C 大多 =0, 少数非 0. GT 方差低但非低于噪声, 模型直接预测均值占便宜
- **contrast** (R=0.07): Expert C 均值 +17 但 std 17, 强烈依赖场景, 仅从 orig 无法判断"要加多少 contrast"
- **saturation** (R=0.06): 同 contrast, 全局 +5 均值但场景相关性弱
- **clarity** (R=0.00): GT 几乎全 0, 模型在 variance 中乱估, MAE 反升

#### 诊断与下一步选项

**问题定位**:
1. backbone 从零训 2.96M 对 4498 数据**欠拟合能力不足**, 前几轮很快饱和
2. 过拟合 Ep 8+ 继续 → 需要 **early stopping** + **更强 aug**
3. 4 个"无信号"参数可能**需要语义条件** (FireRed caption / expert_id) 才能学出风格变化
4. clarity 几乎全零, 训练目标本质退化为"预测 0", 学了也意义不大

**改进路径**:
| 方向 | 做法 | 预期 |
|---|---|---|
| **E1. pretrained** | MobileViT ImageNet 预训练权重 + fine-tune | wb/shadows 进一步 ↑, 其它 +0.1 R |
| **E2. 多任务 multi-expert** | (orig, expert_id ∈ ABCDE) → P, 17.2K samples | 数据量 3.4×, 可能学到 contrast/sat |
| **E3. 条件输入** | 加 expert_id one-hot / caption embedding | contrast/sat 有可能打破 "预测均值" plateau |
| **E4. Stage-A semantic distill** | 重用 `training/main/train_v8_stage_b.py` 的 SemanticDistill 架构 | 更大 backbone + 更好先验 |
| **E5. 物理约束** | 用 diff_isp render(orig, pred_P) vs expert_c JPG (需先下 expert_c) 作 pixel loss | 物理闭环, 可能拯救 contrast 信号 |

**关键限制**: 仅从 orig_jpg 完全推断 Expert C 意图存在天花板 — 因为不同 expert 对同一张图会有不同判断. 若要突破 R=0.3 需要加 **expert_id / caption / user_intent 条件输入**.

#### 交付物

```
training/expert_c_baseline/
├── __init__.py
├── train.py           # 独立训练入口 (dataset + model + loop + eval)
└── print_best.py      # 打印 best.pt vs baseline 对比报告

checkpoints/expert_c_v1/
├── best.pt            # Ep 7, val_loss=0.0248, 11.4 MB
├── last.pt            # Ep 30, val_loss=0.0280
└── history.json       # 30 epoch per-param MAE + Pearson R 轨迹
```

#### 复现命令

```bash
# smoke test (2 epoch, ~1 min)
python training/expert_c_baseline/train.py --epochs 2 --batch_size 16 \
    --num_workers 0 --log_every 100 --image_size 256 \
    --out_dir checkpoints/expert_c_smoke

# full run (30 epoch, ~17 min on RTX 4060)
python training/expert_c_baseline/train.py --epochs 30 --batch_size 16 \
    --num_workers 0 --log_every 100 --image_size 256 --lr 1e-4 \
    --out_dir checkpoints/expert_c_v1

# eval report
python training/expert_c_baseline/print_best.py checkpoints/expert_c_v1/best.pt
```

**Windows 注意**: 设 `$env:KMP_DUPLICATE_LIB_OK='TRUE'` 解决 libiomp5md.dll 冲突.

### Step 9: FireRed 伪标签条件回归 baseline (核心 — 用户实际目标)

**任务重定义**: Step 8 的 Expert C 是"无条件回归"(给定 orig 直接出 expert 风格 7D), 但**用户实际想要的是**:

> "训练这些参数得到 firered 的效果"

也就是 **conditional**: 给定 (orig_jpg, **想做的 action**) → P_7D 满足"按 action 做 FireRed 风格的编辑". 这才是真正面向用户产品的 ISP agent 任务.

#### 实验设定

| 项 | 值 |
|---|---|
| 数据源 | `outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl` (N=499) |
| 过滤 | tier ∈ {A,B,C}, 丢 D fail (372 样本) |
| 划分 | tier-stratified train=299, val=73 |
| 监督 | (orig_jpg + action one-hot 5D) → 7D Lightroom params (fitted_P) |
| backbone | MobileViTSmall(384) + Linear(action_emb 5→32) → fused(416) → MLP→7 + Tanh |
| 总参数量 | **2.94M** |
| sample 权重 | tier A=1.2, B=1.0, C=0.5 (训练时按 tier 加权) |
| param 权重 | clarity=0.1, 其它 1.0 |
| optim | AdamW lr=1e-4, wd=1e-3, dropout=0.3 |
| schedule | CosineAnnealingLR T_max=50 |
| batch | 8 (因为 N=299 太小) |
| epochs | 50 + early stop patience=10 |
| aug | hflip 50% + 轻 ColorJitter (brightness/contrast/sat = 0.05) |
| device | RTX 4060 Laptop, num_workers=0 |

整套训练耗时 **~2.5 min** (3.3s/epoch × 32 epoch = early stopped).

#### "predict per-action mean" baseline

每个 action 用其训练集 primary param 均值作预测, val MAE:

```
contrast    primary=contrast       mean_pred=  +21.18  mae=   19.15  n=17
saturation  primary=saturation     mean_pred=  +26.75  mae=   15.60  n=15
shadows     primary=shadows        mean_pred=   -1.12  mae=    2.34  n=18
highlights  primary=highlights     mean_pred=   -9.91  mae=   15.47  n=16
wb          primary=white_balance  mean_pred=+7953.98  mae= 1553.31  n=7
```

#### 训练最佳 (Ep 22, val_loss=0.0492, early stop @ Ep 32)

train: 0.103 → 0.026 (↓74%); val: 0.103 → 0.049@Ep22 → 0.057@Ep32 (轻微 overfit, early stop 救场).

**整体 7D per-param** (跨所有 action 一起评):

| param | val_MAE | Pearson R | 信号 |
|---|---|---|---|
| white_balance | 787.6 | +0.44 | 中 |
| brightness | 16.34 | **+0.59** | 强 |
| contrast | 21.28 | **+0.57** | 强 |
| shadows | 5.81 | +0.40 | 中 |
| highlights | 15.31 | **+0.60** | 强 |
| saturation | 14.63 | +0.42 | 中 |
| clarity | 21.93 | +0.30 | 中 |

**Per-action primary param** (核心评估 — 模型在每个 action 主导参数上的能力):

| action | primary | base_MAE | model_MAE | dMAE | R | 结论 |
|---|---|---|---|---|---|---|
| **contrast** | contrast | 19.15 | **18.50** | -3% | **+0.47** | ✓ 战胜 baseline |
| **saturation** | saturation | 15.60 | **13.79** | **-12%** | **+0.60** | ✓ 战胜 baseline |
| **highlights** | highlights | 15.47 | **11.51** | **-26%** | **+0.64** | ✓ 战胜 baseline |
| shadows | shadows | 2.34 | 4.22 | +80% | +0.55 | △ R 高但 baseline 太低 |
| wb | white_balance | 1553 | 2279 | +47% | -0.17 | ✗ 学不出 (n_train=28 太少) |

#### 与 Step 8 (Expert C unconditional) 对比

| param | Expert C R (4498 train, unconditional) | FireRed R (299 train, **action conditional**) | dR |
|---|---|---|---|
| white_balance | **+0.66** | +0.44 | -0.22 |
| brightness | +0.02 | **+0.59** | **+0.57** |
| contrast | +0.07 | **+0.57** | **+0.50** |
| shadows | +0.40 | +0.40 | = |
| highlights | +0.32 | **+0.60** | +0.28 |
| saturation | +0.06 | **+0.42** | **+0.36** |
| clarity | +0.00 | +0.30 | +0.30 |

**6/7 维度 FireRed conditional baseline 胜过 Expert C unconditional baseline, 训练数据少 15×**.

#### 视觉验证 (`outputs/firered_v1_val_compare/`)

每个 action 抽 1 张 tier 最高的 val 图, 4-panel 对比:

| panel | 含义 |
|---|---|
| 1 | orig (raw 渲染 JPEG) |
| 2 | FireRed edit (target) |
| 3 | **model→ISP→render** (我们的预测 7D 用 diff_isp 渲染) |
| 4 | inverse_fit→ISP→render (P_inferred 直接渲染, GT 上限) |

观察 (5 张样本):
- **contrast**: model 预测 cont=77 vs GT 93 (under-shoot 17), 视觉接近但稍弱
- **saturation**: model 预测 sat=58 vs GT 43 (over-shoot 15), 但风格对得上
- **shadows / highlights**: 这两 action 的 GT 本身有时 ≈0 (inverse_fit 没找到强解), 因此 model 也学不出强变化
- **wb**: GT 9205 vs model 6381, 偏差最大 — 与 R=-0.17 一致

**核心洞察**: 模型在 contrast / saturation / highlights 这三个 action 上学到了**可视化的真正风格学习**, 不只是数值上的相关性.

#### 核心发现 (本步最重要的科研产出)

1. **action 条件输入是关键**: 同样 ~300 张 N (15× 比 Expert C 少) 反而学得更好 — 因为任务本身可解 (给定 action 后 P 的不确定性大幅降低)
2. **数据小不是问题**: tier-stratified split + tier-weighted loss + 强 dropout + 轻 aug 让 N=299 都能稳定训出 R>0.4
3. **wb 是 outlier**: train n=28 (5 actions 中最少) + wb 物理 scale 100× 大于其它 → 学习严重不足. 解决: 单独加大 wb 数据或调 normalization
4. **inverse_fit 伪标签的限制**: shadows/highlights 上有些样本 inverse_fit 解 ≈0 (没找到强信号), 这部分样本本身就是噪声训练目标

#### 下一步选项

| 方向 | 做法 | 预期 |
|---|---|---|
| **G1. 扩大数据** | 跑 FireRed 至 N=2000 (4×), 重训 | 整体 R 再 +0.05~0.1 |
| **G2. wb 单独补数据** | 多挑 100 张 wb action, 重点训 wb 维度 | wb R 从 -0.17 → +0.3+ |
| **G3. 加 caption embedding** | 用 sentence-transformer 把 caption 编码为 384D, 替代 one-hot | 支持 free-form 指令 |
| **G4. inverse_fit 质量过滤** | 只保留 tier A+B (207 张), 丢 C; 或重新跑 inverse_fit (更高 maxiter) | 减少噪声标签污染 |
| **G5. expert C 联合训练** | (orig, action_or_expert_c_token) → P, 把 4498 Expert C + 372 FireRed 一起训 | 大数据量 + 多任务 |

#### 交付物

```
training/firered_baseline/
├── __init__.py
├── train.py              # FireRed7DModel + tier-aware loss + early stop
├── print_best.py         # best.pt vs baseline 对比
└── render_val_compare.py # 5×4-panel 视觉验证

checkpoints/firered_v1/
├── best.pt               # Ep 22, val=0.0492, 11.4 MB
├── last.pt
└── history.json          # 32 epoch 完整轨迹

outputs/firered_v1_val_compare/
├── index.html            # 5 个 4-panel + P_pred vs P_inv 表
└── {action}_{img}.png    # 5 张拼接对比图
```

#### 复现命令

```bash
# smoke test (3 epoch, ~30s)
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python training/firered_baseline/train.py --epochs 3 --batch_size 8 \
    --num_workers 0 --log_every 20 --out_dir checkpoints/firered_smoke

# full train (50 epoch + early stop, ~2.5 min)
python training/firered_baseline/train.py --epochs 50 --batch_size 8 \
    --lr 1e-4 --dropout 0.3 --patience 10 --num_workers 0 --log_every 30 \
    --out_dir checkpoints/firered_v1

# eval report
python training/firered_baseline/print_best.py checkpoints/firered_v1/best.pt

# 视觉对比 (5 actions × 4-panel)
python training/firered_baseline/render_val_compare.py
# 然后开浏览器看 outputs/firered_v1_val_compare/index.html
```

### Step 9.1: FireRed baseline 演进 (v2 / v3 / v4 — 修复 wb dimension)

Step 9 训出 v1 后, 用户视觉检验提出"orig (panel 1) ≈ model_render (panel 3)"的观察, 提示 v1 在视觉上还达不到 FireRed 风格. 进一步实证发现 **wb dimension 是元凶** (R=-0.17, 训练样本仅 28), 后续做了三次架构/损失调整尝试修复.

#### 诊断: 真问题不是 "under-prediction" 而是 "mis-direction"

写 `training/firered_baseline/diagnose_under_prediction.py`, 对全 73 val 样本算 panel 间像素 L1 + 参数偏离:

```
原图->FireRed_edit 平均改动幅度:  L1 = 0.0717  (真实改动)
inv_fit 渲染相比 orig 改动幅度:   L1 = 0.0437  (61% 改动, 7D 表达上限)
model 渲染相比 orig 改动幅度:     L1 = 0.0488  (68% 改动, 跟 inv 接近)
inv_render vs FireRed:           L1 = 0.0458  (7D ISP 上限)
model_render vs FireRed:         L1 = 0.0732  (模型离 FR)
```

**关键发现**:
- v1 模型**改动幅度 OK** (0.0488 vs 0.0437 inv), 不是 under-prediction
- 但 L1(model, FR) = 0.0732 > L1(inv, FR) = 0.0458, 说明模型**改动方向不对**
- 罪魁是 wb dimension: wb action 7 张样本 P_pred=6297 vs P_inv=8283, 差 1986K — diff_isp 中 wb 是第一步 multiplicative gain, 决定整体色调

#### v2: pixel reconstruction loss (失败)

**设计**: 加 `apply_diff_isp(orig, P_pred) vs apply_diff_isp(orig, P_inv)` L1 作辅助 loss, 让模型对像素效果负责, 试图通过渲染监督修复 wb. `pixel_loss_weight=0.5`, 5 epoch warmup.

**修复 NaN 问题**:
- diff_isp 通过 backward 梯度 boundary 偶发 NaN → 污染模型权重
- 加 `torch.nan_to_num` 在 pred_render/grad 上 + skip NaN batch + warmup 从 0 起

**结果 (60 epoch, early stop @ Ep 48, val=0.1038)**:
- L1(model, FR) 0.0732 → **0.0713** (仅 -2.6%)
- |dP|_phys 126 → **176** (恶化 +40%)
- per-action primary R 平均 +0.42 → **+0.20** (腰斩)

**为什么失败**: **7D ISP 是非单射的** — 多组参数能产生类似图像. pixel loss 让模型找到一个"渲染相似但参数偏离"的解, 像素改善微乎其微但参数学习严重退化. v2 路线放弃.

#### v3: action-aware param weighting (wb 突破, shadows 反向)

**设计**: 对每个样本根据 action 调整 7D loss 权重:
- action 主参数权重 5.0 (e.g. wb action 上 wb=5.0)
- secondary params 权重 1.0
- 其它 params 权重 0.2 (强抑制)
- clarity 全局压低 0.1

```
weight matrix (5 actions × 7 params):
            wb    bri   cont  shad  high  sat   clar
contrast    0.20  1.00  5.00  1.00  1.00  0.20  0.10
saturation  0.20  0.20  0.20  0.20  0.20  5.00  0.10
shadows     0.20  1.00  0.20  5.00  1.00  0.20  0.10
highlights  0.20  1.00  0.20  1.00  5.00  0.20  0.10
wb          5.00  0.20  0.20  0.20  0.20  1.00  0.10
```

**结果 (60 epoch, early stop @ Ep 25, best Ep 13, val=0.0427)**:
- **wb R: -0.17 → +0.80** (突破!)
- contrast R: +0.47 → **+0.71** (强化)
- saturation R: +0.60 → +0.47 (略降)
- highlights R: +0.64 → +0.43 (降)
- **shadows R: +0.55 → -0.41** (反向!)
- L1(model, FR): 0.0732 → 0.0810 (反而变差)
- |dP|_phys: 126 → 211 (恶化)

**为什么 shadows 反向**: shadows action 的 GT primary param 本身就小 (MAE base ~2.34, 近 0), `primary_w=5` 强加权放大了对小 GT 的过拟合, 模型学到一个奇怪的负相关. **过度抑制非主参数破坏了均衡**.

#### v4: wb 全局 ×3 (minimal fix, 折中)

**设计**: 复用 v1 的 weighted_mse_loss, 不引入 action-aware 复杂逻辑. 直接针对 wb 物理 scale 100× 大的根因, 全局加权 wb=3.0, 其它 1.0, clarity 0.1.

**结果 (60 epoch, early stop @ Ep 21, best Ep 9, val=0.0764)**:
- wb R: -0.17 → +0.18 (改善但远不如 v3 的 +0.80)
- 其它 dimension R 跟 v1 持平 (差 ±0.1)
- shadows R: +0.55 → -0.29 (仍轻微反向)
- L1(model, FR): 0.0732 → 0.0783 (略差)

**结论**: wb 全局加权 = "对 wb action 强化不够, 对其它 action 噪声引入". 没有 v3 那么戏剧化但也没显著改善.

#### 四代综合对比

| action | v1 | v2 | v3 | v4 | 最佳 |
|---|---|---|---|---|---|
| contrast | +0.47 | +0.08 | **+0.71** | +0.49 | v3 |
| saturation | **+0.60** | +0.37 | +0.47 | +0.39 | v1 |
| shadows | **+0.55** | +0.16 | -0.41 | -0.29 | v1 |
| highlights | **+0.64** | +0.30 | +0.43 | +0.37 | v1 |
| wb | -0.17 | -0.12 | **+0.80** | +0.18 | v3 |
| **L1(m, FR)** | **0.0732** | **0.0713** | 0.0810 | 0.0783 | v1/v2 |

**无单一胜者** — v1 在 4/5 action 上仍是最稳, v3 在 wb 上完胜但 shadows 反向, v2/v4 是失败 / 折中.

#### 核心科研产出

1. **7D ISP 非单射性 = pixel-only 监督不可行** (v2 实证). 必须用 param-space loss 才能保持参数学习信号.
2. **action-aware weighting 是一把双刃剑** (v3): 能解锁难学的 wb, 但同时让 GT 信号弱的 dimension (shadows) 反向. 需更精细控制权重 schedule.
3. **wb 的根本问题不是数据少而是 loss balance**: 物理 scale 100× 大但 normalized loss 跟其它等权, 模型保守输出中性 6000. v3 strong weighting 证明 wb 是可学的.
4. **均衡 + 突破的张力**: 当前架构无法既保 v1 的 4/5 均衡又拿 v3 的 wb. 真正破局需:
   - 数据扩容 (G1) — wb action 28 → 200 样本
   - 双头分流 (G2) — wb 独立 head 不影响其它参数
   - 主线切换 (Qwen3-VL LoRA) — 用 MLLM 大模型自然处理 scale 不均衡

#### 交付物

```
training/firered_baseline/
├── train.py / print_best.py / render_val_compare.py  # v1
├── train_v2.py                       # pixel loss
├── train_v3.py                       # action-aware
├── train_v4.py                       # wb global x3
├── diagnose_under_prediction.py      # L1 像素诊断
├── compare_3way.py                   # N-way per-action R + L1
└── write_compare_html.py             # 生成 outputs/firered_compare.html

checkpoints/firered_v{1,2,3,4}/best.pt + history.json

outputs/
├── firered_compare.html              # 4-way 总览页 (推荐入口)
├── firered_v{1,2,3,4}_val_compare/   # 各 5×4-panel viewer + index.html + L1 标注
└── firered_v{2,3,4}_train.log        # 训练日志
```

#### 复现命令

```bash
$env:KMP_DUPLICATE_LIB_OK='TRUE'

# v1: baseline
python training/firered_baseline/train.py --epochs 50 --batch_size 8 \
    --patience 10 --out_dir checkpoints/firered_v1

# v2: + pixel loss (失败实验, 仅复现用)
python training/firered_baseline/train_v2.py --epochs 50 --batch_size 8 \
    --pixel_loss_weight 0.5 --pixel_warmup_epochs 5 \
    --patience 10 --out_dir checkpoints/firered_v2

# v3: action-aware
python training/firered_baseline/train_v3.py --epochs 60 --batch_size 8 \
    --primary_w 5.0 --secondary_w 1.0 --other_w 0.2 \
    --patience 12 --out_dir checkpoints/firered_v3

# v4: wb global x3
python training/firered_baseline/train_v4.py --epochs 60 --batch_size 8 \
    --wb_weight 3.0 --patience 12 --out_dir checkpoints/firered_v4

# 全套评估 + 4-way 表
python training/firered_baseline/compare_3way.py \
    --ckpts checkpoints/firered_v{1,2,3,4}/best.pt --names v1 v2 v3 v4

# 渲染 viewer + 生成对比 index
python training/firered_baseline/render_val_compare.py --ckpt {ckpt} \
    --out_dir outputs/firered_v{x}_val_compare
python training/firered_baseline/write_compare_html.py
python -m http.server 9123 --directory outputs
# 浏览器开 http://localhost:9123/firered_compare.html
```
