# IntelligenceCamera 全版本实验记录

> 更新时间: 2026-04-30
> 项目目标: 基于语义理解的智能 ISP 参数预测，最终目标 MUSIQ-AVA 8.5+
> 评估数据: MIT-Adobe FiveK 验证集 (50张)
> 训练平台: AutoDL (RTX 5090)

---

## 版本演进总览

```
Baseline → v1-v3(语义蒸馏探索) → v4-v5(精度优化) → v6(Venus NL) → v7(退化增强)
    → v8(Stage C文本条件) → v9(美学感知) → v10(端到端E2E) → v11(RefinementNet)
```

| 版本 | 核心改进 | PSNR | MUSIQ-AVA (512) | 状态 |
|------|---------|------|-----------------|------|
| Baseline | MobileViT + 8参数 | 32.05 | — | ✅ |
| v5 | Expert C 单专家精准监督 | **34.11** | — | ✅ |
| v6 | Venus NL 语义对齐 + 6参数 | 32.30 | 4.413 | ✅ |
| v7 | 退化增强 + 对比学习 | 25.99 | 4.425 | ✅ |
| v8 | Stage C 文本条件融合 | — | **4.446** | ✅ |
| v9 | 美学感知微调 (MUSIQ 监控) | — | 4.418 | ✅ |
| v10 | 端到端 image_loss 通过 diff_isp | — | 训练中 | 🔄 |
| v11 | RefinementNet + MUSIQ loss | — | 训练中 | 🔄 |

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
- `scripts/train_v6_stage_a.py`: Stage A 训练
- `scripts/train_v6_stage_b.py`: Stage B 训练
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
- `scripts/train_v7_stage_b.py`: 含 `batch_degrade()` 和对比学习
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
- `scripts/train_stage_c.py`: Stage C 训练
- `tools/data/generate_instruction_data.py`: 指令数据生成
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
- `scripts/train_v9_aesthetic.py`
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
- `eval_hires.py`: 高分辨率评估脚本

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
- `eval_ceiling.py`: 天花板评估脚本

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
- `scripts/train_v10_e2e.py`

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
- `scripts/train_v11_refine.py`: 训练脚本

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
- `scripts/train_v12_refine_hd.py`: 统一训练脚本

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
- AutoDL 训练就绪 (`scripts/train_v13_multiscale.py`)
- 目标: val_MUSIQ ≥ 4.30

### 关键代码
- `scripts/train_v13_multiscale.py`: 训练脚本 (含 EMA, 多尺度, WarmRestarts)
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
python scripts/train_v13_multiscale.py --param_version v14
```

### 关键代码
- `tools/data/rescore_with_aesexpert.py`: AesExpert 全量重打分 (checkpoint/resume)
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
- `scripts/train_neural_isp.py`: 训练脚本

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
| `scripts/train_v6_stage_a.py` | v6 | Stage A 语义对齐 |
| `scripts/train_v6_stage_b.py` | v6 | Stage B 参数预测 |
| `scripts/train_v7_stage_b.py` | v7 | 退化增强 + 对比学习 |
| `scripts/train_stage_c.py` | v8 | Stage C 文本条件 |
| `scripts/train_v9_aesthetic.py` | v9 | 美学感知微调 |
| `scripts/train_v10_e2e.py` | v10 | 端到端 image_loss |
| `scripts/train_v11_refine.py` | v11 | RefinementNet 精修 |
| `scripts/train_neural_isp.py` | — | Neural ISP (已搁置) |

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
| `eval_hires.py` | 512×512 高分辨率评估 |
| `eval_ceiling.py` | Expert C 天花板评估 |
| `eval_neural_isp.py` | Neural ISP 对比评估 |
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
