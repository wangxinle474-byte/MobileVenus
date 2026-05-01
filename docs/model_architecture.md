# MobileVenus 模型架构文档

> 更新时间: 2026-04-30

本文档描述 IntelligenceCamera 项目中各模块的架构设计、来源及相互关系。

---

## 架构总览

```
三阶段训练 → 部署模型
───────────────────────────────────────────────────
Stage A: 视觉-语义对齐 (MobileViT + MiniLM)
Stage B: 视觉→ISP参数蒸馏 (冻结encoder + LightroomDecoder)
Stage C: 文本+视觉→ISP参数 (冻结backbone + LightTextEncoder + FiLM + Decoder)
───────────────────────────────────────────────────

推理路径:
  Image + Text → MobileViT → SemanticProjector → FiLM(TextEncoder) → Decoder → 6 ISP Params
```

---

## 模块清单

### 1. MobileViTSmall — 视觉编码器

| 属性 | 值 |
|------|-----|
| 文件 | `models/vision_encoder.py` |
| 来源 | 基于 Apple MobileViT (ICLR 2022) 自行重写 |
| 参数量 | ~5.6M |
| 输入 | `(B, 3, 224, 224)` |
| 输出 | `(B, 384)` 全局特征 |

**与原版 MobileViT 的区别：**
- 每个 stage 添加 **SE (Squeeze-and-Excitation)** 通道注意力
- 新增 **FPN (Feature Pyramid Network)** 多尺度特征融合
- Stage 通道配置：16 → 32 → 64 → 80 → 96
- 支持返回多尺度中间特征（用于蒸馏）

**内部结构：**
```
Conv Stem (3→16, stride=2)
  → Stage 2: MV2Block (16→32) + SE
  → Stage 3: MV2Block (32→48) + MobileViTBlock (48→64, T=96, 2层) + SE
  → Stage 4: MV2Block (64→64) + MobileViTBlock (64→80, T=120, 4层) + SE
  → Stage 5: MV2Block (80→80) + MobileViTBlock (80→96, T=144, 3层) + SE
  → FPN([Stage3, Stage4, Stage5] → 384-D)
  → GlobalAvgPool → (B, 384)
```

### 2. SemanticProjector — 语义投影

| 属性 | 值 |
|------|-----|
| 文件 | `models/semantic_bridge.py` |
| 来源 | 自行设计 |
| 参数量 | ~130K |
| 输入 | `(B, 384)` 视觉特征 |
| 输出 | `(B, 256)` L2 归一化语义向量 |

**结构：**  
渐进式 MLP 投影 `384 → mid → 256`，每层 `Linear + LayerNorm + GELU + Dropout`，最终 L2 归一化。

### 3. TextProjector (Stage A) — 文本投影

| 属性 | 值 |
|------|-----|
| 文件 | `training/semantic_distill/model.py` |
| 来源 | 自行设计（使用预训练 MiniLM-L6-v2 生成 embedding） |
| 文本模型 | `all-MiniLM-L6-v2` (sentence-transformers, 384-D) |
| 输入 | `(B, 384)` 文本 embedding |
| 输出 | `(B, 256)` L2 归一化语义向量 |

**说明：**  
Stage A 训练时，Venus 生成的自然语言美学描述先通过预训练 MiniLM 编码为 384-D 向量，再由 TextProjector 投影到 256-D 统一语义空间，与视觉侧的 SemanticProjector 输出做对齐。

### 4. LightroomDecoder — ISP 参数解码器

| 属性 | 值 |
|------|-----|
| 文件 | `models/semantic_bridge.py` |
| 来源 | 自行设计 |
| 参数量 | ~50K |
| 输入 | `(B, 256)` 语义向量 |
| 输出 | 6 个 Lightroom 参数 + 置信度 |

**参数与范围：**

| 参数 | 范围 | 激活函数 |
|------|------|---------|
| ev_compensation | [-3, 3] | Tanh × 3 |
| white_balance | [2000, 10000] K | Sigmoid × 8000 + 2000 |
| contrast | [-100, 100] | Tanh × 100 |
| shadows | [-100, 100] | Tanh × 100 |
| highlights | [-100, 100] | Tanh × 100 |
| saturation | [-100, 100] | Tanh × 100 |

**结构：**  
共享 MLP `(256 → 256, LayerNorm, GELU)` → 6 个独立参数头 `(256 → 64 → 1)` + 置信度头 `(256 → 32 → 6, Sigmoid)`

### 5. LightTextEncoder (Stage C) — 轻量文本编码器

| 属性 | 值 |
|------|-----|
| 文件 | `training/text_condition/model.py` |
| 来源 | 自行设计 |
| 参数量 | ~0.5M |
| 输入 | `(B, L)` 字符 token IDs |
| 输出 | `(B, 256)` L2 归一化文本特征 |

**设计动机：**  
Stage C 部署时不依赖预训练 MiniLM，而是用完全自主训练的轻量编码器直接理解口语化中文指令。

**结构：**
```
字符嵌入 (vocab=5000, dim=128) + 位置编码 (max_len=64)
  → 2 层 Transformer Encoder (nhead=4, hidden=256, GELU, norm_first)
  → LayerNorm
  → CLS pooling (取第0位)
  → 投影 MLP (128 → 256 → 256)
  → L2 归一化
```

### 6. FiLMFusion — 文本-视觉融合

| 属性 | 值 |
|------|-----|
| 文件 | `training/text_condition/model.py` |
| 来源 | 基于 FiLM (Perez et al., AAAI 2018)，自行实现 |
| 参数量 | ~130K |

**原理：**
```
γ = 1.0 + gamma_net(text_emb) × 0.5   # ∈ [0.5, 1.5]
β = beta_net(text_emb) × 0.3           # ∈ [-0.3, 0.3]
fused = γ ⊙ visual_emb + β
```

**关键设计：**
- 初始化为恒等变换 (γ≈1, β≈0)，保证初始行为等价于无文本的 Stage B
- 无文本输入时直接返回 visual_emb（优雅退化）

### 7. CrossAttentionFusion — 备选融合方案

| 属性 | 值 |
|------|-----|
| 文件 | `training/text_condition/model.py` |
| 来源 | 自行设计 |

Gated cross-attention：文本作 query，视觉作 key/value，门控残差连接。初始 gate≈0 保持 Stage B 行为。

### 8. RefinementNetV4 — 图像精修网络

| 属性 | 值 |
|------|-----|
| 文件 | `models/refinement_net_v4.py` |
| 来源 | 自行设计 |
| 参数量 | ~16M |

**双分支架构：**
- **全局色彩分支**：自适应 3×3 色彩矩阵 + per-channel gamma 曲线
- **局部细节分支**：3 级 U-Net 解码器 + 空间细节残差

### 9. DiffISP — 可微渲染管线

| 属性 | 值 |
|------|-----|
| 文件 | `models/diff_isp.py` |
| 来源 | 自行设计 |

模拟 Lightroom 的 ISP 处理流程，支持反向传播：
```
sRGB → Linear → WB → EV → Tone Curve → Shadows/Highlights → Saturation → Clarity → sRGB
```

---

## 三阶段组装关系

### Stage A: SemanticDistillModel

```python
# training/semantic_distill/model.py
class SemanticDistillModel:
    vision_encoder    = MobileViTSmall(...)      # 学习中
    semantic_projector = SemanticProjector(...)   # 学习中
    text_projector    = TextProjector(...)        # 学习中 (teacher 侧)
```
- **训练目标**：`align(student_emb, teacher_emb)` — 视觉语义对齐文本语义
- **数据**：Venus 生成的美学文本 + 对应图片

### Stage B: DistillParamModel

```python
# training/semantic_distill/model.py
class DistillParamModel:
    vision_encoder     = from Stage A  # 冻结
    semantic_projector = from Stage A  # 冻结
    decoder           = LightroomDecoder(...)    # 新增，学习中
```
- **训练目标**：`param_loss(predicted, Expert C GT)`
- **数据**：MIT-Adobe FiveK Expert C 标注

### Stage C: TextConditionedModel

```python
# training/text_condition/model.py
class TextConditionedModel:
    vision_encoder     = from Stage B  # 冻结
    semantic_projector = from Stage B  # 冻结
    base_decoder      = from Stage B  # 冻结 (作为基准)
    text_encoder      = LightTextEncoder(...)    # 新增，学习中
    fusion            = FiLMFusion(...)           # 新增，学习中
    cond_decoder      = LightroomDecoder(...)     # 从 Stage B 复制初始化，学习中
```
- **训练目标**：`param_loss + consistency_loss + base_align_loss`
- **数据**：口语化指令 + 对应图片

---

## 来源总结

| 模块 | 来源 | 原始论文/库 |
|------|------|------------|
| MobileViTSmall | 基于已有工作 + 增强 | Apple MobileViT (ICLR 2022) |
| MV2Block | 复用 | MobileNetV2 (CVPR 2018) |
| SE 注意力 | 复用 | SENet (CVPR 2018) |
| FPN 融合 | 自行设计轻量版 | — |
| MiniLM-L6-v2 | 预训练模型 | sentence-transformers |
| FiLM 融合 | 自行实现 | FiLM (AAAI 2018) |
| SemanticProjector | 自行设计 | — |
| LightroomDecoder | 自行设计 | — |
| LightTextEncoder | 自行设计 | — |
| CrossAttentionFusion | 自行设计 | — |
| RefinementNetV4 | 自行设计 | — |
| DiffISP | 自行设计 | — |
| 三阶段训练框架 | 自行设计 | — |

---

## 参数量统计

| 组件 | 参数量 | 说明 |
|------|--------|------|
| MobileViTSmall | ~5.6M | 含 SE + FPN |
| SemanticProjector | ~130K | |
| LightroomDecoder | ~50K | |
| LightTextEncoder | ~0.5M | Stage C 新增 |
| FiLMFusion | ~130K | Stage C 新增 |
| **部署总计 (Stage C)** | **~6.4M** | Image + Text → 6 params |
| RefinementNetV4 | ~16M | 可选精修模块 |
