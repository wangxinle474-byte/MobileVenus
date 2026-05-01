# MobileVenus 项目架构总览

> 最后更新: 2026-03-26

## 一、核心创新

**自然语言语义 → 可解释相机参数**

Venus (7B VLM) 输出的自然语言图像分析是连接**视觉理解**和**参数预测**的桥梁。
我们将 Venus 的语义知识蒸馏到轻量端侧模型，实现：
- 推理时**无需大模型**，仅用 MobileViT (~6M) 即可预测 8 个 Lightroom 参数
- 参数**完全可解释**、用户可编辑
- 支持端侧实时推理

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    训练阶段                               │
│                                                         │
│  ┌──────────────────── Stage A: 语义对齐 ──────────────┐ │
│  │                                                     │ │
│  │  Venus 文本分析 ──→ MiniLM ──→ text_emb (384d)     │ │
│  │       ↕ 对齐                                        │ │
│  │  图片 ──→ MobileViT ──→ SemanticProjector ──→       │ │
│  │                         student_emb (256d)          │ │
│  │                                                     │ │
│  │  Loss: cosine_align + uniformity                    │ │
│  │  数据: Venus Stage1 图文对 (~5.3K)                   │ │
│  └─────────────────────────────────────────────────────┘ │
│                          ↓ 冻结backbone                   │
│  ┌──────────────────── Stage B: 参数微调 ──────────────┐ │
│  │                                                     │ │
│  │  图片 ──→ MobileViT ──→ SemanticProjector ──→       │ │
│  │                         semantic_emb                 │ │
│  │                             ↓                        │ │
│  │                      LightroomDecoder ──→ 8 参数     │ │
│  │                                                     │ │
│  │  Loss: weighted_param_loss (多专家共识加权)            │ │
│  │  数据: MIT-Adobe FiveK (5000张, 5专家)                │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                         │
│  ┌──────────────────── 辅助: 美学评分训练 ─────────────┐ │
│  │  AADB 数据集 → MobileViT → AestheticScorer (5维)   │ │
│  │  用于: 评估图像质量、辅助参数选择                      │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    推理阶段                               │
│                                                         │
│  输入图片 ──→ MobileViT ──→ SemanticProjector           │
│                                    ↓                     │
│                             LightroomDecoder             │
│                                    ↓                     │
│                           8 个 Lightroom 参数            │
│                                    ↓                     │
│                         ISP Pipeline 渲染                │
│                                    ↓                     │
│                              增强后图片                   │
│                                                         │
│  (无需 Venus 大模型, 端侧 ~6M 参数, <10ms 推理)          │
└─────────────────────────────────────────────────────────┘
```

---

## 三、数据使用

### 3.1 训练数据

| 数据集 | 规模 | 用途 | 路径 |
|--------|------|------|------|
| **Venus Stage1** | 42K 图 + 169K 对话 | Stage A 语义对齐 (提取图文对) | `E:\dataset\data\Stage1\` |
| ↳ 实际使用子集 | ~5,377 张有文本embedding的图片 | Stage A 训练 | `venus_text_embeddings.npz` |
| **MIT-Adobe FiveK** | 5,000 张 × 5 专家 = 46,167 条 | Stage B 参数微调 | `E:\dataset\fivek_jpeg\` |
| ↳ 训练/验证 | 4,500 / 500 | 参数预测 + 评估 | `data/fivek_expert_params.json` |
| **AADB** | ~10K 张, 11维属性标注 | 美学评分器训练 | `E:\dataset\AADB\` |

### 3.2 标注数据 (本地生成/处理)

| 文件 | 大小 | 内容 |
|------|------|------|
| `data/fivek_expert_params.json` | 14.3 MB | 5专家×5000张的 Lightroom 调参记录 |
| `data/fivek_expert_consensus.json` | 1.9 MB | 多专家共识权重 (加权训练) |
| `data/venus_text_embeddings.npz` | 20 MB | MiniLM 编码的 Venus 文本分析 embedding |
| `data/fivek_aesthetic_scores.json` | 1.2 MB | AADB 模型给 FiveK 图片的美学评分 |
| `data/venus_pseudo_labels.json` | 3.0 MB | Venus 生成的美学伪标签 (AutoDL) |

### 3.3 8 个 Lightroom 参数

| 参数 | 范围 | 语义含义 |
|------|------|----------|
| ev_compensation | [-3, +3] | 曝光补偿 (EV) |
| white_balance | [2000, 10000] K | 色温 (开尔文) |
| contrast | [-100, +100] | 对比度 |
| brightness | [-100, +100] | 亮度 |
| shadows | [-100, +100] | 阴影恢复 |
| highlights | [-100, +100] | 高光压缩 |
| saturation | [-100, +100] | 饱和度 |
| vibrance | [-100, +100] | 自然饱和度 |

---

## 四、模型架构

### 4.1 核心模型

| 模块 | 文件 | 参数量 | 功能 |
|------|------|--------|------|
| **MobileViT-Small** | `models/vision_encoder.py` | ~5.6M | 视觉特征编码 (含 SE + FPN) |
| **SemanticProjector** | `models/semantic_bridge.py` | ~0.2M | 384d → 256d 语义投影 |
| **LightroomDecoder** | `training/fivek_8param/model.py` | ~0.2M | 256d → 8 个参数 |
| **AestheticScorer** | `models/aesthetic_scorer.py` | ~1M | 5维美学评分 [0-10] |
| **ISP Pipeline** | `models/isp_pipeline.py` | - | 参数→图像渲染 (gamma-aware) |

### 4.2 模型变体

| 变体 | 结构 | Stage A | Stage B |
|------|------|---------|---------|
| **Baseline** | MobileViT → Projector → Decoder | ❌ 无 | FiveK 直接训练 |
| **Distill v1/v2** | 同上 | ✅ Venus 语义对齐 | FiveK 微调 |
| **Distill v3** (计划) | 同上 | ✅ 扩展 5.3K 数据 | FiveK 微调 |

### 4.3 数据流

```
训练:
  Image → MobileViT(384d) → SemanticProjector(256d) → [对齐 text_emb] → LightroomDecoder → 8参数

推理:
  Image → MobileViT(384d) → SemanticProjector(256d) → LightroomDecoder → 8参数 → ISP渲染 → 增强图
```

---

## 五、已完成的有效工作

### ✅ 1. AADB 美学评分器训练
- **数据**: AADB 8458 train / 500 val / 1000 test
- **结果**: 测试集 SRCC=0.4407
- **权重**: `checkpoints/aadb_aesthetic_full/best.pt`
- **作用**: 美学评分基础能力验证

### ✅ 2. Baseline (8参数直接回归)
- **训练**: AutoDL RTX 5090, 40 epochs
- **数据**: FiveK 4500 train / 500 val, 多专家共识加权
- **结果**: PSNR=32.05 dB, SSIM=0.9269
- **权重**: `checkpoints/fivek_8param/best.pt`

### ✅ 3. 语义蒸馏 Distill v1 + v2
- **Stage A**: Venus Stage1 3K 图文对, MiniLM embedding 对齐
- **Stage B v1**: 30 epochs → PSNR=33.09
- **Stage B v2**: 50 epochs, 优化解冻策略 → **PSNR=33.15 (+1.10 dB vs Baseline)**
- **权重**: `checkpoints/semantic_distill/` + `checkpoints/semantic_distill_v2/`

### ✅ 4. Gamma-aware ISP Pipeline
- sRGB ↔ 线性空间正确转换
- 色温基于 Planckian locus RGB 增益
- Shadows/Highlights 使用 luminance mask
- 处理顺序: WB → EV → Contrast → S/H → Sat → Vib

### ✅ 5. 完整评估体系
- `tools/eval_psnr_ssim.py` — PSNR/SSIM + 参数 MAE 评估
- `tools/eval_param_validity.py` — AADB 美学验证 (结论: AADB 不够敏感)
- `tools/gen_eval_images.py` — 批量生成评估图片 (50张×3组)
- `tools/venus_aesthetic_eval.py` — Venus 美学评价脚本 (AutoDL 端)

### ✅ 6. 数据准备
- Venus Stage1 图文对提取 + embedding 生成 (`tools/pack_stage1_data.py`)
- FiveK 多专家共识权重计算
- FiveK 美学评分 (AADB模型 + Venus伪标签)

---

## 六、关键实验结果

### 6.1 PSNR/SSIM (500张验证集)

| 模型 | PSNR ↑ | SSIM ↑ | vs Baseline |
|------|--------|--------|-------------|
| Baseline | 32.05 | 0.9269 | - |
| Distill v1 | 33.09 | 0.9257 | +1.04 dB |
| **Distill v2** | **33.15** | **0.9311** | **+1.10 dB** |

> +1.1 dB PSNR 提升在图像质量评估中是**显著**的

### 6.2 关键发现

1. **语义蒸馏有效**: 仅 3K Stage A 数据就带来 +1.1 dB 提升
2. **参数协调 > 单参数准确**: WB MAE 更高但整体 PSNR 更好
3. **EV/Shadows/Highlights 受益最大**: 与场景语义（光照、明暗）密切相关
4. **AADB 美学分不够敏感**: 对 FiveK 这类高质量图片的微调差异无法区分

---

## 七、项目文件结构

```
MobileVenus/
├── models/                          # 模型定义
│   ├── vision_encoder.py            # MobileViT-Small (5.6M, SE+FPN)
│   ├── semantic_bridge.py           # SemanticProjector + DistillLoss
│   ├── aesthetic_scorer.py          # 5维美学评分器
│   ├── isp_pipeline.py              # Gamma-aware ISP 渲染
│   ├── text_encoder.py              # MiniLM 文本编码器
│   ├── mobile_venus.py              # 主模型集成 (全模块整合)
│   ├── parameter_predictor.py       # 参数预测器 + 问题检测
│   ├── language_model.py            # TinyLLaMA 语言模型
│   └── suggestion_generator.py      # 建议生成模块
│
├── training/
│   ├── fivek_8param/                # Baseline 8参数训练
│   │   ├── config.py                # 参数范围 + 训练超参
│   │   ├── dataset.py               # 数据集 + 一致性权重
│   │   ├── model.py                 # FiveK8ParamModel + LightroomDecoder
│   │   ├── loss.py                  # 一致性加权损失
│   │   └── trainer.py               # 训练循环
│   ├── semantic_distill/            # ★ 语义蒸馏训练
│   │   ├── config.py                # 蒸馏配置
│   │   ├── model.py                 # SemanticDistillModel + DistillParamModel
│   │   ├── text_dataset.py          # Stage A 图文对数据集
│   │   ├── loss.py                  # 蒸馏损失
│   │   └── trainer.py               # Stage A/B 训练器
│   ├── train_aadb_aesthetic.py      # AADB 美学训练
│   ├── distillation.py              # 蒸馏辅助模块
│   └── dataset.py                   # 通用数据加载器
│
├── data/                            # 数据文件
│   ├── fivek_expert_params.json     # 5专家 Lightroom 参数 (46K条)
│   ├── fivek_expert_consensus.json  # 多专家共识权重
│   ├── fivek_aesthetic_scores.json  # AADB 美学评分
│   ├── venus_pseudo_labels.json     # Venus 文本分析 (伪标签)
│   └── venus_text_embeddings.npz    # MiniLM 文本 embedding (384d)
│
├── checkpoints/                     # 训练权重
│   ├── aadb_aesthetic_full/best.pt  # AADB 美学 (SRCC=0.44)
│   ├── fivek_8param/best.pt         # Baseline (PSNR=32.05)
│   ├── semantic_distill/stage_a/    # Stage A 语义对齐 (cos=0.999)
│   ├── semantic_distill/stage_b/    # Distill v1 (PSNR=33.09)
│   └── semantic_distill_v2/stage_b/ # ★ Distill v2 (PSNR=33.15)
│
├── tools/                           # 工具脚本
│   ├── demo_8param.py               # 8参数 demo
│   ├── demo_distill_compare.py      # ★ Baseline vs Distill 对比
│   ├── demo_predict_params.py       # 参数预测推理 demo
│   ├── eval_psnr_ssim.py            # ★ PSNR/SSIM + MAE 评估
│   ├── eval_param_validity.py       # AADB 美学验证
│   ├── gen_eval_images.py           # 批量生成评估图片
│   ├── venus_aesthetic_eval.py      # Venus 美学评价 (AutoDL)
│   ├── autodl_distill_full.sh       # AutoDL 完整蒸馏训练
│   ├── autodl_8param_train.sh       # AutoDL 8参数训练
│   ├── autodl_all_in_one.py         # AutoDL Venus 伪标签
│   ├── pack_stage1_data.py          # Stage1 数据打包
│   ├── compute_expert_consensus.py  # 专家一致性分析
│   ├── convert_fivek_dng_to_jpeg.py # FiveK DNG→JPEG
│   ├── score_fivek_aesthetic.py     # AADB 美学评分
│   ├── extract_pseudo_labels.py     # 参数伪标签提取
│   ├── analyze_expert_variance.py   # 专家方差分析
│   ├── check_data_quality.py        # 数据质量检查
│   └── plot_training.py             # 训练曲线绘制
│
├── evaluate/                        # 评估框架
│   ├── baselines.py                 # SOTA 方法基线
│   ├── run_comparison.py            # 对比评估
│   └── results/                     # 评估结果 (JSON + LaTeX)
│
├── docs/                            # 文档
│   ├── training_log.md              # 实验记录
│   ├── experiment_plan.md           # SOTA 对比方案
│   └── architecture_overview.md     # 本文件
│
├── inference/camera_controller.py   # 推理控制器
├── examples/smart_camera_demo.py    # 智能相机示例
└── images/architecture/             # 论文结构图 (v4-v7)
```

---

## 八、下一步计划

### 近期 (本周)

| 优先级 | 任务 | 状态 | 说明 |
|--------|------|------|------|
| **P0** | Stage1 图片上传 AutoDL | 🔄 进行中 | 百度网盘/直传 |
| **P0** | Distill v3 训练 (5.3K) | ⏳ 待数据就绪 | `autodl_distill_full.sh` |
| **P0** | Venus 美学评价 | ⏳ 待上传 | `venus_aesthetic_eval.py` 已就绪 |

### 中期 (1-2周)

| 优先级 | 任务 | 说明 |
|--------|------|------|
| **P1** | 统一 SOTA 评估协议 | 480p, Expert C, 与文献一致 |
| **P1** | 效率指标 | #Params, FLOPs, 推理延迟 |
| **P1** | PPR10K 跨数据集验证 | Zero-shot 泛化能力 |
| **P2** | 可解释性实验 | t-SNE, 参数可视化, 用户编辑 demo |

### 远期 (论文提交前)

| 任务 | 说明 |
|------|------|
| 完整 SOTA 对比表 | HDRNet, 3D LUT, CSRNet, SepLUT 等 |
| 用户研究 | A/B 测试主观偏好 |
| 端侧部署 | ONNX 导出, Android/iOS demo |

---

## 九、硬件环境

| 环境 | 配置 | 用途 |
|------|------|------|
| **本地** | RTX 4060 Laptop 8GB + 15.7GB RAM | 推理、评估、demo |
| **AutoDL** | RTX 5090 32GB (按需) | 训练、Venus 推理 |
| **数据存储** | 本地 `E:\dataset\` + 百度网盘 SVIP | 数据集管理 |
