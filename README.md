# IntelligenceCamera: 移动端实时美学指导系统

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![PSNR](https://img.shields.io/badge/PSNR-34.11_dB-brightgreen.svg)]()
[![SSIM](https://img.shields.io/badge/SSIM-0.9449-brightgreen.svg)]()
[![Venus](https://img.shields.io/badge/Venus_Aesthetic-5.38-blueviolet.svg)]()

> **基于 Venus (CVPR 2026) 的移动端实时美学指导系统。核心创新：通过语义桥接蒸馏，将 Venus 7B 大模型的自然语言美学理解能力压缩到手机端轻量模型中，实现从语义理解到 6 个核心 Lightroom 参数的端到端预测。v7 版本引入退化增强对比学习，在牺牲技术还原度的前提下大幅提升美学质量和退化修复泛化能力。**

---

## 核心创新：语义桥接 (Semantic Bridge)

传统方法直接从图像特征预测相机参数，缺乏语义理解。IntelligenceCamera 的核心创新是引入 **语义桥接蒸馏**：

```
训练阶段:
  图片 → Venus 教师 → 自然语言美学分析 → 文本编码器 → 语义embedding (teacher)
  图片 → MobileViT  → 语义投影层       →              语义embedding (student)
                                                            ↓
                                        对齐 student ≈ teacher (蒸馏损失)
                                                            ↓
                                                      参数解码器 → 相机参数

推理阶段 (手机端):
  图片 → MobileViT → 语义投影层 → 语义embedding → 参数解码器 → 相机参数
  (无需 Venus，无需文本处理，轻量级实时运行)
```

**关键思想**：Venus 的自然语言分析（如 "光线偏暗，建议提高曝光"）蕴含了对图像的深层理解。通过语义蒸馏，轻量模型学会了类似的 "理解" 能力，再将这种理解转化为具体的参数建议。

### 与 Venus 的对比

| 特性 | Venus | IntelligenceCamera |
|------|-------|-------------|
| **模型大小** | 14GB (7B) | ~30MB (8M) |
| **推理延迟** | 2-3s | <80ms |
| **运行平台** | 服务器 GPU | 手机/边缘设备 |
| **输出** | 自然语言分析 | 6 个 Lightroom 参数 (EV/WB/Contrast/Shadows/Highlights/Saturation) |
| **语义理解** | 原生 (语言模型) | 蒸馏习得 (语义桥接) |

---

## 项目结构

```
IntelligenceCamera/
├── models/                              # 模型定义
│   ├── vision_encoder.py               # MobileViT-Small (SE + FPN, ~5.6M)
│   ├── semantic_bridge.py              # ★ SemanticProjector + ParameterDecoder
│   ├── diff_isp.py                     # ★ 可微 ISP 渲染管线 (Lightroom 风格)
│   ├── parameter_predictor.py          # 参数预测器
│   ├── refinement_net_v4.py            # 图像精修网络 V4 双分支 (~16M)
│   ├── isp_pipeline.py                 # Gamma-aware Lightroom ISP
│   ├── aesthetic_scorer.py             # 美学评分器
│   ├── neural_isp.py                   # Neural ISP (已搁置)
│   ├── language_model.py               # 语言模型接口
│   ├── text_encoder.py                 # 文本编码器
│   ├── intelligence_camera.py          # 智能相机主模型
│   └── suggestion_generator.py         # 建议生成器
│
├── training/
│   ├── fivek_8param/                   # Baseline 8参数训练
│   │   ├── config.py                   #   参数范围 + 训练超参
│   │   ├── dataset*.py                 #   FiveK / Expert / Augment / PPR10K / Synth 数据集
│   │   ├── model.py                    #   FiveK8ParamModel + LightroomDecoder
│   │   ├── loss.py                     #   ConsensusWeightedLoss + CombinedLoss
│   │   └── trainer.py                  #   训练/验证循环
│   ├── semantic_distill/               # ★ Stage A/B: Venus 语义蒸馏
│   │   ├── model.py                    #   SemanticDistillModel + DistillParamModel
│   │   ├── text_dataset.py             #   Stage A 图文对数据集
│   │   ├── trainer.py                  #   Stage A / Stage B 训练器
│   │   ├── embed_texts.py              #   文本 embedding 生成 (MiniLM)
│   │   ├── loss.py                     #   对齐损失
│   │   └── config.py                   #   DistillConfig
│   ├── text_condition/                 # ★ Stage C: 文本条件化 ISP 参数预测
│   │   ├── model.py                    #   TextConditionedModel + FiLM + LightTextEncoder
│   │   ├── dataset.py                  #   指令数据集
│   │   └── config.py                   #   TextCondConfig
│   ├── aesthetic_loss.py               # MUSIQ 美学损失封装
│   └── dataset.py                      # 通用数据集
│
├── scripts/                              # 训练脚本 (全版本)
│   ├── training/main/train_v6_stage_a.py              #   v6 Stage A 语义对齐
│   ├── train_v6_stage_b.py              #   v6 Stage B 参数预测
│   ├── train_v7_stage_b.py              #   v7 退化增强 + 对比学习
│   ├── training/main/train_stage_c.py                 #   v8 Stage C 文本条件
│   ├── training/main/train_v9_aesthetic.py            #   v9 美学感知微调
│   ├── training/main/train_v10_e2e.py                 #   v10 端到端 image_loss
│   ├── training/main/train_v11_refine.py              #   v11 RefinementNet 精修
│   ├── training/main/train_v12_refine_hd.py           #   v12 大模型 512 训练
│   ├── training/main/train_v13_multiscale.py          #   v13 多尺度 MUSIQ + EMA
│   └── training/main/train_neural_isp.py              #   Neural ISP (已搁置)
│
├── tools/                               # 工具脚本
│   ├── data/                           # 数据处理 (50+ 脚本, 详见 docs/data_preparation.md)
│   ├── eval/                           # 评估工具
│   │   ├── eval_psnr_ssim.py           #   主评估: PSNR/SSIM/MS-SSIM
│   │   ├── eval_multi_expert.py        #   多专家一致性验证
│   │   ├── eval_param_validity.py      #   参数有效性检查
│   │   ├── eval_unified_iqa.py         #   统一 IQA 评估
│   │   ├── eval_robustness.py          #   鲁棒性评估
│   │   ├── eval_v8_autodl.py           #   v6-v9 统一评估 (AutoDL)
│   │   └── standard_protocol.py        #   标准评估协议
│   ├── demo/                           # Demo + 可视化
│   ├── plot/                           # 论文图表生成
│   └── train/                          # AutoDL 训练脚本
│
├── evaluate/                            # 论文评估框架
│   ├── baselines.py                    # Baseline 方法实现
│   ├── run_comparison.py               # 对比实验
│   ├── run_venus_inference.py          # Venus 推理
│   └── results/                        # 评估结果 (JSON + LaTeX)
│
├── data/                                # 预处理数据 (详见 docs/data_preparation.md)
│   ├── fivek_expert_params.json        #   FiveK 5专家 Lightroom 参数
│   ├── fivek_expert_consensus.json     #   专家共识权重
│   ├── fivek_aesthetic_scores.json     #   FiveK 美学评分
│   ├── venus_pseudo_labels.json        #   Venus 伪标签
│   ├── venus_text_embeddings.npz       #   Venus 文本 MiniLM embedding
│   ├── fivek_text_embeddings.npz       #   FiveK 文本 MiniLM embedding
│   └── coco5k_text_embeddings.npz      #   COCO 文本 MiniLM embedding
│
├── images/
│   ├── architecture/                   # 架构图 (PNG/SVG/PDF)
│   └── results/                        # 论文结果图表
│
├── docs/
│   ├── experiment_log.md               # ★ 全版本实验记录 (Baseline → v14)
│   ├── model_architecture.md           # ★ 模型架构与来源文档
│   ├── data_preparation.md             # ★ 数据准备与处理流程
│   ├── architecture.md                 # 系统架构设计
│   ├── paper_figures.md                # 论文图表说明
│   ├── autodl_setup.md                 # AutoDL 环境配置
│   ├── prd.md                          # 产品需求文档
│   └── history.md                      # 项目历史
│
├── inference/                           # 推理接口
├── APP/                                 # Web 应用 (Django + Vue3)
└── requirements.txt
```

---

## 训练流程

### Phase 1: 美学评分器预训练 ✅

```bash
python training/train_aadb_aesthetic.py
```

- 数据: AADB 10K 图像, 11 维属性标注
- 结果: 测试集 SRCC = 0.4407
- 权重: `checkpoints/aadb_aesthetic_full/best.pt`

### Phase 2: FiveK 8参数 Baseline 训练 ✅

```bash
python -m training.fivek_8param --jpeg_dir /path/to/fivek_jpeg --epochs 40
```

- 数据: FiveK 5000 图, 5 专家共识加权
- 结果: PSNR=32.05, SSIM=0.9269
- 权重: `checkpoints/fivek_8param/best.pt`

### Phase 3: Venus 语义蒸馏 ✅

**Step 1 — 生成文本 embedding**:
```bash
python training/semantic_distill/embed_texts.py \
    --json outputs/fivek_stage_a.json \
    --image_root E:/dataset/fivek_jpeg \
    --output data/fivek_text_embeddings.npz
```

**Step 2 — Stage A 语义对齐** (在 AutoDL 运行):
```bash
python tools/train_distill_v4_autodl.py --skip_embed
# Stage A: best cos_sim=0.999, 约 20 min (RTX 5090)
```

**Step 3 — Stage B 参数微调**:
```bash
python tools/train_distill_v4_autodl.py --skip_embed --skip_stage_a
# Stage B: val_loss=0.0034, 约 14 min
```

- Distill v2 (COCO Stage A): PSNR=33.15 (+1.10 dB), SSIM=0.9311
- Distill v4 (FiveK Stage A): PSNR=33.11 (+1.06 dB), SSIM=0.9326
- **Distill v5 (Expert C 单专家精准监督): PSNR=34.11 (+2.06 dB), SSIM=0.9449 ★ 技术最优**
- **Distill v6 (Venus NL + 6参数): PSNR=32.05, SSIM=0.9289**
- **Distill v7 (退化增强 + 对比学习): PSNR=25.97, Venus=5.38 ★ 美学最优**

---

## 模型架构

### 核心组件

| 组件 | 参数量 | 说明 |
|------|--------|------|
| **MobileViT-Small** | ~1.5M | 视觉编码器 (SE + FPN) |
| **SemanticProjector** | ~0.1M | 视觉→语义投影 (核心创新) |
| **LightroomDecoder** | ~0.17M | 语义→6参数解码 |
| **总计 (推理)** | **~1.93M** | **FP16 约 4MB** |

### 6 参数预测 (Lightroom)

| 参数 | 范围 | 语义来源 | 专家一致性 |
|------|------|---------|----------|
| **曝光补偿 (EV)** | [-3.0, +3.0] | 光线语义 | 17.5% (难) |
| **白平衡 (WB)** | [2000K, 10000K] | 色彩语义 | 6.9% |
| **对比度 (Contrast)** | [-100, 100] | 清晰度语义 | 14.5% (难) |
| **阴影 (Shadows)** | [-100, 100] | 动态范围 | 8.3% |
| **高光 (Highlights)** | [-100, 100] | 动态范围 | 18.3% (难) |
| **饱和度 (Saturation)** | [-100, 100] | 色彩语义 | 2.2% (易) |

> v6 起移除 Brightness 和 Vibrance：Brightness 与 EV Compensation 功能冗余（均控制整体亮度），Vibrance 与 Saturation 效果高度重叠。减少参数降低了学习难度，提升训练稳定性。
>
> 专家一致性 = 5位专家间标准差 / 参数范围。越低表示专家越一致，越好预测。

---

## 数据来源

| 数据集 | 规模 | 用途 | 状态 |
|--------|------|------|------|
| **AADB** | ~10K 图 + 11维标注 | 美学评分器预训练 | ✅ 完成 |
| **MIT-Adobe FiveK** | 5000图 × 5专家 × 9参数 = 46K条 | 参数预测训练 (v1–v7) | ✅ 完成 |
| **COCO val2017** | 5000 图 | Venus 文本分析 (伪标签) | ✅ 生成完成 |
| **FiveK (multi-expert)** | 46K条 × 3专家 | 跨标注者泛化验证 | ✅ Expert-A +1.35 dB |

---

## 快速开始

### 环境配置

```bash
conda create -n intelligence_camera python=3.10 -y
conda activate intelligence_camera
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 推理示例

```python
import torch
from training.fivek_8param import FiveK8ParamModel, PARAM_NAMES

# 加载 8 参数模型
model = FiveK8ParamModel(image_size=224)
state = torch.load('checkpoints/fivek_8param/best.pt', map_location='cpu')
model.load_state_dict(state['model_state_dict'])
model.eval()

# 推理
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
img = transform(Image.open('photo.jpg').convert('RGB')).unsqueeze(0)
with torch.no_grad():
    out = model(img)
for p in PARAM_NAMES:
    print(f"{p}: {out['raw_params'][p].item():.1f}")
```

### 评估 (复现论文结果)

```bash
# 主评估: PSNR/SSIM/MS-SSIM (全部模型, 500张验证集)
python tools/eval/eval_psnr_ssim.py --model all --num_images 500 \
    --output outputs/eval/eval_psnr_ssim.txt

# 多专家一致性验证
python tools/eval/eval_multi_expert.py --model all --num_images 200 \
    --output outputs/eval/eval_multi_expert.txt

# 生成论文图表
python tools/plot/plot_results.py
# → images/results/ 下生成 8 张图表
```

### Demo (可视化对比)

```bash
# Baseline vs Distill v5 对比
python tools/demo/demo_distill_compare.py
# 输出: outputs/demo/demo_distill_compare.png

# 单图参数预测
python tools/demo/demo_8param.py --image photo.jpg
```

---

## 论文

- **标题**: IntelligenceCamera: Semantic Bridge Distillation for Mobile Camera Parameter Prediction
- **目标**: CVPR 2027 / ICCV 2027

### 主要贡献

1. **语义桥接蒸馏** — 首次将 VLM 的自然语言理解能力蒸馏到端侧视觉模型，通过语义 embedding 空间对齐实现
2. **语义驱动的参数预测** — 基于图像语义理解（而非底层统计量）预测相机参数
3. **完整的移动端系统** — 从 Venus 7B 到 7M 参数，<80ms 延迟，支持实时拍摄指导

---

## 实验结果

### PSNR/SSIM/MS-SSIM 评估 (500张验证集, gamma-aware ISP)

| 模型 | Stage A 数据 | PSNR ↑ | SSIM ↑ | MS-SSIM ↑ | vs Baseline |
|------|------------|--------|--------|-----------|-------------|
| Baseline | — | 32.05 | 0.9269 | 0.9805 | — |
| Distill v1 | 3K COCO | 33.09 | 0.9257 | 0.9802 | +1.04 dB |
| Distill v2 | 3K COCO | 33.15 | 0.9311 | 0.9813 | +1.10 dB |
| Distill v3 | 5.3K COCO | 32.85 | 0.9278 | 0.9811 | +0.80 dB |
| Distill v4 | 2K FiveK | 33.11 | 0.9326 | 0.9813 | +1.06 dB |
| **Distill v5** | **2K FiveK + Expert C** | **34.11** | **0.9449** | **0.9849** | **+2.06 dB** |
| Distill v6 | 4.5K FiveK Venus NL | 32.05 | 0.9289 | 0.9779 | +0.00 dB |
| Distill v7 | 复用 v6 + 退化增强 | 25.97 | 0.8676 | 0.9459 | −6.08 dB |

> **v5** = 技术还原最优 (Expert C 单专家精准监督)。v7 PSNR 下降但美学评分大幅提升（见下文）。

### 多专家一致性验证 (200张, FiveK Expert-A 独立测试)

| 模型 | Expert-Default | Expert-A (独立) | 泛化提升 |
|------|---------------|----------------|----------|
| Baseline | 32.17 dB | 32.47 dB | — |
| **Distill v2** | **33.10 dB** | **33.82 dB** | **+1.35 dB** |
| Distill v4 | 33.11 dB | 33.73 dB | +1.26 dB |

> Distill 模型在**从未见过的独立专家标注**上同样超越 Baseline，验证跨标注者泛化性。

### 参数 MAE (Distill v4 vs Baseline)

| 参数 | Baseline | Distill v4 | Delta |
|------|----------|------------|-------|
| EV | 0.09 | **0.06** | ↓ 33% |
| WB | **501.98** | 618.73 | ↑ 23% |
| Contrast | **7.47** | 7.31 | ↓ 2% |
| Brightness | **3.77** | **3.77** | = |
| Shadows | 6.38 | **5.85** | ↓ 8% |
| Highlights | 7.53 | **7.10** | ↓ 6% |
| Saturation | **1.35** | 1.52 | ↑ 13% |
| Vibrance | **6.06** | 6.81 | ↑ 12% |

> **关键发现**: WB MAE 更高但整体 PSNR/SSIM 更好 → 语义蒸馏学到了参数间协调，而非逐参数精度

### Venus 美学评估 (50张干净图, AutoDL)

| 维度 | 原图 | Distill v6 | Distill v7 |
|------|------|------------|------------|
| composition | 5.23 | 4.98 | **5.34** |
| lighting | 5.08 | 5.08 | **5.52** |
| color | 5.10 | 5.08 | **5.34** |
| **overall** | **5.10** | **5.00 (−0.10)** | **5.38 (+0.28)** |

> **v7** 在所有美学维度超越原图和 v6，尤其 lighting (+0.44) 和 color (+0.24)。

### 退化修复泛化能力 (NR-IQA, v7 vs v6)

| 指标 | v6 退化修复 | v7 退化修复 | v7 更好？ |
|------|------------|------------|----------|
| NIQE ↓ | 5.18 | **5.04** | ✓ |
| BRISQUE ↓ | 28.53 | **27.77** | ✓ |
| MUSIQ ↑ | 56.42 | **58.68** | ✓ |
| DBCNN ↑ | 0.42 | **0.45** | ✓ |
| HyperIQA ↑ | 0.48 | **0.49** | ✓ |

> **v7 在 8/9 个 NR-IQA 指标上优于 v6**，退化增强对比学习显著提升修复泛化能力。

### 核心发现

1. **PSNR ≠ 美学**: v7 PSNR −6 dB 但 Venus +0.28，偏离 Expert C 不等于变差
2. **退化增强有效**: v7 退化修复在 8/9 个 NR-IQA 指标上优于 v6
3. **推荐评估方案**: PSNR (技术还原) + Venus (美学) + MUSIQ (感知质量) 三维评估

## 项目路线图

### 已完成 ✅
- [x] 核心模型实现 (MobileViT + SemanticBridge)
- [x] 美学评分器训练 (AADB, SRCC=0.4407)
- [x] FiveK 数据准备 (DNG→JPEG + 46K 专家参数)
- [x] 8参数 Baseline 训练 (PSNR=32.05 dB)
- [x] Gamma-aware ISP Pipeline
- [x] Distill v1–v4 迭代 (Stage A 数据源/域对齐实验)
- [x] Distill v5 (PSNR=34.11, SSIM=0.9449, Expert C 单专家精准监督) ← **技术最优**
- [x] Distill v6 (Venus NL 语义蒸馏 + 8→6 参数精简 + ISP NaN 修复)
- [x] Distill v7 (退化增强 + 对比学习, Venus=5.38) ← **美学最优**
- [x] PSNR/SSIM/MS-SSIM 完整评估体系 (500张)
- [x] 多专家一致性验证 (Expert-A +1.35 dB)
- [x] Venus 美学评价 (v6/v7 对比)
- [x] 9 指标 NR-IQA 全面评估 (退化检测 + 修复泛化)

### 计划中 📋
- [ ] SOTA 方法对比 (HDRNet / 3D LUT / CSRNet)
- [ ] 移动端部署 (ONNX / CoreML)
- [ ] 论文撰写 (CVPR 2027)

---

## 致谢

- [Venus (CVPR 2026)](https://github.com/PKU-ICST-MIPL/Venus_CVPR2026) — 教师模型
- [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) — 专家调参数据
- [AADB](https://github.com/aimerykong/AADB) — 美学属性数据集
- [sentence-transformers](https://www.sbert.net/) — 文本编码

---

**创建日期**: 2026-03-19  
**最后更新**: 2026-04-30  
**版本**: v14.0 (三阶段训练 + RefinementNet V4 + AesExpert 增强数据)  
**作者**: IntelligenceCamera Team
