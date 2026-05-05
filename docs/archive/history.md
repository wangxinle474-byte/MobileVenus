# MobileVenus 文档归档 (History)

> 本文件合并了项目早期（v1/v2，9参数版）的所有设计文档。
> 当前版本为 **v3（5参数精简版）**，仅保留 `README.md` 和 `docs/OVERVIEW_FIGURE_PROMPT.md`。
> 
> **归档日期**: 2026-03-20

---

## 📂 归档文档索引

| # | 原文件 | 标题 | 行数 | 说明 |
|---|--------|------|------|------|
| 1 | `TECHNICAL_DESIGN.md` | 技术设计文档 | 721 | 系统架构、蒸馏训练、量化、自适应调度、移动端推理引擎 |
| 2 | `MODEL_ARCHITECTURE.md` | 模型架构详解 | 578 | MobileViT-Small、美学评分器、TinyLLaMA、建议生成器、蒸馏损失、QAT |
| 3 | `SEMANTIC_TO_PARAMETER.md` | 语义到参数转换设计 | 437 | 参数映射规则库（9参数）、预测器原型、验证/平滑、UI 交互 |
| 4 | `APP_DESIGN.md` | App 设计文档 | 497 | UI/UX 布局、AR 辅助、iOS (Swift+CoreML)、Android (Kotlin+TFLite) |
| 5 | `FLOWCHART.md` | Venus vs MobileVenus 对比 | 485 | 训练流程、推理流程、架构对比、数据流对比 |
| 6 | `PROJECT_ROADMAP.md` | 项目路线图 | 532 | 7 阶段实施计划、成本预算、时间线、预期成果 |
| 7 | `DATASET_GUIDE.md` | 数据集构建指南 | 455 | 数据源、标注方案、格式定义 |
| 8 | `TRAINING_COMPLETE_GUIDE.md` | 完整训练指南 | 550 | 从环境配置到部署的完整流程 |
| 9 | `AUTODL_SETUP_GUIDE.md` | AutoDL 配置指南 | 504 | RTX 5090 租赁、环境安装、训练启动 |
| 10 | `PYTORCH28_COMPATIBILITY.md` | PyTorch 2.8 兼容性 | 248 | PyTorch 2.8.0 + CUDA 12.8 说明 |
| 11 | `README_WITH_VENUS_DATA.md` | Venus 数据训练方案 | 428 | 使用 Venus 数据集训练 MobileVenus |
| 12 | `old_readmes/README_SMART_CAMERA.md` | 智能相机 README | 316 | 早期智能相机系统说明 |
| 13 | `old_readmes/README_TRAINING.md` | 训练 README | 267 | 早期参数预测训练说明 |
| 14 | `OVERVIEW_FIGURE_PROMPT.md` (v1) | 单面板概览图提示词 | ~450 | 左→右流程图，9参数，中英文 |
| 15 | `OVERVIEW_FIGURE_PROMPT.md` (v2) | 双面板学术图提示词 | ~350 | GM-MOE 风格 (a)+(b)，9参数，中英文 |

---

## 1. 技术设计文档 (TECHNICAL_DESIGN.md)

### 核心架构
```
相机帧 → 预处理 → 特征提取 → 评分 → (条件)建议生成 → UI 渲染
  ↓                                                    ↑
  └──────────── 自适应调度控制 ────────────────────────┘
```

### 模型组件 (原版)
- **MobileViT-Small**: ~5M 参数, 输入 224×224, 输出 384-d
- **Vision Projection**: Linear(384→512) + GELU
- **AestheticScorer**: 5 维度评分头, ~1M 参数
- **TinyLLaMA-1B**: 16 层 Transformer, vocab=32000
- **SuggestionGenerator**: 条件编码器 + 10 类建议模板

### 知识蒸馏配置
- Teacher: Venus-Q-Stage2 (7B)
- Student: MobileVenus (1B)
- 损失: α=0.5 评分 + β=0.3 特征 + γ=0.2 响应
- Temperature: 4.0, LR: 1e-4, Epochs: 10

### 量化
- INT8: 500MB → 125MB, 精度损失 <2%
- INT4: 500MB → 62.5MB, 精度损失 3-5%

### 自适应调度器
- 场景复杂度 + 运动检测 + 电量感知
- 动态采样率: 0.5-2 fps
- 模式: ultra_lite / balanced / full

### 移动端推理
- CoreML (iOS): AVFoundation + Neural Engine
- TFLite (Android): NNAPI 加速
- 推理缓存 + 异步处理

### 性能目标
| 指标 | 目标 |
|------|------|
| 模型大小 | <500MB |
| 推理延迟 (iPhone) | <100ms |
| 电量消耗 (1h) | <10% |
| 准确率 (vs Venus) | >90% |

---

## 2. 模型架构详解 (MODEL_ARCHITECTURE.md)

### MobileViT-Small 各 Stage
| Stage | 输出通道 | Transformer dim | Heads | Layers |
|-------|---------|----------------|-------|--------|
| 3 | 64 | 96 | 4 | 2 |
| 4 | 80 | 120 | 4 | 4 |
| 5 | 96 | 144 | 4 | 3 |

### 美学评分权重
| 维度 | 权重 |
|------|------|
| Composition | 25% |
| Lighting | 20% |
| Color | 20% |
| Clarity | 20% |
| Subject | 15% |

### 推理延迟分解
| 阶段 | 延迟 |
|------|------|
| 预处理 | 5ms |
| 视觉编码 | 30ms |
| 美学评分 | 20ms |
| 建议生成 | 20ms |
| 后处理 | 5ms |
| **总计** | **80ms** |

### 优化效果
| 技术 | 大小减少 | 延迟减少 | 精度损失 |
|------|---------|---------|---------|
| 知识蒸馏 | 85% | 70% | 8% |
| INT8 量化 | 75% | 30% | 2% |
| 算子融合 | - | 15% | 0% |

---

## 3. 语义到参数转换 (SEMANTIC_TO_PARAMETER.md)

### 参数映射规则 (原 9 参数版)

**光线问题**:
- 轻微欠曝 → EV +0.5
- 中度欠曝 → EV +1.0, ISO 400, HDR ON
- 严重欠曝 → EV +1.5, ISO 800, HDR ON
- 逆光 → EV +2.0, HDR ON

**色彩问题**:
- 偏暖 → WB 4500K
- 偏冷 → WB 7000K
- 正常 → WB 5500K

**构图问题**:
- 主体过小 → Zoom 2x
- 背景杂乱 → 人像模式 + Zoom 2x

### 执行流程
```
预测参数 → 安全性检查 → 用户确认(可选) → 平滑过渡 → 调用相机 API → 重新评分
```

---

## 4. App 设计 (APP_DESIGN.md)

### 核心功能
1. 实时美学评分 (5维度)
2. 智能建议 (低分时触发)
3. AR 辅助 (三分线、水平仪、主体追踪)
4. 个性化学习
5. Before/After 对比

### UI 布局
- 顶部: 工具栏
- 中间: 相机预览 + AR 叠加
- 评分面板: 可折叠 5 维度评分条
- 建议卡片: 滑动查看
- 底部: 模式/网格/闪光/拍摄控制

### 技术栈
- **iOS**: Swift + SwiftUI + CoreML + AVFoundation
- **Android**: Kotlin + Jetpack Compose + TFLite + Camera2/CameraX

### 性能优化
- 异步推理 (后台线程)
- 帧跳过 (自适应频率)
- 图像降采样 224×224
- 对象池复用
- 温度监控降载

---

## 5. 流程图对比 (FLOWCHART.md)

### Venus vs MobileVenus 关键指标
| 指标 | Venus | MobileVenus |
|------|-------|-------------|
| 模型大小 | 14GB | 500MB |
| 参数量 | 7B | 1B |
| 推理延迟 | 2.5s | 80ms |
| 运行环境 | 服务器 GPU | 手机 |
| 分析频率 | 单次 | 1-2 fps |
| 准确率 | 100% | 92% |

### Venus 训练
- Stage 1: AesGuide + 开源 → LoRA 微调 Qwen-7B → 美学批判能力
- Stage 2: Stage2 数据 → 继续 LoRA → 裁剪+CoT 能力

### MobileVenus 训练
- 知识蒸馏 (Venus-7B → MobileVenus-1B) + INT8 量化 + CoreML/TFLite 转换

---

## 6. 项目路线图 (PROJECT_ROADMAP.md)

### 7 阶段计划
| Phase | 内容 | 时间 | 成本 |
|-------|------|------|------|
| 1 | 环境搭建 | 1 周 | ¥50-100 |
| 2 | 数据准备 | 1-2 周 | ¥0 |
| 3 | 模型训练 | 2-3 周 | ¥770-1350 |
| 4 | 模型评估 | 1 周 | ¥50 |
| 5 | 移动端部署 | 2-3 周 | ¥5000-10000 |
| 6 | 用户测试 | 2-3 周 | ¥0-500 |
| 7 | 论文撰写 | 1-2 月 | ¥0 |

**总计**: 3-4 个月, ¥5870-11950 (最低 ¥100)

### 目标会议
- CVPR 2027 (截稿 2026年11月)
- ICCV 2027 (截稿 2027年3月)

---

## 7. 数据集指南 (DATASET_GUIDE.md)

### 推荐数据集
- MIT-Adobe FiveK: 5000 张专业修图
- AVA: 255K 美学评分
- Venus Stage 1/2: 美学理解 + 高级分析

### 数据格式
图像 + 美学评分 + 问题类型 + 目标参数

---

## 8. 完整训练指南 (TRAINING_COMPLETE_GUIDE.md)

从环境配置 → 数据准备 → 模型训练 → 评估 → 部署的完整流程。
包含 DeepSpeed ZeRO-2/3 配置、AutoDL 平台操作步骤。

---

## 9. AutoDL 配置指南 (AUTODL_SETUP_GUIDE.md)

RTX 5090 (32GB) 租赁配置:
- PyTorch 2.8.0 + CUDA 12.8
- 一键安装脚本: `scripts/setup_autodl_pytorch28.sh`
- 预计训练成本: ¥70 (参数预测) / ¥200-400 (蒸馏)

---

## 10. PyTorch 2.8 兼容性 (PYTORCH28_COMPATIBILITY.md)

PyTorch 2.8.0 + CUDA 12.8 完全兼容 MobileVenus。
torch.compile 支持、flash attention 默认开启。

---

## 11. Venus 数据训练方案 (README_WITH_VENUS_DATA.md)

使用 Venus 官方数据集训练的完整方案:
- `tools/convert_venus_data.py` 转换数据
- `training/train_mobilevenue.sh` 启动训练
- 支持 `--merge_fivek` 合并 FiveK 数据

---

## 12-13. 早期 README (old_readmes/)

### README_SMART_CAMERA.md
早期智能相机系统说明，介绍语义到参数的自动转换概念。

### README_TRAINING.md
早期参数预测训练说明，基于 LLaVA 多模态大模型的训练方案。

---

## 14-15. OVERVIEW 图提示词 v1/v2

### v1: 单面板概览图
- 左→右流程图布局
- 6 个主要组件 (输入→编码器→评分+参数→语言→控制→输出)
- 9 参数输出 (EV/ISO/WB/快门/HDR/变焦/对焦/模式/闪光)
- 中英文双版本

### v2: 双面板学术架构图
- 参考 GM-MOE (CVPR) 风格
- (a) 核心模块内部: ProblemClassifier → CrossAttn(9 token) → Validation
- (b) 整体系统架构: 自上而下端到端管线
- 9 参数 + 完整数学符号标注
- 中英文双版本

> **注意**: v1/v2 均为 9 参数版本，已被 v3（5参数版）替代。
> v3 保留在 `docs/OVERVIEW_FIGURE_PROMPT.md` 中。

---

## 其他归档文件

### prompt.md
项目对话记录 (24 问)，记录了 2026-03-19 的设计讨论过程。

### scripts/quick_start_autodl.md
AutoDL 5 分钟快速开始指南。

---

**归档版本**: v3.0
**归档日期**: 2026-03-20
**当前活跃文档**: `README.md` + `docs/OVERVIEW_FIGURE_PROMPT.md` (仅 v3)
