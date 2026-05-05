# IntelligenceCamera 资产清单

> 更新时间: 2026-05-01  
> 三端: 本地 Windows (全量代码) + AutoDL GPU (仅训练代码) + GitHub (全量备份)

---

## 〇、目录分工

### 本地 (全量，完整项目)
```
E:\智能相机\Venus_CVPR2026-main\IntelligenceCamera\
├── models/           ← 模型定义
├── training/         ← 训练框架
├── scripts/          ← 训练脚本 + 同步脚本
├── tools/            ← 全部工具 (data/eval/train/demo/plot)
├── APP/              ← Django+Vue3 演示应用 (仅本地)
├── docs/             ← 文档+架构图 (仅本地)
├── images/           ← 论文图片 (仅本地)
├── examples/         ← 示例代码 (仅本地)
├── inference/        ← 推理接口 (仅本地)
├── evaluate/         ← 评估框架 (仅本地)
├── data/             ← 数据配置模板 (仅本地)
├── training/main/train_v12_refine_hd.py
└── training/main/train_v6_stage_a.py
```

### AutoDL (仅训练相关)
```
/root/autodl-tmp/IntelligenceCamera/
├── models/           ← 6 个核心模型文件 (排除 DEPRECATED)
├── training/         ← 完整训练框架
├── scripts/          ← 训练脚本
├── tools/eval/       ← 评估工具
├── tools/data/       ← 数据处理工具
├── tools/train/      ← 训练辅助工具
├── training/main/train_v12_refine_hd.py
├── training/main/train_v6_stage_a.py
└── requirements.txt

/root/autodl-tmp/
├── checkpoints/      ← 训练产出 (1.4G)
├── data/             ← 标注+embedding (129M)
├── fivek_jpeg/       ← 原图 5121张 (227M)
├── fivek_expert_c/   ← Expert C GT (289M)
└── PPR10K/           ← PPR10K 数据集 (111G)
```

### 同步方式
```bash
# Linux/Mac:
bash scripts/local/sync_to_autodl.sh <autodl_ssh_host> <port>

# Windows PowerShell:
.\scripts\local\sync_to_autodl.ps1 -Host "root@connect.xxx.seetacloud.com" -Port 12345
```

---

## 一、Checkpoints (AutoDL: `/root/autodl-tmp/checkpoints/`, 共 1.4G)

### 核心管线 (当前使用)

| 版本 | 阶段 | 路径 | 大小 | 日期 | 说明 |
|------|------|------|------|------|------|
| **v6** | Stage A | `distill_v6/stage_a/best.pt` | 7.4M | 03-31 | ✅ **所有下游共享的冻结 backbone** |
| v6 | Stage A | `distill_v6/stage_a/final.pt` | 7.4M | 03-31 | 最终 epoch |
| v6 | Stage B | `distill_v6/stage_b/best.pt` | 7.4M | 04-07 | 基础参数预测 |
| **v7** | Stage B | `distill_v7/stage_b/best.pt` | 7.4M | 04-07 | ✅ **Stage C 基座** (退化增强+对比学习, Venus=5.38) |
| **v8** | Stage B | `distill_v8/stage_b/best.pt` | 7.4M | 04-13 | ✅ **RefinementNet 基座** (MUSIQ=4.446) |
| **v8** | Stage C | `distill_v8/stage_c/best.pt` | 13M | 04-13 | ✅ **文本条件化模型** (LightTextEncoder+FiLM) |
| **v4** | Refine | `refinement_v4/best.pt` | 61M | 04-16 | ✅ **RefinementNetV4** (双分支, ~16M params) |

### 历史迭代

| 版本 | 阶段 | 大小 | 日期 | 说明 |
|------|------|------|------|------|
| v4 | Stage A/B | 7.4M | 03-27 | 早期实验 |
| v5 | Stage A/B | 7.4-7.5M | 03-29 | 中间迭代 |
| v9 | Stage B | 7.4M | 04-13 | 实验分支 |
| v10 | Stage B | 7.4M | 04-13 | 实验分支 (含 ep10-50) |
| neural_isp | — | 6.3M | 04-13 | NeuralISP 实验 (ep10-80) |
| refinement | v1 | 529K | 04-13 | 最早的轻量 RefineNet (~130K params) |
| refinement_v2 | — | 7.4M | 04-14~15 | U-Net RefineNet (~1M params, ep10-90) |
| refinement_v3 | — | 35M | 04-15~16 | 中等 RefineNet (ep10-120) |
| refinement_v5 | ❌ | 35M | 04-16 | 失败实验 (w8_failed) |

---

## 二、数据集 (AutoDL: `/root/autodl-tmp/`)

### 图像数据

| 数据集 | 路径 | 大小 | 数量 | 说明 |
|--------|------|------|------|------|
| **FiveK JPEG** | `fivek_jpeg/` | 227M | 5121 张 | 原始图片 (模型输入) |
| **FiveK Expert C** | `fivek_expert_c/` | 289M | ~5000 张 | Expert C 处理后 GT |
| **PPR10K** | `PPR10K/` | 111G | — | 大规模人像修图数据集 |

### 标注/Embedding 数据

| 文件 | 路径 | 大小 | 说明 |
|------|------|------|------|
| `fivek_expert_params.json` | `data/` | 14M | 5 位专家的 Lightroom 参数标注 |
| `fivek_expert_consensus.json` | `data/` | 1.8M | 专家一致性权重 |
| `fivek_text_embeddings.npz` | `data/` | 19M | MiniLM 文本 embedding (Stage A) |
| `fivek_venus_embeddings.npz` | `data/` | 6.2M | Venus 语义 embedding |
| `fivek_venus_labels.json` | `data/` | 2.9M | Venus 美学描述标签 |
| `instruction_data.json` | `data/` | 65M | Stage C 口语化指令数据 |
| `ppr10k_params.json` | `data/` | 20M | PPR10K 参数标注 |
| `venus_eval_results_all.json` | `data/` | 180K | Venus 评估结果 |

### 预训练模型缓存

| 路径 | 大小 | 说明 |
|------|------|------|
| `hf_cache/` | 33G | HuggingFace 模型缓存 (MiniLM, MUSIQ 等) |
| `models/` | 32G | 大模型存储 |
| `Venus-Q-Stage1/` | 18G | Venus 量化模型 (Stage 1) |
| `LLaVA-main/` | 21M | LLaVA 代码 (参考用) |

---

## 三、代码仓库 (GitHub + 本地)

**GitHub**: `https://github.com/wangxinle474-byte/MobileVenus`  
**本地**: `E:\智能相机\Venus_CVPR2026-main\IntelligenceCamera\`  
**AutoDL**: `/root/autodl-tmp/IntelligenceCamera/` (12M, 需同步)

### 当前架构核心文件

```
models/
  vision_encoder.py      ← MobileViTSmall (SE+FPN, ~5.6M)
  semantic_bridge.py      ← SemanticProjector + TextProjector + LightroomDecoder
  diff_isp.py             ← 可微 Lightroom ISP v2 (8步渲染)
  refinement_net_v4.py    ← 双分支像素精修 (~16M)
  isp_pipeline.py         ← 非可微 ISP 渲染
  aesthetic_scorer.py     ← AADB + MUSIQ 美学评分

training/
  semantic_distill/       ← Stage A (语义对齐) + Stage B (参数预测)
    model.py              ← SemanticDistillModel + DistillParamModel
    config.py, loss.py, trainer.py, embed_texts.py, text_dataset.py
  text_condition/         ← Stage C (文本条件化)
    model.py              ← TextConditionedModel + LightTextEncoder + FiLMFusion
  fivek_8param/           ← 数据加载 + 配置
    config.py, dataset.py, dataset_expert.py, dataset_ppr10k.py, model.py, loss.py

scripts/                  ← 训练脚本
  training/main/train_v6_stage_a.py, train_v6_stage_b.py, train_v7_stage_b.py, training/main/train_stage_c.py
training/main/train_v12_refine_hd.py    ← RefinementNet v12 训练 (512×512, MUSIQ主导)

evaluate/                 ← 评估框架 + 结果
inference/                ← 推理接口 (predict.py, camera_controller.py)
tools/                    ← 工具链 (data/demo/eval/plot/train)
APP/                      ← Django + Vue3 演示应用
docs/                     ← 文档 + 架构图脚本
```

### 遗留文件 (已标记 DEPRECATED)

```
models/mobile_venus.py         ← 旧版 1B 主模型
models/parameter_predictor.py  ← 旧版 5 参数预测器
models/language_model.py       ← TinyLLaMA
models/suggestion_generator.py ← 建议生成器
models/text_encoder.py         ← 旧版文本编码器
```

---

## 四、评估结果 (AutoDL)

| 路径 | 大小 | 说明 |
|------|------|------|
| `unified_eval/` | 7.8M | 标准评估结果 |
| `unified_eval_hires/` | 23M | 高分辨率评估结果 |
| `unified_eval_nisp/` | 8.1M | NeuralISP 评估结果 |
| `evaluate/results/` | — | 对比结果 JSON + LaTeX 表格 |

---

## 五、训练管线总结

```
Stage A (v6)                     → distill_v6/stage_a/best.pt (冻结)
  ↓
Stage B (v6→v7→v8)              → distill_v7/stage_b/best.pt (Stage C 基座)
  │                              → distill_v8/stage_b/best.pt (Refine 基座)
  ↓
Stage C (v8)                     → distill_v8/stage_c/best.pt
  ↓
RefinementNet (v1→v2→v3→v4)     → refinement_v4/best.pt (当前最优)
```

### 关键指标

| 模型 | MUSIQ | Venus评分 | 备注 |
|------|-------|-----------|------|
| v7 Stage B | — | 5.38 | 美学最优参数预测 |
| v8 Stage B | 4.446 | — | MUSIQ最优参数预测 |
| RefinementNetV4 | 目标8.5+ | — | 突破全局参数天花板4.75 |

---

## 六、待办 / 需要注意

1. **AutoDL 代码同步**: `/root/autodl-tmp/IntelligenceCamera/` 是旧版本，需要从 GitHub pull 最新代码
2. **PPR10K 数据集**: 111G 大数据集已准备好，可用于扩展训练
3. **清理建议**: `refinement_v5_w8_failed/` 可删除 (失败实验)；历史 checkpoint 的中间 epoch 可酌情清理节省空间
