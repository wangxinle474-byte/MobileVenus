# 数据准备与处理流程

> 更新时间: 2026-04-30

---

## 数据集概览

| 数据集 | 用途 | 规模 | 存储位置 |
|--------|------|------|---------|
| MIT-Adobe FiveK | Stage B 参数训练 + 评估 | 5,000 张 | `E:/dataset/fivek_jpeg` |
| Venus Stage1 | Stage A 语义对齐 | 5,377 张 | `E:/dataset/data/Stage1/` |
| COCO 5K | Stage A 补充 | 5,000 张 | `E:/dataset/coco/` |
| 口语化指令数据 | Stage C 文本条件训练 | 合成生成 | `data/` |

---

## 预处理产物 (`data/` 目录)

| 文件 | 大小 | 生成脚本 | 说明 |
|------|------|---------|------|
| `venus_text_embeddings.npz` | ~20MB | `training/semantic_distill/embed_texts.py` | Venus 美学文本的 MiniLM-L6-v2 编码 |
| `fivek_text_embeddings.npz` | ~20MB | `tools/data/gen_fivek_embeddings.py` | FiveK 图片的 Venus 文本描述编码 |
| `coco5k_text_embeddings.npz` | ~7MB | `tools/data/gen_coco_embeddings.py` | COCO 图片的 Venus 文本描述编码 |
| `fivek_expert_params.json` | ~14MB | 外部提取 | FiveK 5位专家的 Lightroom 参数标注 |
| `fivek_expert_consensus.json` | ~2MB | `tools/data/compute_expert_consensus.py` | 5 专家共识权重 |
| `fivek_aesthetic_scores.json` | ~1MB | `tools/data/score_fivek_aesthetic.py` | FiveK 图片美学评分 |
| `venus_pseudo_labels.json` | ~3MB | `tools/data/extract_pseudo_labels.py` | Venus 伪标签 (美学分析结构化输出) |

---

## Stage A 数据流

```
原始图片 (Venus Stage1 / FiveK / COCO)
  │
  ├── Venus 7B 推理 → 自然语言美学描述
  │      (光线充足，色彩和谐，建议微调对比度...)
  │
  └── MiniLM-L6-v2 编码 → 384-D text embedding
         │
         └── 保存为 *.npz → Stage A 训练时加载
```

**生成文本 embedding：**
```bash
# Venus 文本 embedding
python training/semantic_distill/embed_texts.py

# FiveK 文本 embedding
python tools/data/gen_fivek_embeddings.py

# COCO 文本 embedding
python tools/data/gen_coco_embeddings.py
```

---

## Stage B 数据流

```
FiveK Expert C 标注 (fivek_expert_params.json)
  │
  ├── 5 专家共识加权 (compute_expert_consensus.py)
  │      → fivek_expert_consensus.json
  │
  └── 训练数据: (image, expert_params_normalized)
         │
         ├── dataset.py: 基础 FiveK 加载
         ├── dataset_expert.py: 多专家标注加载
         ├── dataset_aug.py: 数据增强版本
         ├── dataset_synth.py: 合成退化增强 (v7)
         └── dataset_ppr10k.py: PPR10K 补充数据
```

---

## Stage C 数据流

```
FiveK + PPR10K 图片
  │
  └── generate_instruction_data.py
         │
         ├── 参数差异分析 → 口语化指令模板生成
         │   例: Δ(EV)=+0.5 → "有点暗，提亮一些"
         │   例: Δ(WB)=-500K → "偏黄了，调冷一点"
         │
         └── 指令数据: (image, text_instruction, target_params)
```

**指令模板类型：**
- 单参数调整: "太暗了" → EV+
- 多参数联合: "冷色调清爽感" → WB↓ + Saturation↓
- 风格描述: "电影感" → Contrast+ + Shadows- + Saturation-
- 否定指令: "不要太鲜艳" → Saturation-

---

## 增强数据 (v14)

```
aug_venus_guided.json + aug_lut.json (22,394 条)
  │
  ├── rescore_with_aesexpert.py → AesExpert MLLM 全量重打分
  ├── rescore_with_aadb.py → AADB 美学重打分
  │
  └── merge_aesexpert_aadb.py → 双门槛过滤
         │
         └── AesExpert ≥ 7.0 AND AADB ≥ 4.5
               → ~3,400 条高质量增强样本
               → 与 FiveK Expert C 混合训练
```

---

## 评估数据

| 集合 | 数量 | 用途 |
|------|------|------|
| FiveK 验证集 | 50 张 | 主要评估 (PSNR/SSIM/MUSIQ) |
| FiveK 测试集 | 500 张 | 完整评估 |
| 退化测试集 | 合成生成 | v7 退化修复能力验证 |
