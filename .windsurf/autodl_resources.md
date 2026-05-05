# AutoDL 服务器资源参考

> 本文件供 AI 助手快速了解 AutoDL 服务器上的文件结构和资源分布。
> 当上下文丢失时，读取此文件即可恢复对 AutoDL 环境的认知。
> 
> 最后更新: 2026-05-01

---

## 服务器信息

- **容器 ID**: autodl-container-0a774196a2-bccdf9f2
- **系统**: Linux, root 用户
- **数据盘**: `/root/autodl-tmp/` (持久存储)
- **公共数据**: `/root/autodl-pub/` (只读)
- **Python**: miniconda3

---

## 目录结构

```
/root/autodl-tmp/
│
├── IntelligenceCamera/           ← 2M   训练代码 (主工作区)
│   ├── models/                   ← 模型定义 (16 个 .py)
│   │   ├── vision_encoder.py         MobileViTSmall (SE+FPN, ~5.6M params)
│   │   ├── semantic_bridge.py        SemanticProjector + TextProjector + LightroomDecoder
│   │   ├── diff_isp.py               可微 Lightroom ISP (8步渲染)
│   │   ├── refinement_net_v4.py      双分支像素精修 (~16M params) ← 当前使用
│   │   ├── refinement_net_v3.py      中等 RefineNet (历史)
│   │   ├── refinement_net_v2.py      U-Net RefineNet (历史)
│   │   ├── refinement_net.py         最早轻量 RefineNet (历史)
│   │   ├── neural_isp.py             NeuralISP 实验
│   │   ├── isp_pipeline.py           非可微 ISP 渲染
│   │   ├── aesthetic_scorer.py       AADB + MUSIQ 美学评分
│   │   ├── intelligence_camera.py    IntelligenceCamera 集成模型
│   │   ├── __init__.py               模块导出
│   │   └── (deprecated: mobile_venus, parameter_predictor, language_model,
│   │        suggestion_generator, text_encoder)
│   │
│   ├── training/                 ← 训练框架
│   │   ├── semantic_distill/         Stage A (语义对齐) + Stage B (参数预测)
│   │   │   ├── model.py              SemanticDistillModel + DistillParamModel
│   │   │   ├── config.py             DistillConfig (visual_dim=384, semantic_dim=256, decoder_hidden=256)
│   │   │   ├── loss.py               蒸馏损失
│   │   │   ├── trainer.py            训练器
│   │   │   ├── embed_texts.py        MiniLM embedding 预计算
│   │   │   └── text_dataset.py       文本数据集
│   │   │
│   │   ├── text_condition/           Stage C (文本条件化)
│   │   │   ├── model.py              TextConditionedModel + LightTextEncoder + FiLMFusion
│   │   │   ├── config.py             TextCondConfig
│   │   │   └── dataset.py            指令数据集
│   │   │
│   │   ├── fivek_8param/             数据加载 + 配置
│   │   │   ├── config.py             PARAM_NAMES (6个), PARAM_RANGES, normalize/denormalize
│   │   │   ├── dataset.py            FiveK 数据集
│   │   │   ├── dataset_expert.py     Expert 配对数据集 (orig + expert GT)
│   │   │   ├── dataset_ppr10k.py     PPR10K 数据集
│   │   │   ├── model.py              FiveK 模型
│   │   │   ├── loss.py               损失函数
│   │   │   └── trainer.py            训练器
│   │   │
│   │   ├── aesthetic_loss.py         美学损失
│   │   ├── dataset.py                通用数据集
│   │   └── distillation.py           蒸馏工具
│   │
│   ├── train_v6_stage_b.py       ← Stage B v6 训练
│   ├── train_v7_stage_b.py       ← Stage B v7 (+退化增强)
│   ├── train_v8_stage_b.py       ← Stage B v8 (MUSIQ优化)
│   ├── train_stage_c.py          ← Stage C 文本条件化
│   ├── train_v9_aesthetic.py     ← 美学实验
│   ├── train_v10_e2e.py          ← 端到端实验
│   ├── train_v11_refine.py       ← RefineNet v1
│   ├── train_v12_refine_hd.py    ← RefineNet v4 (512+MUSIQ) ← 当前主力
│   ├── train_v13_multiscale.py   ← 多尺度实验
│   ├── train_neural_isp.py       ← NeuralISP 实验
│   │
│   ├── scripts/                  ← 辅助脚本
│   │   ├── train_v6_stage_a.py       Stage A 训练
│   │   ├── train_v6_stage_b.py       Stage B 训练
│   │   ├── train_v5_clean.py         v5 训练
│   │   └── deploy_v8_to_autodl.sh    部署脚本
│   │
│   ├── tools/
│   │   ├── data/                     数据处理 (16+ 脚本)
│   │   ├── eval/                     评估工具 (10+ 脚本)
│   │   └── demo/                     演示工具
│   │
│   ├── evaluate/                 ← 评估框架
│   └── inference/                ← 推理接口
│
├── checkpoints/                  ← 317M  模型权重
│   ├── distill_v4/stage_a/           best.pt, final.pt    (7.4M each)  早期实验
│   ├── distill_v5/stage_a+b/         best.pt              (7.4M)       中间迭代
│   ├── distill_v6/stage_a+b/         best.pt, final.pt    (7.4M each)  ★ Stage A 冻结 backbone
│   ├── distill_v7/stage_b/           best.pt, final.pt    (7.4M each)  ★ Stage C 基座
│   ├── distill_v8/stage_b+c/         best.pt              (7.4M+13M)   ★ Refine 基座 + Stage C
│   ├── distill_v9/stage_b/           best.pt              (7.4M)       实验分支
│   ├── distill_v10/stage_b/          best.pt              (7.4M)       实验分支
│   ├── neural_isp/                   best.pt              (6.3M)       NeuralISP
│   ├── refinement/                   best.pt              (529K)       RefineNet v1
│   ├── refinement_v2/                best.pt, final.pt    (7.4M each)  RefineNet v2
│   ├── refinement_v3/                best.pt, final.pt    (35M each)   RefineNet v3
│   └── refinement_v4/                best.pt              (61M)        ★ RefineNet v4 当前最优
│
├── data/                         ← 129M  标注 + embedding
│   ├── fivek_expert_params.json      14M   5位专家 Lightroom 参数标注
│   ├── fivek_expert_consensus.json   1.8M  专家一致性权重
│   ├── fivek_text_embeddings.npz     19M   MiniLM 文本 embedding (Stage A)
│   ├── fivek_venus_embeddings.npz    6.2M  Venus 语义 embedding
│   ├── fivek_venus_labels.json       2.9M  Venus 美学描述标签
│   ├── instruction_data.json         65M   Stage C 口语化指令数据
│   ├── ppr10k_params.json            20M   PPR10K 参数标注
│   └── venus_eval_results_all.json   180K  Venus 评估结果
│
├── fivek_jpeg/                   ← 227M  FiveK 原图 (5121 张 JPEG)
├── fivek_expert_c/               ← 289M  Expert C 处理后 GT 图
├── PPR10K/                       ← 111G  PPR10K 完整数据集
│
├── eval_results/                 ← 71M   所有评估结果 (合并)
│   ├── venus_eval/                   Venus 评估图片
│   ├── venus_extreme/                极端退化评估
│   ├── venus_degraded/               退化评估
│   ├── unified_eval/                 标准统一评估
│   ├── unified_eval_hires/           高分辨率评估
│   └── unified_eval_nisp/            NeuralISP 评估
│
├── logs/                         ← 704K  训练日志
│   ├── v8_1_train.log ~ v13_w5_train.log
│   └── neural_isp_train.log
│
├── Venus-Q-Stage1/               ← 18G   Venus 量化模型
├── hf_cache/                     ← 33G   HuggingFace 模型缓存 (MiniLM, MUSIQ 等)
├── models/                       ← 32G   大模型存储
├── LLaVA-main/                   ← 21M   LLaVA 参考代码
└── assets/                       ← 11M   素材文件
```

---

## 核心训练管线

```
Stage A (v6)  ─→ distill_v6/stage_a/best.pt   (冻结 backbone)
  ↓
Stage B (v7)  ─→ distill_v7/stage_b/best.pt   (Stage C 基座, Venus=5.38)
Stage B (v8)  ─→ distill_v8/stage_b/best.pt   (Refine 基座, MUSIQ=4.446)
  ↓
Stage C (v8)  ─→ distill_v8/stage_c/best.pt   (文本条件化, FiLM融合)
  ↓
RefineNet v4  ─→ refinement_v4/best.pt         (双分支像素精修, 目标 MUSIQ≥8.5)
```

## 6 个 Lightroom ISP 参数

| 参数 | 范围 | 归一化方式 |
|------|------|-----------|
| ev_compensation | [-3, 3] | tanh × 3 |
| white_balance | [2000, 10000] K | sigmoid × 8000 + 2000 |
| contrast | [-100, 100] | tanh × 100 |
| shadows | [-100, 100] | tanh × 100 |
| highlights | [-100, 100] | tanh × 100 |
| saturation | [-100, 100] | tanh × 100 |

---

## 常用命令

```bash
# 启动训练
cd /root/autodl-tmp/IntelligenceCamera
python training/legacy/train_v6_stage_a.py  # Stage A
python train_v7_stage_b.py                   # Stage B
python train_stage_c.py                      # Stage C
python train_v12_refine_hd.py               # RefinementNet v4

# 查看训练日志
tail -f /root/autodl-tmp/logs/v12_train.log

# 查看 checkpoint
ls -lh /root/autodl-tmp/checkpoints/*/
```
