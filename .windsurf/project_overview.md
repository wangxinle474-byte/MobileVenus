# MobileVenus 项目概览

> 本文件供 AI 助手快速了解项目架构和当前进度。
> 当上下文丢失时，读取 .windsurf/ 下的文件即可恢复认知。
>
> 最后更新: 2026-05-01

---

## 项目目标

为移动端智能相机开发轻量化 ISP 参数预测模型，通过 3 阶段知识蒸馏 + 像素级精修，
实现从图像/文本到 6 个 Lightroom 参数的自动预测，达到专业修图效果。

目标论文: **CVPR 2026**

---

## 架构概览

```
┌─────────────────────────────────────────────────────┐
│  Stage A: 语义对齐                                    │
│  Image → MobileViTSmall → SemanticProjector          │
│          ← Align →  MiniLM(Text) → TextProjector     │
│  输出: distill_v6/stage_a/best.pt (冻结 backbone)     │
├─────────────────────────────────────────────────────┤
│  Stage B: 参数预测                                    │
│  Image → (frozen) Semantic Features → LightroomDecoder│
│  → 6 Lightroom Parameters                            │
│  v7: +退化增强+对比学习 → distill_v7/stage_b/best.pt  │
│  v8: MUSIQ最优         → distill_v8/stage_b/best.pt  │
├─────────────────────────────────────────────────────┤
│  Stage C: 文本条件化                                  │
│  Image + Text("warmer") → LightTextEncoder            │
│  → FiLM modulation → 6 Parameters                    │
│  输出: distill_v8/stage_c/best.pt                     │
├─────────────────────────────────────────────────────┤
│  RefinementNet v4: 像素级精修                         │
│  Image → ParamModel(frozen v8) → diff_isp render      │
│  → RefinementNetV4 (双分支: Global + Local)           │
│  输出: refinement_v4/best.pt                          │
│  目标: MUSIQ ≥ 8.5 (突破参数天花板 4.75)              │
└─────────────────────────────────────────────────────┘
```

---

## 关键模型组件

| 组件 | 文件 | 参数量 | 说明 |
|------|------|--------|------|
| MobileViTSmall | `models/vision_encoder.py` | ~5.6M | SE+FPN 视觉编码器 |
| SemanticProjector | `models/semantic_bridge.py` | ~130K | 视觉→语义投影 |
| TextProjector | `models/semantic_bridge.py` | ~100K | 文本→语义投影 |
| LightroomDecoder | `models/semantic_bridge.py` | ~50K | 语义→6参数 |
| LightTextEncoder | `training/text_condition/model.py` | ~200K | 字符级 Transformer |
| FiLMFusion | `training/text_condition/model.py` | ~30K | 特征调制 |
| RefinementNetV4 | `models/refinement_net_v4.py` | ~16M | 双分支精修网络 |
| DiffISP | `models/diff_isp.py` | 0 (无参数) | 可微 Lightroom 渲染 |

---

## 训练脚本版本线

| 版本 | 脚本 | 内容 | 状态 |
|------|------|------|------|
| v5 | `scripts/train_v5_clean.py` | 早期清理版 | 已完成 |
| v6 | `scripts/train_v6_stage_a.py` + `train_v6_stage_b.py` | Stage A/B 基础 | ✅ 已完成 |
| v7 | `train_v7_stage_b.py` | +退化增强+对比学习 | ✅ 已完成 |
| v8 | `train_v8_stage_b.py` + `train_stage_c.py` | +Stage C | ✅ 已完成 |
| v9 | `train_v9_aesthetic.py` | 美学实验 | ✅ 已完成 |
| v10 | `train_v10_e2e.py` | 端到端实验 | ✅ 已完成 |
| v11 | `train_v11_refine.py` | RefineNet v1 | ✅ 已完成 |
| v12 | `train_v12_refine_hd.py` | RefineNet v4 (512+MUSIQ) | ✅ 当前主力 |
| v13 | `train_v13_multiscale.py` | 多尺度实验 | ✅ 已完成 |

---

## 三端同步状态

| 位置 | 路径 | 内容 | 同步方式 |
|------|------|------|----------|
| **本地** | `E:\智能相机\...\IntelligenceCamera\` | 全量代码+文档+APP | 主开发环境 |
| **GitHub** | `wangxinle474-byte/MobileVenus` | 全量代码备份 | `git push` (需代理 127.0.0.1:7897) |
| **AutoDL** | `/root/autodl-tmp/IntelligenceCamera/` | 仅训练代码 | `scripts/sync_to_autodl.sh` 或手动上传 |

---

## .windsurf/ 文件索引

| 文件 | 用途 |
|------|------|
| `project_overview.md` | 项目架构、组件、版本线 (本文件) |
| `autodl_resources.md` | AutoDL 服务器文件结构、checkpoint、数据、常用命令 |
| `local_resources.md` | 本地 E 盘数据集清单、与 AutoDL 对照 |
