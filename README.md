#  IntelligenceCamera: 端侧 ISP 参数化编辑模型

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![PSNR](https://img.shields.io/badge/PSNR-25.22_dB-brightgreen.svg)]()
[![Params](https://img.shields.io/badge/Params-6.7M-blue.svg)]()
[![Actions](https://img.shields.io/badge/Actions-7-orange.svg)]()
[![Target](https://img.shields.io/badge/Target-CVPR_2026-blueviolet.svg)]()

> **基于 FireRed 1.1 多模态教师蒸馏的端侧 7-action 图像编辑系统。从 NamedCurves (ECCV'24) 7D ISP 参数化基线出发，叠加轻量级 image-domain 残差精修器，在 6537 张 FireRed pseudo-label 上达到 25.22 dB 验证 PSNR，比 ISP base 提升 +1.23 dB，突破参数化 ISP 的表达力天花板。整体推理 6.7M 参数，可在手机端 ISP pipeline 中实时运行。**

---

## 核心架构: Two-stage MLLM Distillation

```
训练阶段 (offline, 仅一次):
  FiveK 输入图 ──┐
                 ├──► FireRed 1.1 (MLLM 教师) ──► pseudo-label (6537 张 png)
  text prompt ───┘     "Increase contrast..." × 7 actions
                       (强教师做开放域编辑, 不参与推理)

推理阶段 (端侧, 完全无文本):
  输入图 ──┐
           │
  one-hot ─┼──► v11-d ISP base (frozen) ────► base_out ──┐
  [0..6]   │     7D Bezier + 6-color naming               │
           │     (3 M params)                             ├──► v13 refiner ──► final
           └────────────────────────────────────────────► │     (3.7 M U-Net + FiLM)
                                                                final = base + 0.5·tanh(δ)
```

**关键思想**:
- **v11-d** 提供物理可解释的 7D ISP 参数化基础, 但被 1D Bezier 表达力所限 (≈24 dB ceiling)
- **v13** 用 image-domain per-pixel 残差**突破**参数化空间限制 (+1.23 dB), 直接修正 ISP 流形外的颜色/光照
- 整体推理输入只有 image + 7-dim action one-hot, **完全没有 text encoder / tokenizer / LLM**

### 7 个支持的 action (one-hot 接口)

| Action | One-hot | FireRed 教师 prompt | base PSNR | v13 PSNR | Δ |
|---|---|---|---:|---:|---:|
| contrast   | `[1,0,0,0,0,0,0]` | Increase contrast               | 27.06 | 27.44 | +0.38 |
| saturation | `[0,1,0,0,0,0,0]` | Enhance saturation              | 22.36 | 24.37 | **+2.01** |
| shadows    | `[0,0,1,0,0,0,0]` | Lift shadows                    | 23.95 | 25.03 | +1.08 |
| highlights | `[0,0,0,1,0,0,0]` | Recover highlights              | 23.51 | 25.26 | **+1.75** |
| wb         | `[0,0,0,0,1,0,0]` | Apply warmer white balance      | 21.67 | 23.22 | +1.55 |
| brightness | `[0,0,0,0,0,1,0]` | Increase brightness             | 23.02 | 24.33 | +1.31 |
| clarity    | `[0,0,0,0,0,0,1]` | Enhance clarity and sharpness   | 27.27 | 26.87 | −0.40 |
| **Overall** |                  |                                 | **23.99** | **25.22** | **+1.23** |

> 涨幅最大的 4 个动作 (sat / hig / wb / bri) 对应 ISP 全局曲线表达不出来的 "空间变化类" 编辑;
> contrast/clarity 涨幅小或回归是因为 7D ISP 已经能很好处理它们 (clarity 还被 SSIM loss 反向拉)。


---

## 项目结构

```
IntelligenceCamera/
├── models/
│   ├── named_curves.py                          # ★ v11-d ISP 基础 (NamedCurves, 7D Bezier + 6-color)
│   ├── firered_residual_refiner.py              # ★ v13 残差精修器 (4-level U-Net + FiLM)
│   ├── diff_isp.py                              # 可微 ISP 渲染管线
│   ├── isp_pipeline.py                          # Lightroom 风格 ISP
│   ├── image_domain_resunet.py                  # Path Z 图像域基线
│   ├── aesthetic_scorer.py                      # 美学评分器
│   └── ... (其他模型组件)
│
├── training/
│   ├── firered_baseline/
│   │   ├── train_lut.py                         # ★ v11-d NamedCurves 训练 (LUT/Bezier 基线)
│   │   ├── train_v13_firered_refine.py          # ★ v13 残差精修器训练 (frozen base + U-Net)
│   │   ├── bezier.py                            # Bezier 曲线工具
│   │   └── color_naming.py                      # 6 色基颜色命名
│   ├── fivek_8param/                            # 旧 8 参数 baseline (历史)
│   ├── expert_c_baseline/                       # Expert C 单专家精准监督 (历史)
│   ├── distillation.py, dataset.py              # 通用蒸馏/数据集组件
│   └── aesthetic_loss.py                        # MUSIQ 美学损失封装
│
├── tools/                                       # ★ 按场景分类的 ~210 个脚本
│   ├── inference/                               # ★ 端到端推理 CLI (v13_edit_image.py 等)
│   ├── eval_runs/                               # ★ 每版本 eval 入口 (eval_v13_firered_refine_viewer.py 等)
│   ├── viz/                                     # 视觉对比 / 网格 / viewer 生成
│   ├── data_prep/                               # 训练数据组装 (build_*_jsonl 等)
│   ├── audit/                                   # 检查 / 诊断 / 分析
│   ├── autodl/                                  # AutoDL 远程基础设施 (sync/launch/monitor)
│   ├── _lib/                                    # hub 模块 (被其他脚本 import: eval_track2 等)
│   ├── _archive/                                # 一次性 scratch (_tmp_*)
│   ├── eval/                                    # 通用 eval 辅助 (PSNR/SSIM/IQA/Multi-expert)
│   ├── demo/                                    # 离线 demo 脚本
│   └── data/                                    # 数据流水线 (analysis/data_prep/scoring/...)
│
├── scripts/
│   ├── local/run_firered_all_actions.py         # ★ FireRed pseudo-label 生成 (ModelScope API)
│   ├── autodl/                                  # AutoDL 训练 / 数据下载 shell 脚本
│   └── README.md
│
├── data/
│   ├── teacher_edits_fivek_full_<action>.json   # ★ FireRed prompts × 7 (per-action 模板)
│   ├── aug_fivek_params*.json                   # FiveK 8 参数标注
│   ├── fivek_expert_*.json                      # FiveK 5 专家 Lightroom 参数 / 共识
│   ├── fivek_aesthetic_scores.json              # FiveK 美学评分
│   └── ...
│
├── outputs/
│   ├── teacher_edits/fivek_full/<action>/       # ★ FireRed pseudo-labels (6537 png)
│   ├── firered_v11_existing_7actions/           # ★ 训练数据 jsonl (image↔target↔action 映射)
│   ├── v13a_firered_refine_viewer/              # ★ v13a 视觉对比 viewer.html
│   └── ...
│
├── checkpoints/
│   ├── lut_v11d_firered_7actions_6537/best.pt   # ★ v11-d base (23.99 dB)
│   └── v13a_firered_refine_7actions/best.pt     # ★ v13a refiner (25.22 dB)
│
├── docs/
│   ├── paper_draft.md                           # ★ 论文草稿 (CVPR 2026)
│   ├── paper_outline.md                         # ★ 论文大纲
│   ├── related_work_references.md               # ★ Related work 参考
│   ├── v11_plan.md, track2_plan.md              # 当前活跃实验计划
│   ├── architecture_overview.md, model_architecture.md, design_rationale.md
│   ├── PROJECT_STRUCTURE.md, asset_inventory.md, data_preparation.md
│   ├── experiment_log.md, experiments_v1_to_v9.md
│   └── archive/                                 # 历史文档归档 (early_experiment_log, v10_plan, ...)
│
├── inference/                                   # 推理接口 (旧, 待重构为 v13 版本)
├── APP/                                         # Web 应用 (Django + Vue3) — 独立子项目
└── requirements.txt
```

---

## 训练流程 (Three-stage)

### Stage 1: FireRed pseudo-label 生成 (offline, 已完成 ✅)

```powershell
$env:MODELSCOPE_API_KEY = "your_key_here"
python scripts/local/run_firered_all_actions.py
# → outputs/teacher_edits/fivek_full/<action>/*.png  (6537 张, 7 actions)
```

- **教师**: FireRed-Image-Edit 1.1 (通过 ModelScope API 调用)
- **数据规模**: 5000 张 FiveK 输入 × 7 actions = 35000 (实际成功 6537)
- **prompt 模板**: 每个 action 一个 (见 `data/teacher_edits_fivek_full_<action>.json`)
- **配额**: API 调用费用约 $80 (一次性)

### Stage 2: v11-d ISP 基线训练 (NamedCurves, 已完成 ✅)

```bash
python -m training.firered_baseline.train_lut \
    --jsonl outputs/firered_v11_existing_7actions/pseudo_labels.jsonl \
    --actions contrast,saturation,shadows,highlights,wb,brightness,clarity \
    --named_curves --nc_n_colors 6 --nc_n_control_points 11 \
    --image_size 256 --epochs 50 --batch_size 8 \
    --output_dir checkpoints/lut_v11d_firered_7actions_6537
# 结果: val_psnr ≈ 23.99 dB
```

- **架构**: NamedCurves (ECCV'24): 6 色基 × 11 控制点 Bezier 曲线
- **参数量**: 3 M
- **训练时间**: 约 6 h (RTX 4060)

### Stage 3: v13 残差精修器训练 (frozen base, 已完成 ✅)

```bash
python -m training.firered_baseline.train_v13_firered_refine \
    --base_ckpt checkpoints/lut_v11d_firered_7actions_6537/best.pt \
    --jsonl outputs/firered_v11_existing_7actions/pseudo_labels.jsonl \
    --base_ch 32 --delta_scale 0.5 \
    --l1_weight 1.0 --ssim_weight 0.5 --lab_weight 0.3 \
    --grad_weight 0.2 --delta_l1_weight 0.05 \
    --epochs 30 --batch_size 8 \
    --output_dir checkpoints/v13a_firered_refine_7actions
# 结果: val_psnr ≈ 25.22 dB (+1.23 over base)
```

- **架构**: 4-level U-Net + FiLM action conditioning, zero-init final conv
- **参数量**: 3.7 M (refiner only; v11-d base 冻结)
- **损失**: L1 + SSIM + Lab L1 + gradient L1 + delta L1 sparsity
- **训练时间**: 约 4 h (RTX 4060)

---

## 模型架构

### v11-d NamedCurves (frozen base, 3 M)

| 组件 | 参数量 | 说明 |
|------|---:|------|
| MobileViT-Small encoder | 1.5 M | 视觉特征 (输入 256×256) |
| Action embedding | 0.01 M | 7-dim → 32-dim |
| Bezier curve heads × 7 | 0.5 M | 每个 action 预测 6 条 (R/G/B × 2-axis) Bezier 曲线 |
| 6-color naming | 0.1 M | 颜色基分割 |
| 渲染器 | 0 (解析) | 应用曲线到原图 |
| **总计** | **~3 M** | 推理 ~12 MB FP32 |

### v13 Residual Refiner (3.7 M)

| 组件 | 参数量 | 说明 |
|------|---:|------|
| 6-channel input | — | concat(orig, base_out) |
| 4-level U-Net | 3.5 M | base_ch=32, GroupNorm + GELU |
| FiLM conditioning × 7 | 0.1 M | 7-dim action 向量调制每层 feature |
| Final 1×1 conv | <0.01 M | zero-init → identity at start |
| **总计** | **~3.7 M** | tanh-bounded (±0.5) 残差 |

### 总推理预算

- **v11-d + v13**: ~6.7 M params, ~27 MB FP32, ~14 MB FP16
- 256×256 输入下: 单次 forward ~8 ms (RTX 4060)
- 端侧目标: <30 ms (中端骁龙 8 Gen 2)

---

## 数据来源

| 数据集 | 规模 | 用途 | 状态 |
|--------|------|------|------|
| **MIT-Adobe FiveK** | 5000 图 (jpg) | v13 训练输入图 | ✅ 完成 |
| **FireRed pseudo-label** | 6537 张 png × 7 actions | v13 训练目标 | ✅ 完成 |
| FiveK 5-Expert | 46K 条参数 | 旧 baseline (历史) | ✅ 完成 |
| AADB | 10K 图 | 美学评分器预训练 (旧) | ✅ 完成 |

> 详细 prompt 模板与 pseudo-label 生成流程见 `docs/data_preparation.md`。

---

## 快速开始

### 环境配置

```bash
conda create -n Venus python=3.10 -y
conda activate Venus
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 单图编辑 (单 action, CLI)

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH='.'
python tools/inference/v13_edit_image.py `
    --image "E:/Data/dataset/fivek_jpeg/a0001-jmac_DSC1459.jpg" `
    --action contrast `
    --out "outputs/edit_demo/a0001_contrast.jpg" `
    --save_comparison
```

`--action` 可选: `contrast / saturation / shadows / highlights / wb / brightness / clarity`。
`--save_comparison` 会额外保存一张 `orig | v11-d base | v13 refined` 三联对比图。

### 一次跑全部 7 个 action (推荐, 看完整效果)

```powershell
python tools/inference/v13_edit_image.py `
    --image "your_photo.jpg" `
    --all_actions `
    --out_dir "outputs/edit_demo/your_photo_all"
```

输出包含: 7 张 v13 精修结果 + 7 张 base 对照 + 1 张 4×2 网格汇总图。

### Python API

```python
import torch
from PIL import Image
from torchvision import transforms
from models.firered_residual_refiner import FireRedResidualRefiner
from training.firered_baseline.train_lut import ACTIONS, set_actions
from training.firered_baseline.train_v13_firered_refine import load_v11d_base

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载 v13 + v11-d
v13_ckpt = torch.load('checkpoints/v13a_firered_refine_7actions/best.pt',
                      map_location=device, weights_only=False)
set_actions(v13_ckpt['args']['actions'])  # 必须在 base 加载之前
base_model, _ = load_v11d_base(v13_ckpt['args']['base_ckpt'], device)
refiner = FireRedResidualRefiner(base_ch=32, n_actions=7,
                                  delta_scale=0.5).to(device)
refiner.load_state_dict(v13_ckpt['model_state_dict'])
refiner.eval()

# 预处理
to_tensor = transforms.Compose([transforms.Resize((256, 256)),
                                 transforms.ToTensor()])
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
orig = to_tensor(Image.open('photo.jpg').convert('RGB')).unsqueeze(0).to(device)
enc_input = norm(orig.squeeze(0)).unsqueeze(0)

# Action one-hot (e.g. contrast)
action_oh = torch.zeros(1, 7, device=device); action_oh[0, ACTIONS.index('contrast')] = 1

# 推理
with torch.no_grad():
    base_out, _, _, _ = base_model(enc_input, orig, action_oh)
    refined, _ = refiner(orig, base_out, action_oh)
# refined: (1, 3, 256, 256) in [0, 1]
```

### 评估 + 视觉对比 viewer

```powershell
python tools/eval_runs/eval_v13_firered_refine_viewer.py `
    --ckpt checkpoints/v13a_firered_refine_7actions/best.pt `
    --jsonl outputs/firered_v11_existing_7actions/pseudo_labels.jsonl `
    --out_dir outputs/v13a_firered_refine_viewer `
    --max_samples 80 `
    --tag "v13a 7-action refine"
```

打开 `outputs/v13a_firered_refine_viewer/viewer.html`, 包含:

- 整体 PSNR + per-action PSNR 表 + 7 条 FireRed prompt 模板
- 80 张样本的 5 列对照: 原图 / v11-d base / v13 refined / FireRed target / 误差热图

---

## 论文

- **标题** (临时): *MLLM-as-Editor: Two-stage Distillation from FireRed to End-side Action-conditioned ISP*
- **目标**: CVPR 2026 (Venus 系列工作之一)
- **草稿**: 见 `docs/paper_draft.md`, `docs/paper_outline.md`

### 主要贡献

1. **首次将 MLLM 作为图像编辑教师** — 用 FireRed 1.1 作为 6537 张 FiveK pseudo-label 的开放域编辑教师, 蒸馏到端侧 7-action 模型
2. **Two-stage frozen-base + image-residual 蒸馏范式** — 提出 v11-d (NamedCurves frozen ISP) + v13 (image-domain residual refiner) 的两阶段架构, 突破参数化 ISP 的表达力天花板 (+1.23 dB)
3. **Per-action 失败模式分析** — 在 7 个 action 上系统量化 ISP 表达力差异, 揭示 spatial-varying 编辑 (sat/hig/wb/bri) 是 ISP 的盲区, 而 contrast/clarity 已被 ISP 充分覆盖
4. **可发布数据集** — 6537 张 FireRed pseudo-label (FiveK × 7 actions) + per-action 训练 jsonl

---

## 实验结果

### 主结果: FireRed 6537 验证集 (256×256, 7-action)

| 模型 | 参数量 | 推理输入 | PSNR ↑ | 备注 |
|---|---:|---|---:|---|
| 输入图 (do nothing) | 0 | image | 16.50 | trivial baseline (action-blind) |
| v11-d base (NamedCurves) | 3 M | image + action one-hot | 23.99 | 7D ISP ceiling |
| **v13a (本工作)** | **6.7 M** | image + action one-hot | **25.22** | **+1.23 dB**, 突破 ISP ceiling |

### Per-action 提升 (Δ vs frozen base)

涨幅最大的动作正是 ISP 全局曲线表达不出来的 "空间变化类":

| Action      | base PSNR | v13 PSNR | Δ          |
|-------------|----------:|---------:|-----------:|
| saturation  | 22.36     | 24.37    | **+2.01**  |
| highlights  | 23.51     | 25.26    | **+1.75**  |
| wb          | 21.67     | 23.22    | +1.55      |
| brightness  | 23.02     | 24.33    | +1.31      |
| shadows     | 23.95     | 25.03    | +1.08      |
| contrast    | 27.06     | 27.44    | +0.38      |
| clarity     | 27.27     | 26.87    | −0.40      |

### 与已发表方法对比 (FiveK Expert C 监督, 仅供参考)

> 注意: 我们的监督来自 FireRed pseudo-label 而非 Expert C, 所以下表只用于定位整体级别。

| 方法                  | 监督来源        | PSNR | 备注 |
|-----------------------|----------------|-----:|------|
| NamedCurves (ECCV'24) | FiveK Expert C | 24.91 | 我们的 base 架构来源 |
| CLUT-Net              | FiveK Expert C | 25.78 | 3D LUT |
| **v13a (本工作)**     | **FireRed pseudo-label** | **25.22** | 不同教师, 7-action one-hot |
| VeraRetouch (2026)    | LR API + 100× 数据 | 26.85 | 70× 参数, 不可比 |

### 关键发现

1. **ISP ceiling 假说被验证**: v11-d 用 7D ISP 全局曲线只能到 23.99 dB, 加 image-domain residual 后 +1.23 dB
2. **Spatial-varying 编辑是 ISP 的盲区**: sat/hig/wb/bri 涨幅 +1.3~+2.0, 而 ISP 已能很好处理的 contrast 只 +0.4
3. **clarity 回归 (−0.4)**: SSIM loss 对锐化做反向梯度, 是 v14a 频率分解残差的目标

详见 `docs/paper_draft.md` 第 4 节 "Per-action analysis" 与 `docs/paper_outline.md` ablation 计划。

---

## 路线图

### 已完成 ✅
- [x] FireRed 1.1 pseudo-label 生成 (6537 张, 7 actions, 一次性 offline)
- [x] v11-d NamedCurves baseline 训练 (23.99 dB)
- [x] v13a 残差精修器训练 (25.22 dB, +1.23 dB)
- [x] Per-action 失败模式分析 (clarity 回归, wb 偏置)
- [x] Viewer.html 视觉对比工具 (5 列对照 + 误差热图)
- [x] 单图编辑 CLI (`tools/inference/v13_edit_image.py`)
- [x] `tools/` 重构 (83 个根层脚本 → 8 个功能子目录)
- [x] 文档清理 (本次, 删 3 + 归档 8)

### 进行中 / 计划 📋
- [ ] **v14a**: 频率分解残差 (低频 color + 高频 sharpen 分支), 解决 clarity 回归 — 预期 +0.4 dB
- [ ] **v14b**: WB Off-Planckian 结构头, 解决 mixed lighting — 预期 wb +0.5 dB
- [ ] **v14c**: 跨阶段语义跳连 (复用 v11-d 的 color naming map) — 预期 +0.2 dB
- [ ] WB cooler 反向数据补充 (修复单向 warmer 偏置)
- [ ] 与 SOTA 全面对比 (CLUT-Net, SDA-LUT, VeraRetouch)
- [ ] 移动端部署 (ONNX → CoreML / TFLite)
- [ ] CVPR 2026 论文撰写 + 提交

---

## 致谢

- [Venus (CVPR 2026)](https://github.com/PKU-ICST-MIPL/Venus_CVPR2026) — 母项目
- [FireRed-Image-Edit 1.1](https://www.modelscope.cn/) — MLLM 图像编辑教师
- [NamedCurves (ECCV'24)](https://github.com/davidserra9/named-curves) — ISP 基线架构 (v11-d 来源)
- [MIT-Adobe FiveK](https://data.csail.mit.edu/graphics/fivek/) — 输入图源
- [AADB](https://github.com/aimerykong/AADB) — 美学属性数据集 (旧 baseline)
- [MUSIQ](https://github.com/google-research/google-research/tree/master/musiq) / [NIMA](https://github.com/idealo/image-quality-assessment) — 美学/感知评估指标

---

**创建日期**: 2026-03-19  
**最后更新**: 2026-05-24  
**当前版本**: v13a (FireRed 7-action residual refine, 25.22 dB)  
**作者**: Venus IntelligenceCamera Team
