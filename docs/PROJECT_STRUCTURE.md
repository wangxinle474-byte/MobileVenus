# IntelligenceCamera 项目结构

> CVPR 2026 投稿项目, 端侧美学+ISP agent. 本文档让首次接触代码的人快速找到自己要的东西.

## 顶层目录速查

```
IntelligenceCamera/
├── README.md                  ← 项目主介绍
├── PROJECT_STRUCTURE.md       ← 本文件 (结构索引)
├── TRAINING_LOG.md            ← 训练记录
├── requirements.txt
├── .gitignore                 ← 排除 checkpoints/、outputs/、大 JSON 等
│
├── APP/                       ← Web 应用 (前后端, 与研究代码解耦)
├── configs/                   ← 训练配置 YAML (如 lora_qwen3vl_4b_sft.yaml)
├── checkpoints/               ← 训练权重 (gitignore)
├── data/                      ← 数据集 / 标签 / captions
├── docs/                      ← 论文素材 + 设计文档
├── eval_results/              ← 评估结果
├── evaluate/                  ← 评估流水线 (run_comparison.py 等)
├── examples/                  ← 用例展示
├── images/                    ← 论文配图 + 训练曲线
├── inference/                 ← 推理引擎 (predict.py, camera_controller.py)
├── logs/                      ← 训练日志 (gitignore)
├── models/                    ← 模型定义 (intelligence_camera.py 等)
├── outputs/                   ← 实验输出 (gitignore, 见下文 §4)
├── scripts/                   ← 运行脚本 (autodl/ 远端 + local/ 本地)
├── tools/                     ← 数据处理 + 评估工具
└── training/                  ← 训练子模块 (dataset, distillation, losses, legacy/)
```

## 1. 主要入口

### 训练
- 完整流水线: `train_v13_multiscale.py` (最新 v13)
- 主版本演进: `train_v6_stage_a.py` → `train_v8_stage_b.py` → `train_v10_e2e.py` → `train_v11_refine.py` → `train_v12_refine_hd.py` → `train_v13_multiscale.py`
- 配置: `configs/lora_qwen3vl_4b_sft.yaml`

### 推理 (Web app 后端)
- `inference/predict.py` — 单张推理入口
- `inference/camera_controller.py` — 镜头控制器
- `models/intelligence_camera.py` — 主模型类

### 评估
- `evaluate/run_comparison.py` — 对比评估
- `evaluate/run_venus_inference.py` — Venus 推理
- `tools/eval/score_longcat_edits.py` — Qwen3-VL 多维度打分 (1-10)

### 编辑模型对比 (本次工作核心)
- `tools/data/local_run_longcat_turbo.py` — LongCat-Turbo 本地推理
- `tools/data/run_firered_online.py` — FireRed-Lightning 在线推理 (ModelScope Studio)
- `tools/data/autodl_rewrite_edit_to_scene.py` — Qwen3-VL 把"指令"重写成"场景描述"

## 2. data/ 数据布局

```
data/
├── compare_5_captions.json          ← 5 张图 × 场景描述 (sceneA)
├── compare_5_captions_edit.json     ← 5 张图 × 编辑指令 (editB)
├── instruction_data.json (67MB, gitignore)
├── aug_*.json (gitignore)           ← 数据增强 (训练用)
├── fivek_*.json/.npz (gitignore)    ← FiveK 数据
├── ppr10k_params.json (gitignore)
└── pseudo_labels/ (gitignore)       ← 伪标签
```

## 3. tools/ 工具脚本

```
tools/
├── data/                      ← 数据处理 (~60 脚本)
│   ├── local_run_longcat_turbo.py        ← LongCat 本地推理
│   ├── run_firered_online.py             ← FireRed Studio 在线推理
│   ├── autodl_rewrite_edit_to_scene.py   ← Qwen3-VL 重写器
│   ├── autodl_score_*.py                 ← 各种 AutoDL 评分
│   ├── analyze_*.py / inspect_*.py       ← 分析/检查工具
│   ├── compare_*.py                      ← 对比工具
│   └── ... (其他数据生成/转换)
│
└── eval/                      ← 评估脚本
    ├── score_longcat_edits.py            ← 主评分脚本 (Qwen3-VL 1-10 多维度)
    ├── present_3way_10pt.py              ← 多组评分对比展示
    ├── build_compare_html.py             ← 视觉对比 HTML 生成
    ├── infer_lora.py                     ← LoRA 推理
    ├── eval_lora_on_artedit.py           ← LoRA 在 ArtEdit 上评估
    └── compare_isp_modes.py / eval_psnr_ssim.py / eval_nima.py
```

## 4. outputs/ 实验产物布局 (★ 关键约定 ★)

**核心原则: 同一实验的所有产物归到一个子目录, 原图(orig)与编辑图(edit)分别放在不同子目录, 不要混在同一文件夹.**

### compare_5/ 5 张图编辑模型对比 (本次工作)
```
outputs/compare_5/
├── README 见 docs/compare_5_experiment.md
├── originals/                 ← 5 张原图 (源, 共享, 不重复)
│   └── <idx>.png              ← 文件名只用 idx (如 0071.png), 不带后缀
│
├── longcat_sceneA/            ← LongCat × 场景描述 prompt
│   └── <idx>.png
├── longcat_editB/             ← LongCat × 编辑指令 prompt
│   └── <idx>.png
├── firered_editB/             ← FireRed × 编辑指令 (rewrite=False)
│   └── <idx>.png
├── firered_editB_rewrite/     ← FireRed × 编辑指令 (rewrite=True)
│   └── <idx>.png
│
├── scores/                    ← Qwen3-VL 1-10 评分结果
│   └── <group>_10pt.json
├── meta/                      ← 推理 metadata (用了什么参数/seed)
│   └── <group>.json
├── legacy_1to5/               ← 旧 1-5 评分归档 (历史保留)
│   └── <group>.json
└── viewer.html                ← 4 栏对比 HTML (浏览器直开)
```

**文件名约定:**
- 原图: `outputs/compare_5/originals/<idx>.png` (只有 idx)
- 编辑图: `outputs/compare_5/<group>/<idx>.png` (只有 idx, group 名标识来源)
- 评分: `outputs/compare_5/scores/<group>_10pt.json`

### 其他 outputs/ 子目录
```
outputs/
├── compare_5/                 ← 见上
├── ip2p_pilot_100/            ← IP2P 100 张训练样本 (~972MB, 不动)
├── ip2p_test/                 ← IP2P 烟囱测试
├── ip2p_data/                 ← IP2P 相关 JSONs + tar.gz 归档
├── sdxl_turbo_compare/        ← 历史 SDXL Turbo 对比
├── diagnose_param_conflicts/  ← 参数冲突诊断热图
├── validate_*/                ← 各种验证结果
└── legacy/                    ← 单文件级历史 outputs (eval_full.json 等)
```

## 5. scripts/ 运行脚本

```
scripts/
├── README.md                              ← 索引表
├── autodl/                                ← 远端 AutoDL 上跑 (21 files)
│   ├── autodl_download_*.sh                   ← 模型权重下载 (5)
│   ├── autodl_install_*.sh                    ← 环境安装 (2)
│   ├── autodl_phase1_rewrite_and_rescore.sh   ← 主流水线: Qwen3-VL 重写 + 4 组评分
│   ├── autodl_phase2_score_longcat_rewritten.sh  ← 第 5 组评分
│   ├── autodl_rescore_all_10pt.sh             ← 4 组 1-10 评分
│   ├── autodl_run_longcat_compare.sh          ← LongCat 推理 + 评分
│   ├── autodl_run_lora_sft.sh                 ← LoRA SFT 训练
│   ├── autodl_serve_*.sh                      ← 服务部署 (2)
│   ├── autodl_watch_pipeline.sh               ← 实时监控
│   └── run_validate_*.sh / inspect_artedit.sh  ← 验证 (3)
├── local/                                 ← 本地 Windows/Linux (10 files)
│   ├── sync_to_autodl.{sh,ps1}                ← 本地↔AutoDL 同步 (2)
│   ├── local_resume_*.ps1                     ← HF 断点续传 (3)
│   ├── fetch_pseudo_labels.ps1                ← 拉伪标签
│   ├── upload_compare_5.ps1                   ← 上传 5 张对比
│   ├── tail_autodl_log.ps1                    ← tail 远端日志
│   ├── watch_download.ps1                     ← 监视下载
│   └── check_dl_progress.sh                   ← 进度查看
└── (legacy_training 已迁到 training/legacy/, 见 §训练)
```

## 6. docs/ 设计文档

```
docs/
├── PROJECT_STRUCTURE.md       ← 本文件 (新人入口)
├── research_prd.md            ← 研究 PRD (面向论文/方法)
├── product_prd.md             ← 产品 PRD (面向 C 端 App)
├── architecture_overview.md   ← 总体架构
├── design_rationale.md        ← 设计理由
├── experiment_plan.md         ← 对比实验方案
├── experiment_log.md          ← 主实验日志 (Distill v6-v14)
├── early_experiment_log.md    ← 早期实验 (Baseline + Distill v1-v4)
├── data_preparation.md        ← 数据准备
├── model_architecture.md      ← 模型结构
├── compare_5_experiment.md    ← 5 张图编辑模型对比实验
├── asset_inventory.md         ← 资产清单
├── plot_*.py                  ← 论文配图生成脚本
├── archive/                   ← 历史归档 (v1/v2 9-param 早期)
│   └── history.md
├── surveys/                   ← 调研资料
│   └── REUSE_SURVEY_2026.md
├── artedit_samples/, diff_isp_validation/, lora_v1/
└── *.jpg / *.png              ← 已生成的论文配图
```

## 7. 端到端工作流 (5 张图编辑对比)

| 步骤 | 命令 | 产物 |
|------|------|------|
| 1. LongCat 推理 (sceneA) | `python tools/data/local_run_longcat_turbo.py --captions data/compare_5_captions.json --out_dir outputs/compare_5/longcat_sceneA --skip_download --use_4bit --steps 4` | `compare_5/longcat_sceneA/<idx>.png` |
| 2. LongCat 推理 (editB) | 同上, `--captions data/compare_5_captions_edit.json --out_dir outputs/compare_5/longcat_editB` | `compare_5/longcat_editB/<idx>.png` |
| 3. FireRed 推理 (editB) | `python tools/data/run_firered_online.py --captions data/compare_5_captions_edit.json --input_dir outputs/compare_5/originals --out_dir outputs/compare_5/firered_editB --lora Lightning` | `compare_5/firered_editB/<idx>.png` |
| 4. FireRed (editB rewrite) | 同上 + `--rewrite_prompt --out_dir outputs/compare_5/firered_editB_rewrite` | `compare_5/firered_editB_rewrite/<idx>.png` |
| 5. AutoDL 评分 (4 组) | `bash scripts/autodl/autodl_rescore_all_10pt.sh` | `compare_5/scores/<group>_10pt.json` |
| 6. Qwen3-VL 重写 + 5 组完整对比 | `bash scripts/autodl/autodl_phase1_rewrite_and_rescore.sh` 然后本地用重写 prompt 跑 LongCat 上传, 再 `bash scripts/autodl/autodl_phase2_score_longcat_rewritten.sh` | `scores/longcat_editB_rewritten_10pt.json` |
| 7. 视觉对比 HTML | `python tools/eval/build_compare_html.py` | `outputs/compare_5/viewer.html` (浏览器开) |
| 8. 表格化展示 | `python tools/eval/present_3way_10pt.py` | 控制台 5-way 总均值表 |

## 8. 命名约定 (新人必读)

- **图片**: 只用 `<idx>.png` 作文件名 (4 位数字), 用所在目录名标识"操作". e.g. `compare_5/firered_editB/0071.png` 表示 idx=71 在 firered_editB 这一组的图.
- **不要混放**: 原图归 `originals/`, 不重复; 编辑图归各自 `<group>/` 目录. 不要在同一目录里塞 `_orig` + `_<model>` 后缀的文件.
- **评分**: `<group>_10pt.json` 格式; 脚本生成的, 不手编.
- **meta**: 推理时的参数+seed+runtime 等, 用于复现.

## 9. 常见问题

**Q: 我要新增一个对比实验, 应该放哪?**
A: 在 `outputs/<experiment_name>/` 下新建, 沿用 `originals/<idx>.png` + `<group>/<idx>.png` 结构. 评分写到 `<experiment_name>/scores/`.

**Q: 旧的 `outputs/longcat_compare_*` 目录在哪?**
A: 已合并重构到 `outputs/compare_5/`. 见本次重构 commit.

**Q: 训练数据为什么不进 git?**
A: `instruction_data.json` 67MB、`aug_*.json` 几 MB+, 体积大. 在 `.gitignore` 里. 单独管理或上传 LFS.

**Q: AutoDL 实例怎么用?**
A: 见 `scripts/sync_to_autodl.sh` (本地→远端同步). 远端路径: `/root/autodl-tmp/IntelligenceCamera/`.
