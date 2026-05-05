# tools/data/ 索引

> 65 个数据相关脚本，按"做什么事"分了 6 个子目录。

## 目录速查

| 子目录 | 数量 | 用途 |
|--------|:-:|------|
| [`editor_models/`](#editor_models) | 9 | 编辑模型推理 (LongCat / FireRed / CSGO / SDXL / Qwen-CN) + Caption 生成 |
| [`scoring/`](#scoring) | 11 | 美学/质量评分 (FiveK / IP2P / AesExpert / Venus) |
| [`data_prep/`](#data_prep) | 24 | 数据准备/转换/打包/嵌入/反推参数 |
| [`analysis/`](#analysis) | 8 | 分析/对比 (pair scores / prompts / before-after) |
| [`viz/`](#viz) | 7 | 可视化/网格图/HTML |
| [`model_io/`](#model_io) | 6 | 模型权重下载/续传/监控 |

> 一次性 inspect/check/diagnose helper (12 个) 已删除. v1 续传脚本已删, 保留 v2.

---

## editor_models/

编辑模型推理脚本（本地 + AutoDL 远端）+ Caption 生成。

| 脚本 | 用途 |
|------|------|
| `local_run_longcat_turbo.py` | LongCat-Image-Edit-Turbo 本地推理 (8 GB VRAM 适配) |
| `local_run_csgo.py` | CSGO 本地推理 |
| `local_run_sdxl_turbo.py` | SDXL-Turbo 本地推理 |
| `run_firered_online.py` | FireRed-Lightning 在线推理 (ModelScope Studio) |
| `autodl_run_csgo.py` | CSGO AutoDL 推理 |
| `autodl_run_qwen_cn.py` | Qwen-Image-CN-Inpainting AutoDL 推理 |
| `autodl_rewrite_edit_to_scene.py` | Qwen3-VL: 编辑指令重写为场景描述 prompt |
| `local_qwen3vl_rewrite.py` | Qwen3-VL: 本地生成"a photo of..." 风格 caption (用于 SDXL) |
| `sdxl_with_vl_captions.py` | SDXL + VL caption 联合推理对比 |

---

## scoring/

各种美学/质量评分器（含监控工具）。

| 脚本 | 评分模型 | 数据 |
|------|---------|------|
| `autodl_score_fivek.py` | (AutoDL 远端) | FiveK |
| `autodl_score_ip2p_pairs.py` | (AutoDL 远端) | IP2P 编辑对 |
| `score_fivek_aesexpert.py` | AesExpert (LLaVA) | FiveK |
| `score_fivek_aesthetic.py` | NIMA / MUSIQ | FiveK |
| `score_ip2p_pair.py` | 通用 | IP2P 对 |
| `rescore_with_aesexpert.py` | AesExpert | 任意 augmented set |
| `rescore_with_venus.py` | Venus 7B (4D) | 任意 augmented set |
| `venus_aesthetic_eval.py` | Venus 7B | 评估 pipeline |
| `monitor_rescore.py` | (监控工具) | 监视 AesExpert 重打分进度 |
| `run_autodl_score.sh` / `run_fivek_score.sh` | shell wrapper | — |

---

## data_prep/

数据准备核心区，覆盖增强/转换/嵌入/伪标签/反推参数等。

### 数据增强 + 训练数据构建
- `aug_low_score_images.py` — 对低分图片做增强生成
- `build_aug_training_data.py` — 构建增强训练数据
- `build_sharegpt_from_labels.py` — 从标签生成 ShareGPT 格式
- `convert_pseudo_to_sharegpt.py` — 伪标签 → ShareGPT
- `merge_final_dataset.py` — 合并最终训练集

### FiveK 专项
- `build_fivek_params_data.py` — FiveK 参数数据构建
- `convert_fivek_dng_to_jpeg.py` — DNG → JPEG
- `gen_fivek_stage_a.py` — Stage A 数据
- `gen_fivek_embeddings.py` — FiveK 文本 embeddings
- `gen_coco_embeddings.py` — COCO 文本 embeddings
- `filter_fivek_by_score.py` — 按分数过滤 FiveK

### 伪标签 / 提取
- `extract_pseudo_labels.py` — 提取 Venus 伪标签
- `generate_cot_pseudo_labels.py` — 生成 CoT 伪标签
- `extract_edit_prompts.py` — 提取编辑提示
- `extract_pixel_params.py` — 提取像素级参数
- `compute_expert_consensus.py` — 计算 5 专家共识

### 数据质量 / 诊断
- `check_data_quality.py` — 数据质量综合检查
- `diagnose_param_conflicts.py` — 参数冲突分析（带可视化）

### Bucket / Reparse
- `bucket_pair_scores.py` — 按分数桶分类
- `reparse_pair_scores.py` — 重解析评分

### 反推参数 (inverse fitting)
- `inverse_fit.py` — Lightroom 参数反推优化
- `autodl_vlm_inverse_params.py` — VLM 输出反推为参数

### 推理/打包
- `package_ip2p_for_autodl.py` — IP2P 打包给 AutoDL
- `generate_venus_eval_images.py` — 生成 Venus 评估图

---

## analysis/

中性分析/对比脚本，不改数据，只读取并产生 report/数字。

| 脚本 | 用途 |
|------|------|
| `analyze_current_prompts.py` | 当前 prompt 分析 |
| `analyze_expert_variance.py` | 专家间方差分析 |
| `analyze_pair_scores.py` | pair 评分分析 |
| `analyze_rescued.py` | 救回的样本分析 |
| `compare_all_scores.py` | 全量评分对比 |
| `compare_before_after.py` | 前后对比 |
| `compare_per_image.py` | 逐图对比 |
| `compare_prompt_opt.py` | prompt 优化对比 |

---

## viz/

可视化输出 - HTML/网格图/Top-K 展示。

| 脚本 | 输出 |
|------|------|
| `make_ip2p_html.py` | IP2P 对比 HTML |
| `make_multi_model_grid.py` | 多模型对比网格图 |
| `make_sdxl_comparison_grid.py` | SDXL 对比网格 |
| `make_top5_grid.py` | Top-5 网格图 |
| `show_top5.py` / `get_top5_descriptions.py` | Top-5 描述提取 |
| `visualize_delta_samples.py` | Delta 样本可视化 |

---

## model_io/

模型权重的 I/O 工具（下载/断点续传/监控）。

| 脚本 | 用途 |
|------|------|
| `autodl_download_longcat.py` | LongCat → AutoDL 后台下载 |
| `local_download_longcat_ms.py` | LongCat → 本地 ModelScope 下载 |
| `local_resume_aesexpert_v2.py` | AesExpert HF 镜像断点续传 (v2 改进版) |
| `local_resume_longcat_v2.py` | LongCat HF 镜像断点续传 (v2 改进版) |
| `load_aesexpert_4bit.py` | 4-bit 量化加载 AesExpert |
| `watch_downloads.py` | 实时监控 HF 下载进度 |

---

## 常见用法快照

```powershell
# 本地跑 LongCat 推理 (compare_5)
python tools/data/editor_models/local_run_longcat_turbo.py `
  --captions data/compare_5_captions_edit.json `
  --out_dir outputs/compare_5/longcat_editB `
  --skip_download --use_4bit --steps 4

# 本地跑 FireRed 在线推理
python tools/data/editor_models/run_firered_online.py `
  --captions data/compare_5_captions_edit.json `
  --input_dir outputs/compare_5/originals `
  --out_dir outputs/compare_5/firered_editB `
  --lora Lightning

# AutoDL Qwen3-VL 重写 prompt
python tools/data/editor_models/autodl_rewrite_edit_to_scene.py `
  --captions_in data/compare_5_captions_edit.json `
  --out data/compare_5_captions_edit_rewritten.json

# 本地分析对比评分
python tools/data/analysis/compare_all_scores.py
python tools/data/analysis/analyze_pair_scores.py

# 数据准备（FiveK）
python tools/data/data_prep/gen_fivek_embeddings.py
python tools/data/data_prep/build_fivek_params_data.py

# 模型续传
.\scripts\local\local_resume_longcat.ps1   # 调用 model_io/local_resume_longcat_v2.py
```

---

## 已删除的文件 (12)

Cleanup commit 删除了一次性 inspect/check helper：
- `check_aesguide.py`、`check_ip2p_diff.py`、`check_venus_labels.py`
- `inspect_fivek_settings.py`、`inspect_pre_aug_data.py`、`inspect_pseudo_labels.py`、`inspect_score_sources.py`、`inspect_vlm_output.py`
- `diagnose_fivek_params.py`、`verify_autodl_models.py`
- `local_resume_aesexpert.py` (v1, 留 v2)、`local_resume_longcat.py` (v1, 留 v2)

需要回滚: `git checkout <prev_commit> -- tools/data/<file>.py`
