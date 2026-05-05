# IntelligenceCamera LoRA SFT 训练记录

实时日志路径（AutoDL 远程）: `/root/autodl-tmp/IntelligenceCamera/logs/`

| Log 文件 | 说明 |
|----------|------|
| `install.log` | LLaMA-Factory 安装进度 |
| `lora_sft_train.log` | LoRA SFT 训练日志 |

实时查看方法（本地 PowerShell）:
```powershell
.\scripts\tail_autodl_log.ps1                                  # 默认 install.log
.\scripts\tail_autodl_log.ps1 -Log lora_sft_train.log          # 训练日志
```

---

## 配置基线（lora_qwen3vl_4b_sft.yaml）

| 项 | 值 |
|------|------|
| Base model | Qwen3-VL-4B-Instruct |
| LoRA rank / alpha | 16 / 32 |
| target | all (全部 linear) |
| freeze vision tower | true |
| template | qwen3_vl_nothink |
| cutoff_len | 12288 |
| batch size × accum | 1 × 8 = 8 |
| lr | 2e-4 |
| epochs | 3.0 |
| scheduler | cosine, warmup 5% |
| dtype | bf16 |
| grad ckpt | true |

数据：`ArtEdit_LoRA_train` (682) / `ArtEdit_LoRA_val` (75)
预计 steps: 682 × 3 / 8 ≈ **256 steps**

---

## 时间线

### 2026-05-05

#### 12:08 — 第一次 install 尝试（失败）
- `git clone` 直连 GitHub 卡住 3 分钟无进度
- PID 1567 hang 在 `git clone --depth 1 https://github.com/LYL1015/JarvisEvo.git`
- **原因**: AutoDL 直连 GitHub 经常超时
- **修复**: 改 `autodl_install_llama_factory.sh` 走 `ghfast.top` 镜像

#### 12:13 — 第二次 install（带 ghproxy 镜像）✅
- nohup 后台 PID 1839, 日志 → `logs/install.log`
- `ghfast.top` 镜像首次尝试成功
- LLaMA-Factory `0.9.4.dev0` 安装完成
- 仅警告 `vllm 0.11.0 requires pydantic>=2.11.7, but you have pydantic 2.10.6`（非阻塞，训练用不到 vllm）
- `llamafactory-cli` 在 `/root/miniconda3/bin/llamafactory-cli`

#### 12:14 — 训练前置条件验证 ✅
| 项 | 值 |
|------|------|
| Qwen3-VL-4B | 8.3 GB（5/3 已下）|
| Train dataset | 1.1 MB / 682 samples |
| Val dataset | 116 KB / 75 samples |
| Disk free | 29 GB |
| GPU | RTX 5090, 32607 MiB total / 32110 MiB free |

#### 12:15 — 第一次训练尝试（失败：YAML 重复 key）❌
- nohup PID 2130, 立即崩溃
- **错误**: `yaml.constructor.ConstructorError: found duplicate key per_device_eval_batch_size`
- **原因**: `lora_qwen3vl_4b_sft.yaml` 同时在 line 37 和 line 51 出现 `per_device_eval_batch_size: 1`
- **修复**: 删除 `### eval` 段中的重复行（line 51），保留 `### train` 段中的（line 37）

#### 12:23 — 第二次训练尝试（失败：dataset 路径双重 data/）❌
- nohup PID 2x, 在加载数据集时崩溃
- **错误**: `ValueError: File data/data/pseudo_labels/sharegpt/ArtEdit_LoRA_train.json not found.`
- **原因**: LLaMA-Factory 的 `_load_single_dataset` 会自动在 `file_name` 前面加 `data/` (LF root 下的 data 目录)。我们的 `dataset_info_snippet.json` 里 file_name 是 `data/pseudo_labels/sharegpt/...`，导致双重 `data/`
- **修复**:
  - 在 AutoDL 上直接修改 `$LF_ROOT/data/dataset_info.json` 的 `file_name` 改为绝对路径 `/root/autodl-tmp/datasets/ArtEdit-Bench/sharegpt/*.json`
  - 同步修复本地 `data/pseudo_labels/sharegpt/dataset_info_snippet.json` 用绝对路径
  - 修复本地 `tools/data/data_prep/build_sharegpt_from_labels.py` 加 `--lf_data_root` 参数，默认输出绝对路径

#### 12:34 — 第三次训练尝试（失败：snippet 没重传）❌
- 仍然报 `data/data/...not found` —— 因为 AutoDL 上的 `dataset_info_snippet.json` 还是旧的有 `data/` 前缀的版本
- `autodl_run_lora_sft.sh` 的 Step 1 用 AutoDL 上的 snippet 覆盖了 LF 的 dataset_info.json
- **修复**: scp 重传修正后的本地 snippet 到 AutoDL

#### 12:42 — 第四次训练尝试 ✅ **成功**
- nohup 后台启动，258 steps 9 分 35 秒训练完成
- **关键指标**:
  | 指标 | 值 |
  |------|-----|
  | train_loss | **0.5077** |
  | eval_loss (ep 3.0) | **0.5559** |
  | eval_loss (ep 2.33) | 0.5649 (still improving) |
  | train_samples/sec | 3.557 |
  | train_runtime | 575s (9:35) |
  | total optim steps | 258 |
- **输出物**: `/root/autodl-tmp/checkpoints/intelligence_camera/lora_v1/`
  - `adapter_model.safetensors` (127 MB)
  - `adapter_config.json`
  - `checkpoint-200/`, `checkpoint-258/`
  - `training_loss.png`, `training_eval_loss.png`
  - `trainer_log.jsonl`, `trainer_state.json`

---

## Eval 结果（v1）

完整报告 → `docs/lora_v1/EVAL_REPORT.md`

#### 13:12 — Full eval on 75 val samples ✅
- **格式合规率: 75/75 (100%)** — CN 35/35, EN 40/40 都完美输出 `<think>` + `<tool_call>` 格式
- **Eval 时间**: 476 秒 (8 min, 6.3s/sample)
- **关键指标 (pred vs GT MAE, paired keys only)**:
  | Key | MAE | 备注 |
  |-----|----:|------|
  | exposure | 0.13 | ⭐ 极准 |
  | dehaze | 0.93 | ⭐ 极准 |
  | clarity | 2.55 | 不错 |
  | vibrance | 3.79 | 不错 |
  | contrast | 5.76 | 可接受 |
  | shadows | 12.70 | 一般 |
  | blacks | 8.02 | 一般 |
  | temp | **812.80** | ❌ 量纲不统一 (Kelvin vs offset 混了) |
  | whites | 110.00 | ❌ 仅 1 个 paired sample |

### Eval 发现的问题

1. **Mode collapse on rare keys** — `dehaze=10`, `tint=10`, `texture=10`, `whites=10` 全部 std=0.0，模型把这些当固定值输出
2. **`temp` 量纲混乱** — 训练数据里有的 sample 用 Kelvin (4000)，有的用 offset (5)；模型学得乱套，需要在数据生成阶段统一
3. **稀有 key 覆盖不足** — `highlights`, `whites`, `texture` 训练样本太少，pred 几乎不用

### Eval 改进方向 (v2 候选)

- 数据清洗: 把 `temp` 全部转成 offset (-100~+100) 或 Kelvin，二选一
- 数据增强: 对 `highlights/whites/texture` 这些稀有 key 加权采样
- 训练: 加 `lora_alpha` (32→64), 多跑 1-2 个 epoch 看 eval_loss 是否还降
- Loss 设计: 对 mode-collapse 的 key 加 KL 正则鼓励多样性

---

## 教训汇总

1. **LLaMA-Factory yaml 不能有重复 key** — `omegaconf` 严格模式
2. **snippet 的 file_name 用绝对路径** — LF 会在相对路径前自动加 `data/`，导致双重前缀
3. **修改 LF dataset_info.json 没用** — `autodl_run_lora_sft.sh` 每次跑都会重新合并 snippet 进去；要修就改 snippet
4. **torchrun 错误吞了 root cause** — 实际报错在日志中间，不在 `error_file: <N/A>` 的尾部

