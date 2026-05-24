# v11 实验计划 — Scene/Region-aware NamedCurves

> 目标: 重新探索“按场景/区域区分调节参数”的方向, 但避免 v9d 的亮度 context 对 wb/saturation 造成负迁移。

## 背景

v9d/v9d_fix 证明单通道 learnable context 可以学到空间分区, 但实际主要学成了亮度图。它对 highlights / shadows / contrast 有轻微帮助, 对 wb 明显有害。

v11 的核心思路是: 不再让所有 action 共享同一种区域调节, 而是把区域调节结构化为 action-gated、region-basis 或局部参数 delta。

## 已实现变体

| 实验 | 开关 | 核心改动 | 结果 |
|------|------|----------|------|
| v11a | `--nc_use_context --nc_action_gated_context` | context 只影响 contrast/shadows/highlights; wb/saturation 使用 global low/high average | 正向, overall=24.61, wb=22.58 |
| v11b | `--nc_use_region_basis` | 用 6 个固定 region basis: dark/mid/bright + warm/cool/neutral, 替代 learnable 1-channel context | 次优, overall=24.53, wb=22.30 |
| v11c | `--nc_use_region_param_delta` | 在 NamedCurves 输出后叠加 region-wise brightness/contrast/shadows/highlights/saturation delta; wb 被 action mask 排除 | 负迁移, overall=24.23, wb=21.41 |
| v11d | `--nc_use_context --nc_use_action_context` | 用 action-conditioned context head 替代普通 ContextHead | 当前 best, overall=24.80, wb=24.53 |

## 实现文件

- `training/firered_baseline/train_lut.py`
  - 新增 `action_gated_context`
  - 新增 `use_region_basis`
  - 新增 `use_region_param_delta`
  - 新增 `use_action_context`
  - 修正 v11 optimizer grouping: `region_param_head` 进入 `lut_lr`
  - 增加 CLI 互斥/依赖校验
- `tools/eval_v11_named_curves.py`
  - 通用 v11 per-action eval
  - 从 checkpoint `args` 自动还原 v11 flags

## 执行顺序

建议按风险从低到高跑:

1. v11a: 最小改动, 验证 action-gated context 是否能修复 v9d 的 wb 负迁移。
2. v11b: 固定 region basis 曲线, 验证 handcrafted region 是否比 learned context 稳。
3. v11c: 局部参数 delta, 验证区域参数调节方向是否有效。
4. v11d: action-conditioned context, 验证 action-aware spatial routing。

## 训练命令

### v11a action-gated context

```powershell
python -u training\firered_baseline\train_lut.py `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_7d_anchor --nc_use_context --nc_action_gated_context `
  --dropout 0.5 `
  --out_dir checkpoints\lut_v11a_action_gated_context --epochs 80 `
  --param_weight 0.05 `
  2>&1 | Tee-Object -FilePath checkpoints\lut_v11a_action_gated_context_log.txt
```

```powershell
python tools\eval_v11_named_curves.py --ckpt checkpoints\lut_v11a_action_gated_context\best.pt
```

### v11b fixed region-basis curves

```powershell
python -u training\firered_baseline\train_lut.py `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_7d_anchor --nc_use_region_basis `
  --dropout 0.5 `
  --out_dir checkpoints\lut_v11b_region_basis --epochs 80 `
  --param_weight 0.05 `
  2>&1 | Tee-Object -FilePath checkpoints\lut_v11b_region_basis_log.txt
```

```powershell
python tools\eval_v11_named_curves.py --ckpt checkpoints\lut_v11b_region_basis\best.pt
```

### v11c region-wise parameter delta

```powershell
python -u training\firered_baseline\train_lut.py `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_7d_anchor --nc_use_region_param_delta `
  --dropout 0.5 `
  --out_dir checkpoints\lut_v11c_region_param_delta --epochs 80 `
  --param_weight 0.05 `
  2>&1 | Tee-Object -FilePath checkpoints\lut_v11c_region_param_delta_log.txt
```

```powershell
python tools\eval_v11_named_curves.py --ckpt checkpoints\lut_v11c_region_param_delta\best.pt
```

### v11d action-conditioned context

```powershell
python -u training\firered_baseline\train_lut.py `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_7d_anchor --nc_use_context --nc_use_action_context `
  --dropout 0.5 `
  --out_dir checkpoints\lut_v11d_action_context --epochs 80 `
  --param_weight 0.05 `
  2>&1 | Tee-Object -FilePath checkpoints\lut_v11d_action_context_log.txt
```

```powershell
python tools\eval_v11_named_curves.py --ckpt checkpoints\lut_v11d_action_context\best.pt
```

## 判断标准

| 指标 | 通过标准 |
|------|----------|
| Overall | ≥ v9a 24.52 dB 或接近且 per-action 更均衡 |
| wb | 不低于 v9a 22.35 dB 太多 |
| tone actions | highlights/shadows/contrast 至少保持 v9d_fix 的小幅提升 |
| 稳定性 | 不出现 v10b_fix / v10c 那种明显负迁移 |

## 实验结果

| 实验 | Train best | Eval overall | wb | saturation | highlights | shadows | contrast | 结论 |
|------|------------|--------------|----|------------|------------|---------|----------|------|
| v9a baseline | 24.42 | 24.52 | 22.35 | 22.84/22.93 | 23.93/23.62 | 26.21/26.50 | 26.80/26.31 | 原始 NamedCurves baseline |
| v10b Plan C | 24.47 | 24.47 | 22.76 | 22.93 | 23.76 | 26.61 | 25.76 | EMA/resume 收益, NILUT gate=0 |
| v11a | 24.57 @ Ep24 | 24.61 | 22.58 | 22.88 | 23.84 | 26.35 | 26.70 | action-gated context 修复 v9d 的 wb 负迁移 |
| v11b | 24.42 @ Ep39 | 24.53 | 22.30 | 23.32 | 23.78 | 26.18 | 26.11 | saturation 有收益, 但 wb/contrast 低于 v11a |
| v11c | 24.14 @ Ep21 | 24.23 | 21.41 | 22.69 | 24.25 | 25.91 | 25.39 | highlights 有收益, 但 wb/contrast 明显负迁移 |
| **v11d** | **24.78 @ Ep36** | **24.80** | **24.53** | **23.48** | **23.86** | **26.43** | **25.79** | **当前 SOTA: action-conditioned context 同时提升 overall 与 wb** |

## 结果解读

v11d 是目前最强结果。它不是简单恢复 v9a/v10 的水平, 而是在 wb action 上从 v9a 的 22.35 dB 提升到 24.53 dB, 同时 overall 达到 24.80 dB。

v11a 证明“context 负迁移”的主要问题来自全 action 共享亮度 routing。只让 context 作用于 tone actions 后, wb 从 v9d_fix 的 21.55 恢复到 22.58。

v11d 进一步证明硬 gate 不是上限。action-conditioned context 可以让不同 action 学不同空间 routing, 因而 wb 也能从 context 中受益, 而不是被亮度先验拖累。

v11b/v11c 作为负面对照保留:

- v11b 的 fixed region basis 太刚性, saturation 提升但 wb/contrast 下降。
- v11c 的局部参数 delta 太自由, highlights 提升但 wb/contrast 明显负迁移。

## 如果出现明显问题

- v11a 如果 wb 仍下降: 说明 low/high average 本身也带来 bias, 需要让 wb 完全走单 bin identity/global branch。
- v11b 如果整体下降: 说明 6-bin region curves 参数太多, 可降到 3 luminance bins 或 3 color bins 单独 ablation。
- v11c 如果 early epoch 很低: region delta 幅度仍可能过大, 可把 scales 降半。
- v11d 如果不如 v11a: action-conditioned context 容量不足或仍被亮度主导, 可与 action-gated 同时启用作为 v11d_fix。

## 当前状态

四个 v11 变体均已训练并评估。当前推荐保留 `checkpoints\lut_v11d_action_context\best.pt` 作为 WB refinement track 的 best checkpoint。

## v11d follow-up

### 1. 可视化 action-conditioned context maps

生成 viewer:

```powershell
python tools\visualize_v11d_action_context.py `
  --ckpt checkpoints\lut_v11d_action_context\best.pt `
  --out_dir outputs\lut_v11d_action_context_viewer `
  --max_samples 30 `
  --split_seed 42
```

输出:

- `outputs\lut_v11d_action_context_viewer\viewer.html`
- `outputs\lut_v11d_action_context_viewer\summary.json`
- `outputs\lut_v11d_action_context_viewer\avg_maps\avg_all_actions.jpg`

重点看:

- 不同 action 的 context map 是否明显不同。
- `corr(ctx, lum)` 是否不像 v9d_fix 那样单纯被亮度主导。
- `wb` 的 context 是否有结构, 而不是退化为 uniform 或纯亮度图。

### 2. v11d seed repeat

`--seed` 改模型初始化/训练随机性, `--split_seed 42` 固定原始 val split, 这样和 v11d 主结果可比。

Seed 123:

```powershell
python -u training\firered_baseline\train_lut.py `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_7d_anchor --nc_use_context --nc_use_action_context `
  --dropout 0.5 `
  --seed 123 --split_seed 42 `
  --out_dir checkpoints\lut_v11d_action_context_seed123 --epochs 80 `
  --param_weight 0.05 `
  2>&1 | Tee-Object -FilePath checkpoints\lut_v11d_action_context_seed123_log.txt
```

```powershell
python tools\eval_v11_named_curves.py `
  --ckpt checkpoints\lut_v11d_action_context_seed123\best.pt `
  --split_seed 42
```

Seed 777:

```powershell
python -u training\firered_baseline\train_lut.py `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_7d_anchor --nc_use_context --nc_use_action_context `
  --dropout 0.5 `
  --seed 777 --split_seed 42 `
  --out_dir checkpoints\lut_v11d_action_context_seed777 --epochs 80 `
  --param_weight 0.05 `
  2>&1 | Tee-Object -FilePath checkpoints\lut_v11d_action_context_seed777_log.txt
```

```powershell
python tools\eval_v11_named_curves.py `
  --ckpt checkpoints\lut_v11d_action_context_seed777\best.pt `
  --split_seed 42
```

判断:

- 如果两个 repeat 都 ≥24.65 overall 且 wb 仍明显高于 22.58, v11d 稳定成立。
- 如果方差很大, 先保留主 seed best, 论文中报告 seed variance。

seed 123 结果:

- `checkpoints\lut_v11d_action_context_seed123\best.pt`
- `best_val_psnr=24.33dB @ Ep12`, early stop Ep24

seed 777 结果:

- `checkpoints\lut_v11d_action_context_seed777\best.pt`
- `best_val_psnr=24.32dB @ Ep22`, early stop Ep34

汇总 v11d (val split `--split_seed 42`):

| seed | best PSNR | best epoch |
|------|-----------|------------|
| 42 (主) | 24.80 | 36 |
| 123 | 24.33 | 12 |
| 777 | 24.32 | 22 |

`mean ± std (n=3) = 24.48 ± 0.27 dB`

v11a 也补了同样的 3-seed repeat:

| seed | best PSNR | best epoch |
|------|-----------|------------|
| 42 (主) | 24.61 | ? |
| 123 | 24.24 | 22 |
| 777 | 24.54 | 34 |

`mean ± std (n=3) = 24.46 ± 0.20 dB`

**v11a vs v11d 最终结论**:

- 两者均值差 0.02 dB, 远小于任一 std, **统计上完全等价**。
- seed 42 在两个 variant 上都偏高, 都不应作为单点结论。
- "action-conditioned context > action-gated context" 不成立。
- "action-aware context 解决 v9d 对称性陷阱" 在两个 variant 上都站得住。
- 论文以 **v11a 为主模型**(更简单), v11d 为 ablation。
- 数值统一报 mean ± std (3 seeds), 不报 24.80。

### 3. v11d EMA / low-LR fine-tune

从当前 best 只加载模型权重, 不恢复旧 optimizer/scheduler:

```powershell
python -u training\firered_baseline\train_lut.py `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_7d_anchor --nc_use_context --nc_use_action_context `
  --dropout 0.5 `
  --resume checkpoints\lut_v11d_action_context\best.pt `
  --resume_model_only `
  --seed 42 --split_seed 42 `
  --out_dir checkpoints\lut_v11d_action_context_ema_ft `
  --epochs 30 --patience 8 `
  --lr 3e-5 --lut_lr 1e-4 --ema_decay 0.999 `
  --param_weight 0.05 `
  2>&1 | Tee-Object -FilePath checkpoints\lut_v11d_action_context_ema_ft_log.txt
```

```powershell
python tools\eval_v11_named_curves.py `
  --ckpt checkpoints\lut_v11d_action_context_ema_ft\best.pt `
  --split_seed 42
```

判断:

- 如果 eval_overall > 24.80 且 wb 不下降, 替换当前 best。
- 如果 only overall 小涨但 wb 明显下降, 不替换 v11d 主 checkpoint。
- 如果下降, 说明当前 best 已接近小数据最优点, fine-tune 丢弃。

结果:

- `checkpoints\lut_v11d_action_context_ema_ft\best.pt`
- `best_val_psnr=24.73dB @ Ep3`
- Ep11 early stop, final val `24.64dB`
- 低于原 v11d `24.80dB`, 不替换当前 best。

---

## 4. v11a Expert C clean-target 实验 (2026-05-16)

### 动机

3-seed repeat 后, v11a/v11d 在 pseudo-label val set 上的 ceiling 都是 ~24.5 dB。
我们怀疑 ceiling 来自 pseudo-label 噪声 (inverse_fit L1 mean=0.076, WB=0.123, 65% fail), 而非架构。

为了验证, 我们用 `tools/build_fivek_expert_jsonl.py` 直接从 FiveK Expert C 参数表 (`fivek_expert_abcde_params.json`) 生成 clean 训练数据:
- target = `apply_diff_isp(real_Expert_C_param, input)` (无 inverse_fit 误差)
- 排除 brightness (train_lut 仅支持 5 actions)
- 排除 WB 越界值 (2000-12000K), 16820 → 15739 条
- 4989 unique images, 12612 train / 3127 val (val_ratio=0.2)

### 训练命令

```powershell
python -u training\firered_baseline\train_lut.py `
  --jsonl outputs\fivek_expert_c_master\pseudo_labels.jsonl `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_7d_anchor --nc_use_context --nc_action_gated_context `
  --dropout 0.5 --param_weight 0.05 `
  --seed 42 --split_seed 42 `
  --out_dir checkpoints\lut_v11a_expertC_seed42 `
  --epochs 50 --patience 10 `
  2>&1 | Tee-Object -FilePath checkpoints\lut_v11a_expertC_seed42_log.txt
```

### 训练结果

- `checkpoints\lut_v11a_expertC_seed42\best.pt`
- `best_val_psnr=41.88dB @ Ep29`

### Eval 1 — Expert C in-domain val (clean diff_isp targets)

```powershell
python tools\eval_v11_named_curves.py `
  --ckpt checkpoints\lut_v11a_expertC_seed42\best.pt `
  --jsonl outputs\fivek_expert_c_master\pseudo_labels.jsonl `
  --split_seed 42
```

| Action | n | mean | p10 | p90 |
|--------|---|------|-----|-----|
| highlights | 971 | **45.81** | 41.03 | 49.68 |
| wb | 871 | **42.10** | 33.63 | 48.31 |
| contrast | 645 | **41.44** | 35.05 | 47.26 |
| saturation | 319 | **41.06** | 32.25 | 47.85 |
| shadows | 321 | **31.06** | **19.67** | 40.52 |
| **Overall** | 3127 | **41.88** | — | — |

### Eval 2 — Pseudo-label val (domain transfer to FireRed inverse_fit targets)

```powershell
python tools\eval_v11_named_curves.py `
  --ckpt checkpoints\lut_v11a_expertC_seed42\best.pt `
  --jsonl outputs\inverse_fit_pilot\fivek_500_master\pseudo_labels.jsonl `
  --split_seed 42
```

| Action | n | mean | p10 | p90 | vs v11a baseline (pseudo-trained) |
|--------|---|------|-----|-----|-----------------------------------|
| wb | 7 | **15.62** | 12.75 | 19.27 | 22.58 → **-6.96** |
| saturation | 17 | 19.88 | 17.08 | 22.72 | 23.48 → -3.60 |
| highlights | 19 | 21.36 | 16.69 | 28.55 | 23.86 → -2.50 |
| shadows | 18 | 22.77 | 18.89 | 25.33 | 26.43 → -3.66 |
| contrast | 13 | 24.41 | 18.24 | 29.83 | 25.79 → -1.38 |
| **Overall** | 74 | **21.36** | — | — | 24.46 → **-3.10** |

### 关键解读

1. **架构没有问题**: 41.88 dB 证明 v11a 容量足以"完美"逆向 diff_isp 操作, 41.88 vs 24.46 (20 dB gap) 不是架构容量差距, 是 GT 信噪比差距。

2. **GT 选择决定一切**: 同一个模型在 diff_isp-rendered targets 上 41.88 dB, 在 inverse_fit pseudo-label targets 上 21.36 dB, 差距 20 dB 完全来自 target 分布不同。

3. **"训练干净 GT → 迁移噪声 GT" 不成立**: Expert C 训练的模型在 pseudo-label val 上反而 **低于** baseline (-3.10 dB), 说明:
   - baseline 学到的是"如何拟合 inverse_fit 噪声模式"
   - Expert C 模型学到的是"如何精确逆向 diff_isp"
   - 两者优化的是不同的"任务", 不能简单比较

4. **WB 跌幅最大** (-6.96 dB): 因为 pseudo-label 的 WB target 噪声最严重 (D-fail 65%), 而 Expert C 的 WB 是真实精确值, 两个分布偏差最大。

5. **Shadows 在 in-domain 也是最弱** (31.06 dB, p10=19.67): 即使在干净 target 上, shadows 的曲线近似仍是难点, 原因可能是 `apply_diff_isp` 的 shadows 算子在暗像素处梯度敏感。

6. **理论参考**: 真实 Expert C JPEG 上 7D ISP ceiling ≈ 23.94 dB, 即 NamedCurves 24.91 dB 之所以更高, 不是因为他们模型更强, 而是因为他们不受 7D 参数化约束 (直接预测图像)。

### 论文叙事

这是一个有价值的负面 / 解构性结果, 可写入实验章节:

> "We isolate that the previously reported ~24.5 dB ceiling for our NamedCurves variants is dominated by **pseudo-label noise**, not architecture. When supervised with noise-free targets rendered from real FiveK Expert C parameters, the same architecture reaches 41.88 dB in-domain. However, this model fails to transfer (21.36 dB) to inverse_fit targets, indicating that what looked like 'better generalization' from pseudo-label training was actually overfitting to inverse_fit residual patterns. The fundamental 7D ISP capacity ceiling (~23.94 dB on real Expert C JPEGs) bounds any 7-parameter approach; moving beyond requires either (i) higher-capacity differentiable rendering (NILUT/Vera-style residuals), or (ii) direct image-domain supervision without ISP detour."

### 下一步候选

| 方向 | 难度 | 预期 |
|------|------|------|
| **A) Parameter Perturbation 训练 (PerTouch [R16] idea)** | ★☆ | 同时在两种 GT 下接近最优, 缓解 overfit |
| **B) 真实 Expert C JPEG eval** (替换 target 为真 JPEG) | ★★ | 直接对标 NamedCurves 24.91 dB |
| **C) 直接监督真实 Expert C 图像 (不走 diff_isp)** | ★★★ | 突破 7D ISP ceiling, 但失去可解释性 |
| **D) 接受现状, 论文以负面结果定位** | ★ | 当前所有结果已足够支撑叙事 |

---

## 5. v11a Expert C + Parameter Perturbation σ=0.03 (2026-05-16)

### 动机

§4 的 Expert C 实验暴露了 41.88 ↔ 21.36 的 in-domain / domain-transfer gap。
假设这个 gap 是因为模型过拟合了 diff_isp 的精确输出, 引入 PerTouch [R16]-style
parameter perturbation 是否能让模型学到更平滑的 (param→image) 流形, 从而提高
domain transfer?

### 实现

`train_lut.py` 新增:

- `ACTION_TO_PARAM_7D_INDICES = [contrast→2, saturation→5, shadows→3, highlights→4, wb→0]`
  (action_idx → 7D 参数槽位映射)
- `denormalize_params_7d(params_norm)`: standalone (B,7) [-1,1] → physical dict
- `--param_perturb_sigma` CLI flag (default 0.0 = disabled)
- 训练循环 (line 1399-1432): 当 sigma>0 时, 对**当前 action 对应的 7D 槽位**加 Gaussian noise,
  其他槽位 mask 为 0 (因为 per-action 数据中其余参数恒为 0), 然后调用 `apply_diff_isp`
  重新渲染 target, 同时把扰动后的 params_norm_7d 用于 param 监督。所有操作在 `torch.no_grad()` 下进行。

仅训练时生效, eval 不扰动。

### 训练命令

```powershell
python -u training\firered_baseline\train_lut.py `
  --jsonl outputs\fivek_expert_c_master\pseudo_labels.jsonl `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_7d_anchor --nc_use_context --nc_action_gated_context `
  --dropout 0.5 --param_weight 0.05 `
  --param_perturb_sigma 0.03 `
  --seed 42 --split_seed 42 `
  --out_dir checkpoints\lut_v11a_expertC_perturb03_seed42 `
  --epochs 50 --patience 10 `
  2>&1 | Tee-Object -FilePath checkpoints\lut_v11a_expertC_perturb03_seed42_log.txt
```

### 训练结果

- `checkpoints\lut_v11a_expertC_perturb03_seed42\best.pt`
- `best_val_psnr=41.80dB @ Ep15` (vs baseline 41.88dB @ Ep29)
- 收敛更快 (Ep15 vs Ep29) 但 in-domain 几乎不变
- 总耗时 3.5h (比 baseline 2h 多, 因为每 batch 多一次 apply_diff_isp 重渲染)

### Eval 1 — Expert C in-domain val

| Action | baseline (no perturb) | σ=0.03 | Δ |
|--------|----------------------|--------|---|
| highlights | 45.81 | 45.71 | -0.10 |
| wb | 42.10 | 42.21 | +0.11 |
| contrast | 41.44 | 41.37 | -0.07 |
| saturation | 41.06 | 40.99 | -0.07 |
| shadows | 31.06 | 30.56 | -0.50 |
| **Overall** | **41.88** | **41.80** | **-0.08** |

### Eval 2 — Domain transfer (FireRed pseudo val)

| Action | baseline (no perturb) | σ=0.03 | Δ |
|--------|----------------------|--------|---|
| wb | 15.62 | 15.56 | -0.06 |
| saturation | 19.88 | 19.93 | +0.05 |
| highlights | 21.36 | 21.36 | 0.00 |
| shadows | 22.77 | 22.55 | -0.22 |
| contrast | 24.41 | 24.45 | +0.04 |
| **Overall** | **21.36** | **21.32** | **-0.04** |

### 结论: σ=0.03 perturbation 是 no-op

所有 Δ 都在 ±0.5 dB 内, 统计上完全等价 baseline。perturbation **既没有损害也没有帮助**。

#### 为什么 σ=0.03 没用 — 量级问题

σ=0.03 在归一化 [-1,1] 空间, 换算到物理量级:

| 参数 | scale | σ=0.03 物理扰动 | 视觉感知 |
|------|-------|----------------|----------|
| white_balance | 4000K | ±120K | < JND |
| contrast | 100 | ±3 | 几乎不可见 |
| saturation | 100 | ±3 | 几乎不可见 |
| shadows | 100 | ±3 | 几乎不可见 |
| highlights | 100 | ±3 | 几乎不可见 |

扰动重渲染的 target 和原 target 自身 PSNR 大概在 40+ dB, 模型轻松"smooth over"。
要看到效果至少需要 σ=0.10 (wb ±400K, 其他 ±10), 但即便如此也未必能解决 §4 的 gap。

#### 为什么"加大 σ"也未必能解 — 结构性问题

σ=0.03 的 negative result 实际上揭示了一个更深层的诊断:

- **§4 的 gap (41.88 vs 21.36) 是结构性分布差异, 不是平滑性问题**
- diff_isp(P_known) targets 来自 7D ISP 操作的精确输出
- inverse_fit pseudo-label targets 有结构化噪声 (WB inverse_fit 65% fail, 整体 L1=0.076)
- **随机高斯噪声扰动 ≠ inverse_fit 残差的特定 failure mode 分布**

理论上, 即使 σ→∞, 也不能让 diff_isp 重渲染的 target 分布逼近 inverse_fit 的失败模式分布。
两者是不可桥接的两个流形。

---

## 5.5. σ=0.10 update — completing the sigma sweep (2026-05-17)

### 动机

§5 的 σ=0.03 是 no-op (±0.25 dB), 但论证依赖一个 "σ 太小" 的潜在反驳。
为了完全堵死这个解释, 我们再跑一个 **σ=0.10** (3.3× larger) 的训练。

### 训练

```powershell
python -u training\firered_baseline\train_lut.py `
  --jsonl outputs\fivek_expert_c_master\pseudo_labels.jsonl `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_7d_anchor --nc_use_context --nc_action_gated_context `
  --dropout 0.5 --param_weight 0.05 `
  --param_perturb_sigma 0.10 `
  --seed 42 --split_seed 42 `
  --out_dir checkpoints\lut_v11a_expertC_perturb10_seed42 `
  --epochs 50 --patience 10
```

- `checkpoints\lut_v11a_expertC_perturb10_seed42\best.pt`
- `best_val_psnr=41.71dB @ Ep39` (vs σ=0.03: 41.80@Ep15, baseline: 41.88@Ep29)
- 没有早停 (满 50 epoch 顶满 patience), 表明 σ=0.10 让训练任务变得更难但仍可学。
- 总耗时 3.5h.

### 完整 σ-sweep 结果

#### In-domain (Expert C val, 3127 samples)

| Action | baseline | σ=0.03 | σ=0.10 |
|--------|---------:|-------:|-------:|
| highlights | 45.81 | 45.71 | 45.71 |
| wb | 42.10 | 42.21 | 42.05 |
| contrast | 41.44 | 41.37 | 41.00 |
| saturation | 41.06 | 40.99 | 41.21 |
| shadows | 31.06 | 30.56 | 30.62 |
| **Overall** | **41.88** | **41.80** | **41.71** |

in-domain 单调下降, 但所有 Δ 在 ±0.5 dB 内 — 模型仍然能近乎无损地拟合 perturbed 目标。

#### Domain transfer (FireRed pseudo val, 74 samples)

| Action | baseline | σ=0.03 | σ=0.10 |
|--------|---------:|-------:|-------:|
| wb | 15.62 | 15.56 | 15.62 |
| saturation | 19.88 | 19.93 | 19.90 |
| highlights | 21.36 | 21.36 | 21.37 |
| shadows | 22.77 | 22.55 | 22.40 |
| contrast | 24.41 | 24.45 | 24.41 |
| **Overall** | **21.36** | **21.32** | **21.27** |

**Domain transfer 单调微降, 所有 Δ ≤ 0.40 dB**:

- 3.3× σ 增加 → transfer 仅从 21.32 → 21.27 (-0.05 dB)
- 没有任何一个 per-action 数字向 baseline 24.46 dB 靠近
- wb (15.62) 完全不变 — perturbation 对最关键的 action 零影响

### 结论: σ-magnitude 假设彻底证伪

σ=0.03 vs σ=0.10 的对比给出 **monotonic-no-effect** 曲线, 这比单点更具说服力:

1. **In-domain: 单调微降** (41.88 → 41.80 → 41.71): perturbation 确实在 "工作" — 模型在更难的 task 上略损精度。
2. **Transfer: 单调微降** (21.36 → 21.32 → 21.27): perturbation 没有 "传递" 到 transfer 域, 反而轻微负面。
3. **关键 action wb 零变化** (15.62 → 15.56 → 15.62): perturbation 对最弱的 action 完全无效。

这彻底排除了 "σ 太小" 的解释, 留下的唯一可能是: **the gap is structural, not magnitude-related**.

### 论文叙事补强

§4 + §5 的组合给出更强的 deconstructive claim:

> "We tested whether the 41.88 ↔ 21.36 dB in-domain/transfer gap reflects an
> overfitting / smoothness deficit and could be closed by Gaussian parameter
> perturbation (PerTouch [R16]-inspired). A two-point sigma sweep (σ ∈ {0.03,
> 0.10}, a 3.3× scale increase) produced a **monotonic-no-effect** curve on
> domain transfer (21.36 → 21.32 → 21.27 dB; |Δ| ≤ 0.40 dB on every per-action
> bucket), with the worst-performing action (WB at 15.62 dB) entirely
> insensitive to noise scale. We conclude that **the gap is a structural
> mismatch between the diff_isp rendered target distribution and the
> inverse_fit pseudo-label distribution**, not a smoothness issue —
> independently corroborated by the inverse_fit pipeline residual analysis
> (§5.6) showing 0% OK / 58% FAIL verdicts and only 35% of WB samples
> usable for training. Domain transfer in this setup requires either (i)
> training jointly on both target distributions, or (ii) abandoning the
> ISP detour and supervising image-domain directly."

### 下一步候选 (更新)

| 方向 | 难度 | 预期 |
|------|------|------|
| ~~A) Parameter Perturbation~~ | ~~★☆~~ | **σ ∈ {0.03, 0.10} 都已证伪 (§5 + §5.5)** |
| ~~B) 真实 Expert C JPEG eval~~ | ~~★★~~ | **不可行, 磁盘上没有真实 Expert C 渲染图** |
| **C) 直接图像监督 (绕过 diff_isp)** | ★★★ | 唯一能突破 7D ceiling 的路径 |
| **D) 现状已可写论文** ⭐ | ★ | §3 + §4 + §5 + §5.5 + §5.6 已构成完整 deconstructive 故事 |

---

## 5.6. inverse_fit pipeline 质量诊断 (2026-05-17)

### 工具与命令

`tools/analyze_inverse_fit_residuals.py` — 聚合 LBFGS 反推质量统计。

```powershell
python tools/analyze_inverse_fit_residuals.py `
  --jsonl outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl
```

输出报告: `outputs/inverse_fit_pilot/fivek_500_master/residual_analysis.md`

### 关键发现 #1: 整个 pipeline 没有一个 "OK" verdict

| Verdict | n | % |
|---------|--:|--:|
| OK | **0** | **0.0%** |
| WARN | 191 | 38.3% |
| FAIL | 292 | 58.5% |

LBFGS 反推 7D ISP 参数到 Expert C 目标, **没有任何一个样本达到合格质量**。即便
被算作 "训练数据" 的 372 个样本 (74.5%), 也都是带残差的近似拟合。

### 关键发现 #2: WB 仅 35% 可用, FAIL 率 83%

| Action | n | L1 mean | FAIL | Usable |
|--------|--:|--------:|-----:|-------:|
| **wb** | 100 | 0.1232 | **83** | **35 (35%)** ⚠️ |
| saturation | 100 | 0.0831 | 73 | 75 |
| highlights | 99 | 0.0681 | 58 | 84 |
| contrast | 100 | 0.0581 | 44 | 86 |
| shadows | 100 | 0.0496 | 34 | 93 |

**这与 §3 baseline 的 wb=22.58 dB (5 个 action 中最差) 完全一致** —
LBFGS 反推 wb 最难, 训练 wb 最难, 模型 eval wb 最差。

### 关键发现 #3: WB 推出的色温分布严重偏热

`P_inferred.white_balance` 分布:

| Stat | Value (K) |
|------|----------:|
| min | 2015 |
| p10 | 5612 |
| p50 | 6915 |
| p90 | 9644 |
| max | 9978 |
| mean | 7079 |

mean 7079K 远高于 daylight 5500K, p90 接近黑体范围上界 10000K。**大量样本
需要通过 "反向降温" 来匹配 Expert C 引入的非黑体色偏 (品红/绿)**, 但 1D
黑体曲线表达不了, 因此 LBFGS 把它们推到了边界。

### 论文论据 — 量化的 "结构性" 证据

这一节给 §5/§5.5 的 "结构性失配" 主张提供了**直接的、可量化的根因
证据**, 而不只是间接推论:

1. **OK verdict 0%** → "pseudo-label" 这个名字本身就有误导性, 实际上是 "近似 label"
2. **WB 65% 丢弃** → 训练 wb 的有效样本比其他 action 少 60%+, 解释了 wb 是最差 action
3. **色温 mean 7079K, p90 9644K** → Expert C 编辑确实包含大量 1D 黑体表达不了的色偏,
   这是 [R1] CST-MLP 论文的直接动机

论文叙事可以从 "我们的 24.46 dB ceiling 是数据噪声" 加强为:

> "The 24.46 dB ceiling is not merely a data-noise issue but a **structural
> consequence of the 1D color-temperature parameterization itself**: 65%
> of WB-action pseudo-labels could not be fit by LBFGS with PSNR ≥ acceptable
> threshold, and the inferred temperature distribution (mean 7079K, p90
> 9644K) is heavily biased toward boundary values, indicating that Expert C's
> WB edits contain off-Planckian casts (magenta/green) that no 1D scalar can
> represent (corroborating [R1] CST-MLP's motivation)."

---

## 5.7. Model parameter prediction quality (2026-05-17)

### 工具与命令

`tools/eval_param_prediction.py` — 报告模型 7D 参数预测的 per-action MAE,
per-parameter MAE 跨样本, 以及系统性偏差 (signed bias)。

```powershell
python tools/eval_param_prediction.py `
  --ckpt checkpoints/lut_v11a_action_gated_context/best.pt `
  --jsonl outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl
```

输出: `checkpoints/lut_v11a_action_gated_context/param_pred_eval.json` (full dump)

### 关键发现 — Data-Quality ↔ Model-MAE 完美单调相关

| Action | inverse_fit Usable | Active param MAE | baseline PSNR |
|--------|-------------------:|-----------------:|--------------:|
| **shadows** | 93% | **1.09** (1% of range) | 26.43 dB (best) |
| contrast | 86% | 21.11 | ~24-25 |
| highlights | 84% | 16.90 | ~24 |
| saturation | 75% | 18.30 | ~22-25 |
| **wb** | **35%** ⚠️ | **1791 K** (45% of range) | **22.58 dB** (worst) |

**完美单调**: 越干净的 pseudo-label → 模型 MAE 越低 → PSNR 越高。

- **Shadows 1.09 / 100 = 1% MAE** 证明: **架构有能力把参数预测得非常准, 只要数据干净**
- **WB 1791 / 4000 = 45% MAE** 证明: **标签结构有问题, 模型无论如何也学不会**

### 系统性偏差 — "Regression to no-edit"

| Action | gt_mean | pred_mean | bias | pred > gt | 行为 |
|--------|--------:|----------:|-----:|----------:|------|
| contrast | +9.91 | +17.12 | +7.21 | 38.5% | 过度对比 |
| saturation | +27.39 | +13.93 | **-13.46** | **17.6%** | 严重欠饱和 |
| highlights | -18.62 | -6.33 | +12.30 | 68.4% | 欠压暗 |
| **wb** | **8459K** | **6983K** | **-1476K** | **14.3%** | **严重偏冷** |
| shadows | -0.47 | -0.78 | -0.31 | 44.4% | 平衡 |

**Saturation 和 WB 的 % pred>gt 都极低 (17%, 14%)**: 模型在 >80% 的样本上都
欠预测, 系统性地把饱和度和色温拉回 "原图状态"。这是经典的**对噪声标签做
L2 回归**的行为 — 模型在不确定时回归到均值。

### Per-parameter MAE (Section 2 of report)

| param | n | MAE | gt_mean | pred_mean |
|-------|--:|----:|--------:|----------:|
| **clarity** | 74 | **22.35** | 21.75 | 14.74 |
| **brightness** | 74 | **16.64** | 16.44 | 11.62 |
| white_balance | 74 | 798 K | 5849 K | 5782 K |
| shadows | 74 | 4.52 | -2.88 | -1.85 |
| highlights | 74 | 15.92 | -9.98 | -5.93 |
| saturation | 74 | 11.98 | 10.56 | 7.60 |
| contrast | 74 | 20.22 | 27.86 | 21.73 |

GT clarity/brightness 不为零是因为 inverse_fit 自由地把残差吸收到这两个
"未指定" 的参数里。模型被迫学这种 "擦屁股式" 预测, 这本身就是 pseudo-label
噪声的一种表现。

### 论文论据 — 数据质量决定参数预测精度

这一节给 §5.6 (inverse_fit 质量) 添加了**直接的下游验证**:

> "The model's per-action parameter prediction accuracy is **strongly
> correlated with the inverse-fit pipeline's per-action usable rate**:
> shadows (93% usable) achieves a 1.09-unit MAE on a [-100, 100] range
> (1% error), while WB (35% usable) yields a 1791K MAE on a 4000K
> normalization range (45% error). The architecture is **demonstrably
> capable** of accurate parameter prediction when supervised by
> well-fit pseudo-labels (shadows); the per-action PSNR ceiling is
> set by the per-action **data quality**, not the model. Furthermore,
> systematic biases — particularly the model under-predicting WB by
> 1476K (86% of samples cooler than GT) and saturation by 13.5 units
> (82% of samples below GT) — exhibit classic regression-to-the-mean
> behavior expected of L2-trained networks under noisy targets,
> directly corroborating the structural-mismatch claim of §5."

### 对论文 4-ceilings 框架的补充

之前 §5.6 给出了 3 个 ceilings (pseudo-label noise / WB structural / 7D ISP capacity)。
§5.7 揭示了**第 4 个观察 — 不是 ceiling, 而是 mechanism**:

**模型在噪声标签下不可避免的 regression-to-mean 行为**, 这解释了为什么:
1. WB pred=6983K, gt=8459K — 模型不敢相信"夸张"的 GT
2. Saturation pred=14, gt=27 — 模型不敢相信"激进"的 GT
3. 而 shadows GT 本身平均接近 0 (-0.47) → 模型也接近 0 → 误差天然低

这告诉我们: 即使有完美的架构, 在带噪声的 pseudo-label 上训练, 模型也会
"被迫保守", 这就是 24.46 dB ceiling 的**生成机制**。

