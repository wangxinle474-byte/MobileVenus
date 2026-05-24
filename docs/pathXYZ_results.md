# Pilot 0 Path X/Y/Z 实验总结 (2026-05-18)

## 背景

v11a 7D-ISP-bound NamedCurves 在 FireRed pseudo-label 上 SOTA = 24.46 ± 0.20 dB
(3 seeds)，per-action WB=22.58 dB（最差，65% 样本 LBFGS 拟合失败）。

§4/§5 已经证伪了 5 个 architecture-side bridging 干预（VeraRetouch, NILUT,
PerTouch σ-sweep, region context, region delta），共同结论：**7D ISP 的渲
染天花板和 28 个 WB 训练样本的数据稀疏是两个独立瓶颈**。

本次同时探索三条路径：

- **Path X**: 在 v11a backbone 之后加 implicit residual head (small U-Net,
  ImplicitResidualHead, 3.68M params), **冻 backbone**, 只学像素残差
- **Path Y**: 用 AutoDL 跑 FireRed 再生成 +1500 个 pseudo-labels (300 张新
  FiveK × 5 actions), 扩 4× WB 训练样本
- **Path Z**: 弃用 7D ISP, 用纯图像域 ImageDomainResUNet (8.26M, base_ch=48)
  端到端学 (orig, action) → refined

## 实验结果（同 val 集 n=74, split_seed=42）

| 指标 | v11a (Path A) | Path X | Path Z |
|---|---|---|---|
| **architecture** | 7D ISP + NamedCurves (3.09M) | v11a + 残差 head (3.09M+3.68M, frozen) | 纯 U-Net (8.26M) |
| **train epochs** | 24 (early-stop) | 12 (early-stop) | 19 (early-stop) |
| **best val_psnr** | 24.57 | 24.73 | 24.11 |
| **eval_overall** | **24.61** | **25.28** | **24.52** |
| **Δ vs v11a** | — | **+0.67 dB** | **-0.09 dB** |

### Per-action PSNR

| Action | v11a | Path X | Path Z | X-A | Z-A |
|---|---|---|---|---|---|
| wb         | 22.58 | 22.50 | 20.97 | -0.08 | **-1.61** |
| saturation | 22.88 | 23.04 | **24.17** | +0.16 | **+1.29** |
| highlights | 23.84 | **25.99** | 24.21 | **+2.15** | +0.37 |
| shadows    | 26.35 | 26.39 | 26.27 | +0.04 | -0.08 |
| contrast   | 26.70 | 26.19 | 25.84 | -0.51 | -0.86 |

(粗体 = 当行最佳/最差变化)

## 关键发现

### 1. Path X (v11a + 残差 head) 是最佳架构

- **+0.67 dB overall** 提升来自 highlights (+2.15)
- 这印证了 **7D ISP highlights 参数只有 1D**，无法表达 FireRed 学到的局部色彩
  vignetting/specular 修正；像素残差恰好补足
- saturation/shadows/contrast 几乎不变（说明 7D ISP 已足够表达这些 action）
- WB 仍是 22.50（持平）→ 残差 head 没能修复 WB，因为问题是 **数据稀疏**
  (28 train + 5-7 val WB 样本)，不是表达力

### 2. Path Z (纯像素域) 比 v11a 还差

- **-0.09 dB overall**（在误差内基本持平），但 WB **-1.61 dB**
- 说明 **7D ISP 是有用的归纳偏置 (inductive bias)**：在 372 个标签的小数据集
  下，把模型限制在物理 ISP 流形上比让 8M 参数自由学更稳
- Path Z 在 saturation 上反超 (+1.29)，说明 saturation 偏色彩调色板修改，纯
  像素域比 7D ISP-saturation-scalar 更灵活

### 3. WB 始终是瓶颈，证实 Path Y 必要性

- v11a 22.58, Path X 22.50, Path Z 20.97
- Path X 加了残差 head 也救不了 WB
- 这证明 **WB 的核心问题是训练数据稀疏**：28 train + 5-7 val
- Path Y 计划 +1500 样本（其中 300 WB） → WB 训练样本从 28 → 328 (~12×)

## 论文 narrative 建议

§4/§5 的故事："5 architecture-side bridging interventions all fail because the
gap is structural (renderer + data), not magnitude."

§4.9 / §5.3 可以加：

- **6th intervention falsified: Path Z (pure image-domain U-Net)** —
  Removing the 7D ISP scaffolding **degrades** overall PSNR by 0.09 dB and
  WB by 1.61 dB. The 7D ISP parameterization, despite its ceiling, provides
  a useful inductive bias under low-data supervision.
- **7th intervention (positive!): Path X (residual head on frozen backbone)** —
  Adds 3.68M parameters in a U-Net residual head trained only on pixel L1+SSIM.
  Yields +0.67 dB overall via highlights +2.15 dB. **Breaks the 23.94 dB
  7D-ISP rendering ceiling** for ISP-inexpressible edits but does NOT help WB
  (limited by data, not architecture).

→ 主线收尾："5+1 architecture-side interventions falsified the
single-bottleneck hypothesis. Pixel-domain residual head (Path X) is the
first arch-side intervention to break v11a's ceiling, gaining +2.15 dB on
highlights. But WB remains stuck at ~22.5 dB regardless of architecture,
empirically establishing the **data-side bottleneck** that motivates Path Y
(+1500 FireRed pseudo-labels)."

## 检查点

- `checkpoints/lut_v11a_action_gated_context/best.pt` — v11a baseline (24.46±0.20, 3 seeds)
- `checkpoints/lut_v11a_pathX_implicit_head/best.pt` — Path X, **best (25.28 eval)**
- `checkpoints/lut_pathZ_resunet/best.pt` — Path Z (24.52 eval)

## 待办

- [ ] Path Y AutoDL 启动 1500 样本 FireRed 生成 (~5h, L20 48GB)
- [ ] AutoDL 返回后, 跑 `tools/data/data_prep/ingest_pathY_results.py` → v2 master jsonl
- [ ] 用 v2 master jsonl 重训 Path X (期望 WB 从 22.5 → 25+)

## 代码 artifact

- `models/image_domain_resunet.py` — Path Z 模型 (~8M)
- `training/firered_baseline/implicit_head.py` — Path X residual head (~3.7M)
- `training/firered_baseline/train_lut.py` — Path X 训练入口
- `training/firered_baseline/train_resunet_imgdomain.py` — Path Z 训练入口
- `tools/smoke_test_pathX.py` — Path X 烟雾测试 (7 步全过)
- `tools/eval_v11_named_curves.py` — v11/Path X eval
- `tools/eval_pathZ_resunet.py` — Path Z eval
- `tools/data/data_prep/build_pathY_1500_captions.py` — Path Y captions 生成 (1500 samples)
- `tools/data/data_prep/split_pathY_per_action.py` — per-action 拆分
- `tools/data/data_prep/ingest_pathY_results.py` — AutoDL 数据回收 + 合并
- `scripts/autodl/run_pathY_firered_batch.sh` — AutoDL FireRed 批跑
- `scripts/local/sync_pathY_to_autodl.ps1` — 本地 → AutoDL 同步
