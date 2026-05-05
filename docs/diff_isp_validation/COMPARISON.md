# diff_isp vs NeuralISP — 验证对比

## 1. 总体指标 (100 样本, FiveK, 512px)

| Metric | diff_isp | NeuralISP | Baseline (orig=GT) |
|---|---:|---:|---:|
| PSNR (dB) | 19.280 | **22.440** | 18.235 |
| SSIM | 0.729 | **0.813** | 0.734 |
| ΔE | 10.704 | **7.571** | 11.821 |

## 2. 胜负分桌

- NeuralISP 胜出 diff_isp (>+0.5 dB PSNR): **80/100 (80%)**
- NeuralISP 败于 diff_isp (<-0.5 dB): 12/100 (12%)
- 平均 PSNR 提升: **+3.16 dB**

## 3. 参数相关修复检验

表示 corr(|参数值|, PSNR). 负值 = 参数越大时该 ISP 表现越差。
“修复” = diff_isp 负相关 → NeuralISP 近 0。

| 参数 | diff_isp corr | NeuralISP corr | 状态 |
|---|---:|---:|:-:|
| `ev_compensation` | -0.017 | +0.089 | 〇 中性 |
| `white_balance` | -0.045 | -0.088 | 〇 中性 |
| `contrast` | -0.290 | -0.305 | 〇 中性 |
| `brightness` | -0.179 | -0.111 | 〇 中性 |
| `shadows` | -0.073 | -0.194 | ⚠️ |
| `highlights` | +0.245 | +0.116 | ✅ 保持 |
| `saturation` | -0.008 | -0.000 | 〇 中性 |

## 4. 结论

- **NeuralISP (1.6M FiLM U-Net) 明显优于 手写 diff_isp**: PSNR 提升 +3.16 dB, ΔE 下降 29%, SSIM 提升 +8.4%
- 但 NeuralISP 仍然 < 30 dB (SOTA 在 25 dB 左右, e.g. 3DLUT)
- 需要决定: 是否 drop-in 补代 diff_isp 在训练 pipeline