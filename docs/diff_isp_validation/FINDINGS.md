# `diff_isp` vs Real Lightroom \u2014 \u9a8c\u8bc1\u62a5\u544a

**\u65e5\u671f**: 2026-05-03
**\u6837\u672c**: 100 \u5f20 FiveK \u968f\u673a\u62bd\u6837 (seed=42)
**\u56fe\u50cf\u5c3a\u5bf8**: 512\u00d7512
**\u53c2\u6570\u805a\u5408**: mean (\u540c\u4e00\u56fe ~10 \u6761 XMP \u8bb0\u5f55\u6c42\u5747)
**GPU**: RTX 5090 (AutoDL)

---

## 1. \u603b\u4f53\u6307\u6807 \u2014 diff_isp \u4ec5\u5fae\u8f7b\u5b9c\u4e8e "\u4ec0\u4e48\u90fd\u4e0d\u505a"

| Metric | `diff_isp(orig, xmp)` vs Expert | Baseline (orig vs Expert) | Delta |
|---|---:|---:|---:|
| **PSNR** | **19.28 dB** | 18.24 dB | **+1.05 dB** |
| **SSIM** | 0.729 | 0.734 | \u22120.005 |
| **\u0394E (CIEDE2000)** | 10.70 | 11.82 | \u22121.12 |

**\u89e3\u8bfb**:
- PSNR \u7edd\u5bf9\u503c **19 dB** \u5f88\u4f4e (\u4e00\u822c"\u4e0d\u53ef\u611f\u77e5\u5dee\u522b" \u9700 30 dB+, \u6211\u4eec\u8fde 20 dB \u90fd\u8fbe\u4e0d\u5230)
- \u0394E=10.70 \u610f\u5473\u7740 **\u80c9\u773c\u53ef\u89c1\u7684\u989c\u8272\u5dee\u5f02** (\u0394E<2 \u624d\u662f"\u96be\u4ee5\u5206\u8fa8")
- \u6211\u4eec\u76f8\u5bf9 baseline \u53ea\u63d0\u5347 1 dB PSNR \u2014 **\u5f62\u540c\u5c55\u793a\u4ec5\u5e26\u6765\u8f7b\u5fae\u51c0\u6536\u76ca**

## 2. \u5206\u684c\u6548\u7387 \u2014 1/4 \u6837\u672c\u53cd\u800c\u66f4\u5dee

| \u7c7b\u522b | \u5360\u6bd4 | \u6837\u672c\u6570 |
|---|---:|---:|
| \u2705 diff_isp \u6539\u5584 (>+0.5 dB gain) | **62%** | 62 / 100 |
| \ud83d\udd36 \u65e0\u663e\u8457\u5dee\u522b (\u00b10.5 dB) | 15% | 15 / 100 |
| \u274c **diff_isp \u53cd\u800c\u66f4\u5dee** (<\u22120.5 dB) | **23%** | 23 / 100 |

**PSNR gain \u5206\u5e03**: mean +1.05 \u00b1 2.00, median +1.03

## 3. \u6839\u672c\u539f\u56e0 \u2014 \u54ea\u4e9b Op "\u62cd\u8111\u888b"\u5e72\u6b7b\u4e86\uff1f

**Correlation(|\u53c2\u6570\u503c|, PSNR gain)**:
\u8d1f\u6570\u8868\u793a\u53c2\u6570\u8d8a\u5927\u65f6 diff_isp \u8d8a\u5dee

| \u53c2\u6570 | Corr(\|\u503c\|, gain) | \u79d1\u5b66\u4f9d\u636e | \u89e3\u8bfb |
|---|---:|---|---|
| `ev_compensation` | **+0.507** | \u2b50\u2b50\u2b50\u2b50\u2b50 \u7269\u7406\u786e\u786e (2^EV) | \u6b63\u5411\u76f8\u5173 \u2192 op \u6b63\u786e, \u7528\u8d8a\u591a\u53cd\u800c\u6548\u679c\u8d8a\u597d |
| `brightness` | +0.159 | \u2b50\u2b50 gamma \u8fd1\u4f3c | \u5f31\u6b63\u5411 |
| `white_balance` | +0.055 | \u2b50\u2b50\u2b50 Planckian | \u8fd1 0 (\u6b63\u5e38) |
| `saturation` | \u22120.022 | \u2b50\u2b50\u2b50\u2b50 \u6807\u51c6 | \u8fd1 0 (\u6b63\u5e38) |
| `contrast` | \u22120.083 | \u2b50\u2b50 cubic S-curve | \u5f31\u8d1f (\u4e0d\u5339\u914d LR) |
| `vibrance` | \u22120.085 | \u2b50 | \u5f31\u8d1f |
| **`highlights`** | **\u22120.306** \u26a0\ufe0f | \u2b50 sigmoid mask | **\u4e2d\u8d1f \u2192 op \u660e\u663e\u4e0d\u5339\u914d** |
| **`shadows`** | **\u22120.476** \u26a0\ufe0f\u26a0\ufe0f | \u2b50 sigmoid mask | **\u5f3a\u8d1f \u2192 op \u4e25\u91cd\u4e0d\u5339\u914d** |

**\u51b3\u5b9a\u6027\u53d1\u73b0**: `shadows` \u548c `highlights` \u7684\u5904\u7406 **\u5f88\u660e\u663e\u4e0d\u5339\u914d\u771f\u5b9e LR**\u3002
\u8fd9\u4e24\u4e2a op \u7528\u4e86\u5168\u5c40 sigmoid \u8499\u7248 (center=0.15/0.65, width=0.12/0.15) \u2014
\u5e38\u6570\u5168\u662f\u6211\u4eec\u81ea\u5df1\u731c\u7684\u3002**\u771f LR \u7528\u7684\u662f\u5c40\u90e8\u8272\u8c03\u6620\u5c04 (\u7c7b\u4f3c bilateral filter)**, \u6839\u672c\u4e0d\u4e00\u56de\u4e8b\u3002

## 4. \u6700\u4f73/\u6700\u5dee \u6837\u672c\u5bf9\u6bd4

### \u2705 BEST \u6837\u672c (diff_isp \u975e\u5e38\u6709\u6548)

| image | PSNR gain | \u0394E gain | ev | wb | C | B | S | H | sat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a3794-kme_583 | **+5.58** | +5.59 | +0.28 | 5253 | +22 | +34 | **+6** | **+7** | +3 |
| a3412-IMG_4112 | +5.31 | +3.97 | +0.49 | 5940 | +7 | +9 | **+3** | **+10** | +4 |
| a2689-jmac_DSC3218 | +5.21 | +0.56 | +1.04 | 5461 | +9 | +10 | **+1** | **+17** | \u22121 |

**\u6a21\u5f0f**: shadows \u548c highlights \u5747 **\u8f83\u5c0f** (<20)

### \u274c WORST \u6837\u672c (diff_isp \u4e25\u91cd\u7834\u574f)

| image | PSNR gain | \u0394E gain | ev | wb | C | B | S | H | sat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a4802-DSC_0133 | **\u22126.94** | \u22127.88 | +0.18 | 5414 | +22 | +20 | **+48** \u26a0 | +23 | +1 |
| a4577-DSC_0111 | \u22123.59 | \u22122.62 | +0.14 | 4910 | +7 | +17 | +12 | **+18** | +3 |
| a2467-jmacDSC_0043 | \u22122.83 | \u22122.15 | +0.04 | 5591 | +7 | +9 | **+21** | +6 | +5 |
| a3956-IMG_1808 | \u22122.29 | \u22121.61 | \u22120.43 | 5700 | +14 | +10 | +12 | **+32** \u26a0 | 0 |

**\u6a21\u5f0f**: shadows \u6216 highlights \u5927 (>20), diff_isp \u4fbf\u5f00\u59cb\u7834\u574f

## 5. \u5c40\u9650\u5206\u6790

### 5.1 \u672c\u8bd5\u9a8c\u7684\u52a3\u52bf
1. **\u53c2\u6570 mean \u805a\u5408\u5f15\u5165\u566a\u58f0**: \u540c\u4e00\u56fe\u6709 ~10 \u6761 XMP (\u7248\u672c\u5386\u53f2), \u5747\u503c\u4e0d\u662f\u4efb\u4f55\u5355\u4e00\u4e13\u5bb6
2. **Expert filter \u89c4\u5219\u4e0d\u6e05**: `fivek_expert_c/*.jpg` \u662f"Expert C"\u5bcc\u5382, \u4f46\u53c2\u6570 JSON \u65e0 c \u6807\u7b7e, \u53ea\u6709 "default" / UUID
3. **Input gap**: diff_isp \u7528 rawpy \u89e3\u9a6c (camera WB, no auto-bright) \u4f5c\u8d77\u70b9, \u800c LR \u4ece DNG \u5f00\u59cb\u5e26\u81ea\u5df1\u7684\u9ed8\u8ba4\u6e32\u67d3\u51fa\u53d1

### 5.2 \u4f46\u4e0d\u5f71\u54cd\u4e3b\u7ed3\u8bba
\u5373\u4f7f\u6709\u4e0a\u8ff0\u566a\u58f0, **62% / 23% \u7684\u5bf9\u6bd4\u548c shadows=\u22120.48 \u7684\u76f8\u5173\u90fd\u5f88\u5f3a\u5217**, \u8db3\u4ee5\u5f97\u51fa:
- `diff_isp` \u5bf9\u8f7b\u5ea6\u8c03\u6574 (ev+wb \u4e3a\u4e3b) \u6709\u6548
- \u5bf9 shadows/highlights \u7684\u91cd\u5ea6\u56de\u590d\u4e0d\u5339\u914d
- \u603b\u4f53\u6bd4 "\u4ec0\u4e48\u90fd\u4e0d\u505a" \u53ea\u597d 1 dB

---

## 6. \u5bf9\u8bba\u6587/\u4ea7\u54c1\u7684\u542b\u4e49

### 6.1 \u6838\u5fc3\u95ee\u9898
\u6211\u4eec\u7684\u8bad\u7ec3 loop \u662f:
```
expert_xmp (GT) \u2192 predictor \u2192 pred_xmp
                                      \u2193
                              apply_diff_isp(orig, pred_xmp)
                                      \u2193 L1 loss
                              expert_jpeg (real LR render)
```

\u5982\u679c `apply_diff_isp \u2260 real_LR`, \u90a3 **loss \u5373\u4f7f\u4e3a 0 \u4e5f\u4e0d\u4ee3\u8868 pred_xmp = expert_xmp**\u3002

**\u5efa\u6a21 \u5047\u8bbe**: \u76ee\u524d\u7684 Venus \u8bad\u7ec3 loss \u6709 \u2264 1 dB PSNR \u7684 **\u7ed3\u6784\u6027\u4e0a\u9650**\u3002\u7531 PSNR=19 dB \u56fa\u5b9a, L1 pixel loss \u4e5f\u4e0d\u4f1a\u4f4e\u4e8e\u7c7b\u4f3c\u6c34\u5e73\u3002

### 6.2 \u4e09\u6761\u8def\u5f84

#### \u65b9\u6848 A: **\u6821\u51c6 diff_isp** \u5e38\u6570 \ud83d\udee0
\u76ee\u6807: \u5728 FiveK 5000 \u5bf9\u4e0a fit shadows/highlights \u7b49 op \u7684\u5e38\u6570
- **\u5f97\u4f4e\u6302\u8db3\u5b9e**: ev \u4e0d\u52a8 (\u5df2\u51c6), \u4e3b\u8981\u6539 shadows/highlights mask \u7684 center, width, lift
- **\u9884\u671f\u63d0\u5347**: PSNR \u4ece 19 \u2192 24-26 dB (5-7 dB gain) \u662f\u53ef\u80fd\u7684
- **\u5de5\u4f5c\u91cf**: ~0.5 \u5929 (\u4fee `diff_isp.py` \u5e38\u6570 + scipy.optimize fit 10 \u4e2a\u53c2\u6570)
- **\u98ce\u9669**: \u62df\u5408 FiveK \u540e\u53ef\u80fd\u8fc7\u62df\u5408 \u2014 \u5176\u4ed6\u76f8\u673a\u53ef\u80fd\u6210\u8d25

#### \u65b9\u6848 B: **\u66ff\u6362 \u4e3a\u8f7b\u578b NeuralISP** \ud83e\udd16
- \u5220\u6389 `apply_diff_isp`, \u7528 CNN (~300K params): `(orig, xmp_params) \u2192 predicted_render`
- \u7528 FiveK \u5bf9\u76d1\u7763\u5b66\u4e60 LR \u884c\u4e3a
- **\u9884\u671f\u63d0\u5347**: PSNR \u6d88\u706d\u5f0f\u8d8a 30+ dB (\u5b66\u5230\u771f LR \u89c4\u5f8b)
- **\u4ee3\u4ef7**: **\u4e22\u5931\u53ef\u89e3\u91ca\u6027** \u2014 \u4e0d\u518d\u80fd\u8bf4"7 \u53c2\u6570 \u9e1f\u6211\u770b LR \u4e3a"
- **\u5de5\u4f5c\u91cf**: 2-3 \u5929

#### \u65b9\u6848 C: **\u8c03\u6574\u8bba\u6587\u5b9a\u4f4d** \ud83d\udcc4
- \u73b0\u72b6\u4ee5\u6587\u5b57\u6559\u6e05\u695a: \u201c\u6211\u4eec\u7684 diff_isp \u4e3a **differentiable proxy**, \u4e0d\u9700\u5b8c\u7f8e\u5339\u914d LR\u201d
- \u8bc1\u636e: \u53c2\u6570\u9884\u6d4b\u7684\u6253\u5206 (\u7528\u771f\u5b9e LR \u6e32\u67d3\u4e0b\u6e38\u6253\u5206) \u80fd\u8d62
- **\u5de5\u4f5c\u91cf**: 0 \u5929 (\u8bba\u6587\u7a0b\u5e8f\u8c03\u6574)
- **\u4ee3\u4ef7**: \u8d4f\u9762\u4e0d\u8db3, \u7ade\u54c1 (VeraRetouch/JarvisArt \u7528 real LR API) \u4f1a\u538b\u8fc7

## 7. \u63a8\u8350\u65b9\u6848 \u2014 \u65b9\u6848 A \u4f18\u5148 (0.5 \u5929\u9a6c\u50ac)

### \u4e3a\u4ec0\u4e48 A \u6bd4 B \u597d
- \u53ef\u89e3\u91ca\u6027\u662f\u6211\u4eec\u4e3b\u8981\u5356\u70b9 \u2192 \u4fdd\u7559 A
- B \u7684 CNN \u6a21\u5f0f\u8bf4\u53bb\u8de8 NamedCurves/AirNet \u91cd\u53e0 \u2192 \u63d0\u4e0d\u4e0a\u65b0\u7a81\u7834\u53e3
- A \u5141\u8bb8 paper \u5199\u5f97\u786c\u7829: \"We calibrate our diff_isp against 5000 FiveK pairs with scipy.optimize to minimize pixel loss. Before: 19 dB. After: N dB.\"

### \u6267\u884c\u6b65\u9aa4
1. \u5c06 `apply_diff_isp` \u7684\u786c\u7f16\u5e38\u6570 (center, width, lift_factor \u7b49 ~10 \u4e2a) \u63d0\u53d6\u4e3a\u53c2\u6570\u5316
2. \u5199 `fit_diff_isp_calibration.py`: L-BFGS minimize L1(apply_diff_isp(orig, xmp, calibration), expert_jpeg) \u6c47\u6574\u4e2a FiveK
3. \u5b58\u5165 `diff_isp_calibration.json`, \u8bad\u7ec3\u65f6\u52a0\u8f7d
4. \u91cd\u8dd1\u672c\u811a\u672c \u2192 \u770b PSNR \u662f\u5426\u4ece 19 \u2192 >25

\u8fd9\u5361\u8def\u7684\u826f\u5904: \u5b83 \u76f4\u63a5\u56de\u7b54\u6587\u7ae0\u6700\u6838\u5fc3\u95ee\u9898 (\"\u6d82\u6539\u5fd5\u4f3c\u771f LR \u600e\u4e48\u6837?\"), \u800c\u4e14\u6709\u660e\u786e\u6570\u6570\u6307\u6807\u8003\u8bc4\u662f\u5426\u6210\u529f\u3002
