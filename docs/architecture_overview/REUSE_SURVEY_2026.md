# IntelligenceCamera 2026 \u2014 \u9010\u6a21\u5757 SOTA \u91c7\u7528\u8c03\u7814

**\u9879\u76ee\u540d**: IntelligenceCamera (\u667a\u80fd\u76f8\u673a) \u2014 \u72ec\u7acb\u9879\u76ee, \u53d7 PKU Venus (CVPR 2026) \u542f\u53d1
**\u76ee\u7684**: \u907f\u514d\u91cd\u590d\u9020\u8f6e\uff0c\u6309\u6a21\u5757\u9010\u4e2a\u8bc4\u4f30 \u201c\u73b0\u6709\u8f6e\u5b50\u201d \u662f\u5426\u53ef\u7528\u3002
**\u8c03\u7814\u65f6\u95f4**: 2026-05-03 (\u8986\u76d6 2024-2026 \u91cd\u8981\u8bba\u6587)

---

## \ud83c\udfaf \u9879\u76ee\u5b9a\u4f4d\u91cd\u51c6

| | **PKU Venus** (CVPR'26) | **JarvisEvo** (CVPR'26) | **IntelligenceCamera** (\u6211\u4eec) |
|---|---|---|---|
| \u4efb\u52a1 | \u7f8e\u5b66\u5f15\u5bfc + \u88c1\u526a (\u62cd\u524d/\u62cd\u540e\u6784\u56fe) | \u52a8\u6001 LR \u8c03\u53c2 agent (\u62cd\u540e\u4fee\u56fe) | **AG + LR \u8c03\u53c2 + \u7aef\u4fa7** |
| MLLM | 7B-13B \u4e91\u7aef | 7B+ \u4e91\u7aef | **4B \u7aef\u4fa7** (Qwen3-VL-4B) |
| \u5de5\u5177 | \u65e0 LR \u63a5\u5165 | 200+ LR tools | **30+ LR tools \u5b50\u96c6** (\u79fb\u52a8\u4f18\u5316) |
| \u72ec\u521b\u70b9 | AG benchmark | iMCoT + SEPO RL | **\u9996\u4e2a\u7aef\u4fa7\u7f8e\u5b66\u52a0 ISP agent** |

**\u201c\u7ad9\u5728\u5de8\u4eba\u80a9\u8180\u201d**:
- PKU Venus \u63d0\u4f9b: AesGuide \u6570\u636e\u96c6 + Venus-Q \u9884\u8bad\u7ec3\u6743\u91cd (Qwen-VL-Chat \u63a5\u7eed)
- JarvisEvo \u63d0\u4f9b: ArtEdit-170K + iMCoT/SEPO \u8bad\u7ec3\u8303\u5f0f + 200+ \u5de5\u5177 schema

**\u51b3\u7b56\u9047\u5219**:
- \ud83d\udfe2 **\u76f4\u63a5\u91c7\u7528** \u2014 \u4e0a\u6e38\u4ed3\u5e93\u6210\u719f, \u4e3b\u8981\u5199\u63a5\u5165\u4ee3\u7801
- \ud83d\udfe1 **\u5fae\u8c03\u91c7\u7528** \u2014 \u6709 checkpoint, \u9700\u5728\u6211\u4eec\u6570\u636e\u4e0a fine-tune
- \ud83d\udfe0 **\u91cd\u5199 \u4e0d\u53ef\u907f** \u2014 \u4efb\u52a1\u72ec\u7279, \u65e0\u73b0\u6210\u5339\u914d
- \ud83d\udd34 **\u4fdd\u7559\u73b0\u6709** \u2014 \u73b0\u6709\u5df2\u591f\u7528, \u4e0d\u52a8

---

# \ud83d\udce6 2026 \u91cd\u8981\u8bba\u6587\u6e05\u5355

| \u8bba\u6587 | \u4f1a\u8bae | \u4ed3\u5e93 | \u53ef\u91cd\u7528\u4ef7\u503c |
|---|:-:|---|:-:|
| **JarvisEvo** | CVPR 2026 | [LYL1015/JarvisEvo](https://github.com/LYL1015/JarvisEvo) | \u2b50\u2b50\u2b50\u2b50\u2b50 \u8bad\u7ec3\u8303\u5f0f + 170K \u6570\u636e\u96c6 |
| **PKU Venus** | CVPR 2026 | [PKU-ICST-MIPL/Venus_CVPR2026](https://github.com/PKU-ICST-MIPL/Venus_CVPR2026) | \u2b50\u2b50\u2b50\u2b50\u2b50 Venus-Q \u9884\u8bad\u7ec3 + AesGuide |
| **JarvisArt** | NeurIPS 2025 | [LYL1015/JarvisArt](https://github.com/LYL1015/JarvisArt) | \u2b50\u2b50\u2b50\u2b50 MMArt-55K + Agent-to-Lightroom Protocol |
| **RetouchIQ** | arXiv 2602 | (\u672a\u5f00\u6e90?) | \u2b50\u2b50\u2b50 Generalist Reward + PGRT |
| **Agentic Retoucher** | arXiv 2601 | (\u4f8b\u5b50\u4e0d\u540c\u573a\u666f) | \u2b50\u2b50 GenBlemish-27K |
| **MobileIE** | arXiv 2507 (CVPR'25) | github | \u2b50\u2b50\u2b50 1100 FPS \u79fb\u52a8\u7aef IE |
| **Q-Align (one-align)** | ICML 2024 | [Q-Future/Q-Align](https://github.com/Q-Future/Q-Align) | \u2b50\u2b50\u2b50\u2b50 LMM-based aesthetic, pyiqa \u4e00\u884c |
| **NamedCurves** | ECCV 2024 | [davidserra9/namedcurves](https://github.com/davidserra9/namedcurves) | \u2b50\u2b50\u2b50 baseline, \u8272\u540d \u8c03\u8c03\u80a1\u66f2\u7ebf |
| **3D LUT (Zeng)** | TPAMI 2022 | [HuiZeng/Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) | \u2b50\u2b50\u2b50\u2b50 baseline, <600K params |
| **SepLUT** | ECCV 2022 | [ImCharlesY/SepLUT](https://github.com/ImCharlesY/SepLUT) | \u2b50\u2b50 baseline 1D+3D LUT |
| **SigLIP-2** | 2025/02 | google | \u2b50\u2b50\u2b50\u2b50 \u4ee3\u73b0 visual encoder |

---

# \ud83d\udd11 JarvisEvo \u8bad\u7ec3\u8303\u5f0f \u8be6\u89e3 (\u6700\u9ad8\u4ef7\u503c)

JarvisEvo \u63d0\u4f9b\u4e86\u6211\u4eec\u9700\u8981\u7684\u5168\u90e8\u8bad\u7ec3\u903b\u8f91:

## Stage 1: Cold-Start SFT (iMCoT)
- 110K \u7f16\u8f91 + 40K \u8bc4\u4f30 = 150K \u6837\u672c
- \u5b66\u4e60 \u6807\u7b7e\u8bed\u6cd5: `<think>`, `<tool_call>`, `<answer>`
- \u591a\u6b65\u63a8\u7406 + \u5de5\u5177\u8c03\u7528 + \u81ea\u8bc4\u4f30

## Stage 2: SEPO (Synergistic Editor-Evaluator Policy Optimization)
- 10K + 10K = 20K \u6837\u672c
- **\u53cc\u5faa\u73af RL**:
  - \u7f16\u8f91\u5668\u5faa\u73af: \u6a21\u578b\u81ea\u8bc4\u5206 \u2192 \u5fae\u8c03\u7f16\u8f91\u7b56\u7565
  - \u8bc4\u4f30\u5668\u5faa\u73af: \u4eba\u5de5\u6807\u6ce8\u5206\u6570 \u2192 \u5fae\u8c03\u8bc4\u4f30\u7b56\u7565
- \u5956\u52b1: format + tool accuracy + pairwise preference

## Stage 3: Reflection FT
- 5K \u81ea\u53cd\u601d\u6837\u672c (SEPO \u8fd0\u884c\u4e2d\u751f\u6210)
- \u5f3a\u5316\u9519\u8bef\u68c0\u6d4b + \u81ea\u6211\u4fee\u6b63

## ArtEdit-170K \u6570\u636e\u6e90 (\u6211\u4eec\u90fd\u80fd\u62ff)
- **PPR10K** (Liang 2021)
- **MMArt-55K** (JarvisArt 2025) \u2190 \u9700\u4ece\u4ed6\u4eec\u62c9
- **CADB** (Zhang 2021)
- **RPCD** (Nieto 2022)
- **FiveK** (\u6211\u4eec\u6709 5000 \u5bf9)
- **AesMMIT** (Huang 2024)
- **LAION-Aesthetics** (Schuhmann 2022)
- **pico-banana-400K** (Qian 2025)

---

# \u5168\u666f\u51b3\u7b56\u8868

| \u6a21\u5757 | \u73b0\u72b6 | \u63a8\u8350 SOTA | \u51b3\u7b56 | \u5de5\u4f5c\u91cf |
|---|---|---|:-:|:-:|
| **1. \u53cd\u63a8\u4ea7\u751f GT** (\u5408\u6210 \u8bad\u7ec3\u6570\u636e) | inverse_fit + \u624b\u5199 diff_isp | **JarvisArt MMArt-55K** \u6570\u636e\u96c6 + **Agent-to-Lightroom Protocol** | \ud83d\udfe2 \u76f4\u63a5\u7528 | 1-2 \u5929 |
| **2. ISP \u53c2\u6570 schema** | 7 \u53c2\u6570\u624b\u9009 | **JarvisArt \u7684 200+ LR tool schema** | \ud83d\udfe2 \u76f4\u63a5\u7528 | 0.5 \u5929 |
| **3. \u53ef\u5fae\u6e32\u67d3 forward** | diff_isp (PSNR=19) + NeuralISP (PSNR=22) | **\u4e0d\u9700\u8981** \u2014 \u8c03\u771f LR API | \ud83d\udfe0 \u5e9f\u5f03 | 0 \u5929 |
| **4. \u89c6\u89c9\u7f16\u7801\u5668** | MobileViT-S 5.6M (2021) | **SigLIP-2 ViT-Small** (2025/2) \u6216 **DINOv2 ViT-S/14** | \ud83d\udfe1 swap \u4e3b\u5e72 | 2 \u5929 \u91cd\u8bad |
| **5. \u53c2\u6570\u9884\u6d4b\u5934** | Stage A/B/C semantic distill | **JarvisArt \u7684 CoT-SFT + GRPO-R** | \ud83d\udfe1 \u5fae\u8c03\u67b6\u6784 | 5 \u5929 |
| **6. Caption \u751f\u6210** | Qwen3-VL-4B (\u4e0b\u8f7d\u4e2d) | \u5408\u9002 (Qwen3-VL \u5df2\u662f 2025/9 \u6700\u65b0) | \ud83d\udd34 \u4fdd\u7559 | 0 \u5929 |
| **7. \u7f8e\u5b66\u8bc4\u5206** | MUSIQ + AADB (2021) | **Q-Align (one-align)** ICML 2024, \u652f\u6301 IQA/IAA/VQA | \ud83d\udfe2 \u76f4\u63a5\u63d2 | 0.5 \u5929 |
| **8. \u7cbe\u4fee\u7f51 RefineNet** | V4 16M (\u624b\u5199 U-Net) | **MobileIE** (CVPR 2025, 1100 FPS) | \ud83d\udfe1 swap | 1-2 \u5929 |
| **9. \u591a\u4e13\u5bb6\u805a\u5408** | \u624b\u5199 consensus \u6743\u91cd | **MMArt-55K \u5df2\u5305\u542b** \u4e13\u5bb6\u591a\u7248\u672c | \ud83d\udfe2 \u4f9d\u8d56\u6570\u636e | 0 \u5929 |
| **10. baseline \u5bf9\u6bd4** | \u65e0 | **3D LUT** (Zeng) + **NamedCurves** (ECCV 24) + **JarvisArt** | \ud83d\udfe2 \u62c9\u9879\u76ee | 3-5 \u5929 |

---

# \u9010\u6a21\u5757\u8be6\u8c08

## \u6a21\u5757 1: \u5408\u6210\u8bad\u7ec3\u6570\u636e (\u53cd\u63a8 GT)

### \u73b0\u72b6
- \u624b\u5199 inverse_fit \u6cd5: \u624b\u5199 diff_isp + LBFGS \u53cd\u63a8 7 \u53c2\u6570
- \u9a8c\u8bc1 PSNR=19 dB \u2014 **\u53cd\u63a8\u51fa\u6765\u7684\u4e0d\u662f\u771f LR \u53c2\u6570**, \u662f\u201c\u62df\u5408\u624b\u5199 diff_isp\u201d

### SOTA: **JarvisArt MMArt-55K \u6570\u636e\u96c6**
- **55K \u4e13\u5bb6\u6807\u6ce8\u7684 LR \u7f16\u8f91**, \u5168 200+ tool calls \u90fd\u6709\u7cbe\u786e\u8bb0\u5f55
- \u5305\u542b\u539f\u56fe + \u4e13\u5bb6 LR \u8c03\u53c2 + \u9010\u6b65 reasoning chain
- \u7531 NeurIPS 2025 \u8bba\u6587\u516c\u5f00\u91ca\u51fa
- \u53ef\u4ece <https://github.com/LYL1015/JarvisArt> \u83b7\u5f97

### \u51b3\u7b56: \ud83d\udfe2 **\u76f4\u63a5\u91c7\u7528 MMArt-55K**
- \u7701 inverse_fit \u91cd\u8bad\u6570\u5341\u6b21\u7684\u5de5\u4f5c\u91cf
- \u7701\u9a8c\u8bc1 \u201c\u53cd\u63a8\u662f\u5426\u51c6\u201d
- \u6211\u4eec\u8bba\u6587\u53ef\u76f4\u63a5\u8bf4 \u201c\u91c7\u7528 MMArt-55K \u6807\u51c6\u8bad\u7ec3\u96c6\u201d
- **\u7565\u8c03**: 55K \u53ef\u80fd\u4e0d\u591f\u8bad\u8f7b\u91cf\u7aef\u4e0a\u6a21\u578b, \u53ef\u8865\u5145\u6211\u4eec\u539f\u6709 instruction_data.json (65M)

### \u4ee3\u4ef7
- 0 (\u63a5\u53d3\u6210\u719f\u6570\u636e\u96c6)

---

## \u6a21\u5757 2: ISP \u53c2\u6570 schema (\u8bbe\u8ba1\u7a7a\u95f4)

### \u73b0\u72b6
- \u624b\u9009 7 \u4e2a\u53c2\u6570: white_balance, contrast, brightness, shadows, highlights, saturation, clarity
- \u539f\u521b 9 \u53c2\u6570\u540e\u53d1\u73b0 ev/vibrance \u4e0e\u5176\u4ed6\u51b2\u7a81 \u2192 \u7cbe\u7b80

### SOTA: **JarvisArt \u4ee3 200+ LR tools**
- \u8bbe\u8ba1: \u5168\u9762\u8986\u76d6 LR \u5168\u90e8\u8c03\u53c2 (Light, Color, Detail, Tone Curve, HSL, Calibration, Local Adjust)
- **Agent-to-Lightroom Protocol** \u662f\u5b8c\u6574\u7684 JSON Schema
- \u5305\u542b\u53c2\u6570\u8303\u56f4 + \u8c03\u7528\u987a\u5e8f + \u4f9d\u8d56\u5173\u7cfb

### \u51b3\u7b56: \ud83d\udfe2 **\u91c7\u7528 JarvisArt \u7684 schema**
- \u4e2d\u4ec0\u4e48\u53d6\u4ec0\u4e48: \u6253\u4f4d Light + Color + Tone Curve + HSL (\u7ea6 30 \u4e2a tools)
- \u4f7f\u7528\u4ed6\u4eec\u7684\u53c2\u6570\u8303\u56f4 (\u4ee5 LR \u540c\u6b65)
- **\u7559\u7740\u5c0f\u5b50\u96c6\u5b9a\u4f4d**: \u201c\u6211\u4eec\u53ea\u770b 30 \u4e2a\u6700\u5e38\u7528\u53c2\u6570 (vs JarvisArt 200+)\uff0c\u4e3a\u79fb\u52a8\u7aef\u52a0\u901f\u201d

### \u4ee3\u4ef7
- 0.5 \u5929 \u8bfb\u4ed6\u4eec\u7684 protocol.md, \u5c06\u6211\u4eec\u539f 7 \u53c2\u6620\u5c04\u4e0a

---

## \u6a21\u5757 3: \u53ef\u5fae\u6e32\u67d3 forward (\u4e0b\u6e38\u8bad\u7ec3\u7528)

### \u73b0\u72b6
- diff_isp \u624b\u5199 (PSNR=19, \u5df2\u5224\u201c\u4e0d\u592a\u80fd\u7528\u201d)
- NeuralISP FiLM U-Net (PSNR=22, \u8fd8\u53ef\u4ee5)

### SOTA: **\u4e0d\u9700\u8981\u53ef\u5fae forward**
- JarvisArt \u8def\u7ebf\u4e0d\u8d70 reverse-engineered ISP, \u76f4\u63a5\u8c03 LR API (\u201c\u9ed1\u76d2\u201d)
- Agent-to-Lightroom Protocol \u8c03 lightroom.set_param() \u2192 \u771f LR \u6e32\u67d3
- \u8bad\u7ec3\u65f6\u7528\u771f LR \u6e32\u67d3\u7684 GT \u56fe + RL \u56de\u62a5

### \u51b3\u7b56: \ud83d\udfe0 **\u53ef\u80fd\u5e9f\u5f03 diff_isp \u6574\u4e2a\u6982\u5ff5**
- **\u6362\u601d\u8def**: \u9700\u8981 \u5728 train \u65f6 score, \u4e0d\u9700\u8981 differentiable
- \u8bad\u7ec3 \u53d8 supervised: \u6709 GT \u53c2\u6570 \u2192 L1 loss \u5728\u6570\u503c\u4e0a (\u4e0d\u9700\u6e32\u67d3)
- \u9700\u9a8c\u8bc1\u65f6: \u8c03\u771f LR (\u6216 LR\u514b\u9686 darktable) \u770b\u6548\u679c
- **\u4ec5\u672a\u96be\u4ee5\u8c03 LR API \u573a\u666f\u4e0b\u4fdd\u7559 NeuralISP \u4f5c\u4e3a fallback**

### \u4ee3\u4ef7
- \u91ca\u653e: -1 \u5468 (\u4e0d\u518d\u5728 \u624b\u5199 isp \u4e0a\u6298\u817e)
- \u96c6\u6210 LR API: 1-2 \u5929 (\u4e0b\u53e5)

---

## \u6a21\u5757 4: \u89c6\u89c9\u7f16\u7801\u5668

### \u73b0\u72b6
- MobileViT-S (2021), 5.6M params
- \u8bba\u6587\u539f\u521b\u70b9: \u201c\u8f7b\u91cf\u201d\u3001\u201c\u79fb\u52a8\u7aef\u201d

### SOTA \u9009\u9879

| \u9009\u9879 | \u53d1\u8868 | \u53c2\u6570\u91cf | \u4f18\u70b9 | \u7f3a\u70b9 |
|---|---|---|---|---|
| **SigLIP-2 ViT-S/16** | 2025/2 | ~22M | \u591a\u8bed\u8a00 + \u80dc\u8fc7 SigLIP \u6240\u6709\u7248\u672c | \u91cf\u4e0d\u8db3\u8f7b |
| **DINOv2 ViT-S/14** | 2023 | 21M | \u65e0\u76d1\u7763\u9884\u8bad\u7ec3\u3001\u5e76 detector \u7279\u5f81 | \u4ec5\u89c6\u89c9, \u65e0 text |
| **C-RADIO v2.5** | CVPR 2025 | \u53ef\u8c03 | \u4e00\u4e2a\u6a21\u578b\u4ee3\u66ff CLIP+DINO+SAM+SigLIP | \u8f83\u65b0, \u5b9e\u9a8c\u6027 |
| **EVA-02 Tiny** | 2023 | 5.6M | \u4e0e MobileViT \u540c\u91cf\u7ea7\u4f46\u8868\u73b0\u4f18 | \u8f7b\u91cf\u7248\u9009\u62e9\u5c11 |
| MobileViT-S (\u73b0\u6709) | 2021 | 5.6M | \u79fb\u52a8\u7aef\u4f18 | \u592a\u8001 |

### \u51b3\u7b56: \ud83d\udfe1 **\u6362 SigLIP-2 ViT-S/16 \u4f5c\u4e3a\u4e3b backbone**
- \u53ef\u80fd 22M \u6709\u70b9\u5927, \u4f46\u4e3a CVPR \u8bba\u6587\u8003\u8651"\u73b0\u4ee3\u7f16\u7801\u5668"\u662f\u5fc5\u9700
- \u4fdd\u7559 MobileViT-S \u4e3a Mobile fallback (\u53ef\u8bf4 \u201c\u6211\u4eec\u4e24\u4e2a\u7248\u672c\u201d)
- \u672a\u6765 distill SigLIP-2 \u2192 MobileViT-S, \u7167\u987e\u8f7b\u91cf

### \u4ee3\u4ef7
- 2 \u5929 swap + \u91cd\u8bad Stage A/B
- \u53ef\u80fd\u9700 +1 GPU-day

---

## \u6a21\u5757 5: \u53c2\u6570\u9884\u6d4b\u5934 (\u6838\u5fc3 \u5168\u6587)

### \u73b0\u72b6
- Stage A: visual \u2192 semantic
- Stage B: visual \u2192 ISP \u53c2\u6570
- Stage C: visual + text \u2192 ISP \u53c2\u6570 (FiLM \u878d\u5408)

### SOTA: **JarvisArt CoT-SFT + GRPO-R**
- **\u4e24\u9636\u6bb5\u8bad\u7ec3**:
  1. CoT-SFT (Chain-of-Thought Supervised Fine-Tune): \u5b66 \u201c\u9605\u8bfb\u56fe \u2192 \u63a8\u7406 \u2192 \u8c03\u53c2\u201d
  2. GRPO-R (Group Relative Policy Optimization for Retouching): RL \u8d4f\u52b1\u4f30\u4f30 better\u2026

### \u51b3\u7b56: \ud83d\udfe1 **\u91c7\u7528 GRPO-R \u8bad\u7ec3\u8303\u5f0f, \u4f46\u540e\u7aef\u8f6c\u4e3a\u8f7b\u91cf**
- \u6539\u9020 Stage C: \u4ece "FiLM \u878d\u5408 \u2192 \u53c2\u6570" \u6539\u4e3a "VLM CoT \u63a8\u7406 \u2192 tool call sequence"
- VLM \u7528\u4e0e \u7f16\u7801\u5668\u67b6\u63a5 (Qwen3-VL-4B)
- LR API \u63a5\u8c03\u5e9f
- \u91c7\u7528 GRPO-R \u8bad\u7ec3\u4ee5\u5728 MMArt-55K \u4e0a fine-tune (\u8c03 reward = LR \u6e32\u67d3 \u6253\u5206)

### \u4ee3\u4ef7
- 5 \u5929 (\u6700\u5927\u91cd\u5199, \u4f46\u53ef\u62fc PEFT \u8282\u7701\u8bad\u7ec3)

---

## \u6a21\u5757 6: Caption \u751f\u6210

### \u73b0\u72b6
- \u8ba1\u5212\u7528 Qwen3-VL-4B-Instruct (2025/9 \u53d1)

### \u51b3\u7b56: \ud83d\udd34 **\u4fdd\u7559 Qwen3-VL-4B**
- \u5df2\u662f 2025 \u6700\u65b0
- \u5408\u9002 4B \u5927\u5c0f, \u80fd\u8dd1\u672c\u5730 / \u5c0f GPU
- (\u6216 \u5347\u7ea7 Qwen3-VL-7B \u5982 \u8d44\u6e90\u5141\u8bb8)

### \u4ee3\u4ef7
- 0 (\u4e0d\u52a8)

---

## \u6a21\u5757 7: \u7f8e\u5b66\u8bc4\u5206

### \u73b0\u72b6
- MUSIQ-AVA + AADB CNN (2021)
- val_MUSIQ=4.20 \u662f\u73b0\u9636\u6bb5\u4e3b\u8981\u4f18\u5316\u76ee\u6807

### SOTA: **Q-Align / one-align**
- ICML 2024
- \u9886 IQA/IAA/VQA SOTA
- \u57fa\u4e8e LMM (Mantis-Chat \u6216 mPLUG-Owl2)
- **pyiqa \u53ef\u4ee5\u76f4\u63a5\u8c03**: `pyiqa.create_metric('qalign')`

### \u51b3\u7b56: \ud83d\udfe2 **\u63d2 Q-Align \u4f5c\u4e3a Aesthetic loss**
- \u4ee5 \u4ee3\u66ff/\u8865\u5145 MUSIQ
- pyiqa.create_metric('qalign') \u4e00\u884c\u5c31\u63a5\u4e0a
- \u8bba\u6587\u4e0a\u6253 \u201c\u91c7\u7528\u73b0\u4ee3 LMM-based aesthetic\u201d

### \u4ee3\u4ef7
- 0.5 \u5929 \u63a5\u5165 + \u9a8c\u8bc1
- \u63a8\u7406\u8d77\u8d70 8GB+ \u663e\u5b58 (\u9700\u9a8c\u662f\u5426\u80fd\u8dd1)

---

## \u6a21\u5757 8: \u7cbe\u4fee\u7f51 RefineNet

### \u73b0\u72b6
- RefineNet V4 (16M params)
- \u624b\u5199\u53cc\u5206\u652f U-Net
- val_MUSIQ=4.20 \u672c\u8868\u540e\u8d77\u4e0a\u9650

### SOTA: **MobileIE** (arXiv 2507.01838, CVPR 2025)
- **1100 FPS** \u79fb\u52a8\u7aef\u3001\u6781\u8f7b\u91cf
- 3 \u4efb\u52a1 SOTA: noise / blur / low-light
- pre-trained checkpoint \u53ef\u7528

### \u51b3\u7b56: \ud83d\udfe1 **\u8bd5\u63a5 MobileIE \u4ee3\u66ff RefineNet**
- \u9999 1100 FPS \u662f\u8bba\u6587\u4eae\u70b9
- \u9700\u9a8c\u8bc1: PSNR / MUSIQ \u662f\u5426\u4e0d\u9000\u6b65
- \u4fdd\u7559 RefineNet V4 \u4f5c\u4e3a baseline \u9879

### \u4ee3\u4ef7
- 1-2 \u5929 (clone + \u63a5\u5165 + \u8bad\u7ec3)

---

## \u6a21\u5757 9: \u591a\u4e13\u5bb6\u805a\u5408 (consensus)

### \u73b0\u72b6
- fivek_expert_consensus.json \u624b\u5199 weight \u8bf4\u660e

### SOTA: **MMArt-55K \u6570\u636e\u96c6\u5df2\u5305\u542b**
- \u591a\u4e13\u5bb6\u591a\u7248\u672c
- \u5df2\u591a\u8c03\u8d77 reasoning chain

### \u51b3\u7b56: \ud83d\udfe2 **\u5982\u91c7\u7528 MMArt-55K, consensus \u903b\u8f91\u4e0d\u52a8**
- \u4ed6\u4eec\u5df2\u5904\u7406\u4e86

### \u4ee3\u4ef7
- 0

---

## \u6a21\u5757 10: baseline \u5bf9\u6bd4 (\u8bba\u6587\u5fc5\u9700)

### \u73b0\u72b6
- \u65e0 baseline \u5bf9\u6bd4 (CVPR \u4f1a\u88ab\u62d2)

### \u5fc5\u52a0 baselines:

| Baseline | \u53d1\u8868 | github | \u96be\u5ea6 |
|---|---|---|---|
| **Image-Adaptive 3D LUT** | TPAMI 2022 | HuiZeng/Image-Adaptive-3DLUT | \u4f4e |
| **NamedCurves** | ECCV 2024 | davidserra9/namedcurves | \u4f4e |
| **SepLUT** | ECCV 2022 | ImCharlesY/SepLUT | \u4e2d |
| **JarvisArt** | NeurIPS 2025 | LYL1015/JarvisArt | \u4e2d (\u9700 LR API) |
| **MonetGPT** | SIGGRAPH 2025 | niladridutt/monetGPT | \u4e2d |
| **PhotoArtAgent** | arXiv 2505 | (\u672a\u5f00\u6e90?) | \u9ad8 |

### \u51b3\u7b56: \ud83d\udfe2 **\u62c9 3 \u4e2a baseline**
- 3D LUT (\u8f6f\u6307\u6807) + NamedCurves (\u73b0\u4ee3\u8868\u73b0) + JarvisArt (\u96e8\u4f24\u5fc5\u624b)

### \u4ee3\u4ef7
- 3-5 \u5929 (\u62c9 repo + \u8c03 \u8dd1)

---

# \u603b\u8d26\u5355

## \u91cd\u8d77\u52b3\u52a8 \u4f30\u8ba1

| \u4efb\u52a1 | \u5929\u6570 | \u4ee3\u4ef7 |
|---|---:|---|
| \u62c9 MMArt-55K + Agent-to-Lightroom Protocol | 1-2 | \u8bfb\u6587\u6863 + \u7406\u89e3 |
| \u63a5\u5165 LR API (\u901a\u8fc7 darktable / Lightroom CLI) | 2-3 | \u6700\u96be\u9879 |
| swap MobileViT \u2192 SigLIP-2 \u91cd\u8bad Stage A/B | 2 | GPU-day |
| \u63d2 Q-Align \u4ee3\u66ff MUSIQ | 0.5 | trivial |
| swap RefineNet \u2192 MobileIE | 1-2 | \u9700\u9a8c\u8bc1 |
| GRPO-R \u8bad\u7ec3 \u4e8e MMArt-55K | 5-7 | \u6700\u590d\u6742, \u4f46\u8bba\u6587\u4eae\u70b9 |
| \u62c9 3 \u4e2a baseline | 3-5 | \u5fc5\u9700 |
| \u201c\u8bba\u6587\u201d | \u5e76\u884c | \u5199 |
| \u603b\u8ba1 | **15-20 \u5929 \u5b9e\u9645\u5de5\u4f5c** | (CVPR 2026 \u622a\u6b62 11/15) \u4e22\u4ec5\u8db3\u591f \u96be |

## \u67e0\u8d77\u573a\u666f

\u539f\u8bba\u6587\u201c\u8f7b\u91cf 7 \u53c2\u6570 + \u5408\u6210\u8bad\u7ec3\u201d \u6539\u4e3a:

> **\u201cVenus 2026: \u4e00\u4e2a\u8f7b\u91cf VLM-Agent \u5728\u79fb\u52a8\u7aef\u8c03 LR API\u201d**
>
> - **\u8f6f\u4ef6\u5305**: 4B VLM (\u8c03\u53c2 \u63a8\u7406) + LR \u63a5\u5165 (\u53d1\u51fa\u8c03\u7528) + 30+ tool subset
> - **\u521b\u65b0\u70b9**: \
>   1. **\u9996\u4e2a\u8f7b\u91cf VLM Agent**: \u5176\u4ed6 (JarvisArt 7B+, MonetGPT 7B+) \u90fd\u662f cloud, \u6211\u4eec\u662f \u79fb\u52a8\u7aef
>   2. **30+ tool subset**: \u9876\u7ea7 LR \u6709 200+, \u6211\u4eec\u9009\u4e86 \u201c\u6700\u5e38\u7528 30+\u201d \u5e76\u4e3a\u79fb\u52a8\u7aef\u4f18\u5316
>   3. **\u79bb\u7ebf RL \u5fae\u8c03**: \u6240\u6709\u7ade\u54c1\u8bf4 cloud, \u6211\u4eec\u63d0\u4f9b on-device GRPO-R \u5fae\u8c03
> - **\u4e2a\u70b9**: \u6210\u719f\u5305\u88c5+\u5fae\u8c03 (\u4e3b\u8981\u9020\u8f6e\u4ef6: \u63a5\u5165\u5c42 + \u79fb\u52a8\u7aef\u8f6c\u8bd1)

---

# \u63a8\u8350\u4e0b\u4e00\u6b65

\u4f18\u5148\u987a\u5e8f (\u9020\u8f6e\u62b5\u62a5):

1. \u5148 **\u62c9 MMArt-55K \u6570\u636e\u96c6** \u770b\u6837\u672c (1 \u5929) \u2014 \u770b\u80fd\u4e0d\u80fd\u7528
2. **\u63a5\u5165 LR API** (\u6700\u96be, 2-3 \u5929) \u2014 \u8def\u7ebf\u7684\u6307\u540d\u5fc5\u9700\u54c1
3. **\u63d2 Q-Align** \u4ee3 MUSIQ (0.5 \u5929) \u2014 \u624b\u5230\u62c9\u6210
4. **swap MobileViT \u2192 SigLIP-2** (2 \u5929) \u2014 \u624b\u5165\u73b0\u4ee3\u8868\u73b0
5. **\u62c9 baselines: 3D LUT + NamedCurves** (3-5 \u5929) \u2014 \u8bba\u6587\u5fc5\u9700
6. **GRPO-R \u8bad\u7ec3 (\u53ef\u9009)** (5-7 \u5929) \u2014 \u540c\u4e0a

\u8fdb\u4ec5\u671f\u9650:
- \u8bba\u6587 \u67aa\u53e3 (\u5408\u4eea\u62a5 + GPU\u91cf): \u6709 6+ \u5468 (\u6e7e\u53d8 11/15) \u662f\u53ef\u4ee5\u5b8c\u6210
- 6 \u5468 = 30 \u5929, \u6211\u4eec\u8bf4 15-20 \u5929\u5b9e\u9645\u5de5\u4f5c, \u53e6 10 \u5929 \u662f \u8bd5\u9519 \u8c03\u53c2 \u8c03 \u4f8b\u5b50

---

\u672c\u8c03\u7814\u53ef\u4ee5\u4e3a Venus 2026 \u8bba\u6587\u63d0\u4f9b\u660e\u786e\u7684 \u201c\u91cd\u7528+\u5fae\u8c03\u201d \u67b6\u6784\uff0c\u5e76\u907f\u514d\u91cd\u590d\u6280\u672f\u9020\u8f6e\u3002
