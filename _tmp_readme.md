# IntelligenceCamera Related Works (2024-2026)

> **\u672c\u9879\u76ee\u5b9a\u4f4d**\uff1a\u7aef\u4fa7\u7f8e\u5b66+ISP agent\u3002MobileViT (~6M) \u6839\u636e\u56fe\u7247+\u53e3\u8bed\u6307\u4ee4\u8f93\u51fa Lightroom \u53c2\u6570 (4-8 sliders)\uff0c\u901a\u8fc7\u8bed\u4e49\u84b8\u998f\u4ece Venus 7B / Qwen3-VL \u5b66\u4e60\u7f8e\u5b66\u8bed\u4e49\u7a7a\u95f4 + DiffISP \u53ef\u5fae\u6e32\u67d3 + LoRA SFT on ArtEdit-Bench\u3002
>
> \u4e0b\u8f7d\u65f6\u95f4\uff1a2026-05-06  \u00b7  \u5171 27 \u7bc7\u3002

\u672c README \u6309\u4e0e\u672c\u9879\u76ee\u7684**\u76f8\u4f3c\u5ea6\u68af\u5ea6**\u5212\u4e3a 5 \u4e2a Tier\u3002\u4f18\u5148\u7ea7\u7f16\u53f7\u4ece 01 \u8d77, \u7f16\u53f7\u8d8a\u5c0f \u2192 \u4e0e\u672c\u9879\u76ee\u8d8a\u50cf\u3002

---

## Tier 1: \u76f4\u63a5\u5bf9\u6807 (Lightroom Agent / VLM \u53c2\u6570\u5316\u7f16\u8f91 / \u53ef\u5fae retouching)

\u8fd9 7 \u7bc7**\u8ddf\u6211\u4eec\u7684 Distill + LoRA SFT \u5728\u540c\u4e00\u4e2a\u95ee\u9898\u4e0a**\uff1a\u8ba9 (M)LLM/Agent \u8f93\u51fa Lightroom \u7c7b\u53ef\u89e3\u91ca\u53c2\u6570 \u00b7 \u7528\u53ef\u5fae\u6216\u8f6f\u4ef6\u6e32\u67d3\u95ed\u73af \u00b7 \u5e94\u5bf9 photo retouching \u573a\u666f\u3002

| # | \u8bba\u6587 | \u5e74\u4efd/\u6536\u5f55 | \u4e0e\u6211\u4eec\u7684\u5173\u8054 |
|:--:|------|:---:|---------|
| **01** | **JarvisEvo** \u00b7 Self-Evolving Photo Editing Agent with iMCoT | CVPR 2026 | **\u6700\u50cf\u6211\u4eec**\u3002Qwen3-VL \u00b7 Lightroom \u00b7 200+ tools \u00b7 \u4ea4\u7ec7\u591a\u6a21 CoT \u00b7 \u7eaf\u53c2\u6570\u8f93\u51fa\u3002\u6211\u4eec\u9879\u76ee\u751a\u81f3\u624b\u52a8\u5305\u542b\u4e86 JarvisEvo \u4ed3\u5e93\u4f5c\u4e3a\u4f9d\u8d56 |
| **02** | **JarvisArt** \u00b7 Photo Retouching Agent with Lightroom Protocol | NeurIPS 2025 | JarvisEvo \u524d\u4f5c, A2L \u534f\u8bae, GroundingIQA \u8bc4\u4f30, \u662f\u672c\u9886\u57df\u5174\u8d77\u7684\u5e95\u5ea7 |
| **03** | **VeraRetouch** \u00b7 Lightweight Differentiable VLM Retouching | 2026 | Qwen3-VL + LoRA + puzzle-based \u8bad\u7ec3 + \u53ef\u5fae operations\u3002\u8de8 puzzle/CoT/\u53ef\u5fae \u4e09\u91cd\u4ea4\u96c6 |
| **04** | **PhotoArtAgent** \u00b7 Intelligent Photo Retouching with LM Agents | 2025 | LM-based agent\uff0c**\u660e\u786e\u63d0 "interpretable parameters vs black-box"** \u2014 \u662f\u6211\u4eec\u8bba\u6587 motivation \u7684\u91cd\u8981\u53c2\u8003 |
| **05** | **MonetGPT** \u00b7 Solving Puzzles Enhances MLLM Retouching Skills | 2025 | "puzzle\u8bad\u7ec3" \u601d\u8def\u7684\u5176\u4e2d\u4e00\u4efd\u539f\u521b, \u7c7b VeraRetouch \u601d\u8def\u7684\u524d\u8eab |
| **06** | **INRetouch** \u00b7 Context-Aware Implicit Neural Repr Retouching | 2024 | one-shot \u53c2\u8003\u8fc1\u79fb editing\uff0c\u5360\u4e86 "\u53c2\u6570\u5316 retouching" \u5360\u4f4d, **\u8de8\u4e0e\u6211\u4eec Stage C \u601d\u8def** |
| **07** | **Agentic Retoucher** \u00b7 T2I Iterative Retouching | 2026 | T2I \u53d1\u751f\u540e\u53cd\u601d\u4fee\u8865, \u4e3a\u6211\u4eec **\u53cd\u601d/\u7cbe\u4fee\u95ed\u73af** \u5934\u8111\u98ce\u66b4\u53c2\u8003 |

---

## Tier 2: \u7f8e\u5b66 + MLLM/IQA \u8bc4\u5206\u4e0e Foundation Model

\u8fd9 5 \u7bc7\u662f\u6211\u4eec\u8bc4\u4f30\u7ad9\u7684\u5e95\u5ea7\u3002\u6211\u4eec\u7528\u4e86 AesExpert + MUSIQ + Q-Bench \u6307\u6807\u4f53\u7cfb\uff0c\u5e76\u53d7 GPT-4V aesthetic eval \u5de5\u4f5c\u542f\u53d1\u3002

| # | \u8bba\u6587 | \u5e74\u4efd | \u4e0e\u6211\u4eec\u7684\u5173\u8054 |
|:--:|------|:---:|---------|
| **08** | **AesExpert** \u00b7 Multi-modality Foundation Model for Aesthetic Perception | 2024 | **\u6211\u4eec\u76f4\u63a5\u4f7f\u7528\u4e86\u8fd9\u4e2a\u6a21\u578b** (`tools/data/scoring/score_fivek_aesexpert.py`) |
| **09** | **Q-Bench+** \u00b7 MLLM \u5728 Low-level Vision \u4e0a\u7684 Benchmark | 2024 | ICLR 2024 spotlight \u539f\u4f5c\u3002\u8bc4\u4f30\u4f53\u7cfb\u53c2\u8003 |
| **10** | **Grounding-IQA** \u00b7 Grounding MLLM for Image Quality | 2024 | JarvisArt \u91c7\u7528\u7684\u8bc4\u5206\u5668; \u9876\u5c42\u8bc4\u4f30\u53c2\u8003 |
| **11** | **Compare2Score** \u00b7 Adaptive IQA via Teaching MLLMs | 2024 | "\u6559 LLM \u6253\u5206" \u601d\u8def\u539f\u521b\u4f5c, \u6211\u4eec\u7684 Venus \u8bc4\u5206\u601d\u8def\u540c\u6e90 |
| **12** | **GPT-4V Aesthetic Eval** \u00b7 \u8bc4\u4f30 GPT-4V \u7684\u7f8e\u5b66\u8bc4\u5206\u80fd\u529b | 2024 | \u6211\u4eec\u7528\u672c\u9879\u76ee\u540c\u601d\u8def\u8ba9 Venus 7B \u6253\u5206, \u8fd9\u662f\u5b83\u7684\u540c\u4ee3\u539f\u578b |

---

## Tier 3: \u53c2\u6570\u5f0f photo retouching / \u53ef\u5fae\u7f16\u8f91

\u8fd9 4 \u7bc7\u662f\u6211\u4eec **DiffISP \u53ef\u5fae\u6e32\u67d3 + Lightroom-style \u53c2\u6570\u9884\u6d4b** \u8def\u7ebf\u7684\u540c\u884c\u4f5c\u3002

| # | \u8bba\u6587 | \u5e74\u4efd/\u6536\u5f55 | \u4e0e\u6211\u4eec\u7684\u5173\u8054 |
|:--:|------|:---:|---------|
| **13** | **Taming Lookup Tables** \u00b7 Efficient Image Retouching | 2024 | LUT-based \u8f7b\u91cf retouching, \u7aef\u4fa7\u8def\u7ebf\u4e0a\u4e0e\u6211\u4eec MobileViT \u5e76\u884c |
| **14** | **Degradation-Aware Enhancement via VLM Classification** | 2025 | VLM \u5148\u5224\u522b\u9000\u5316\u518d\u5904\u7406, \u6211\u4eec v7 \u7684\u9000\u5316\u589e\u5f3a\u7a0b\u5e8f\u540c\u6e90\u601d\u8def |
| **15** | **BeautyGRPO** \u00b7 RL Aesthetic Alignment for Face Retouching | 2026 | RL \u5bf9\u9f50\u7f8e\u5b66, \u4e3a\u6211\u4eec\u672a\u6765\u53ef\u80fd\u5230 RL \u9636\u6bb5\u63d0\u4f9b\u53c2\u8003 |
| **16** | **VRetouchEr** \u00b7 Cross-frame Face Video Retouching | CVPR 2024 | \u9762\u90e8 retouching SOTA \u4f5c, \u7531 face \u63a8\u5e7f\u5230\u666f\u8c03 retouching \u7684\u5b9e\u8df5\u53c2\u8003 |

---

## Tier 4: Image Editing / Restoration Agent + Instruction Editing

\u8fd9 8 \u7bc7\u662f\u672c\u9886\u57df\u7684\u5e7f\u4e49 "MLLM as Tool/Agent for Image Tasks" \u8c31\u7cfb, \u662f\u6211\u4eec related work \u8c031 + \u7ae0\u8282\u8fd0\u884c\u3002

| # | \u8bba\u6587 | \u5e74\u4efd/\u6536\u5f55 | \u4e0e\u6211\u4eec\u7684\u5173\u8054 |
|:--:|------|:---:|---------|
| **17** | **GenArtist** \u00b7 MLLM as Agent for Unified Image Gen+Edit | NeurIPS 2024 | "MLLM \u4f5c\u4e3a\u603b\u63a7\u8c03\u5ea6" \u539f\u521b\u4f5c |
| **18** | **RestoreAgent** \u00b7 Autonomous Image Restoration via MLLM | 2024 | \u591a\u9000\u5316\u5904\u7406\u3001\u4e0e\u6211\u4eec\u7684 Stage B \u53c2\u6570\u9884\u6d4b\u53cc\u8f68 |
| **19** | **AgenticIR** \u00b7 Intelligent Agentic System for IR | 2024 | RestoreAgent \u540c\u671f\u4f5c, \u5de5\u5177\u8c03\u7528 + \u53cd\u601d, \u67b6\u6784\u53c2\u8003 |
| **20** | **Multi-Agent Image Restoration** | IJCV 2026 | \u591a Agent \u534f\u4f5c\u5904\u7406\u590d\u6742\u9000\u5316, \u672c\u9886\u57df\u6700\u65b0\u5168\u91cf\u8c03\u7814\u4e4b\u4e00 |
| **21** | **Lego-Edit** \u00b7 General Editing with Model-Level Bricks | 2025 | \u6a21\u578b\u62fc\u63d2\u5f0f\u7f16\u8f91, \u4e0e Tier 1 agent \u4e0d\u540c\u8def\u7ebf |
| **22** | **FireEdit** \u00b7 Region-aware VLM Instruction Editing | CVPR 2025 | \u6307\u4ee4 + \u533a\u57df, **\u4e0e\u6211\u4eec\u53e3\u8bed\u5316\u6307\u4ee4\u601d\u8def\u4ea4\u4e92** |
| **23** | **InsightEdit** \u00b7 Better Instruction Following Editing | CVPR 2025 | "instruction following" \u8d28\u91cf\u8bc4\u5206\u4e0e\u6539\u8fdb |
| **24** | **Image Editing as Programs** \u00b7 Diffusion as Program | 2025 | \u7f16\u8f91 = \u53ef\u6267\u884c\u7a0b\u5e8f, \u4e0e\u6211\u4eec\u53c2\u6570\u8f93\u51fa \"=\u7a0b\u5e8f\" \u540c\u54f2\u5b66 |

---

## Tier 5: \u77e5\u8bc6\u84b8\u998f / Efficient VLM (\u4e3a\u6211\u4eec **\"\u8bed\u4e49\u84b8\u998f\"** \u4e3b\u9898\u63d0\u4f9b\u65b9\u6cd5\u8bba\u53c2\u8003)

\u672c\u9879\u76ee\u4ece Venus 7B \u84b8\u998f\u5230 MobileViT (\u62a5\u544a\u4e3a\u4ee5 6M \u8d70\u7aef), \u4e0e\u8fd9\u4e09\u7bc7 KD-VLM \u662f\u540c\u4ee3\u4f5c\u54c1\u3002

| # | \u8bba\u6587 | \u5e74\u4efd/\u6536\u5f55 | \u4e0e\u6211\u4eec\u7684\u5173\u8054 |
|:--:|------|:---:|---------|
| **25** | **VL2Lite** \u00b7 Task-Specific KD: Large VLM \u2192 Lightweight | CVPR 2025 | \u4e0e\u6211\u4eec\u8def\u7ebf\u51e0\u4e4e\u5e73\u884c (\u4efb\u52a1\u7279\u5b9a\u3001\u8f7b\u91cf\u5316), **\u5fc5\u5f15** |
| **26** | **MoVE-KD** \u00b7 KD VLM with Mixture of Visual Encoders | CVPR 2025 | \u591a\u89c6\u89c9 encoder \u84b8\u998f\u5230\u5355\u4e2a, **\u672c\u9879\u76ee MobileViT \u9009\u578b\u53ef\u53c2\u8003** |
| **27** | **VLMs-Guided Repr Distillation** \u00b7 for Efficient VRL | CVPR 2025 | VLM \u5f15\u5bfc\u8868\u793a\u84b8\u998f, \u5fc5\u5f15\u4f5c\u3002\u53ea\u662f\u4ed6\u4eec\u9762\u5411\u7684\u662f RL\u4efb\u52a1 |

---

## \u7edf\u8ba1

| Tier | \u4e3b\u9898 | \u8bba\u6587\u6570 | \u603b\u5927\u5c0f |
|:--:|------|:--:|:--:|
| 1 | Lightroom Agent / VLM \u53c2\u6570\u5316 retouching | 7 | ~245 MB |
| 2 | \u7f8e\u5b66 + MLLM/IQA Foundation | 5 | ~14 MB |
| 3 | \u53c2\u6570\u5f0f retouching / \u53ef\u5fae\u7f16\u8f91 | 4 | ~26 MB |
| 4 | \u7f16\u8f91/\u590d\u539f Agent + Instruction Editing | 8 | ~78 MB |
| 5 | \u77e5\u8bc6\u84b8\u998f / Efficient VLM | 3 | ~10 MB |
| **\u603b** | | **27** | **~370 MB** |

\u6765\u6e90\u5206\u5e03\uff1aarXiv 21\u7bc7  \u00b7  CVF Open Access 6\u7bc7
\u5e74\u4efd\u5206\u5e03\uff1a2024 (10\u7bc7) \u00b7 2025 (11\u7bc7) \u00b7 2026 (6\u7bc7)

---

## \u63a8\u8350\u9605\u8bfb\u987a\u5e8f (\u5bf9\u4e8e\u5199 related work)

1. **\u5148\u770b 01 (JarvisEvo) + 02 (JarvisArt) + 03 (VeraRetouch)**: \u8fd9\u4e09\u7bc7\u51e0\u4e4e\u662f\u6211\u4eec\u7684 baseline + \u5dee\u5f02\u70b9\u53d1\u73b0\u6e90\uff0c\u5fc5\u987b\u9010\u53e5\u8bfb\u3002
2. **\u518d\u770b 04 (PhotoArtAgent)**: \u5bf9\u6211\u4eec\u8bba\u6587 motivation \u90e8\u5206\u51b2\u51fb\u6700\u5927\u3002
3. **08-12 (Tier 2)**: \u6311\u9009\u4f60\u8981\u5f15\u7528\u7684\u7f8e\u5b66\u8bc4\u5206 + IQA \u53c2\u8003\u3002
4. **13-16 (Tier 3) \u5feb\u8bfb**: \u6446 baseline \u8868\u7684\u4f3c\u4f50\u3002
5. **25-27 (Tier 5)**: KD-VLM \u4e3a\u6211\u4eec "\u8bed\u4e49\u84b8\u998f" \u63d0\u4f9b\u65b9\u6cd5\u8bba\u8112\u7edc\u3002
6. **17-24 (Tier 4)**: \u8003\u5bdf\u662f\u5426\u62d3\u5c55\u5230\u7eaf agent \u8def\u7ebf \u00b7 \u9009\u62e9\u6027\u5f15\u3002

---

## \u4e0b\u4e00\u6b65\u5efa\u8bae

- \u91cd\u70b9 vs **JarvisEvo / JarvisArt / VeraRetouch** \u63cf\u8ff0\u672c\u9879\u76ee \u5dee\u5f02\u5316\u70b9\u3002\u6700\u80fd\u5f70\u663e\u4f60\u4eec\u4f18\u52bf\u7684\u53d6\u89d2:
  - **\u7aef\u4fa7\u8f6c\u6362** (MobileViT 6M vs Qwen3-VL 4B): \u4f60\u4eec\u6709\u8c03\u4f18\u540e\u7684\u8f7b\u91cf\u5316\u8bbe\u8ba1
  - **\u8bed\u4e49\u84b8\u998f** \u4e0d\u662f \"\u7b54\u6848\u84b8\u998f\" \u800c\u662f \"\u7406\u89e3\u8fc7\u7a0b\u84b8\u998f\" (\u6709\u540c\u4ee3\u53c2\u8003 25 VL2Lite)
  - **DiffISP \u53ef\u5fae\u6e32\u67d3** \u00b7 \u53ef\u4f5c\u4e3a\u4e0e VeraRetouch \u4e0d\u540c\u7684\u533a\u522b\u5316
  - **6 \u53c2\u6570\u7cbe\u7b80** \u00b7 vs JarvisEvo \u7684 200+ tools \u591a\u4e3b\u9898
- \u6211\u4eec\u9879\u76ee\u8fd8\u6709\u4e00\u4e2a\u72ec\u7279\u70b9: **\u540c\u65f6\u8d70 Distill (v6-v13) \u548c LoRA SFT \u4e24\u6761\u8def**, \u8be5\u53cc\u8f68\u5728\u4e0a\u9762\u4efb\u4f55\u4e00\u7bc7\u91cc\u90fd\u6ca1\u770b\u5230\u3002
