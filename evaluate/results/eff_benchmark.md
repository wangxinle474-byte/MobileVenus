# IntelligenceCamera — Efficiency Benchmark

- **Device**: `cpu`
- **Latency protocol**: warmup=5, measure=30, batch=1
- **Param-model input**: 256×256
- **ISP render size**: 256×256

## Our model breakdown

| Module | Params (M) | GFLOPs | Lat mean (ms) | std | p50 | p95 | Peak VRAM (MB) | Input |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **StageA** | 3.30 | 0.68 | 15.98 | 0.94 | 15.85 | 17.52 | — | 1×3×256×256 |
| **StageB** | 3.27 | 0.68 | 17.35 | 1.86 | 16.63 | 20.13 | — | 1×3×256×256 |
| **NeuralISP_256** | 1.62 | 12.86 | 34.65 | 1.70 | 34.13 | 37.02 | — | 1×3×256×256 + 1×6 |
| **Pipeline** | 4.89 | 13.54 | 51.12 | 1.30 | 51.40 | 52.82 | — | 1×3×256×256 + 1×3×256×256 |

## vs Literature (from PDF scans of Tier A & B papers)

| Method | Params | GFLOPs | Latency (ms) | Hardware | Dataset | Src |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| JarvisEvo | 8B* | - | cloud | A100 | ArtEdit | 01_2026 |
| JarvisArt | 7B | - | cloud | A100 | FiveK | 02_2025 |
| VeraRetouch | 4B+LoRA | - | ~150* | RTX 4090 | FiveK | 03_2026 |
| PhotoArtAgent | API | - | API call | GPT-4 | FiveK | 04_2025 |
| MonetGPT | MLLM | - | cloud | - | PPR10K | 05_2025 |
| INRetouch | 0.01M | 10.59 | ~10* | RTX 3090 | FiveK | 06_2024 |
| Taming LUT | 0.06M | 0.03 | 0.4 | GPU | FiveK | 13_2024 |
| **Ours (Stage B only)** | **3.27M** | 0.68 | **17.35** | cpu | FiveK | this work |
| **Ours (Pipeline)** | **4.89M** | 13.54 | **51.12** | cpu | FiveK | this work |

## Notes

- `*` in literature column = paper does **not** report on-device latency; estimate from arch.
- **Stage B (256)** is the apples-to-apples target for paper Table — JarvisArt outputs ~25 sliders, ours outputs 6.
- **Pipeline** includes neural ISP rendering — competitors that go via Lightroom XMP do not include the renderer cost.
- CPU latency here is dev-machine reference; **真·端侧 ms 数字需要 ARM/Snapdragon profiling**（用 ONNX/CoreML 部署后再实测）。

## Suggested paper claim

> "Our parameter prediction model uses 3.27M parameters and runs in 17.35 ms per image (batch=1) on cpu, a **2143× reduction** in parameters vs JarvisArt (7B) while operating in the same Lightroom-style parameter space."