"""End-side efficiency benchmark for IntelligenceCamera vs literature.

Measures: #Params, GFLOPs, Latency (mean/std/p50/p95), Peak VRAM.

Targets benchmarked:
  StageA       Image (--img_size) → 256-D semantic embedding
  StageB       Image (--img_size) → 6 ISP params + confidence (full param-pred model)
  NeuralISP    (Image, 6 params) → enhanced Image  @ --isp_size
  Pipeline     StageB + NeuralISP                 (端侧总延迟)

Note: MobileViT stage5 requires input size divisible by 64 (256, 320, 384...).
      224 fails (stage5 outputs 7×7, patch=2 cannot divide). Default 256 is
      the correct working size for this architecture.

Outputs (in --out_dir, default evaluate/results/):
  eff_benchmark.json   raw measurements
  eff_benchmark.md     markdown report with literature comparison
  eff_benchmark.tex    LaTeX table for paper

Usage:
  python evaluate/eff_benchmark.py                       # auto GPU/CPU
  python evaluate/eff_benchmark.py --device cpu          # CPU only
  python evaluate/eff_benchmark.py --img_size 320        # change param input
  python evaluate/eff_benchmark.py --isp_size 1024       # high-res ISP
  python evaluate/eff_benchmark.py --runs 200            # more samples

Notes:
  - No checkpoints / dataset required (uses random tensors of correct shape).
  - FLOPs computed via fvcore → thop → skip (graceful fallback).
  - GPU latency uses torch.cuda.Event with synchronize barriers.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Make project imports work regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
from models.neural_isp import NeuralISP


# ============================================================================
# Literature reference numbers (from PDF scans of related papers).
# Cells with '*' are estimates from architecture (paper did not report).
# ============================================================================

LITERATURE = [
    # name              params       gflops    latency_ms    hardware     dataset    src
    ('JarvisEvo',       '8B*',       '-',      'cloud',      'A100',      'ArtEdit', '01_2026'),
    ('JarvisArt',       '7B',        '-',      'cloud',      'A100',      'FiveK',   '02_2025'),
    ('VeraRetouch',     '4B+LoRA',   '-',      '~150*',      'RTX 4090',  'FiveK',   '03_2026'),
    ('PhotoArtAgent',   'API',       '-',      'API call',   'GPT-4',     'FiveK',   '04_2025'),
    ('MonetGPT',        'MLLM',      '-',      'cloud',      '-',         'PPR10K',  '05_2025'),
    ('INRetouch',       '0.01M',     '10.59',  '~10*',       'RTX 3090',  'FiveK',   '06_2024'),
    ('Taming LUT',      '0.06M',     '0.03',   '0.4',        'GPU',       'FiveK',   '13_2024'),
]


# ============================================================================
# Helpers
# ============================================================================

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, *inputs):
    """Try multiple FLOP counters; return (gflops, lib_name) or (None, None)."""
    # 1) fvcore — most accurate, handles ConvT / FiLM
    try:
        from fvcore.nn import FlopCountAnalysis
        analysis = FlopCountAnalysis(model, inputs)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        return analysis.total() / 1e9, 'fvcore'
    except ImportError:
        pass
    except Exception as e:
        print(f'  [fvcore failed: {e}]')
    # 2) thop — lightweight fallback
    try:
        from thop import profile
        flops, _ = profile(model, inputs=inputs, verbose=False)
        return flops / 1e9, 'thop'
    except ImportError:
        pass
    except Exception as e:
        print(f'  [thop failed: {e}]')
    return None, None


def benchmark_latency(model, inputs, device, runs=100, warmup=20):
    """Measure inference latency. Returns dict of stats in ms."""
    model = model.to(device).eval()
    inputs = tuple(t.to(device) for t in inputs)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
        with torch.no_grad():
            for i in range(runs):
                starts[i].record()
                _ = model(*inputs)
                ends[i].record()
        torch.cuda.synchronize()
        times_ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        times = []
        with torch.no_grad():
            for _ in range(runs):
                t0 = time.perf_counter()
                _ = model(*inputs)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
        times_ms = np.array(times)
        peak_mem_mb = None

    return {
        'mean_ms': float(times_ms.mean()),
        'std_ms': float(times_ms.std()),
        'p50_ms': float(np.percentile(times_ms, 50)),
        'p95_ms': float(np.percentile(times_ms, 95)),
        'min_ms': float(times_ms.min()),
        'max_ms': float(times_ms.max()),
        'peak_mem_mb': peak_mem_mb,
        'n_runs': runs,
    }


# ============================================================================
# Forward wrappers — make multi-input / dict-output models work with FLOP libs
# ============================================================================

class _StageAFwd(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, images):
        return self.m(images)['student_emb']


class _StageBFwd(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, images):
        return self.m(images)['norm_params']


class _PipelineFwd(nn.Module):
    """Full on-device path: image (224) → params → render image (isp_size)."""
    def __init__(self, param_model, isp):
        super().__init__()
        self.param_model = param_model
        self.isp = isp
    def forward(self, image_224, image_render):
        out = self.param_model(image_224)
        return self.isp(image_render, out['norm_params'])


# ============================================================================
# Reporters
# ============================================================================

def write_markdown(results, path):
    targets = results['targets']
    lines = [
        '# IntelligenceCamera — Efficiency Benchmark',
        '',
        f'- **Device**: `{results["device"]}`'
        + (f' ({results["gpu_name"]})' if results.get('gpu_name') else ''),
        f'- **Latency protocol**: warmup={results["warmup"]}, measure={results["runs"]}, batch=1',
        f'- **Param-model input**: {results["img_size"]}×{results["img_size"]}',
        f'- **ISP render size**: {results["isp_size"]}×{results["isp_size"]}',
        '',
        '## Our model breakdown',
        '',
        '| Module | Params (M) | GFLOPs | Lat mean (ms) | std | p50 | p95 | Peak VRAM (MB) | Input |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for name, t in targets.items():
        lat = t['latency']
        flops = f'{t["gflops"]:.2f}' if t['gflops'] is not None else '—'
        mem = f'{lat["peak_mem_mb"]:.0f}' if lat.get('peak_mem_mb') is not None else '—'
        lines.append(
            f'| **{name}** | {t["params_M"]:.2f} | {flops} | '
            f'{lat["mean_ms"]:.2f} | {lat["std_ms"]:.2f} | '
            f'{lat["p50_ms"]:.2f} | {lat["p95_ms"]:.2f} | {mem} | {t["input_shape"]} |'
        )

    lines += [
        '',
        '## vs Literature (from PDF scans of Tier A & B papers)',
        '',
        '| Method | Params | GFLOPs | Latency (ms) | Hardware | Dataset | Src |',
        '|---|:---:|:---:|:---:|:---:|:---:|:---:|',
    ]
    for name, params, gflops, lat, hw, ds, src in LITERATURE:
        lines.append(
            f'| {name} | {params} | {gflops} | {lat} | {hw} | {ds} | {src} |'
        )
    # Append our rows
    sb = targets['StageB']
    pipe = targets['Pipeline']
    sb_g = f'{sb["gflops"]:.2f}' if sb['gflops'] is not None else '—'
    pipe_g = f'{pipe["gflops"]:.2f}' if pipe['gflops'] is not None else '—'
    lines += [
        f'| **Ours (Stage B only)** | **{sb["params_M"]:.2f}M** | {sb_g} | '
        f'**{sb["latency"]["mean_ms"]:.2f}** | {results["device"]} | FiveK | this work |',
        f'| **Ours (Pipeline)** | **{pipe["params_M"]:.2f}M** | {pipe_g} | '
        f'**{pipe["latency"]["mean_ms"]:.2f}** | {results["device"]} | FiveK | this work |',
    ]

    lines += [
        '',
        '## Notes',
        '',
        '- `*` in literature column = paper does **not** report on-device latency; estimate from arch.',
        f'- **Stage B ({results["img_size"]})** is the apples-to-apples target for paper Table — JarvisArt outputs ~25 sliders, ours outputs 6.',
        '- **Pipeline** includes neural ISP rendering — competitors that go via Lightroom XMP do not include the renderer cost.',
        '- CPU latency here is dev-machine reference; **真·端侧 ms 数字需要 ARM/Snapdragon profiling**（用 ONNX/CoreML 部署后再实测）。',
        '',
        '## Suggested paper claim',
        '',
        f'> "Our parameter prediction model uses {sb["params_M"]:.2f}M parameters and runs in '
        f'{sb["latency"]["mean_ms"]:.2f} ms per image (batch=1) on {results["device"]}, '
        f'a **{int(7000 / max(sb["params_M"], 0.01))}× reduction** in parameters '
        f'vs JarvisArt (7B) while operating in the same Lightroom-style parameter space."',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_latex(results, path):
    sb = results['targets']['StageB']
    pipe = results['targets']['Pipeline']
    sb_g = f'{sb["gflops"]:.2f}' if sb['gflops'] else '--'
    pipe_g = f'{pipe["gflops"]:.2f}' if pipe['gflops'] else '--'

    tex = r"""% Auto-generated by evaluate/eff_benchmark.py — do not hand-edit.
\begin{table}[t]
\centering
\small
\caption{Efficiency comparison vs prior parametric retouching methods.
Latency measured at batch=1, single image, on the indicated hardware.
$^{*}$ indicates the paper does not report on-device latency;
we estimate from the reported architecture.}
\label{tab:efficiency}
\begin{tabular}{lcccc}
\toprule
Method & \#Params & GFLOPs & Latency (ms) & Hardware \\
\midrule
JarvisArt~\cite{jarvisart}     & 7B       & --     & cloud     & A100   \\
JarvisEvo~\cite{jarvisevo}     & 8B$^{*}$ & --     & cloud     & A100   \\
VeraRetouch~\cite{veraretouch} & 4B+LoRA  & --     & 150$^{*}$ & 4090   \\
INRetouch~\cite{inretouch}     & 0.01M    & 10.59  & 10$^{*}$  & 3090   \\
Taming LUT~\cite{taminglut}    & 0.06M    & 0.03   & 0.4       & GPU    \\
\midrule
"""
    tex += (
        f'\\textbf{{Ours (Stage B)}}    & \\textbf{{{sb["params_M"]:.2f}M}} & '
        f'{sb_g} & \\textbf{{{sb["latency"]["mean_ms"]:.2f}}} & '
        f'{results["device"]} \\\\\n'
    )
    tex += (
        f'\\textbf{{Ours (Pipeline)}} & \\textbf{{{pipe["params_M"]:.2f}M}} & '
        f'{pipe_g} & \\textbf{{{pipe["latency"]["mean_ms"]:.2f}}} & '
        f'{results["device"]} \\\\\n'
    )
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    path.write_text(tex, encoding='utf-8')


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    ap.add_argument('--runs', type=int, default=100)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--img_size', type=int, default=256,
                    help='Input size for param model. Must be divisible by 64 '
                         '(256/320/384). 224 fails due to MobileViT stage5.')
    ap.add_argument('--isp_size', type=int, default=512)
    ap.add_argument('--out_dir', default='evaluate/results')
    args = ap.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f'[device] {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')

    # CPU is much slower — reduce repeats
    if device.type == 'cpu':
        args.runs = min(args.runs, 30)
        args.warmup = min(args.warmup, 5)
        print(f'  (CPU mode → runs={args.runs}, warmup={args.warmup})')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.img_size % 64 != 0:
        print(f'  [warn] --img_size {args.img_size} is not divisible by 64; '
              f'MobileViT stage5 may fail. Recommended: 256.')

    results = {
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'runs': args.runs,
        'warmup': args.warmup,
        'img_size': args.img_size,
        'isp_size': args.isp_size,
        'targets': {},
    }

    # =========== Stage A ===========
    print('\n[1/4] Stage A (visual encoder + semantic projector)')
    stage_a = SemanticDistillModel(image_size=args.img_size)
    total, _ = count_params(stage_a)
    img_param = torch.randn(1, 3, args.img_size, args.img_size)
    flops_g, flops_lib = estimate_flops(_StageAFwd(stage_a), img_param)
    lat = benchmark_latency(_StageAFwd(stage_a), (img_param,),
                            device, args.runs, args.warmup)
    results['targets']['StageA'] = {
        'params_M': total / 1e6,
        'gflops': flops_g, 'flops_lib': flops_lib,
        'latency': lat,
        'input_shape': f'1×3×{args.img_size}×{args.img_size}',
        'output_shape': '1×256',
    }
    fg = f'{flops_g:.2f}G' if flops_g else '—'
    print(f'  Params {total/1e6:.2f}M | FLOPs {fg} ({flops_lib or "skipped"}) | '
          f'Lat {lat["mean_ms"]:.2f}±{lat["std_ms"]:.2f} ms')

    # =========== Stage B ===========
    print('\n[2/4] Stage B (full param-prediction model: 6 sliders + confidence)')
    stage_b = DistillParamModel(stage_a_model=stage_a, image_size=args.img_size)
    total, _ = count_params(stage_b)
    flops_g, flops_lib = estimate_flops(_StageBFwd(stage_b), img_param)
    lat = benchmark_latency(_StageBFwd(stage_b), (img_param,),
                            device, args.runs, args.warmup)
    results['targets']['StageB'] = {
        'params_M': total / 1e6,
        'gflops': flops_g, 'flops_lib': flops_lib,
        'latency': lat,
        'input_shape': f'1×3×{args.img_size}×{args.img_size}',
        'output_shape': '1×6 (params)',
    }
    fg = f'{flops_g:.2f}G' if flops_g else '—'
    print(f'  Params {total/1e6:.2f}M | FLOPs {fg} ({flops_lib or "skipped"}) | '
          f'Lat {lat["mean_ms"]:.2f}±{lat["std_ms"]:.2f} ms')

    # =========== NeuralISP ===========
    print(f'\n[3/4] NeuralISP @ {args.isp_size}×{args.isp_size}')
    isp = NeuralISP(param_dim=6, base_ch=32, n_res_blocks=4)
    total_isp, _ = count_params(isp)
    img_isp = torch.randn(1, 3, args.isp_size, args.isp_size)
    params6 = torch.randn(1, 6)
    flops_g, flops_lib = estimate_flops(isp, img_isp, params6)
    lat = benchmark_latency(isp, (img_isp, params6),
                            device, args.runs, args.warmup)
    isp_key = f'NeuralISP_{args.isp_size}'
    results['targets'][isp_key] = {
        'params_M': total_isp / 1e6,
        'gflops': flops_g, 'flops_lib': flops_lib,
        'latency': lat,
        'input_shape': f'1×3×{args.isp_size}×{args.isp_size} + 1×6',
        'output_shape': f'1×3×{args.isp_size}×{args.isp_size}',
    }
    fg = f'{flops_g:.2f}G' if flops_g else '—'
    print(f'  Params {total_isp/1e6:.2f}M | FLOPs {fg} ({flops_lib or "skipped"}) | '
          f'Lat {lat["mean_ms"]:.2f}±{lat["std_ms"]:.2f} ms')

    # =========== Pipeline ===========
    print(f'\n[4/4] Pipeline = Stage B ({args.img_size}) + NeuralISP ({args.isp_size})')
    pipeline = _PipelineFwd(stage_b, isp)
    total_pipe = total + total_isp  # stage_b includes stage_a so this is correct
    img_param_p = torch.randn(1, 3, args.img_size, args.img_size)
    img_isp_p = torch.randn(1, 3, args.isp_size, args.isp_size)
    lat = benchmark_latency(pipeline, (img_param_p, img_isp_p),
                            device, args.runs, args.warmup)
    sb_g = results['targets']['StageB']['gflops']
    isp_g = results['targets'][isp_key]['gflops']
    pipe_g = sb_g + isp_g if (sb_g is not None and isp_g is not None) else None
    results['targets']['Pipeline'] = {
        'params_M': total_pipe / 1e6,
        'gflops': pipe_g, 'flops_lib': 'sum',
        'latency': lat,
        'input_shape': f'1×3×{args.img_size}×{args.img_size} + 1×3×{args.isp_size}×{args.isp_size}',
        'output_shape': f'1×3×{args.isp_size}×{args.isp_size}',
    }
    fg = f'{pipe_g:.2f}G' if pipe_g else '—'
    print(f'  Params {total_pipe/1e6:.2f}M | FLOPs {fg} (sum) | '
          f'Lat {lat["mean_ms"]:.2f}±{lat["std_ms"]:.2f} ms')

    # =========== Save ===========
    json_path = out_dir / 'eff_benchmark.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n[saved] {json_path}')

    md_path = out_dir / 'eff_benchmark.md'
    write_markdown(results, md_path)
    print(f'[saved] {md_path}')

    tex_path = out_dir / 'eff_benchmark.tex'
    write_latex(results, tex_path)
    print(f'[saved] {tex_path}')

    print('\n=== Summary ===')
    print(f'  Stage B  : {results["targets"]["StageB"]["params_M"]:.2f}M params, '
          f'{results["targets"]["StageB"]["latency"]["mean_ms"]:.2f} ms')
    print(f'  Pipeline : {results["targets"]["Pipeline"]["params_M"]:.2f}M params, '
          f'{results["targets"]["Pipeline"]["latency"]["mean_ms"]:.2f} ms')


if __name__ == '__main__':
    main()
