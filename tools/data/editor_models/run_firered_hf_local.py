"""FireRed-Image-Edit-1.0 + Lightning LoRA \u672c\u5730\u63a8\u7406 (HuggingFace diffusers).

\u57fa\u4e8e https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0-Lightning \u7684\u5b98\u65b9 README \u4ee3\u7801.

\u5173\u952e\u5dee\u5f02 vs run_firered_online.py (\u8c03 ModelScope Studio 1.1 API):
- HF \u672c\u5730\u8dd1 \u672c\u5730\u6743\u91cd, \u53ef\u5b8c\u5168\u63a7\u5236
- \u7248\u672c 1.0 (vs Studio 1.1 \u53ef\u80fd\u7248\u672c\u4e0d\u540c)
- \u53ef\u9009 enable_model_cpu_offload() \u7528\u4e8e 32GB VRAM \u88c5 ~40GB bf16 \u6a21\u578b
- \u540c\u4e00\u7ec4 seed/steps/cfg \u4ee5\u4fbf\u5bf9\u6bd4

\u7528\u6cd5 (AutoDL):
  export HF_HOME=/root/autodl-tmp/hf_cache
  export HF_ENDPOINT=https://hf-mirror.com  # \u7adf\u9009\u955c\u50cf\u52a0\u901f
  /root/miniconda3/bin/python tools/data/editor_models/run_firered_hf_local.py \\
      --captions data/teacher_edits_firered_0600_instr_pilot.json \\
      --input_dir /root/autodl-tmp/firered_hf_pilot/origs \\
      --out_dir /root/autodl-tmp/firered_hf_pilot/out \\
      --cpu_offload
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', required=True, help='caption JSON file')
    ap.add_argument('--input_dir', required=True, help='\u539f\u56fe\u76ee\u5f55 (\u67e5\u627e <idx:04d>.png)')
    ap.add_argument('--out_dir', required=True, help='\u8f93\u51fa\u76ee\u5f55')
    ap.add_argument('--seed', type=int, default=42,
                    help='Default 42 \u4ee5\u4fdd\u6301\u4e0e run_firered_online.py \u4e00\u81f4 (HF README \u793a\u4f8b\u7528 0)')
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--cfg', type=float, default=1.0)
    ap.add_argument('--base_model', default='FireRedTeam/FireRed-Image-Edit-1.0')
    ap.add_argument('--lora_repo', default='FireRedTeam/FireRed-Image-Edit-1.0-Lightning')
    ap.add_argument('--lora_weight_name', default=None,
                    help='\u53ef\u9009: \u660e\u786e\u7684 .safetensors \u6587\u4ef6\u540d')
    ap.add_argument('--cpu_offload', action='store_true',
                    help='\u5f00 enable_model_cpu_offload() (\u8f83\u5feb, \u4f46 32GB VRAM \u88c5\u4e0d\u4e0b\u5168\u90e8 transformer)')
    ap.add_argument('--sequential_offload', action='store_true',
                    help='\u5f00 enable_sequential_cpu_offload() (\u6700\u5c0f\u663e\u5b58, ~3-5x \u6162) - 32GB VRAM \u63a8\u8350')
    ap.add_argument('--vae_slice_tile', action='store_true',
                    help='\u5f00 vae.enable_slicing() + enable_tiling() \u8282\u7701 vae \u663e\u5b58')
    ap.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    args = ap.parse_args()

    # CUDA \u5185\u5b58\u788e\u7247\u4f18\u5316
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = Path(args.input_dir)

    print('[ENV] HF_HOME =', os.environ.get('HF_HOME', '(default ~/.cache/huggingface)'))
    print('[ENV] HF_ENDPOINT =', os.environ.get('HF_ENDPOINT', '(default https://huggingface.co)'))
    print(f'[ARG] base_model={args.base_model}')
    print(f'[ARG] lora_repo ={args.lora_repo}')
    print(f'[ARG] cpu_offload={args.cpu_offload}, dtype={args.dtype}')
    print(f'[ARG] seed={args.seed}, steps={args.steps}, cfg={args.cfg}')

    # Lazy import \u4ee5\u4fbf\u63d0\u524d\u770b\u5230 arg \u9519\u8bef
    import torch
    from PIL import Image
    from diffusers import QwenImageEditPlusPipeline

    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f'\n[LOAD] pipeline \u4ece HF \u4e0b\u8f7d (\u9996\u6b21 ~40GB, \u4f30 5-15 min)...')
    t0 = time.time()
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
    )
    print(f'  pipeline loaded in {time.time()-t0:.1f}s')

    print(f'[LOAD] LoRA: {args.lora_repo}')
    t0 = time.time()
    if args.lora_weight_name:
        pipe.load_lora_weights(args.lora_repo, weight_name=args.lora_weight_name)
    else:
        pipe.load_lora_weights(args.lora_repo)
    print(f'  LoRA loaded in {time.time()-t0:.1f}s')

    if args.sequential_offload:
        pipe.enable_sequential_cpu_offload()
        print('  -> sequential_cpu_offload enabled (\u6700\u5c0f\u663e\u5b58, \u6162)')
    elif args.cpu_offload:
        pipe.enable_model_cpu_offload()
        print('  -> model_cpu_offload enabled')
    else:
        pipe = pipe.to('cuda')
        print('  -> pipeline.to(cuda)')

    if args.vae_slice_tile:
        try:
            pipe.vae.enable_slicing()
            pipe.vae.enable_tiling()
            print('  -> vae.slicing + tiling enabled')
        except Exception as e:
            print(f'  vae optimization failed: {e}')

    # \u5165\u53e3 caption
    with open(args.captions, encoding='utf-8') as f:
        cfg = json.load(f)
    samples = cfg['samples']
    print(f'\n[RUN] {len(samples)} sample(s) -> {out_dir}')

    records = []
    t_start = time.time()
    for i, s in enumerate(samples):
        idx = s['idx']
        caption = s['new_caption']
        src_name = s.get('source_image', f'{idx:04d}.png')

        src_path = input_dir / f'{idx:04d}.png'
        if not src_path.exists():
            src_path = input_dir / src_name
        if not src_path.exists():
            print(f'[{i+1}/{len(samples)}] SKIP idx={idx}: no source in {input_dir}')
            records.append({'idx': idx, 'status': 'no_source'})
            continue

        print(f'[{i+1}/{len(samples)}] idx={idx} src={src_path.name} prompt="{caption[:80]}..."', flush=True)
        ti = time.time()
        try:
            img = Image.open(src_path).convert('RGB')
            result = pipe(
                image=[img],
                prompt=caption,
                height=None,
                width=None,
                num_inference_steps=args.steps,
                generator=torch.manual_seed(args.seed),
                true_cfg_scale=args.cfg,
            ).images[0]
            out_path = out_dir / f'{idx:04d}.png'
            result.save(out_path)
            dt = time.time() - ti
            print(f'  -> saved ({dt:.1f}s)  {out_path.name}', flush=True)
            records.append({
                'idx': idx, 'src': str(src_path), 'caption': caption,
                'out': str(out_path), 'runtime_sec': dt, 'status': 'ok',
            })
        except Exception as e:
            dt = time.time() - ti
            print(f'  FAIL ({dt:.1f}s): {type(e).__name__}: {e}', flush=True)
            records.append({
                'idx': idx, 'src': str(src_path), 'caption': caption,
                'status': 'fail', 'error': f'{type(e).__name__}: {e}',
            })

    total = time.time() - t_start
    n_ok = sum(1 for r in records if r.get('status') == 'ok')

    meta = {
        'base_model': args.base_model,
        'lora_repo': args.lora_repo,
        'lora_weight_name': args.lora_weight_name,
        'cpu_offload': args.cpu_offload,
        'sequential_offload': args.sequential_offload,
        'vae_slice_tile': args.vae_slice_tile,
        'dtype': args.dtype,
        'seed': args.seed, 'steps': args.steps, 'cfg': args.cfg,
        'captions_file': args.captions,
        'input_dir': str(input_dir),
        'n_total': len(samples),
        'n_ok': n_ok,
        'runtime_sec': total,
        'records': records,
        'diffusers_version': __import__('diffusers').__version__,
        'torch_version': __import__('torch').__version__,
    }
    meta_path = out_dir / 'run_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'\n[DONE] {n_ok}/{len(samples)} in {total:.1f}s')
    print(f'[META] {meta_path}')


if __name__ == '__main__':
    main()
