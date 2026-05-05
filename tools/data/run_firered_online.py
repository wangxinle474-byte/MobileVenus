"""\u8c03\u7528 ModelScope Studio \u4e0a\u7684 FireRed-Image-Edit-1.1 (Lightning LoRA) \u5728\u7ebf\u63a8\u7406.

API: https://fireredteam-firered-image-edit-1-1.ms.show  (gradio 6.2.0)

\u7528\u6cd5:
  python tools/data/run_firered_online.py \\
    --captions data/compare_5_captions_edit.json \\
    --input_dir data/compare_5_images \\
    --out_dir outputs/firered_compare_editB \\
    --lora Lightning
"""
import argparse
import json
import shutil
import time
from pathlib import Path

from gradio_client import Client, handle_file

STUDIO_URL = 'https://fireredteam-firered-image-edit-1-1.ms.show'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', required=True, help='compare_5_captions(_edit).json')
    ap.add_argument('--input_dir', default='data/compare_5_images',
                    help='5 \u5f20\u539f\u56fe\u6240\u5728\u76ee\u5f55 (\u6309 source_image \u914d\u5bf9)')
    ap.add_argument('--out_dir', required=True, help='\u8f93\u51fa\u76ee\u5f55')
    ap.add_argument('--lora', default='Lightning',
                    choices=['None', 'Covercraft', 'Lightning', 'Makeup'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--true_cfg', type=float, default=4.0)
    ap.add_argument('--steps', type=int, default=0,
                    help='0 = \u4ece /on_lora_change \u81ea\u52a8\u53d6\u63a8\u8350\u503c')
    ap.add_argument('--rewrite_prompt', action='store_true',
                    help='\u8ba9 FireRed \u5185\u7f6e LLM \u6539\u5199 prompt (\u9ed8\u8ba4\u5173)')
    ap.add_argument('--height', type=int, default=0, help='0 = auto')
    ap.add_argument('--width', type=int, default=0, help='0 = auto')
    args = ap.parse_args()

    cfg = json.load(open(args.captions, encoding='utf-8'))
    samples = cfg['samples']
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[CONNECT] {STUDIO_URL}')
    client = Client(STUDIO_URL, verbose=False)

    def _unwrap(x):
        """gradio Component.update() \u8fd4\u56de dict {'__type__': 'update', 'value': ...}"""
        if isinstance(x, dict) and 'value' in x:
            return x['value']
        return x

    # 1) \u67e5 Lightning LoRA \u7684\u63a8\u8350\u53c2\u6570
    if args.steps == 0:
        print(f'[LORA] querying recommended params for lora={args.lora!r}')
        lora_info = client.predict(lora_name=args.lora, api_name='/on_lora_change')
        rec_steps = _unwrap(lora_info[0])
        rec_cfg = _unwrap(lora_info[1])
        rec_seed = _unwrap(lora_info[2])
        rec_rand = _unwrap(lora_info[3])
        print(f'  rec_steps={rec_steps}, rec_cfg={rec_cfg}, '
              f'rec_seed={rec_seed}, rec_randomize={rec_rand}')
        steps = int(rec_steps)
        # \u5c0a\u91cd\u5b98\u65b9\u63a8\u8350\u7684 cfg (Lightning=1.0, \u4e0d\u662f 4.0), \u5141\u8bb8\u7528\u6237\u8986\u76d6
        if abs(args.true_cfg - 4.0) < 1e-6:
            true_cfg = float(rec_cfg)
        else:
            true_cfg = args.true_cfg
    else:
        steps = args.steps
        true_cfg = args.true_cfg

    print(f'[CONFIG] lora={args.lora}, steps={steps}, cfg={true_cfg}, '
          f'seed={args.seed}, rewrite={args.rewrite_prompt}')
    print(f'[RUN] {len(samples)} samples -> {out_dir}')

    records = []
    t0 = time.time()
    for i, s in enumerate(samples):
        idx = s['idx']
        caption = s['new_caption']
        src_name = s['source_image']
        # \u4f18\u5148\u627e <idx>_orig.png (LongCat \u8f93\u51fa\u547d\u540d), \u518d fallback \u5230 source_image
        src_path = input_dir / f'{idx:04d}_orig.png'
        if not src_path.exists():
            src_path = input_dir / src_name
        if not src_path.exists():
            print(f'[{i+1}/{len(samples)}] SKIP idx={idx}: missing both '
                  f'{idx:04d}_orig.png and {src_name} in {input_dir}')
            continue

        # Gallery \u8f93\u5165\u683c\u5f0f: list[{image: {...}, caption: None}]
        gallery_input = [{'image': handle_file(str(src_path)), 'caption': None}]

        print(f'[{i+1}/{len(samples)}] idx={idx}  src={src_name}  '
              f'prompt={caption[:70]}...')
        ti = time.time()
        try:
            result = client.predict(
                input_images=gallery_input,
                prompt=caption,
                lora_choice=args.lora,
                seed=args.seed,
                randomize_seed=False,
                true_guidance_scale=true_cfg,
                num_inference_steps=steps,
                height=args.height,
                width=args.width,
                rewrite_prompt=args.rewrite_prompt,
                num_images_per_prompt=1,
                api_name='/infer',
            )
        except Exception as e:
            dt = time.time() - ti
            print(f'  FAIL ({dt:.1f}s): {type(e).__name__}: {e}')
            records.append({
                'idx': idx, 'source_image': src_name, 'caption': caption,
                'status': 'fail', 'error': f'{type(e).__name__}: {e}',
            })
            continue

        dt = time.time() - ti
        gallery_out, seed_used = result
        if not gallery_out:
            print(f'  WARN ({dt:.1f}s): empty gallery')
            continue
        # Gallery \u8f93\u51fa\u683c\u5f0f\u5728\u4e0d\u540c gradio \u7248\u672c/\u7ed1\u5b9a\u91cc\u4e0d\u540c, \u5c1d\u8bd5\u591a\u79cd:
        # - str (\u6587\u4ef6\u8def\u5f84)
        # - (path, caption) tuple/list
        # - {'image': {'path': ...}, 'caption': ...}
        # - {'image': 'path', ...}
        out_info = gallery_out[0]
        out_local = None
        if isinstance(out_info, str):
            out_local = out_info
        elif isinstance(out_info, (list, tuple)) and out_info:
            out_local = out_info[0]
        elif isinstance(out_info, dict):
            img = out_info.get('image')
            if isinstance(img, dict):
                out_local = img.get('path') or img.get('url')
            else:
                out_local = img or out_info.get('path') or out_info.get('url')
        if not out_local:
            print(f'  WARN ({dt:.1f}s): cannot extract path from gallery_out[0]={out_info!r}')
            records.append({
                'idx': idx, 'source_image': src_name, 'caption': caption,
                'status': 'fail', 'error': f'parse_gallery:{out_info!r}',
            })
            continue

        # \u62f7\u8d1d\u5230\u76ee\u6807\u540d
        final_orig = out_dir / f'{idx:04d}_orig.png'
        final_edit = out_dir / f'{idx:04d}_firered.png'
        shutil.copy(src_path, final_orig)
        shutil.copy(out_local, final_edit)

        print(f'  -> saved ({dt:.1f}s)  seed={int(seed_used)}  out={final_edit.name}')
        records.append({
            'idx': idx, 'source_image': src_name, 'caption': caption,
            'status': 'ok', 'seed_used': int(seed_used),
            'orig': str(final_orig), 'edit': str(final_edit),
            'runtime_sec': dt,
        })

    total = time.time() - t0
    n_ok = sum(1 for r in records if r.get('status') == 'ok')
    print(f'\n[DONE] {n_ok}/{len(samples)} in {total:.1f}s')

    meta_path = out_dir / 'run_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'studio_url': STUDIO_URL,
            'lora': args.lora,
            'steps': steps,
            'true_cfg': true_cfg,
            'seed': args.seed,
            'rewrite_prompt': args.rewrite_prompt,
            'captions_file': args.captions,
            'n_total': len(samples),
            'n_ok': n_ok,
            'runtime_sec': total,
            'records': records,
        }, f, indent=2, ensure_ascii=False)
    print(f'[META] {meta_path}')


if __name__ == '__main__':
    main()
