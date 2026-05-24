"""\u8c03\u7528 ModelScope Studio \u4e0a\u7684 FireRed-Image-Edit-1.1 (Lightning LoRA) \u5728\u7ebf\u63a8\u7406.

API: https://fireredteam-firered-image-edit-1-1.ms.show  (gradio 6.2.0)

\u7528\u6cd5:
  python tools/data/editor_models/run_firered_online.py \\
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

STUDIO_URL = 'https://studio-fireredteam-firered-image-edit-1-1.api-inference.modelscope.net'
MS_TOKEN = 'ms-28b63a2d-f09f-4593-b181-ffc026abb68b'


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
    ap.add_argument('--resume', action='store_true',
                    help='跳过 out_dir 中已存在的输出文件 (断点续传)')
    ap.add_argument('--max_retries', type=int, default=3,
                    help='每个样本最大重试次数 (API 偶尔超时)')
    ap.add_argument('--retry_delay', type=float, default=10.0,
                    help='重试间隔秒数')
    ap.add_argument('--delay', type=float, default=3.0,
                    help='每次成功请求后的等待秒数 (防限流)')
    args = ap.parse_args()

    cfg = json.load(open(args.captions, encoding='utf-8'))
    samples = cfg['samples']
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[CONNECT] {STUDIO_URL}')
    client = Client(STUDIO_URL, token=MS_TOKEN, verbose=False)

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

    # Resume: load existing records if any
    meta_path = out_dir / 'run_meta.json'
    records = []
    done_idxs = set()
    if args.resume and meta_path.exists():
        old_meta = json.load(open(meta_path, encoding='utf-8'))
        records = old_meta.get('records', [])
        done_idxs = {r['idx'] for r in records if r.get('status') == 'ok'}
        # Also check for output files directly
        for png in out_dir.glob('*.png'):
            try:
                done_idxs.add(int(png.stem))
            except ValueError:
                pass
        print(f'[RESUME] {len(done_idxs)} already done, skipping')

    t0 = time.time()
    n_skip = 0
    for i, s in enumerate(samples):
        idx = s['idx']
        caption = s['new_caption']
        src_name = s.get('source_image', f'{idx:04d}.png')

        # Resume: skip if output already exists
        if args.resume and idx in done_idxs:
            n_skip += 1
            continue

        # 优先找 <idx>.png, 再 <idx>_orig.png, 再 source_image (支持 FiveK .jpg)
        src_path = input_dir / f'{idx:04d}.png'
        if not src_path.exists():
            src_path = input_dir / f'{idx:04d}_orig.png'
        if not src_path.exists():
            src_path = input_dir / src_name
        if not src_path.exists():
            print(f'[{i+1}/{len(samples)}] SKIP idx={idx}: no source image found in {input_dir}')
            continue

        # Gallery \u8f93\u5165\u683c\u5f0f: list[{image: {...}, caption: None}]
        gallery_input = [{'image': handle_file(str(src_path)), 'caption': None}]

        done_so_far = len(done_idxs) + sum(1 for r in records if r.get('status') == 'ok' and r['idx'] not in done_idxs)
        remaining = len(samples) - n_skip - (i - n_skip)
        print(f'[{i+1}/{len(samples)}] idx={idx}  src={src_name}  '
              f'prompt={caption[:60]}...  (done={done_so_far}, left≈{remaining})')

        result = None
        last_err = None
        for attempt in range(args.max_retries):
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
                break
            except Exception as e:
                dt = time.time() - ti
                last_err = f'{type(e).__name__}: {e}'
                if attempt < args.max_retries - 1:
                    print(f'  RETRY {attempt+1}/{args.max_retries} ({dt:.1f}s): {last_err}')
                    time.sleep(args.retry_delay)
                else:
                    print(f'  FAIL ({dt:.1f}s): {last_err}')
                    records.append({
                        'idx': idx, 'source_image': src_name, 'caption': caption,
                        'status': 'fail', 'error': last_err,
                    })
        if result is None:
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

        # 新布局: 仅保存编辑后图, 原图独立代管于 outputs/compare_5/originals/
        final_edit = out_dir / f'{idx:04d}.png'
        shutil.copy(out_local, final_edit)

        print(f'  -> saved ({dt:.1f}s)  seed={int(seed_used)}  out={final_edit.name}')
        records.append({
            'idx': idx, 'source_image': src_name, 'caption': caption,
            'status': 'ok', 'seed_used': int(seed_used),
            'edit': str(final_edit),
            'runtime_sec': dt,
        })

        # Throttle to avoid 403 rate limiting
        if args.delay > 0:
            time.sleep(args.delay)

        # Periodic save (every 50 samples) for crash safety
        if len(records) % 50 == 0:
            _save_meta(meta_path, args, STUDIO_URL, steps, true_cfg,
                       samples, records, time.time() - t0)

    total = time.time() - t0
    n_ok = sum(1 for r in records if r.get('status') == 'ok') + len(done_idxs)
    print(f'\n[DONE] {n_ok}/{len(samples)} in {total:.1f}s (skipped {n_skip} resume)')
    _save_meta(meta_path, args, STUDIO_URL, steps, true_cfg,
               samples, records, total)
    print(f'[META] {meta_path}')


def _save_meta(meta_path, args, studio_url, steps, true_cfg,
               samples, records, total):
    n_ok = sum(1 for r in records if r.get('status') == 'ok')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'studio_url': studio_url,
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


if __name__ == '__main__':
    main()
