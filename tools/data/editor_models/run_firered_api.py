"""FireRed 1.1 API 推理 — 直接用 httpx 调 ModelScope API-inference 端点.

绕过 gradio_client 的上传 auth bug，手动处理:
1. 文件上传 (带 Bearer token)
2. /gradio_api/call/infer 提交任务
3. SSE 轮询获取结果
4. 下载输出图片

用法:
  python tools/data/editor_models/run_firered_api.py \
    --captions data/teacher_edits_fivek_full_contrast.json \
    --input_dir E:/Data/dataset/fivek_jpeg \
    --out_dir outputs/teacher_edits/fivek_full/contrast \
    --resume --delay 3
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import httpx

API_BASE = 'https://studio-fireredteam-firered-image-edit-1-1.api-inference.modelscope.net'
MS_TOKEN = 'ms-28b63a2d-f09f-4593-b181-ffc026abb68b'


class ContentFilterError(Exception):
    """Raised when ModelScope content moderation rejects an image."""
    pass


def _headers():
    return {'Authorization': f'Bearer {MS_TOKEN}'}


def upload_file(filepath: str) -> str:
    """Upload a local file and return the server-side temp path."""
    with open(filepath, 'rb') as f:
        r = httpx.post(
            f'{API_BASE}/gradio_api/upload',
            files={'files': (Path(filepath).name, f.read(), 'image/jpeg')},
            headers=_headers(),
            timeout=60,
        )
    if r.status_code == 400:
        try:
            body = r.json()
            if isinstance(body, list) and body and 'modelscope_code' in body[0]:
                raise ContentFilterError(body[0].get('error', 'content filtered'))
        except (ValueError, KeyError):
            pass
    r.raise_for_status()
    paths = r.json()
    return paths[0]  # "/tmp/gradio/..."


def call_infer(uploaded_path: str, caption: str,
              lora='Lightning', seed=42, steps=8, cfg=1.0,
              height=0, width=0, rewrite=False) -> str:
    """Submit an infer job and return event_id."""
    gallery_input = [{'image': {'path': uploaded_path}, 'caption': None}]
    payload = {
        'data': [
            gallery_input,   # input_images
            caption,          # prompt
            lora,             # lora_choice
            seed,             # seed
            False,            # randomize_seed
            cfg,              # true_guidance_scale
            steps,            # num_inference_steps
            height,           # height
            width,            # width
            rewrite,          # rewrite_prompt
            1,                # num_images_per_prompt
        ]
    }
    r = httpx.post(
        f'{API_BASE}/gradio_api/call/infer',
        json=payload,
        headers=_headers(),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()['event_id']


def poll_result(event_id: str, timeout: float = 300) -> dict | None:
    """Poll SSE stream for the result. Returns parsed data or None."""
    t0 = time.time()
    with httpx.stream(
        'GET',
        f'{API_BASE}/gradio_api/call/infer/{event_id}',
        headers=_headers(),
        timeout=httpx.Timeout(timeout, connect=30),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if time.time() - t0 > timeout:
                return None
            line = line.strip()
            if line.startswith('data:'):
                data_str = line[5:].strip()
                if data_str:
                    try:
                        return json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
    return None


def download_image(server_path: str, local_path: Path) -> bool:
    """Download an output image from the server."""
    if server_path.startswith('http'):
        url = server_path
    else:
        url = f'{API_BASE}/gradio_api/file={server_path}'

    r = httpx.get(url, headers=_headers(), timeout=60)
    if r.status_code == 200:
        local_path.write_bytes(r.content)
        return True
    return False


def extract_image_path(result_data) -> str | None:
    """Extract the output image path from the API result."""
    if not isinstance(result_data, list) or len(result_data) < 1:
        return None
    gallery = result_data[0]
    if not gallery:
        return None
    item = gallery[0]
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        img = item.get('image')
        if isinstance(img, dict):
            return img.get('path') or img.get('url')
        return img or item.get('path') or item.get('url')
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', required=True)
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--lora', default='Lightning')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=8)
    ap.add_argument('--cfg', type=float, default=1.0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--delay', type=float, default=3.0,
                    help='每次成功后等待秒数')
    ap.add_argument('--max_retries', type=int, default=10)
    ap.add_argument('--retry_delay', type=float, default=15.0)
    args = ap.parse_args()

    cfg_data = json.load(open(args.captions, encoding='utf-8'))
    samples = cfg_data['samples']
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / 'run_meta.json'

    # Resume
    done_idxs = set()
    records = []
    if args.resume:
        if meta_path.exists():
            old = json.load(open(meta_path, encoding='utf-8'))
            records = old.get('records', [])
            done_idxs = {r['idx'] for r in records if r.get('status') == 'ok'}
        for png in out_dir.glob('*.png'):
            try:
                done_idxs.add(int(png.stem))
            except ValueError:
                pass
        if done_idxs:
            print(f'[RESUME] {len(done_idxs)} already done')

    print(f'[CONFIG] lora={args.lora} steps={args.steps} cfg={args.cfg} '
          f'seed={args.seed} delay={args.delay}s')
    print(f'[RUN] {len(samples)} samples -> {out_dir}')

    t0 = time.time()
    n_skip = 0
    n_new_ok = 0

    for i, s in enumerate(samples):
        idx = s['idx']
        caption = s['new_caption']
        src_name = s.get('source_image', f'{idx:04d}.png')

        if args.resume and idx in done_idxs:
            n_skip += 1
            continue

        # Find source image
        src_path = input_dir / f'{idx:04d}.png'
        if not src_path.exists():
            src_path = input_dir / f'{idx:04d}_orig.png'
        if not src_path.exists():
            src_path = input_dir / src_name
        if not src_path.exists():
            print(f'[{i+1}/{len(samples)}] SKIP idx={idx}: no source')
            continue

        total_done = len(done_idxs) + n_new_ok
        remaining = len(samples) - total_done
        print(f'[{i+1}/{len(samples)}] idx={idx}  src={src_path.name}  '
              f'prompt={caption[:50]}...  (done={total_done}, left≈{remaining})',
              flush=True)

        success = False
        filtered = False
        attempt = 0
        while attempt < args.max_retries:
            ti = time.time()
            try:
                # 1) Upload
                server_path = upload_file(str(src_path))

                # 2) Submit infer
                event_id = call_infer(
                    server_path, caption,
                    lora=args.lora, seed=args.seed,
                    steps=args.steps, cfg=args.cfg,
                )

                # 3) Poll result
                result = poll_result(event_id, timeout=300)
                if result is None:
                    raise TimeoutError('poll timeout')

                # 4) Extract and download output image
                img_path = extract_image_path(result)
                if not img_path:
                    raise ValueError(f'no image in result: {str(result)[:200]}')

                final = out_dir / f'{idx:04d}.png'
                ok = download_image(img_path, final)
                if not ok:
                    raise ValueError(f'download failed: {img_path}')

                dt = time.time() - ti
                print(f'  -> saved ({dt:.1f}s)  {final.name}', flush=True)
                records.append({
                    'idx': idx, 'source_image': src_name,
                    'caption': caption, 'status': 'ok',
                    'runtime_sec': dt,
                })
                n_new_ok += 1
                success = True
                break

            except ContentFilterError as e:
                dt = time.time() - ti
                print(f'  FILTERED ({dt:.1f}s): {e}', flush=True)
                records.append({
                    'idx': idx, 'source_image': src_name,
                    'caption': caption, 'status': 'filtered',
                    'error': str(e),
                })
                filtered = True
                break
            except Exception as e:
                dt = time.time() - ti
                err = f'{type(e).__name__}: {e}'
                if '429' in err:
                    # 429: wait 5 min, do NOT increment attempt (unlimited)
                    print(f'  429 THROTTLE ({dt:.1f}s, wait=300s)',
                          flush=True)
                    time.sleep(300)
                    continue
                attempt += 1
                if attempt < args.max_retries:
                    print(f'  RETRY {attempt}/{args.max_retries} ({dt:.1f}s, '
                          f'wait={args.retry_delay}s): {err[:100]}', flush=True)
                    time.sleep(args.retry_delay)
                else:
                    print(f'  FAIL ({dt:.1f}s): {err[:120]}', flush=True)
                    records.append({
                        'idx': idx, 'source_image': src_name,
                        'caption': caption, 'status': 'fail',
                        'error': err,
                    })

        if filtered:
            continue
        if success and args.delay > 0:
            time.sleep(args.delay)

        # Periodic save
        if len(records) % 50 == 0 and records:
            _save_meta(meta_path, args, samples, records, time.time() - t0)

    total = time.time() - t0
    n_ok = len(done_idxs) + n_new_ok
    print(f'\n[DONE] {n_ok}/{len(samples)} in {total:.1f}s '
          f'(skip={n_skip}, new={n_new_ok})')
    _save_meta(meta_path, args, samples, records, total)
    print(f'[META] {meta_path}')


def _save_meta(meta_path, args, samples, records, total):
    n_ok = sum(1 for r in records if r.get('status') == 'ok')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'api_base': API_BASE,
            'lora': args.lora, 'steps': args.steps, 'cfg': args.cfg,
            'seed': args.seed, 'delay': args.delay,
            'captions_file': args.captions,
            'n_total': len(samples), 'n_ok': n_ok,
            'runtime_sec': total,
            'records': records,
        }, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
