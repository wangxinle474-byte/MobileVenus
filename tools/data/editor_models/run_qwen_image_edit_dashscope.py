"""Call Alibaba Cloud DashScope qwen-image-edit API to edit a batch of images.

Needs DASHSCOPE_API_KEY environment variable set.

Docs: https://help.aliyun.com/zh/model-studio/qwen-image-edit-api

Usage:
  set DASHSCOPE_API_KEY=sk-xxxxxxxx
  python tools/data/editor_models/run_qwen_image_edit_dashscope.py \\
    --captions data/teacher_edits_fivek_wb_clean_pilot3.json \\
    --out_dir outputs/teacher_edits/fivek_wb_clean_qwen_pilot
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def encode_image_to_base64(path: Path) -> str:
    """Read image, encode to data URL format accepted by DashScope."""
    with open(path, 'rb') as f:
        b = f.read()
    ext = path.suffix.lower().lstrip('.') or 'jpeg'
    if ext == 'jpg':
        ext = 'jpeg'
    return f'data:image/{ext};base64,{base64.b64encode(b).decode("utf-8")}'


def edit_one(caption: str, image_path: Path, seed: int = 42,
              model: str = 'qwen-image-edit', verbose: bool = False):
    """Call DashScope MultiModalConversation for image editing.

    Returns:
        image_url (str) if success, or None on failure.
    """
    import dashscope
    from dashscope import MultiModalConversation

    # Build multimodal message: image + text prompt
    content = [
        {'image': encode_image_to_base64(image_path)},
        {'text': caption},
    ]
    messages = [{'role': 'user', 'content': content}]

    t0 = time.time()
    try:
        resp = MultiModalConversation.call(
            model=model,
            messages=messages,
            result_format='message',
            seed=seed,
            watermark=False,
            negative_prompt='',
        )
    except Exception as e:
        return None, f'{type(e).__name__}: {e}', time.time() - t0

    dt = time.time() - t0

    if resp.status_code != 200:
        return None, f'HTTP {resp.status_code}: {resp.message}', dt

    # Parse response: output.choices[0].message.content[0].image = URL
    try:
        out = resp.output
        choice = out.choices[0]
        msg = choice.message
        # content list may have {"image": "https://..."}
        for item in msg.content:
            if 'image' in item:
                return item['image'], None, dt
        return None, f'no image in response: {msg.content}', dt
    except Exception as e:
        return None, f'parse error: {e} resp={resp}', dt


def download_image(url: str, save_path: Path):
    """Download image from URL to local path."""
    import requests
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    save_path.write_bytes(r.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions', required=True, help='caption JSON file')
    ap.add_argument('--out_dir', required=True, help='output dir for <idx:04d>.png')
    ap.add_argument('--model', default='qwen-image-edit',
                    help='DashScope model name (qwen-image-edit / qwen-image-2.0-pro)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rate_sleep', type=float, default=1.0,
                    help='sleep between samples to respect RPS')
    args = ap.parse_args()

    api_key = os.environ.get('DASHSCOPE_API_KEY', '')
    if not api_key:
        print('[ERROR] DASHSCOPE_API_KEY env var not set', file=sys.stderr)
        sys.exit(1)
    import dashscope
    dashscope.api_key = api_key

    # Load captions
    captions_path = Path(args.captions)
    if not captions_path.is_absolute():
        captions_path = PROJECT_ROOT / captions_path
    with open(captions_path, encoding='utf-8') as f:
        cfg = json.load(f)
    samples = cfg['samples']
    print(f'[INFO] {len(samples)} samples from {captions_path.name}')
    print(f'[INFO] model={args.model}  seed={args.seed}')

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    total_time = 0.0

    for i, s in enumerate(samples):
        idx = s['idx']
        orig_path = PROJECT_ROOT / s['orig_path']
        if not orig_path.exists():
            print(f'[{i+1}/{len(samples)}] idx={idx}: SKIP missing {orig_path}')
            failed += 1
            continue

        out_path = out_dir / f'{idx:04d}.png'
        if out_path.exists():
            print(f'[{i+1}/{len(samples)}] idx={idx}: SKIP already exists')
            success += 1
            continue

        print(f'[{i+1}/{len(samples)}] idx={idx}  {s["source_image"]}')
        print(f'    caption: "{s["new_caption"][:80]}"')

        url, err, dt = edit_one(s['new_caption'], orig_path,
                                  seed=args.seed, model=args.model)
        total_time += dt

        if url is None:
            print(f'    FAIL ({dt:.1f}s): {err}')
            failed += 1
            continue

        try:
            download_image(url, out_path)
            print(f'    OK ({dt:.1f}s) -> {out_path.name}')
            success += 1
        except Exception as e:
            print(f'    FAIL download: {e}')
            failed += 1

        if i < len(samples) - 1:
            time.sleep(args.rate_sleep)

    print(f'\n[DONE] success={success}/{len(samples)}  failed={failed}  '
          f'total_time={total_time:.1f}s  avg={total_time/max(1, success+failed):.1f}s/sample')


if __name__ == '__main__':
    main()
