"""Retry-loop \u7684 ModelScope FireRed \u4e0b\u8f7d, \u76f4\u5230\u6240\u6709\u6587\u4ef6\u6210\u529f.

ModelScope snapshot_download \u652f\u6301 resume, \u53ea\u4f1a\u91cd\u8bd5\u7f3a\u5931\u6216 partial \u6587\u4ef6.
"""
import json
import os
import sys
import time
from pathlib import Path

CACHE = '/root/autodl-tmp/ms_models'
os.makedirs(CACHE, exist_ok=True)

from modelscope import snapshot_download
import modelscope
print(f'[VER] modelscope = {modelscope.__version__}', flush=True)


def download_with_retry(model_id, kind='base', max_retry=15, sleep_between=10):
    """\u5bf9 model_id retry \u76f4\u5230\u6210\u529f, \u6216\u8d85\u8fc7 max_retry"""
    for attempt in range(1, max_retry + 1):
        print(f'\n[ATTEMPT {attempt}/{max_retry} {kind}] {model_id}', flush=True)
        t0 = time.time()
        try:
            path = snapshot_download(model_id, cache_dir=CACHE)
            dt = time.time() - t0
            print(f'[OK {kind}] in {dt:.1f}s -> {path}', flush=True)
            return path
        except Exception as e:
            dt = time.time() - t0
            print(f'[FAIL {kind}] after {dt:.1f}s: {type(e).__name__}: {e}', flush=True)
            if attempt < max_retry:
                print(f'[RETRY] sleep {sleep_between}s then retry...', flush=True)
                time.sleep(sleep_between)
    return None


# Base
print('\n========== BASE MODEL (retry loop) ==========', flush=True)
base_path = download_with_retry('FireRedTeam/FireRed-Image-Edit-1.0', kind='base', max_retry=15)
if not base_path:
    print('[CRITICAL] base failed after 15 retries', flush=True)
    sys.exit(1)

# LoRA
print('\n========== LORA (retry loop) ==========', flush=True)
lora_path = download_with_retry('FireRedTeam/FireRed-Image-Edit-1.0-Lightning', kind='lora', max_retry=10)

# Summary
print('\n========== SUMMARY ==========', flush=True)
print(f'base: {base_path}', flush=True)
print(f'lora: {lora_path}', flush=True)

import subprocess
for p in [base_path, lora_path]:
    if p and Path(p).exists():
        r = subprocess.run(['du', '-sh', p], capture_output=True, text=True)
        print(f'  size: {r.stdout.strip()}', flush=True)

paths_out = {
    'base_path': str(base_path) if base_path else None,
    'lora_path': str(lora_path) if lora_path else None,
    'downloaded_at': time.strftime('%Y-%m-%d %H:%M:%S'),
}
paths_file = Path(CACHE) / 'firered_paths.json'
with open(paths_file, 'w') as f:
    json.dump(paths_out, f, indent=2)
print(f'\n[paths] {paths_file}', flush=True)
print('[DONE]', flush=True)
