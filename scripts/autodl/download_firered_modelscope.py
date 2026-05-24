"""\u4ece ModelScope \u4e0b\u8f7d FireRed-Image-Edit-1.0 base + Lightning LoRA.

ModelScope \u56fd\u5185\u5e26\u5bbd\u901a\u5e38 30-100MB/s, \u6bd4 hf-mirror \u5feb\u5f88\u591a.

\u8f93\u51fa:
  /root/autodl-tmp/ms_models/FireRedTeam/FireRed-Image-Edit-1.0/         (base)
  /root/autodl-tmp/ms_models/FireRedTeam/FireRed-Image-Edit-1.0-Lightning/ (lora)
  /root/autodl-tmp/ms_models/firered_paths.json    (\u8def\u5f84\u8bb0\u5f55, inference \u7528)
"""
import json
import os
import sys
import time
from pathlib import Path

CACHE = '/root/autodl-tmp/ms_models'
os.makedirs(CACHE, exist_ok=True)

# \u4f7f\u7528 ModelScope \u5b98\u65b9 endpoint (\u9ed8\u8ba4\u5c31\u662f\u56fd\u5185)
print(f'[ENV] MODELSCOPE_CACHE = {os.environ.get("MODELSCOPE_CACHE", "(default)")}', flush=True)
print(f'[CACHE] cache_dir = {CACHE}', flush=True)

from modelscope import snapshot_download
import modelscope
print(f'[VER] modelscope = {modelscope.__version__}', flush=True)


def try_download(candidates, kind):
    """\u4f9d\u6b21\u5c1d\u8bd5 candidates \u4e2d\u7684 model_id, \u8fd4\u56de\u6210\u529f\u8def\u5f84"""
    for cid in candidates:
        print(f'\n[TRY {kind}] {cid}', flush=True)
        t0 = time.time()
        try:
            path = snapshot_download(cid, cache_dir=CACHE)
            dt = time.time() - t0
            print(f'[OK {kind}] downloaded in {dt:.1f}s ({dt/60:.1f} min)', flush=True)
            print(f'         -> {path}', flush=True)
            return path
        except Exception as e:
            print(f'[FAIL {kind}] {cid}: {type(e).__name__}: {e}', flush=True)
    return None


# Base model
base_candidates = [
    'FireRedTeam/FireRed-Image-Edit-1.0',
    'FireRedTeam/FireRed-Image-Edit',
]
print('\n========== BASE MODEL ==========', flush=True)
base_path = try_download(base_candidates, 'base')
if not base_path:
    print('[CRITICAL] cannot download base model from any candidate', flush=True)
    sys.exit(1)

# LoRA
lora_candidates = [
    'FireRedTeam/FireRed-Image-Edit-1.0-Lightning',
    'FireRedTeam/FireRed-Image-Edit-Lightning',
]
print('\n========== LORA ==========', flush=True)
lora_path = try_download(lora_candidates, 'lora')

# Summary
print('\n========== SUMMARY ==========', flush=True)
print(f'base: {base_path}', flush=True)
print(f'lora: {lora_path}', flush=True)

# size check
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
