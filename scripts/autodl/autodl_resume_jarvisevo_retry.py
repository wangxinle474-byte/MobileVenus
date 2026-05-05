"""Resume JarvisEvo \u4e0b\u8f7d\u7684\u6700\u540e\u4e00\u4e2a shard (\u5e26\u91cd\u8bd5\u5faa\u73af).

\u4e4b\u524d hf-mirror \u53cd\u590d\u5361\u6b7b, \u7528\u91cd\u8bd5\u673a\u5236\u786e\u4fdd\u6700\u7ec8\u5b8c\u6210\u3002
\u8fd9\u4e2a\u811a\u672c\u7531 AutoDL \u8fd0\u884c\u3002
"""
import os
import sys
import time

# \u6e05\u6389\u4e0d\u7a33\u5b9a\u7684\u73af\u5883\u53d8\u91cf
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'HF_HUB_ENABLE_HF_TRANSFER']:
    os.environ.pop(k, None)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from huggingface_hub import snapshot_download

MODEL_ID = 'JarvisEvo/JarvisEvo'
TARGET_DIR = '/root/autodl-tmp/checkpoints/pretrained/JarvisEvo'
MAX_RETRIES = 60
RETRY_SLEEP = 10


def main():
    print(f'[INFO] HF_ENDPOINT = {os.environ["HF_ENDPOINT"]}', flush=True)
    print(f'[INFO] Target: {TARGET_DIR}', flush=True)

    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            path = snapshot_download(
                repo_id=MODEL_ID,
                local_dir=TARGET_DIR,
                max_workers=4,   # \u964d\u4f4e\u5e76\u53d1 (8 -> 4) \u9632 hf-mirror \u62d2\u8fde
            )
            dt = time.time() - t0
            print(f'[DONE] Attempt {attempt} completed in {dt/60:.1f} min -> {path}',
                  flush=True)
            return 0
        except Exception as e:
            dt = time.time() - t0
            print(f'[RETRY {attempt}/{MAX_RETRIES}] {type(e).__name__}: '
                  f'{str(e)[:200]}  (after {dt:.0f}s)', flush=True)
            time.sleep(RETRY_SLEEP)

    print(f'[FAIL] Max retries {MAX_RETRIES} exceeded', flush=True)
    return 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('[ABORT]', flush=True)
        sys.exit(1)
