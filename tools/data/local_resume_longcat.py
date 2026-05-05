"""Resume LongCat-Image-Edit-Turbo \u4e0b\u8f7d (\u5e26\u81ea\u52a8\u91cd\u8bd5).

\u7f51\u7edc\u4e0d\u7a33\u5b9a\u65f6 hf-mirror \u4f1a\u62a5 SSL UNEXPECTED_EOF, \u6211\u4eec\u5728\u5916\u9762\u5957\u91cd\u8bd5\u5faa\u73af\u3002
\u6bcf\u6b21\u51fa\u9519 sleep 10s \u91cd\u65b0\u8fdb\u5165 snapshot_download (\u5b83\u4f1a\u4ece .incomplete \u7eed\u4f20)\u3002
"""
import os
import sys
import time

# \u6e05\u6389 stale \u4ee3\u7406
for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'HF_HUB_ENABLE_HF_TRANSFER']:
    os.environ.pop(k, None)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = r'E:\cache\huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = r'E:\cache\huggingface\hub'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from huggingface_hub import snapshot_download

MODEL_ID = 'meituan-longcat/LongCat-Image-Edit-Turbo'
MAX_RETRIES = 50
RETRY_SLEEP = 15


def main():
    print(f'[INFO] HF_ENDPOINT = {os.environ["HF_ENDPOINT"]}', flush=True)
    print(f'[INFO] HF_HOME     = {os.environ["HF_HOME"]}', flush=True)
    print(f'[INFO] starting resume of {MODEL_ID}', flush=True)

    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            path = snapshot_download(
                repo_id=MODEL_ID,
                cache_dir=r'E:\cache\huggingface\hub',
                allow_patterns=['*.json', '*.txt', '*.safetensors',
                                '*.model', 'tokenizer*/*'],
                max_workers=4,
            )
            dt = time.time() - t0
            print(f'[DONE] Completed attempt {attempt} in {dt/60:.1f} min -> {path}',
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
        print('[ABORT] user interrupt', flush=True)
        sys.exit(1)
