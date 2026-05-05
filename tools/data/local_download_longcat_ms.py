"""\u4ece ModelScope \u4e0b\u8f7d LongCat-Image-Edit-Turbo (29.32 GB).

\u4e3a\u4ec0\u4e48\u8d70 ModelScope \u800c\u4e0d\u8d70 hf-mirror:
  - MS \u81ea\u5e26 byte-range \u7eed\u4f20, CDN \u65ad\u8fde\u4e0d\u9519\u4f4d
  - \u963f\u91cc\u4e91 CDN \u5728\u56fd\u5185\u901f\u5ea6\u7a33\u5b9a
  - \u6587\u4ef6\u6309\u771f\u5b9e\u8def\u5f84\u653e (\u800c\u4e0d\u662f HF \u7684 blob hash), \u4fbf\u4e8e diffusers \u76f4\u63a5\u52a0\u8f7d

\u4f9d\u8d56:
  pip install modelscope

\u8f93\u51fa:
  E:\\cache\\modelscope\\LongCat-Image-Edit-Turbo\\  (\u6309\u4ed3\u5e93\u7ed3\u6784)
"""
import os
import sys
import time
from pathlib import Path

TARGET = Path(r'E:\cache\modelscope\LongCat-Image-Edit-Turbo')
MODEL_ID = 'meituan-longcat/LongCat-Image-Edit-Turbo'
MAX_RETRIES = 30
RETRY_SLEEP = 10


def main():
    print(f'[INFO] Model : {MODEL_ID}', flush=True)
    print(f'[INFO] Target: {TARGET}', flush=True)
    TARGET.parent.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print('[ERR] modelscope not installed. Run:  pip install modelscope', flush=True)
        sys.exit(1)

    for attempt in range(1, MAX_RETRIES + 1):
        t0 = time.time()
        try:
            path = snapshot_download(
                model_id=MODEL_ID,
                cache_dir=str(TARGET.parent),  # \u6839\u76ee\u5f55, MS \u4f1a\u5728\u4e0b\u9762\u5efa model_id \u5b50\u76ee\u5f55
                allow_file_pattern=[
                    '*.json', '*.txt', '*.md', '*.py',
                    '*.safetensors',
                    'scheduler/**', 'text_encoder/**', 'tokenizer/**',
                    'transformer/**', 'vae/**',
                ],
            )
            dt = time.time() - t0
            print(f'\n[DONE] Attempt {attempt} in {dt/60:.1f} min -> {path}', flush=True)

            # \u5217\u51fa\u4e0b\u597d\u7684\u6587\u4ef6\u603b\u5927\u5c0f
            total = 0
            for root, _, files in os.walk(path):
                for f in files:
                    total += (Path(root) / f).stat().st_size
            print(f'[SIZE] {total/1e9:.2f} GB', flush=True)
            return 0

        except Exception as e:
            dt = time.time() - t0
            print(f'[RETRY {attempt}/{MAX_RETRIES}] {type(e).__name__}: '
                  f'{str(e)[:200]}  ({dt:.0f}s)', flush=True)
            time.sleep(RETRY_SLEEP)

    print(f'[FAIL] Max retries {MAX_RETRIES} exceeded', flush=True)
    return 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('[ABORT]', flush=True)
        sys.exit(1)
