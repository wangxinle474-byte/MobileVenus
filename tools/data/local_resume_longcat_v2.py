"""LongCat \u4e0b\u8f7d v2 \u2014 \u7528 hf_hub_download per-file \u907f\u5f00 snapshot_download \u7684 TLS \u6302.

\u903b\u8f91:
  1. \u5148\u62c9 repo \u6587\u4ef6\u5217\u8868 (HfApi.list_repo_files)
  2. \u6309\u6587\u4ef6\u4e00\u4e2a\u4e00\u4e2a\u4e0b (hf_hub_download, \u6bcf\u4e2a\u80fd\u72ec\u7acb\u91cd\u8bd5)
  3. \u6bcf\u4e2a\u6587\u4ef6 TLS \u9519\u8bef \u2192 \u7b49 10s \u91cd\u8bd5

\u6bd4 snapshot_download \u66f4\u7a33\u5b9a, \u56e0\u4e3a\u5355\u4e2a TLS \u6302\u4e0d\u4f1a\u6253\u6b7b\u6574\u4e2a\u6d41\u7a0b\u3002
"""
import os
import sys
import time

for k in ['HTTP_PROXY', 'HTTPS_PROXY', 'HF_HUB_ENABLE_HF_TRANSFER']:
    os.environ.pop(k, None)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = r'E:\cache\huggingface'
os.environ['HUGGINGFACE_HUB_CACHE'] = r'E:\cache\huggingface\hub'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import filter_repo_objects

MODEL_ID = 'meituan-longcat/LongCat-Image-Edit-Turbo'
CACHE_DIR = r'E:\cache\huggingface\hub'
ALLOW = ['*.json', '*.txt', '*.safetensors', '*.model', 'tokenizer*/*']
PER_FILE_RETRY = 30


def list_files():
    api = HfApi(endpoint=os.environ['HF_ENDPOINT'])
    for attempt in range(10):
        try:
            files = api.list_repo_files(MODEL_ID)
            return files
        except Exception as e:
            print(f'  [list retry {attempt+1}] {type(e).__name__}: {str(e)[:100]}',
                  flush=True)
            time.sleep(5)
    raise RuntimeError('list_repo_files failed after 10 attempts')


def matches_pattern(filename, patterns):
    import fnmatch
    return any(fnmatch.fnmatch(filename, p) for p in patterns)


def download_one(filename):
    for attempt in range(1, PER_FILE_RETRY + 1):
        t0 = time.time()
        try:
            path = hf_hub_download(
                repo_id=MODEL_ID,
                filename=filename,
                cache_dir=CACHE_DIR,
            )
            sz = os.path.getsize(path) / 1e6
            print(f'  OK ({sz:.0f} MB, {time.time()-t0:.0f}s): {filename}', flush=True)
            return path
        except Exception as e:
            print(f'  [{filename} retry {attempt}] {type(e).__name__}: '
                  f'{str(e)[:150]}  ({time.time()-t0:.0f}s)', flush=True)
            time.sleep(10)
    print(f'  FAIL: {filename}', flush=True)
    return None


def main():
    print(f'[INFO] HF_ENDPOINT = {os.environ["HF_ENDPOINT"]}', flush=True)
    print(f'[INFO] Model: {MODEL_ID}', flush=True)
    print(f'[INFO] Cache: {CACHE_DIR}', flush=True)
    print('[STEP 1] listing repo files...', flush=True)

    files = list_files()
    print(f'  total files in repo: {len(files)}', flush=True)

    targets = [f for f in files if matches_pattern(f, ALLOW)]
    print(f'  filtered (safetensors+configs): {len(targets)}', flush=True)

    print('\n[STEP 2] downloading files (per-file retry)...', flush=True)
    n_ok = n_fail = 0
    for i, f in enumerate(targets):
        print(f'\n[{i+1}/{len(targets)}] {f}', flush=True)
        result = download_one(f)
        if result:
            n_ok += 1
        else:
            n_fail += 1

    print(f'\n[SUMMARY] {n_ok} OK / {n_fail} fail / {len(targets)} total', flush=True)
    return 0 if n_fail == 0 else 2


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('[ABORT]', flush=True)
        sys.exit(1)
