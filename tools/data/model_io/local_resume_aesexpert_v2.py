"""本地续传 AesExpert (qyuan/AesMMIT_LLaVA_v1.5_7b_240325) v2.

不依赖 huggingface_hub (httpx 不读 HTTP_PROXY 环境变量), 直接用
requests 走代理 (127.0.0.1:7897) 调 huggingface.co API + LFS CDN.

特性:
- 自动复用 E:\\AesExpert_HF\\.cache\\huggingface\\download\\*.incomplete
  按 blob sha256 匹配 (HF cache 文件名后缀就是 sha256)
- 单文件 Range 续传, 失败自动重试, 超时大胆重连
- 完成后自动 mv 到 final 路径
- 多次启动幂等

Usage:
    python tools/data/model_io/local_resume_aesexpert_v2.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

# 必须在 import requests 前设置代理
os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:7897')
os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:7897')

import requests  # noqa: E402

REPO = 'qyuan/AesMMIT_LLaVA_v1.5_7b_240325'
TARGET = Path(r'E:\AesExpert_HF')
CACHE_DIR = TARGET / '.cache' / 'huggingface' / 'download'
# 用 hf-mirror (huggingface.co 走代理被拦, hf-mirror 走代理通但慢)
BASE = os.environ.get('HF_BASE', 'https://hf-mirror.com')
API_URL = f'{BASE}/api/models/{REPO}?blobs=true'
RESOLVE_URL_FMT = (
    f'{BASE}/{REPO}/resolve/main/{{path}}'
)

# 代理 + hf-mirror 不稳, 大超时 + 多重试
CONNECT_TIMEOUT = 60
READ_TIMEOUT = 180
MAX_RETRIES = 15
CHUNK = 1 << 20  # 1 MiB


def log(msg: str) -> None:
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def fetch_file_list() -> list[dict]:
    """[{path, size, blob_sha256 or None}, ...]"""
    log(f'GET {API_URL}')
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(API_URL,
                             timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            r.raise_for_status()
            info = r.json()
            break
        except Exception as e:
            log(f'  api attempt {attempt + 1}/{MAX_RETRIES} '
                f'{type(e).__name__}: {str(e)[:120]}')
            time.sleep(min(2 ** attempt, 30))
    else:
        raise RuntimeError('Failed to fetch file list after retries')

    out = []
    for s in info['siblings']:
        out.append({
            'path': s['rfilename'],
            'size': s.get('size') or 0,
            'sha256': (s.get('lfs') or {}).get('sha256'),
        })
    log(f'  got {len(out)} files, total '
        f'{sum(f["size"] for f in out) / 1024**3:.2f} GB')
    return out


def find_existing_incomplete(sha256: str | None) -> Path | None:
    """从 .cache/huggingface/download/ 找 .incomplete 文件 (sha256 命名后缀)."""
    if not sha256 or not CACHE_DIR.exists():
        return None
    for p in CACHE_DIR.glob(f'*.{sha256}.incomplete'):
        return p
    return None


def download_one(file_info: dict) -> bool:
    """下载/续传一个文件. 返回 True = 完成, False = 失败."""
    path = file_info['path']
    size = file_info['size']
    sha256 = file_info['sha256']
    final = TARGET / path
    final.parent.mkdir(parents=True, exist_ok=True)

    # 已完成?
    if final.exists() and final.stat().st_size == size:
        log(f'  SKIP (done) {path}  {size / 1024**3:.3f} GB')
        return True

    # 找现有 .incomplete (LFS 用 sha256 关联; 小文件没有 lfs sha)
    inc_path = find_existing_incomplete(sha256)
    if inc_path is None and sha256:
        inc_path = (CACHE_DIR / f'_v2.{sha256}.incomplete')
        inc_path.parent.mkdir(parents=True, exist_ok=True)
    elif inc_path is None:
        # 小文件: 直接写到 final
        inc_path = final.with_suffix(final.suffix + '.tmp')

    have = inc_path.stat().st_size if inc_path.exists() else 0
    log(f'  GET {path}  {size / 1024**3:.3f} GB  '
        f'(have {have / 1024**3:.3f} GB, '
        f'remain {(size - have) / 1024**3:.3f} GB)')

    url = RESOLVE_URL_FMT.format(path=path)
    headers = {}
    if have > 0:
        headers['Range'] = f'bytes={have}-'
        mode = 'ab'
    else:
        mode = 'wb'

    for attempt in range(MAX_RETRIES):
        try:
            with requests.get(url,
                              headers=headers,
                              stream=True,
                              timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                              allow_redirects=True) as r:
                if r.status_code == 416:  # Range not satisfiable
                    log(f'    -> 416 (already complete?), checking...')
                    if have == size:
                        break
                    raise RuntimeError(f'416 but have={have} size={size}')
                if r.status_code not in (200, 206):
                    raise RuntimeError(
                        f'HTTP {r.status_code}: {r.text[:200]}')
                last_log = time.time()
                last_have = have
                with open(inc_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        f.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if now - last_log > 5.0:
                            speed = (have - last_have) / (now - last_log)
                            pct = have / size * 100 if size else 0
                            eta = ((size - have) / speed
                                   if speed > 0 else float('inf'))
                            log(f'    {pct:5.1f}%  '
                                f'{have / 1024**3:6.3f}/'
                                f'{size / 1024**3:6.3f} GB  '
                                f'{speed / 1024**2:5.2f} MB/s  '
                                f'ETA {eta / 60:5.1f} min')
                            last_log = now
                            last_have = have
            break  # success
        except Exception as e:
            have = inc_path.stat().st_size if inc_path.exists() else 0
            log(f'    attempt {attempt + 1}/{MAX_RETRIES} '
                f'{type(e).__name__}: {str(e)[:120]}  have={have}')
            time.sleep(min(2 ** attempt, 30))
            mode = 'ab'  # 重试一定续传
            headers = {'Range': f'bytes={have}-'} if have > 0 else {}
    else:
        log(f'  FAIL {path} after {MAX_RETRIES} retries')
        return False

    # 校验
    final_size = inc_path.stat().st_size
    if final_size != size:
        log(f'  SIZE MISMATCH {path}: got {final_size} expect {size}')
        return False

    # mv to final
    if final.exists():
        final.unlink()
    shutil.move(str(inc_path), str(final))
    log(f'  OK {path}  {final_size / 1024**3:.3f} GB')
    return True


def main() -> int:
    log(f'target: {TARGET}')
    log(f'proxy:  {os.environ.get("HTTPS_PROXY")}')
    files = fetch_file_list()

    # 大文件先下 (失败可早发现)
    files.sort(key=lambda f: -f['size'])

    n_ok = 0
    n_fail = 0
    for i, f in enumerate(files, 1):
        log(f'[{i}/{len(files)}] {f["path"]}  ({f["size"] / 1024**3:.3f} GB)')
        if download_one(f):
            n_ok += 1
        else:
            n_fail += 1
    log(f'DONE  ok={n_ok}  fail={n_fail}  total={len(files)}')
    return 0 if n_fail == 0 else 2


if __name__ == '__main__':
    sys.exit(main())
