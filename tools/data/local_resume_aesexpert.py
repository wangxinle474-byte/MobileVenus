"""本地续传 AesExpert (qyuan/AesMMIT_LLaVA_v1.5_7b_240325) 到 E:/AesExpert_HF
通过 hf-mirror.com 镜像加速国内下载.
被 scripts/local_resume_aesexpert.ps1 调用. 也可独立直跑.

历史: 早期 score_fivek_aesexpert.py 注释里写的 repo_id 是
'huang-lin/AesExpert', 但该 repo 已 401, 实际官方权重在
'qyuan/AesMMIT_LLaVA_v1.5_7b_240325' (commit 89697323..., 跟本地
.metadata 里记录的 hash 完全一致, 可断点续传).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# 在 import huggingface_hub 之前设置 endpoint, 否则不生效
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')

from huggingface_hub import snapshot_download  # noqa: E402

REPO_ID = 'qyuan/AesMMIT_LLaVA_v1.5_7b_240325'
TARGET = Path(os.environ.get('HF_TARGET', r'E:\AesExpert_HF'))
EXPECTED_GB = 14.1


def _local_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob('*'):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total / 1024 ** 3


def main() -> int:
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] HF_ENDPOINT = {os.environ["HF_ENDPOINT"]}', flush=True)
    print(f'[{ts}] target     = {TARGET}', flush=True)
    print(f'[{ts}] expected   = {EXPECTED_GB:.1f} GB', flush=True)
    before = _local_size_gb(TARGET)
    print(f'[{ts}] resume from {before:.2f} GB '
          f'({before / EXPECTED_GB * 100:.1f} %)', flush=True)

    try:
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(TARGET),
            resume_download=True,
            max_workers=4,
        )
    except Exception as e:
        ts2 = time.strftime('%Y-%m-%d %H:%M:%S')
        after = _local_size_gb(TARGET)
        print(f'[{ts2}] FAILED at {after:.2f} GB: {e}', flush=True)
        return 2

    ts2 = time.strftime('%Y-%m-%d %H:%M:%S')
    after = _local_size_gb(TARGET)
    print(f'[{ts2}] DONE  -> {after:.2f} GB '
          f'({after / EXPECTED_GB * 100:.1f} %)', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
