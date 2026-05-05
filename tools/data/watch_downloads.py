"""\u5b9e\u65f6\u76d1\u63a7 HF \u6a21\u578b\u4e0b\u8f7d\u8fdb\u5ea6 (LongCat-Turbo + Qwen3-VL-4B)\u3002

\u7528\u6cd5:
  python tools/data/watch_downloads.py                 # \u5237\u65b0 5 \u79d2
  python tools/data/watch_downloads.py --interval 2    # 2 \u79d2\u4e00\u5237
  python tools/data/watch_downloads.py --once          # \u53ea\u6253\u5370\u4e00\u6b21

Ctrl+C \u9000\u51fa\u3002
"""
import os
import sys
import time
import argparse
from pathlib import Path


CACHE = Path(r'E:\cache\huggingface')

MODELS = [
    {
        'name': 'LongCat-Image-Edit-Turbo',
        'repo_dir_name': 'models--meituan-longcat--LongCat-Image-Edit-Turbo',
        'expected_gb': 30.0,
        'log': Path(r'outputs\logs\longcat_run.log'),
    },
    {
        'name': 'Qwen3-VL-4B-Instruct',
        'repo_dir_name': 'models--Qwen--Qwen3-VL-4B-Instruct',
        'expected_gb': 8.5,
        'log': Path(r'outputs\logs\qwen3vl_run.log'),
    },
]


def dir_size_bytes(p: Path) -> int:
    if not p.exists():
        return 0
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except (FileNotFoundError, PermissionError):
                pass
    return total


def fmt_size(n):
    if n < 1024:
        return f'{n}B'
    if n < 1024**2:
        return f'{n/1024:.1f}KB'
    if n < 1024**3:
        return f'{n/1024**2:.0f}MB'
    return f'{n/1024**3:.2f}GB'


def fmt_time(sec):
    sec = int(sec)
    if sec < 60:
        return f'{sec}s'
    if sec < 3600:
        return f'{sec//60}m{sec%60:02d}s'
    return f'{sec//3600}h{(sec%3600)//60:02d}m'


def find_paths(repo_dir_name):
    """HF cache \u5728\u4e24\u4e2a\u8def\u5f84\u90fd\u53ef\u80fd\u5199: hub/ \u6216\u6839."""
    candidates = [
        CACHE / 'hub' / repo_dir_name,
        CACHE / repo_dir_name,
    ]
    return [p for p in candidates if p.exists()]


def status_of(repo_dir_name, expected_gb):
    paths = find_paths(repo_dir_name)
    if not paths:
        return {'found': False, 'gb': 0.0, 'n_blobs': 0, 'n_incomplete': 0}
    total = sum(dir_size_bytes(p) for p in paths)
    # incomplete files
    n_incomplete = 0
    n_blobs = 0
    for p in paths:
        blob_dir = p / 'blobs'
        if blob_dir.exists():
            for f in blob_dir.iterdir():
                if f.is_file():
                    n_blobs += 1
                    if f.name.endswith('.incomplete'):
                        n_incomplete += 1
    return {
        'found': True,
        'gb': total / 1024**3,
        'n_blobs': n_blobs,
        'n_incomplete': n_incomplete,
        'expected_gb': expected_gb,
    }


def last_log_line(p: Path) -> str:
    if not p.exists():
        return ''
    try:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        for line in reversed(lines):
            s = line.strip()
            if s:
                return s[:120]
    except Exception:
        pass
    return ''


def clear_and_draw(snapshots, prev_snapshots, elapsed):
    """\u753b\u4e00\u4e2a\u5237\u65b0\u53ef\u8986\u76d6\u7684\u9762\u677f."""
    # \u6e05\u5c4f + \u5149\u6807\u5230\u9876 (ANSI)
    sys.stdout.write('\x1b[2J\x1b[H')

    print('=' * 72)
    print(f'  HF \u6a21\u578b\u4e0b\u8f7d\u5b9e\u65f6\u76d1\u63a7 ({time.strftime("%H:%M:%S")})  '
          f'(Ctrl+C \u9000\u51fa)')
    print('=' * 72)

    for m, cur, prev in zip(MODELS, snapshots, prev_snapshots):
        if not cur['found']:
            print(f'\n[{m["name"]}]  \u2b1b \u672a\u5f00\u59cb / \u7f13\u5b58\u76ee\u5f55\u672a\u521b\u5efa')
            continue

        gb = cur['gb']
        pct = min(100, gb / m['expected_gb'] * 100)
        bar_w = 40
        filled = int(bar_w * pct / 100)
        bar = '=' * filled + '-' * (bar_w - filled)

        # \u901f\u7387
        rate_info = ''
        if prev and prev.get('found'):
            dgb = gb - prev['gb']
            dt = elapsed
            if dt > 0 and dgb > 0:
                mbs = dgb * 1024 / dt
                # ETA
                remain_gb = max(0, m['expected_gb'] - gb)
                eta_sec = (remain_gb * 1024) / mbs if mbs > 0 else 0
                rate_info = f'  {mbs:5.2f} MB/s  ETA {fmt_time(eta_sec)}'

        print(f'\n[{m["name"]}]')
        print(f'  [{bar}] {pct:5.1f}%')
        print(f'  {gb:.2f} / {m["expected_gb"]:.1f} GB  '
              f'({cur["n_blobs"]} blobs, {cur["n_incomplete"]} incomplete){rate_info}')
        # \u6700\u540e\u4e00\u6761\u65e5\u5fd7
        log_line = last_log_line(m['log'])
        if log_line:
            print(f'  log: {log_line[:108]}')

    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=float, default=5.0,
                    help='\u5237\u65b0\u95f4\u9694 (\u79d2)')
    ap.add_argument('--once', action='store_true',
                    help='\u53ea\u6253\u5370\u4e00\u6b21 \u4e0d\u5faa\u73af')
    args = ap.parse_args()

    # Windows \u542f\u7528 ANSI (Windows Terminal \u5df2\u9ed8\u8ba4\u652f\u6301)
    if os.name == 'nt':
        os.system('')

    prev_snapshots = [None] * len(MODELS)
    while True:
        try:
            snapshots = [status_of(m['repo_dir_name'], m['expected_gb']) for m in MODELS]
            clear_and_draw(snapshots, prev_snapshots, args.interval)
            prev_snapshots = snapshots
            if args.once:
                break
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print('\nstopped.')
            break


if __name__ == '__main__':
    main()
