"""轮转脚本: 依次跑完所有 7 个 action 的 FireRed 1.1 API 数据生成.

特性:
- 每个 action 跑完后自动切下一个
- 自带 --resume，断点续传
- 自带 --delay 3s 防限流
- API 连接失败时自动等待 5 分钟重试
- 晚上挂着跑，白天查进度即可

用法:
    python scripts/local/run_firered_all_actions.py
    python scripts/local/run_firered_all_actions.py --delay 5  # 更保守
    python scripts/local/run_firered_all_actions.py --max_concurrent 2  # 2路并行
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb', 'brightness', 'clarity']

RUNNER_SCRIPT = PROJECT_ROOT / 'tools' / 'data' / 'editor_models' / 'run_firered_api.py'
DATA_DIR = PROJECT_ROOT / 'data'
INPUT_DIR = Path('E:/Data/dataset/fivek_jpeg')
OUTPUT_BASE = PROJECT_ROOT / 'outputs' / 'teacher_edits' / 'fivek_full'


def count_done(action: str) -> int:
    """Count completed PNGs for an action."""
    action_dir = OUTPUT_BASE / action
    if not action_dir.exists():
        return 0
    return len(list(action_dir.glob('*.png')))


def run_action(action: str, delay: float, max_retries_connect: int = 50,
               connect_wait: float = 300.0) -> bool:
    """Run one action to completion. Returns True if finished, False if fatal."""
    captions = DATA_DIR / f'teacher_edits_fivek_full_{action}.json'
    out_dir = OUTPUT_BASE / action

    if not captions.exists():
        print(f'[SKIP] {action}: captions file not found: {captions}')
        return True

    done = count_done(action)
    if done >= 5000:
        print(f'[DONE] {action}: already complete ({done} files)')
        return True

    print(f'\n{"="*60}')
    print(f'[START] {action}  ({done}/5000 done, resuming)')
    print(f'{"="*60}\n')

    for attempt in range(max_retries_connect):
        cmd = [
            sys.executable, '-u', str(RUNNER_SCRIPT),
            '--captions', str(captions),
            '--input_dir', str(INPUT_DIR),
            '--out_dir', str(out_dir),
            '--resume',
            '--delay', str(delay),
            '--retry_delay', '30',
            '--max_retries', '5',
        ]

        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

        # Check if it finished successfully
        new_done = count_done(action)
        if new_done >= 5000:
            print(f'[COMPLETE] {action}: {new_done} files done!')
            return True

        if result.returncode == 0:
            # Normal exit but not 5000 — might have run out of samples or partial
            print(f'[INFO] {action} exited normally with {new_done}/5000 done')
            if new_done == done:
                # No progress — API might be down
                print(f'[WAIT] No progress made. Waiting {connect_wait:.0f}s before retry '
                      f'({attempt+1}/{max_retries_connect})...')
                time.sleep(connect_wait)
            else:
                done = new_done
                continue  # Made progress, try again immediately
        else:
            # Error exit (likely connection/403 issue)
            print(f'[ERROR] {action} exited with code {result.returncode}')
            print(f'[WAIT] Waiting {connect_wait:.0f}s before retry '
                  f'({attempt+1}/{max_retries_connect})...')
            time.sleep(connect_wait)

        done = new_done

    print(f'[GIVE UP] {action} after {max_retries_connect} connection retries')
    return False


def main():
    ap = argparse.ArgumentParser(description='轮转跑 7 action FireRed 数据')
    ap.add_argument('--delay', type=float, default=3.0,
                    help='每次成功后等待秒数 (防限流, default=3)')
    ap.add_argument('--actions', nargs='+', default=ACTIONS,
                    help=f'要跑的 actions (default: all 7)')
    args = ap.parse_args()

    print(f'[CONFIG] delay={args.delay}s, actions={args.actions}')
    print(f'[CONFIG] input_dir={INPUT_DIR}')
    print(f'[CONFIG] output_base={OUTPUT_BASE}')
    print()

    # Show current progress
    print('[PROGRESS]')
    for action in args.actions:
        done = count_done(action)
        print(f'  {action:12s}: {done:4d}/5000')
    total = sum(count_done(a) for a in args.actions)
    target = len(args.actions) * 5000
    print(f'  {"TOTAL":12s}: {total:4d}/{target}')
    print()

    t0 = time.time()
    for action in args.actions:
        success = run_action(action, delay=args.delay)
        if not success:
            print(f'[WARN] {action} did not complete, moving to next action')

    total_time = time.time() - t0
    print(f'\n{"="*60}')
    print(f'[ALL DONE] Total time: {total_time/3600:.1f} hours')
    print('[FINAL PROGRESS]')
    for action in args.actions:
        done = count_done(action)
        print(f'  {action:12s}: {done:4d}/5000')
    total = sum(count_done(a) for a in args.actions)
    print(f'  {"TOTAL":12s}: {total:4d}/{target}')


if __name__ == '__main__':
    main()
