"""轮询等待 FireRed 完成 → 自动跑 missing 重试 + inverse_fit + combine + viewer.

Usage:
    python tools/data/data_prep/watch_and_finalize.py \
        --target_per_action 100 --poll_interval 60

设计:
1. 每 poll_interval 秒检查 outputs/teacher_edits/fivek_per_action/{action}/ 的 PNG 数
2. 5 个 action 都达到 target_per_action 后, 进入 finalize 阶段
3. finalize: check_missing → retry (最多 N 轮) → inverse_fit → combine → viewer
4. 超时 max_wait_minutes 则放弃等待 (默认 180min)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']


def count_pngs(root: Path):
    counts = {}
    for a in ACTIONS:
        d = root / a
        n = len(list(d.glob('*.png'))) if d.exists() else 0
        counts[a] = n
    return counts


def all_ready(counts: dict, target: int):
    return all(n >= target for n in counts.values())


def run_step(cmd, desc):
    print(f'\n{"="*78}\n[STEP] {desc}\n{"="*78}')
    print(' '.join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    e = time.time() - t0
    flag = 'OK' if rc == 0 else 'ERR'
    print(f'[{flag}] {desc}  elapsed={e:.0f}s', flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--firered_root',
                    default='outputs/teacher_edits/fivek_per_action')
    ap.add_argument('--target_per_action', type=int, default=100,
                    help='每 action 期望 PNG 数 (20 旧 + 80 新 = 100)')
    ap.add_argument('--poll_interval', type=int, default=60,
                    help='轮询间隔秒')
    ap.add_argument('--max_wait_minutes', type=int, default=180)
    ap.add_argument('--suffix', default='extend80')
    ap.add_argument('--max_retry_rounds', type=int, default=2)
    ap.add_argument('--master_dir',
                    default='outputs/inverse_fit_pilot/fivek_500_master')
    ap.add_argument('--skip_wait', action='store_true',
                    help='跳过轮询直接进入 finalize (假设 FireRed 已完成)')
    args = ap.parse_args()

    py = sys.executable
    firered_root = Path(args.firered_root)
    per_action_root = 'outputs/inverse_fit_pilot/per_action'
    per_action_ext_root = 'outputs/inverse_fit_pilot/per_action_ext'

    # ───── 1) 等 FireRed 完成 ─────
    # 结束条件 (任一):
    #   (a) 全部 5 action 达到 target_per_action
    #   (b) 文件计数停滞 stagnation_minutes 分钟 (FireRed 主进程结束)
    #   (c) 总等待超过 max_wait_minutes
    if not args.skip_wait:
        stagnation_minutes = 4
        print(f'[WATCH] target {args.target_per_action} PNG/action, '
              f'poll every {args.poll_interval}s, '
              f'stagnation {stagnation_minutes}min, '
              f'max wait {args.max_wait_minutes}min')
        t0 = time.time()
        last_total = -1
        last_change_t = time.time()
        while True:
            counts = count_pngs(firered_root)
            total = sum(counts.values())
            elapsed = (time.time() - t0) / 60
            since_change = (time.time() - last_change_t) / 60
            if total != last_total:
                line = '  '.join(f'{a}={n}' for a, n in counts.items())
                print(f'[t={elapsed:5.1f}min] total={total}/500  {line}', flush=True)
                last_total = total
                last_change_t = time.time()
            if all_ready(counts, args.target_per_action):
                print(f'[READY] all actions have ≥{args.target_per_action} PNGs')
                break
            if since_change >= stagnation_minutes:
                print(f'[STAGNATION] no new PNG for {since_change:.1f}min, '
                      f'assume FireRed done at total={total}')
                break
            if elapsed > args.max_wait_minutes:
                print(f'[TIMEOUT] {args.max_wait_minutes}min, proceeding')
                break
            time.sleep(args.poll_interval)

    # ───── 2) 检查 missing 并重试 (最多 N 轮) ─────
    for r in range(args.max_retry_rounds):
        print(f'\n[RETRY ROUND {r+1}/{args.max_retry_rounds}]')
        rc = run_step([py, 'tools/data/data_prep/check_missing_per_action.py',
                       '--suffix', args.suffix],
                      f'check missing (round {r+1})')
        if rc != 0:
            print('[WARN] check_missing 失败, 继续')

        # 检查是否还有 retry JSON 生成
        retry_jsons = []
        for a in ACTIONS:
            rp = Path(f'data/teacher_edits_fivek_{a}_{args.suffix}_retry.json')
            if rp.exists():
                import json
                with open(rp, encoding='utf-8') as f:
                    jd = json.load(f)
                if jd.get('samples'):
                    retry_jsons.append((a, rp))

        if not retry_jsons:
            print('[OK] no missing, skip retry')
            break

        for a, rp in retry_jsons:
            run_step([py, 'tools/data/editor_models/run_firered_online.py',
                      '--captions', str(rp),
                      '--input_dir', r'E:\Data\dataset\fivek_jpeg',
                      '--out_dir', f'outputs/teacher_edits/fivek_per_action/{a}',
                      '--lora', 'Lightning', '--seed', '42'],
                     f'retry FireRed {a} ({rp.name})')

    # ───── 3) inverse_fit extend ─────
    rc = run_step([py, 'tools/data/data_prep/run_inverse_fit_extend.py'],
                  'inverse_fit extend (5×80 Adam)')
    if rc != 0:
        print('[ABORT] inverse_fit 失败')
        return rc

    # ───── 4) combine master_500 ─────
    rc = run_step([py, 'tools/data/data_prep/combine_per_action_results.py',
                   '--sources', per_action_root, per_action_ext_root,
                   '--out_dir', args.master_dir],
                  'combine master_500')
    if rc != 0:
        return rc

    # ───── 5) viewer N=500 ─────
    rc = run_step([py, 'tools/data/data_prep/build_per_action_viewer.py',
                   '--master_dir', args.master_dir,
                   '--per_action_roots', per_action_root, per_action_ext_root,
                   '--title', 'FiveK per-action pseudo-labels viewer (N=500)'],
                  'build viewer N=500')
    if rc != 0:
        return rc

    print(f'\n{"="*78}\n[ALL DONE] master = {args.master_dir}')
    print(f'  viewer: {Path(args.master_dir) / "viewer.html"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
