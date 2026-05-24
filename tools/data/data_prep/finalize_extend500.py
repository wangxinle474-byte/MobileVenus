"""FireRed extend 完成后, 一键执行剩余流程: inverse_fit → combine → viewer.

前置: outputs/teacher_edits/fivek_per_action/{action}/ 已含 100 张 PNG (20 原+80 新).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']


def run(cmd, desc):
    print(f'\n{"="*78}\n[STEP] {desc}\n{"="*78}')
    print(' '.join(str(c) for c in cmd))
    t0 = time.time()
    rc = subprocess.call(cmd)
    e = time.time() - t0
    flag = 'OK' if rc == 0 else 'ERR'
    print(f'[{flag}] {desc}  elapsed={e:.0f}s')
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--master_dir',
                    default='outputs/inverse_fit_pilot/fivek_500_master')
    ap.add_argument('--skip_fit', action='store_true',
                    help='假设 inverse_fit_ext 已完成, 只跑 combine+viewer')
    ap.add_argument('--actions', nargs='*', default=ACTIONS)
    args = ap.parse_args()

    py = sys.executable
    per_action_root = 'outputs/inverse_fit_pilot/per_action'
    per_action_ext_root = 'outputs/inverse_fit_pilot/per_action_ext'

    # 1) 校验 FireRed 输出完整 (每 action 期望 100 张 png)
    print('\n[CHECK] FireRed outputs:')
    ok = True
    for a in args.actions:
        d = Path(f'outputs/teacher_edits/fivek_per_action/{a}')
        n = len(list(d.glob('*.png')))
        flag = 'OK' if n >= 100 else 'WARN'
        print(f'  [{flag}] {a:<11s} {n}/100 PNGs')
        if n < 100:
            ok = False
    if not ok:
        print('\n[ABORT] FireRed extend 未完成, 请先确认 5×80 全部生成')
        return 1

    # 2) inverse_fit extend
    if not args.skip_fit:
        rc = run([py, 'tools/data/data_prep/run_inverse_fit_extend.py',
                  '--actions'] + list(args.actions),
                 'inverse_fit extend (5×80, ~30min)')
        if rc != 0:
            print('[ABORT] inverse_fit 失败')
            return rc

    # 3) combine master_500 (per_action + per_action_ext)
    rc = run([py, 'tools/data/data_prep/combine_per_action_results.py',
              '--sources', per_action_root, per_action_ext_root,
              '--out_dir', args.master_dir],
             'combine master_500')
    if rc != 0:
        return rc

    # 4) viewer
    rc = run([py, 'tools/data/data_prep/build_per_action_viewer.py',
              '--master_dir', args.master_dir,
              '--per_action_roots', per_action_root, per_action_ext_root,
              '--title', 'FiveK per-action pseudo-labels viewer (N=500)'],
             'build viewer N=500')
    if rc != 0:
        return rc

    print(f'\n[ALL DONE] master_dir = {args.master_dir}')
    print(f'  viewer: {Path(args.master_dir) / "viewer.html"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
