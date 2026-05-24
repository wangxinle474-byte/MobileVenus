"""一键调度: 对 5 个 action 的 extend80 batch 跑 inverse_fit (Adam + heuristic init).

输出: outputs/inverse_fit_pilot/per_action_ext/{action}/  (pseudo_labels.jsonl + comparison/)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions_dir', default='data')
    ap.add_argument('--suffix', default='extend80')
    ap.add_argument('--firered_root', default='outputs/teacher_edits/fivek_per_action')
    ap.add_argument('--out_root', default='outputs/inverse_fit_pilot/per_action_ext')
    ap.add_argument('--optim', choices=['adam', 'lbfgs'], default='adam')
    ap.add_argument('--maxiter', type=int, default=200)
    ap.add_argument('--n_restarts', type=int, default=3)
    ap.add_argument('--ssim_weight', type=float, default=0.5)
    ap.add_argument('--max_size', type=int, default=512)
    ap.add_argument('--actions', nargs='*', default=ACTIONS,
                    help='只跑指定 action (默认全 5)')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    captions_dir = Path(args.captions_dir)
    firered_root = Path(args.firered_root)

    total_start = time.time()
    summary = []
    for a in args.actions:
        captions = captions_dir / f'teacher_edits_fivek_{a}_{args.suffix}.json'
        firered_dir = firered_root / a
        out_dir = out_root / a
        if not captions.exists():
            print(f'[SKIP] captions 不存在: {captions}')
            continue
        if not firered_dir.exists() or not any(firered_dir.glob('*.png')):
            print(f'[SKIP] FireRed 输出未就绪: {firered_dir}')
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            'tools/data/data_prep/inverse_fit_batch.py',
            '--captions', str(captions),
            '--target_dir', str(firered_dir),
            '--out_dir', str(out_dir),
            '--optim', args.optim,
            '--maxiter', str(args.maxiter),
            '--n_restarts', str(args.n_restarts),
            '--ssim_weight', str(args.ssim_weight),
            '--max_size', str(args.max_size),
        ]
        print(f'\n{"="*78}\n[RUN] action={a}\n{"="*78}')
        print(' '.join(cmd))
        if args.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.call(cmd)
        elapsed = time.time() - t0
        print(f'[DONE] {a}  rc={rc}  elapsed={elapsed:.0f}s')
        summary.append((a, rc, elapsed))

    total = time.time() - total_start
    print(f'\n{"="*78}\n[ALL DONE] total elapsed={total:.0f}s')
    for a, rc, e in summary:
        flag = 'OK' if rc == 0 else 'ERR'
        print(f'  {a:<11s} {flag:>3s}  ({e:.0f}s)')


if __name__ == '__main__':
    main()
