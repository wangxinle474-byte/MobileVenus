from __future__ import annotations

import argparse
import os
import posixpath
import sys
import time
from pathlib import Path

import paramiko

HOST = 'connect.bjb2.seetacloud.com'
PORT = 35345
USER = 'root'
REMOTE_BASE = '/root/autodl-tmp'
REMOTE_REPO = f'{REMOTE_BASE}/IntelligenceCamera'
REMOTE_LOGS = f'{REMOTE_BASE}/logs'
PROJECT = Path(__file__).resolve().parents[1]
LOCAL_JSONL = PROJECT / 'outputs' / 'firered_v12a_existing' / 'pseudo_labels.jsonl'
LOCAL_TEACHER = PROJECT / 'outputs' / 'teacher_edits' / 'fivek_full'
REMOTE_JSONL = f'{REMOTE_REPO}/outputs/firered_v12a_existing/pseudo_labels.jsonl'
REMOTE_TEACHER = f'{REMOTE_REPO}/outputs/teacher_edits/fivek_full'
REMOTE_TRAIN = f'{REMOTE_REPO}/training/main/train_v12a_firered.py'
REMOTE_PREP = f'{REMOTE_REPO}/tools/data/data_prep/build_firered_v12a_jsonl.py'
SESSION = 'v12a_firered'
ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb', 'brightness', 'clarity']

CODE_FILES = [
    PROJECT / 'training' / 'main' / 'train_v12a_firered.py',
    PROJECT / 'tools' / 'data' / 'data_prep' / 'build_firered_v12a_jsonl.py',
    PROJECT / 'models' / 'refinement_net_v4.py',
    PROJECT / 'models' / 'diff_isp.py',
    PROJECT / 'training' / 'fivek_8param' / 'config.py',
    PROJECT / 'training' / 'semantic_distill' / 'model.py',
    PROJECT / 'training' / 'semantic_distill' / 'config.py',
]


def run(cli: paramiko.SSHClient, cmd: str, timeout: int | None = 60) -> tuple[str, str, int]:
    _, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    rc = stdout.channel.recv_exit_status()
    return out, err, rc


def mkdir(cli: paramiko.SSHClient, path: str) -> None:
    run(cli, f"mkdir -p '{path}'")


def remote_exists(sftp: paramiko.SFTPClient, path: str) -> bool:
    try:
        sftp.stat(path)
        return True
    except IOError:
        return False


def upload_file(sftp: paramiko.SFTPClient, cli: paramiko.SSHClient, local: Path, remote: str, dry: bool) -> None:
    parent = posixpath.dirname(remote)
    if dry:
        print(f'  (dry) {local} -> {remote}')
        return
    mkdir(cli, parent)
    sftp.put(str(local), remote)


def rel_remote(local: Path) -> str:
    rel = local.relative_to(PROJECT).as_posix()
    return f'{REMOTE_REPO}/{rel}'


def connect() -> paramiko.SSHClient:
    pwd = os.environ.get('AUTODL_PWD') or os.environ.get('AUTODL_PASS')
    if not pwd:
        print('ERROR: set AUTODL_PWD or AUTODL_PASS first', file=sys.stderr)
        sys.exit(2)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f'[CONNECT] {USER}@{HOST}:{PORT}')
    cli.connect(HOST, port=PORT, username=USER, password=pwd, timeout=30, allow_agent=False, look_for_keys=False)
    return cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--no_launch', action='store_true')
    ap.add_argument('--force_png', action='store_true')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--phase1_epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--accum_steps', type=int, default=8)
    ap.add_argument('--base_ch', type=int, default=64)
    ap.add_argument('--musiq_weight', type=float, default=0.5)
    ap.add_argument('--clipiqa_weight', type=float, default=0.2)
    args = ap.parse_args()

    if not LOCAL_JSONL.exists():
        print(f'ERROR: missing {LOCAL_JSONL}', file=sys.stderr)
        sys.exit(2)

    cli = connect()
    sftp = cli.open_sftp()
    try:
        print('[1/5] Uploading code')
        for local in CODE_FILES:
            if not local.exists():
                print(f'  skip missing {local}')
                continue
            remote = rel_remote(local)
            upload_file(sftp, cli, local, remote, args.dry)
            print(f'  {local.relative_to(PROJECT).as_posix()}')

        print('[2/5] Uploading jsonl')
        upload_file(sftp, cli, LOCAL_JSONL, REMOTE_JSONL, args.dry)
        print(f'  {LOCAL_JSONL.relative_to(PROJECT).as_posix()}')

        print('[3/5] Uploading teacher PNGs')
        n_total = 0
        n_upload = 0
        t0 = time.time()
        for action in ACTIONS:
            local_dir = LOCAL_TEACHER / action
            remote_dir = f'{REMOTE_TEACHER}/{action}'
            if not args.dry:
                mkdir(cli, remote_dir)
            for png in sorted(local_dir.glob('*.png')):
                n_total += 1
                remote = f'{remote_dir}/{png.name}'
                if not args.force_png and remote_exists(sftp, remote):
                    continue
                n_upload += 1
                if not args.dry:
                    sftp.put(str(png), remote)
                if n_upload % 100 == 0:
                    dt = time.time() - t0
                    print(f'  uploaded {n_upload}/{n_total} checked, {dt:.0f}s')
        print(f'  checked={n_total}, uploaded={n_upload}')

        print('[4/5] Remote quick check')
        cmd = (
            f"cd {REMOTE_REPO} && "
            f"python - <<'PY'\n"
            f"from pathlib import Path\n"
            f"p=Path('outputs/firered_v12a_existing/pseudo_labels.jsonl')\n"
            f"print('jsonl', p.exists(), sum(1 for _ in p.open()) if p.exists() else 0)\n"
            f"for a in {ACTIONS!r}:\n"
            f"    d=Path('outputs/teacher_edits/fivek_full')/a\n"
            f"    print(a, len(list(d.glob('*.png'))))\n"
            f"PY"
        )
        if args.dry:
            print('  (dry) skip remote check')
        else:
            out, err, rc = run(cli, cmd, timeout=120)
            print(out.rstrip())
            if err:
                print(err.rstrip())
            if rc != 0:
                print(f'ERROR: remote check failed rc={rc}', file=sys.stderr)
                sys.exit(rc)

        print('[5/5] Launch')
        train_cmd = (
            f"cd {REMOTE_REPO} && "
            f"mkdir -p {REMOTE_LOGS} && "
            f"python -u {REMOTE_TRAIN} "
            f"--jsonl outputs/firered_v12a_existing/pseudo_labels.jsonl "
            f"--asset_root {REMOTE_BASE} "
            f"--orig_dir {REMOTE_BASE}/fivek_jpeg "
            f"--out_dir {REMOTE_BASE}/checkpoints/refinement_v12a_firered "
            f"--epochs {args.epochs} --phase1_epochs {args.phase1_epochs} "
            f"--batch_size {args.batch_size} --accum_steps {args.accum_steps} "
            f"--base_ch {args.base_ch} --musiq_weight {args.musiq_weight} "
            f"--clipiqa_weight {args.clipiqa_weight} "
            f"2>&1 | tee {REMOTE_LOGS}/v12a_firered.log"
        )
        launch_cmd = f"tmux kill-session -t {SESSION} 2>/dev/null || true; tmux new-session -d -s {SESSION} \"{train_cmd}\""
        if args.no_launch or args.dry:
            print(f'  launch command:\n{train_cmd}')
        else:
            out, err, rc = run(cli, launch_cmd, timeout=30)
            if rc != 0:
                print(out)
                print(err, file=sys.stderr)
                sys.exit(rc)
            out, _, _ = run(cli, 'tmux ls 2>&1', timeout=10)
            print(out.rstrip())
            print(f'  log: {REMOTE_LOGS}/v12a_firered.log')
    finally:
        sftp.close()
        cli.close()


if __name__ == '__main__':
    main()
