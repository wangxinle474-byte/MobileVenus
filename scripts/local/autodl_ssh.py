"""AutoDL SSH/SCP \u5305\u88c5\u811a\u672c (paramiko, Windows OpenSSH \u4e0d\u652f\u6301\u547d\u4ee4\u884c\u4f20\u5bc6\u7801\u7684\u66ff\u4ee3\u65b9\u6848).

\u5bc6\u7801\u4ece\u73af\u5883\u53d8\u91cf `AUTODL_PWD` \u8bfb\u53d6, \u4e0d\u5165\u53c2\u4e0d\u5165\u6587\u4ef6.

\u7528\u6cd5:
  $env:AUTODL_PWD = '<password>'

  # \u8fdc\u7a0b\u6267\u884c\u547d\u4ee4 (stdout/stderr \u7d2f\u52a0\u8f93\u51fa)
  python scripts/local/autodl_ssh.py exec "nvidia-smi" \\
      --host root@connect.bjb2.seetacloud.com --port 35345

  # \u4e0a\u4f20\u5355\u4e2a\u6587\u4ef6 (\u4e0d\u9012\u5f52)
  python scripts/local/autodl_ssh.py upload local.py /root/autodl-tmp/dst.py \\
      --host root@connect.bjb2.seetacloud.com --port 35345

  # \u4e0b\u8f7d\u5355\u4e2a\u6587\u4ef6
  python scripts/local/autodl_ssh.py download /root/autodl-tmp/out.png ./out.png \\
      --host root@connect.bjb2.seetacloud.com --port 35345
"""
import argparse
import os
import sys
from pathlib import Path

import paramiko


def parse_host(host_arg):
    if '@' in host_arg:
        user, host = host_arg.split('@', 1)
    else:
        user, host = 'root', host_arg
    return user, host


def get_password():
    pwd = os.environ.get('AUTODL_PWD')
    if not pwd:
        print('[ERROR] AUTODL_PWD environment variable not set', file=sys.stderr)
        sys.exit(2)
    return pwd


def connect(host_arg, port):
    user, host = parse_host(host_arg)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f'[SSH] connecting to {user}@{host}:{port}', file=sys.stderr, flush=True)
    client.connect(host, port=port, username=user, password=get_password(),
                   timeout=30, allow_agent=False, look_for_keys=False)
    return client


def cmd_exec(args):
    client = connect(args.host, args.port)
    try:
        stdin, stdout, stderr = client.exec_command(args.command, get_pty=False, timeout=args.timeout)
        # \u5b9e\u65f6\u8f93\u51fa stdout
        for line in iter(stdout.readline, ''):
            print(line, end='')
        # stderr \u4e00\u6b21\u6027\u8f93\u51fa
        err = stderr.read().decode('utf-8', errors='replace')
        if err:
            print(err, end='', file=sys.stderr)
        rc = stdout.channel.recv_exit_status()
        sys.exit(rc)
    finally:
        client.close()


def cmd_upload(args):
    local = Path(args.local)
    if not local.exists():
        print(f'[ERROR] local path not found: {local}', file=sys.stderr)
        sys.exit(2)
    if not local.is_file():
        print(f'[ERROR] only single-file upload supported (got dir: {local})', file=sys.stderr)
        sys.exit(2)
    client = connect(args.host, args.port)
    try:
        sftp = client.open_sftp()
        remote = args.remote
        # \u5982\u679c remote \u4ee5 / \u7ed3\u5c3e \u89c6\u4e3a\u76ee\u5f55, \u9644\u52a0 filename
        if remote.endswith('/'):
            remote = remote + local.name
        # \u786e\u4fdd\u8fdc\u7a0b\u7236\u76ee\u5f55\u5b58\u5728
        remote_parent = remote.rsplit('/', 1)[0] or '/'
        try:
            sftp.stat(remote_parent)
        except FileNotFoundError:
            print(f'[INFO] creating remote parent dir: {remote_parent}', file=sys.stderr)
            # \u9012\u5f52 mkdir
            parts = remote_parent.strip('/').split('/')
            acc = ''
            for p in parts:
                acc += '/' + p
                try:
                    sftp.stat(acc)
                except FileNotFoundError:
                    sftp.mkdir(acc)
        size = local.stat().st_size
        print(f'[UPLOAD] {local} ({size} bytes) -> {remote}', file=sys.stderr, flush=True)
        sftp.put(str(local), remote)
        print('  OK', file=sys.stderr)
        sftp.close()
    finally:
        client.close()


def cmd_download(args):
    client = connect(args.host, args.port)
    try:
        sftp = client.open_sftp()
        local = Path(args.local)
        # \u5982\u679c local \u662f\u73b0\u5b58\u76ee\u5f55, \u4f7f\u7528 remote \u7684\u6587\u4ef6\u540d
        if local.is_dir():
            local = local / Path(args.remote).name
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f'[DOWNLOAD] {args.remote} -> {local}', file=sys.stderr, flush=True)
        sftp.get(args.remote, str(local))
        size = local.stat().st_size
        print(f'  OK ({size} bytes)', file=sys.stderr)
        sftp.close()
    finally:
        client.close()


def add_conn_args(p):
    p.add_argument('--host', required=True, help='user@host or just host (default user=root)')
    p.add_argument('--port', type=int, required=True)


def main():
    ap = argparse.ArgumentParser(description='AutoDL SSH/SCP wrapper (paramiko)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_exec = sub.add_parser('exec', help='Run a shell command on remote')
    p_exec.add_argument('command', help='Shell command (use quotes for spaces)')
    p_exec.add_argument('--timeout', type=float, default=None,
                        help='Command timeout in seconds (default: no timeout)')
    add_conn_args(p_exec)
    p_exec.set_defaults(func=cmd_exec)

    p_up = sub.add_parser('upload', help='Upload a single file to remote')
    p_up.add_argument('local', help='Local file path')
    p_up.add_argument('remote', help='Remote target path (dir if ends with /)')
    add_conn_args(p_up)
    p_up.set_defaults(func=cmd_upload)

    p_dn = sub.add_parser('download', help='Download a single remote file')
    p_dn.add_argument('remote', help='Remote file path')
    p_dn.add_argument('local', help='Local target path (dir or file)')
    add_conn_args(p_dn)
    p_dn.set_defaults(func=cmd_download)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
