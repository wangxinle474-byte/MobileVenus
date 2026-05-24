"""Kill any running Path Y processes on AutoDL (paramiko)."""
import os
import sys

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"


def run(cli, cmd, timeout=20):
    _, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return out, err, rc


def main():
    pwd = os.environ.get("AUTODL_PASS")
    if not pwd:
        print("ERROR: set AUTODL_PASS env var first", file=sys.stderr)
        sys.exit(2)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=pwd, timeout=20)

    print("[before]")
    out, _, _ = run(cli, "ps -ef | grep -E 'run_pathY_firered_batch|run_firered_hf_local' | grep -v grep")
    print(out.rstrip() or "(none)")
    print()

    # Find python + bash pids, kill them
    out, _, _ = run(
        cli,
        "pkill -9 -f 'run_pathY_firered_batch' ; "
        "pkill -9 -f 'run_firered_hf_local' ; "
        "sleep 2 ; "
        "ps -ef | grep -E 'run_pathY_firered_batch|run_firered_hf_local' | grep -v grep"
    )
    print("[after]")
    print(out.rstrip() or "(none)")

    cli.close()


if __name__ == "__main__":
    main()
