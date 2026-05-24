"""Quick Path Y status check on AutoDL via paramiko."""
import os
import sys

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"

REMOTE_BASE = "/root/autodl-tmp"
REMOTE_LOGS = f"{REMOTE_BASE}/pathY_logs"
REMOTE_OUT = f"{REMOTE_BASE}/pathY_outputs"
MASTER_LOG = f"{REMOTE_LOGS}/_master.log"


def run(cli, cmd, timeout=30):
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

    blocks = [
        ("processes", "ps -ef | grep -E 'run_pathY_firered_batch|run_firered_hf_local' | grep -v grep"),
        ("GPU", "nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv"),
        ("python on GPU", "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv"),
        ("counts per action", (
            "for a in contrast saturation shadows highlights wb; do "
            f"n=$(ls {REMOTE_OUT}/$a/*.png 2>/dev/null | wc -l); "
            "echo \"$a: $n\"; "
            "done"
        )),
        ("master log tail", f"tail -40 {MASTER_LOG} 2>/dev/null"),
        ("contrast log tail", f"tail -10 {REMOTE_LOGS}/contrast.log 2>/dev/null"),
        ("disk", f"df -h {REMOTE_BASE} | tail -1"),
    ]
    for label, cmd in blocks:
        print(f"### {label}")
        print(f"$ {cmd}")
        out, err, _ = run(cli, cmd, timeout=20)
        if out:
            print(out.rstrip())
        if err:
            print(f"(stderr) {err.rstrip()}")
        print()
    cli.close()


if __name__ == "__main__":
    main()
