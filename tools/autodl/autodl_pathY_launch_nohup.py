"""Launch Path Y batch via nohup (tmux unavailable on AutoDL).

Usage:
    $env:AUTODL_PASS = "..."
    python tools/autodl_pathY_launch_nohup.py
"""
import os
import sys
import time

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"

REMOTE_BASE = "/root/autodl-tmp"
REMOTE_LOGS = f"{REMOTE_BASE}/pathY_logs"
REMOTE_REPO = f"{REMOTE_BASE}/IntelligenceCamera"
SCRIPT = f"{REMOTE_REPO}/scripts/autodl/run_pathY_firered_batch.sh"
PIDFILE = f"{REMOTE_LOGS}/_master.pid"
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

    # Check if previous run is still active
    out, _, _ = run(cli, f"[ -f {PIDFILE} ] && cat {PIDFILE} || echo NO_PID")
    out = out.strip()
    if out and out != "NO_PID":
        prior_pid = out
        out2, _, _ = run(cli, f"ps -p {prior_pid} >/dev/null && echo ALIVE || echo DEAD")
        if "ALIVE" in out2:
            print(f"[ABORT] previous run already active: pid={prior_pid}")
            print(f"  log:  {MASTER_LOG}")
            print(f"  to kill: ssh -p {PORT} {USER}@{HOST} 'kill {prior_pid}'")
            sys.exit(0)

    # Launch via setsid+nohup, redirect to master log, write pid
    # setsid detaches from controlling terminal so it survives SSH close
    launch = (
        f"mkdir -p {REMOTE_LOGS} && "
        f"cd {REMOTE_REPO} && "
        f"nohup setsid bash {SCRIPT} > {MASTER_LOG} 2>&1 < /dev/null & "
        f"echo $! > {PIDFILE} && "
        f"sleep 1 && cat {PIDFILE}"
    )
    print(f"[LAUNCH] {launch}\n")
    out, err, rc = run(cli, launch, timeout=30)
    if rc != 0:
        print(f"[ERROR] rc={rc}\nstdout: {out}\nstderr: {err}")
        sys.exit(1)
    pid = out.strip().splitlines()[-1]
    print(f"[OK] pid={pid}")
    print(f"  log:  {MASTER_LOG}")
    print(f"  pid:  {PIDFILE}\n")

    # Verify it's running
    time.sleep(3)
    out, _, _ = run(cli, f"ps -p {pid} -o pid=,etime=,cmd= 2>/dev/null || echo NOT_RUNNING")
    print(f"[STATUS] ps -p {pid}: {out.strip()}")
    out, _, _ = run(cli, f"head -20 {MASTER_LOG} 2>/dev/null")
    print(f"\n[LOG head]\n{out}")

    cli.close()


if __name__ == "__main__":
    main()
