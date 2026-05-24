"""Path Y live progress watcher.

Refreshes every REFRESH_SEC, prints concise status:
- per-action sample count
- current sample being processed + step (from log tail)
- GPU utilization
- rolling sample/min rate + ETA

Usage:
    $env:AUTODL_PASS = "..."
    python tools/autodl_pathY_watch.py
    # Ctrl+C to stop
"""
from __future__ import annotations

import os
import re
import sys
import time
from collections import deque
from datetime import datetime, timedelta

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"

REMOTE_BASE = "/root/autodl-tmp"
REMOTE_LOGS = f"{REMOTE_BASE}/pathY_logs"
REMOTE_OUT = f"{REMOTE_BASE}/pathY_outputs"
MASTER_LOG = f"{REMOTE_LOGS}/_master.log"

ACTIONS = ["wb", "highlights", "saturation", "shadows", "contrast"]
ACTION_N = 300  # samples per action
TOTAL = ACTION_N * len(ACTIONS)

REFRESH_SEC = 30


def run(cli, cmd, timeout=15):
    _, stdout, _ = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    return out.rstrip()


def parse_latest_sample(log_tail: str) -> tuple[str, str, str]:
    """Parse last '[N/300] idx=...' line + step progress.

    Returns (sample_progress, step_progress, current_action).
    """
    sample_line = ""
    step_line = ""
    for line in reversed(log_tail.splitlines()):
        if not step_line and ("it/s]" in line or "s/it]" in line) and "%|" in line:
            step_line = line.strip()
        if not sample_line and re.match(r"\[\d+/\d+\]", line.strip()):
            sample_line = line.strip()
        if sample_line and step_line:
            break

    # Detect current action from latest [START] block
    current_action = ""
    for line in reversed(log_tail.splitlines()):
        m = re.match(r"\[START\] (\w+):", line.strip())
        if m:
            current_action = m.group(1)
            break

    return sample_line, step_line, current_action


def fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds > 86400 * 7:
        return "?"
    td = timedelta(seconds=int(seconds))
    return str(td)


def main():
    pwd = os.environ.get("AUTODL_PASS")
    if not pwd:
        print("ERROR: set AUTODL_PASS env var first", file=sys.stderr)
        sys.exit(2)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=pwd, timeout=20)
    print(f"[CONNECTED] {USER}@{HOST}:{PORT}")
    print(f"[REFRESH] every {REFRESH_SEC}s — Ctrl+C to stop\n")

    history: deque[tuple[float, int]] = deque(maxlen=20)  # (timestamp, total_done)

    try:
        while True:
            tnow = time.time()
            ts = datetime.now().strftime("%H:%M:%S")

            # per-action counts
            counts_cmd = " ; ".join(
                [f"find {REMOTE_OUT}/{a} -maxdepth 1 -name '*.png' -type f 2>/dev/null | wc -l"
                 for a in ACTIONS]
            )
            counts_out = run(cli, counts_cmd, timeout=10)
            counts = []
            for line in counts_out.splitlines():
                try:
                    counts.append(int(line.strip()))
                except ValueError:
                    counts.append(0)
            while len(counts) < len(ACTIONS):
                counts.append(0)
            total_done = sum(counts)

            # process alive?
            proc_out = run(cli, "pgrep -f run_firered_hf_local || echo NONE", timeout=10)
            alive = proc_out.strip() != "NONE" and proc_out.strip() != ""

            # GPU
            gpu_out = run(
                cli,
                "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits",
                timeout=10,
            )
            gpu_used, gpu_util = "?", "?"
            if gpu_out:
                parts = [p.strip() for p in gpu_out.split(",")]
                if len(parts) >= 2:
                    gpu_used, gpu_util = parts[0], parts[1]

            # log tail
            log_tail = run(cli, f"tail -30 {MASTER_LOG} 2>/dev/null", timeout=10)
            sample_line, step_line, current_action = parse_latest_sample(log_tail)

            # rolling rate
            history.append((tnow, total_done))
            rate_str, eta_str = "—", "—"
            if len(history) >= 2:
                dt = history[-1][0] - history[0][0]
                dn = history[-1][1] - history[0][1]
                if dt > 0 and dn > 0:
                    rate = dn / dt * 60.0  # samples per minute
                    remaining = TOTAL - total_done
                    eta_sec = remaining / (dn / dt) if dn > 0 else 0
                    rate_str = f"{rate:.2f} smp/min"
                    eta_str = fmt_eta(eta_sec)

            # render
            print(f"=== {ts}  pid_alive={alive}  GPU {gpu_used} MiB / util {gpu_util}% ===")
            print(f"  total: {total_done}/{TOTAL}  ({100.0*total_done/TOTAL:5.1f}%)  rate={rate_str}  eta_all={eta_str}")
            per_action_str = "  | ".join(
                f"{'*' if current_action == a else ' '}{a}:{c:>3}/{ACTION_N}"
                for a, c in zip(ACTIONS, counts)
            )
            print(f"  {per_action_str}")
            if sample_line:
                print(f"  cur:  {sample_line[:120]}")
            if step_line:
                print(f"  step: {step_line[:120]}")
            print()

            time.sleep(REFRESH_SEC)
    except KeyboardInterrupt:
        print("\n[STOPPED]")
    finally:
        cli.close()


if __name__ == "__main__":
    main()
