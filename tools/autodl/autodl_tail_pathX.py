"""Live tail of AutoDL Path X v2 training log via paramiko.

Filters to interesting lines only by default (epoch summaries, best saves, errors).
Pass --raw for full firehose.
"""
import argparse
import os
import re
import sys
import time

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")
REPO = "/root/autodl-tmp/IntelligenceCamera"

# Patterns of lines worth showing in filtered mode
INTERESTING = re.compile(
    r"(\bEp\s+\d+/\d+|best val_psnr|patience|Training complete|"
    r"\[ERR|Error|Traceback|saved|coarse=|=== |stats)",
    re.IGNORECASE,
)


def find_latest_log(c) -> str:
    _, o, _ = c.exec_command(
        f"ls -t {REPO}/logs/pathX_v2_*.log 2>/dev/null | head -1", timeout=10)
    return o.read().decode().strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="Specific log file path (default: latest pathX_v2_*.log)")
    ap.add_argument("--raw", action="store_true", help="Show all lines (no filter)")
    ap.add_argument("--interval", type=float, default=3.0, help="Poll interval seconds")
    ap.add_argument("--max_seconds", type=int, default=0,
                    help="Auto-exit after N sec (0 = forever)")
    args = ap.parse_args()

    if not PASS:
        sys.exit("AUTODL_PASS not set; do: $env:AUTODL_PASS='WNkCYK59VeTd'")

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15)

    log = args.log or find_latest_log(c)
    if not log:
        print("[no pathX_v2 log found]")
        sys.exit(1)
    print(f"[tailing] {log}", flush=True)
    print(f"[mode]    {'raw' if args.raw else 'filtered (epoch/best/err only)'}", flush=True)
    print(f"[hint]    Ctrl+C to stop", flush=True)
    print("-" * 70, flush=True)

    last_size = 0
    start = time.time()
    seen_done = False
    while True:
        try:
            sftp = c.open_sftp()
            st = sftp.stat(log)
            cur = st.st_size
            sftp.close()
        except Exception as e:
            print(f"[err stat] {e}", flush=True)
            time.sleep(args.interval)
            continue

        if cur > last_size:
            # Pull the new chunk via cat | tail
            chunk_cmd = f"tail -c +{last_size+1} {log}"
            _, o, _ = c.exec_command(chunk_cmd, timeout=30)
            data = o.read().decode(errors="ignore")
            last_size = cur
            for line in data.splitlines():
                if args.raw or INTERESTING.search(line):
                    print(line, flush=True)
                if "Training complete" in line:
                    seen_done = True

        # Termination conditions
        if seen_done:
            print("-" * 70, flush=True)
            print("[training complete -- exiting tail]", flush=True)
            break
        if args.max_seconds and (time.time() - start) > args.max_seconds:
            print("-" * 70, flush=True)
            print(f"[max_seconds {args.max_seconds} reached -- exiting tail]", flush=True)
            break
        time.sleep(args.interval)

    c.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[stopped]", flush=True)
