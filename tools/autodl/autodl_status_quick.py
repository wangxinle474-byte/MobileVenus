"""Quick AutoDL status snapshot.

Usage:
    $env:AUTODL_PASS = "..."
    python tools/autodl_status_quick.py
"""
import os
import sys
import paramiko

CHECKS = [
    ("nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader",
     "GPU"),
    ("ls /root/autodl-tmp/pathY_outputs/wb/ 2>/dev/null | wc -l",
     "wb PNGs synced from local"),
    ("ls /root/autodl-tmp/pathY_outputs/wb_cooler/ 2>/dev/null | wc -l",
     "wb_cooler PNGs synced"),
    ("stat -c '%s bytes  %y' /root/autodl-tmp/pathY_viewer.html 2>/dev/null",
     "viewer HTML"),
    ("ps aux | grep -E 'firered|run_pathY|inverse_fit|jupyter' | grep -v grep",
     "Running processes"),
    ("df -h /root/autodl-tmp/ | tail -1",
     "Disk"),
]


def main():
    pwd = os.environ.get("AUTODL_PASS")
    if not pwd:
        print("ERROR: set AUTODL_PASS env var first", file=sys.stderr)
        sys.exit(2)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect("connect.bjb2.seetacloud.com", port=35345,
                username="root", password=pwd, timeout=20)
    print("=== AutoDL status snapshot ===\n")
    for cmd, label in CHECKS:
        _, stdout, _ = cli.exec_command(cmd, timeout=15)
        out = stdout.read().decode("utf-8", errors="replace").rstrip()
        print(f"-- {label}:")
        print(out if out else "  (empty)")
        print()
    cli.close()


if __name__ == "__main__":
    main()
