"""One-shot AutoDL connection + FireRed model check (paramiko).

Reads password from env AUTODL_PASS to avoid putting it in source.

Usage:
    $env:AUTODL_PASS = "...password..."
    python tools/autodl_check.py
"""
import os
import sys

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"

CHECKS = [
    ("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader", "GPU"),
    ("ls -la /root/autodl-tmp/ms_models/FireRedTeam/FireRed-Image-Edit-1___0/ 2>&1 | head -20", "FireRed base"),
    ("ls -la /root/autodl-tmp/ms_models/FireRedTeam/FireRed-Image-Edit-1___0-Lightning/ 2>&1 | head -20", "FireRed Lightning LoRA"),
    ("ls /root/autodl-tmp/fivek_jpeg/ 2>/dev/null | wc -l", "FiveK image count on remote"),
    ("ls /root/autodl-tmp/IntelligenceCamera/ 2>/dev/null | head -20", "IntelligenceCamera repo"),
    ("python3 -c 'import diffusers, torch; print(\"diffusers\", diffusers.__version__, \"torch\", torch.__version__)'", "Python deps"),
    ("df -h /root/autodl-tmp/ | tail -1", "autodl-tmp disk space"),
    ("ls /root/autodl-tmp/pathY_outputs 2>/dev/null && echo '---' && ls /root/autodl-tmp/pathY_captions 2>/dev/null", "Path Y dirs (may be empty)"),
]


def main():
    pwd = os.environ.get("AUTODL_PASS")
    if not pwd:
        print("ERROR: set AUTODL_PASS env var first", file=sys.stderr)
        sys.exit(2)
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[CONNECT] {USER}@{HOST}:{PORT}")
    cli.connect(HOST, port=PORT, username=USER, password=pwd, timeout=20)
    print("[CONNECTED]\n")
    for cmd, label in CHECKS:
        print(f"### {label}")
        print(f"$ {cmd}")
        stdin, stdout, stderr = cli.exec_command(cmd, timeout=30)
        out = stdout.read().decode("utf-8", errors="replace").rstrip()
        err = stderr.read().decode("utf-8", errors="replace").rstrip()
        if out:
            print(out)
        if err:
            print(f"(stderr) {err}")
        print()
    cli.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
