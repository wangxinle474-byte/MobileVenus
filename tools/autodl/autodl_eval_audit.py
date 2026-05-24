"""Audit AutoDL: list ckpts + available eval jsonls before running eval_track2."""
import os, sys, paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")
REPO = "/root/autodl-tmp/IntelligenceCamera"


def run(c, cmd, t=60):
    _, o, _ = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="ignore").rstrip()


def main():
    if not PASS:
        sys.exit("AUTODL_PASS not set")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15)

    print("=== Checkpoints ===")
    print(run(c, f"find {REPO}/checkpoints -name 'best.pt' -mtime -90 "
              f"| xargs -I {{}} ls -la {{}} 2>/dev/null | sort"))

    print()
    print("=== Eval jsonls in outputs/ ===")
    print(run(c, f"find {REPO}/outputs -name 'pseudo_labels.jsonl' -mtime -180 "
              f"-exec wc -l {{}} \\; 2>/dev/null"))

    print()
    print("=== eval_track2.py present ===")
    print(run(c, f"ls -la {REPO}/tools/eval_track2.py"))

    print()
    print("=== existing track2_eval results ===")
    print(run(c, f"find {REPO}/checkpoints -name 'track2_eval*.json' -exec ls -la {{}} \\;"))

    c.close()


if __name__ == "__main__":
    main()
