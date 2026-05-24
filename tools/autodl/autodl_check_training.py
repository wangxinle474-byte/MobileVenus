"""Quick status check for AutoDL training: procs + log."""
import os, sys, paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")
REPO = "/root/autodl-tmp/IntelligenceCamera"


def run(c, cmd, t=30):
    _, o, _ = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="ignore").rstrip()


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15)

    print("=== procs ===")
    print(run(c, "pgrep -af 'python.*train_lut'") or "(none)")
    print()
    print(run(c, "pgrep -af 'bash.*train_pathX_v2'") or "(none)")
    print()
    print("=== recent logs/pathX_v2_*.log ===")
    print(run(c, f"ls -la {REPO}/logs/pathX_v2_*.log 2>/dev/null | tail -5"))
    print()
    # Latest log tail
    out = run(c, f"ls -t {REPO}/logs/pathX_v2_*.log 2>/dev/null | head -1")
    if out:
        latest = out.strip()
        print(f"=== latest log: {latest} ===")
        print(run(c, f"wc -l {latest}"))
        print()
        print(run(c, f"tail -40 {latest}", t=30))
    else:
        print("(no pathX_v2 logs yet)")
    print()
    print("=== GPU ===")
    print(run(c, "nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader"))
    print(run(c, "nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader"))

    c.close()


if __name__ == "__main__":
    main()
