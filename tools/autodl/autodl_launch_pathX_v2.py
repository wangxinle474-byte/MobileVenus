"""Launch Path X v2 training on AutoDL via SSH nohup, then tail log to verify."""
import os
import sys
import time
import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")
REPO = "/root/autodl-tmp/IntelligenceCamera"
LOG = f"{REPO}/logs/pathX_v2_$(date +%Y%m%d_%H%M%S).log"


def run(c, cmd, t=60):
    _, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="ignore").rstrip(), e.read().decode(errors="ignore").rstrip()


def main():
    if not PASS:
        sys.exit("AUTODL_PASS not set")

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15)
    print("[ssh] connected")

    # Make sure logs dir exists
    run(c, f"mkdir -p {REPO}/logs")

    # Pre-flight: confirm jsonl + backbone exist
    print("\n=== Pre-flight checks ===")
    out, _ = run(c, f"ls -la {REPO}/outputs/inverse_fit_pilot/pathY_2000_master/pseudo_labels.jsonl")
    print(f"  jsonl:    {out}")
    out, _ = run(c, f"ls -la {REPO}/checkpoints/lut_v11a_action_gated_context/best.pt")
    print(f"  backbone: {out}")
    out, _ = run(c, f"ls -la {REPO}/scripts/autodl/train_pathX_v2.sh")
    print(f"  launcher: {out}")

    # Check no existing training running (filter out the pgrep cmd itself + the bash wrapper)
    out, _ = run(c, "pgrep -af 'python.*train_lut.py' | grep -v pgrep | head -5")
    if out:
        print(f"\n[WARN] train_lut.py already running:\n  {out}")
        print("  Aborting to avoid double-launch")
        c.close()
        sys.exit(1)

    # Launch nohup
    log_name = f"pathX_v2_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path = f"{REPO}/logs/{log_name}"
    print(f"\n=== Launch ===")
    print(f"  log: {log_path}")
    cmd = (f"cd {REPO} && nohup bash scripts/autodl/train_pathX_v2.sh "
           f"> {log_path} 2>&1 < /dev/null & echo $!")
    out, err = run(c, cmd)
    print(f"  PID: {out}  err: {err}")
    pid = out.strip()

    # Wait + tail log
    print(f"\n=== Wait 60s then tail log ===")
    for i in range(6):
        time.sleep(10)
        out, _ = run(c, f"ps -p {pid} -o pid,etime,rss,comm 2>/dev/null | tail -1")
        n_lines, _ = run(c, f"wc -l < {log_path} 2>/dev/null")
        print(f"  [{(i+1)*10}s] proc: {out}  log_lines: {n_lines}")
        if not out or "<defunct>" in out:
            print("  [WARN] process died — dumping log")
            break

    # Tail log
    print(f"\n=== Log tail (60 lines) ===")
    out, _ = run(c, f"tail -n 60 {log_path}", t=30)
    print(out)

    # Final pid check
    out, _ = run(c, f"pgrep -af 'train_lut.py'")
    print(f"\n=== train_lut.py procs ===\n{out}")

    # GPU check
    print(f"\n=== GPU ===")
    out, _ = run(c, "nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader")
    print(f"  {out}")

    print(f"\n[handoff]")
    print(f"  log:  {log_path}")
    print(f"  pid:  {pid}")
    print(f"  tail live with:")
    print(f"    ssh root@{HOST} -p {PORT} 'tail -f {log_path}'")

    c.close()


if __name__ == "__main__":
    main()
