"""Hard kill + GPU clean wait for Path Y."""
import os
import sys
import time

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

    print("[BEFORE] GPU + processes")
    out, _, _ = run(cli, "nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; echo '---'; ps -ef | grep -E 'python|run_firered|run_pathY' | grep -v grep")
    print(out)

    # Kill all python / bash related processes
    print("[KILL] all firered / pathY processes ...")
    run(cli, "pkill -9 -f 'run_pathY_firered_batch' || true")
    run(cli, "pkill -9 -f 'run_firered_hf_local' || true")
    run(cli, "pkill -9 -f 'firered' || true")
    # Also any orphan python on the GPU
    out, _, _ = run(cli, "nvidia-smi --query-compute-apps=pid --format=csv,noheader")
    pids = [p.strip() for p in out.strip().splitlines() if p.strip()]
    for p in pids:
        run(cli, f"kill -9 {p} || true")
        print(f"  killed gpu pid {p}")

    # Wait for GPU to clear
    print("\n[WAIT] for GPU memory to release ...")
    for i in range(30):
        out, _, _ = run(cli, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
        used = int(out.strip().split()[0])
        if used < 500:
            print(f"  iter {i}: memory.used = {used} MiB ✓ clean")
            break
        print(f"  iter {i}: memory.used = {used} MiB, waiting ...")
        time.sleep(2)

    print("\n[AFTER] GPU + processes")
    out, _, _ = run(cli, "nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader; echo '---'; ps -ef | grep -E 'python|run_firered|run_pathY' | grep -v grep")
    print(out)

    cli.close()


if __name__ == "__main__":
    main()
