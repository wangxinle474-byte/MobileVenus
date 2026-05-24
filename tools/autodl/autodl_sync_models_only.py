"""One-shot: sync local models/ dir to AutoDL (no ckpt re-upload)."""
import os
import sys
import time
from pathlib import Path, PurePosixPath

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")
LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/autodl-tmp/IntelligenceCamera"


def sftp_mkdir_p(sftp, remote_dir):
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def main():
    if not PASS:
        sys.exit("AUTODL_PASS not set")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=20)
    sftp = c.open_sftp()

    local_dir = LOCAL_ROOT / "models"
    remote_dir = f"{REMOTE_ROOT}/models"
    sftp_mkdir_p(sftp, remote_dir)

    n = 0
    total = 0
    t0 = time.time()
    for p in local_dir.rglob("*"):
        if p.is_dir():
            continue
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        rel = p.relative_to(local_dir).as_posix()
        target = f"{remote_dir}/{rel}"
        sftp_mkdir_p(sftp, str(PurePosixPath(target).parent))
        sftp.put(str(p), target)
        n += 1
        total += p.stat().st_size
        print(f"  [{n:2d}] {rel}  ({p.stat().st_size/1024:.1f} KB)")
    print(f"\nsynced {n} files, {total/1024:.1f} KB in {time.time()-t0:.1f}s")
    sftp.close()
    c.close()


if __name__ == "__main__":
    main()
