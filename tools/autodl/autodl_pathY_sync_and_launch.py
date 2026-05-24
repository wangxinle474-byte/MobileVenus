"""Path Y: paramiko-based sync + remote launch.

Steps:
  1. Connect to AutoDL via paramiko.
  2. Ensure remote dirs exist.
  3. Upload 5 per-action caption JSONs.
  4. Check which Path Y source images are missing on remote, upload them.
  5. Upload updated batch script + inference script.
  6. Launch the batch inside a detached tmux session.

Usage:
    $env:AUTODL_PASS = "..."
    python tools/autodl_pathY_sync_and_launch.py [--dry] [--no-launch]

The script is idempotent — safe to re-run if interrupted.
"""
from __future__ import annotations

import argparse
import os
import posixpath
import sys
import time
from pathlib import Path

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"

REMOTE_BASE = "/root/autodl-tmp"
REMOTE_FIVEK = f"{REMOTE_BASE}/fivek_jpeg"
REMOTE_CAPTIONS = f"{REMOTE_BASE}/pathY_captions"
REMOTE_OUTPUTS = f"{REMOTE_BASE}/pathY_outputs"
REMOTE_LOGS = f"{REMOTE_BASE}/pathY_logs"
REMOTE_REPO = f"{REMOTE_BASE}/IntelligenceCamera"

PROJECT = Path(r"E:\智能相机\Venus_CVPR2026-main\IntelligenceCamera")
LOCAL_FIVEK = Path(r"E:\Data\dataset\fivek_jpeg")
SOURCE_LIST = PROJECT / "data" / "teacher_edits_fivek_pathY_300_sources.txt"
CAPTIONS_DIR = PROJECT / "data" / "pathY_captions"

UPLOAD_FILES = [
    # (local_abs, remote_abs)
    (PROJECT / "tools" / "data" / "editor_models" / "run_firered_hf_local.py",
     f"{REMOTE_REPO}/tools/data/editor_models/run_firered_hf_local.py"),
    (PROJECT / "scripts" / "autodl" / "run_pathY_firered_batch.sh",
     f"{REMOTE_REPO}/scripts/autodl/run_pathY_firered_batch.sh"),
]

TMUX_SESSION = "pathY"
LAUNCH_CMD = (
    f"tmux new-session -d -s {TMUX_SESSION} "
    f"'bash {REMOTE_REPO}/scripts/autodl/run_pathY_firered_batch.sh "
    f"2>&1 | tee {REMOTE_LOGS}/_master.log'"
)


def run(cli: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[str, str, int]:
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    rc = stdout.channel.recv_exit_status()
    return out, err, rc


def upload(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    """Upload one file with mtime preserved."""
    sftp.put(str(local), remote)


def ensure_remote_dir(cli: paramiko.SSHClient, path: str) -> None:
    run(cli, f"mkdir -p {path}")


def remote_listdir_set(sftp: paramiko.SFTPClient, path: str) -> set[str]:
    try:
        return set(sftp.listdir(path))
    except IOError:
        return set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="don't upload or launch, just plan")
    ap.add_argument("--no-launch", action="store_true", help="sync only, skip tmux launch")
    ap.add_argument("--force-images", action="store_true",
                    help="re-upload all source images even if present remotely")
    args = ap.parse_args()

    pwd = os.environ.get("AUTODL_PASS")
    if not pwd:
        print("ERROR: set AUTODL_PASS env var first", file=sys.stderr)
        sys.exit(2)

    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[CONNECT] {USER}@{HOST}:{PORT}")
    cli.connect(HOST, port=PORT, username=USER, password=pwd, timeout=20)
    sftp = cli.open_sftp()
    print("[CONNECTED]\n")

    # 1. mkdirs
    print("[1/5] Ensuring remote directories ...")
    for d in (REMOTE_CAPTIONS, REMOTE_OUTPUTS, REMOTE_LOGS,
              f"{REMOTE_REPO}/scripts/autodl",
              f"{REMOTE_REPO}/tools/data/editor_models"):
        if args.dry:
            print(f"  (dry) mkdir -p {d}")
        else:
            ensure_remote_dir(cli, d)
    print()

    # 2. captions
    print("[2/5] Uploading per-action caption JSONs ...")
    caption_files = sorted(CAPTIONS_DIR.glob("pathY_*.json"))
    for cf in caption_files:
        remote_path = f"{REMOTE_CAPTIONS}/{cf.name}"
        size_kb = cf.stat().st_size / 1024
        if args.dry:
            print(f"  (dry) {cf.name} ({size_kb:.1f} KB)")
        else:
            upload(sftp, cf, remote_path)
            print(f"  uploaded {cf.name} ({size_kb:.1f} KB)")
    print()

    # 3. source images
    print("[3/5] Checking source images on remote ...")
    sources = [s.strip() for s in SOURCE_LIST.read_text(encoding="utf-8").splitlines() if s.strip()]
    print(f"  source list: {len(sources)} images")
    remote_set = remote_listdir_set(sftp, REMOTE_FIVEK)
    print(f"  remote fivek_jpeg: {len(remote_set)} files")
    missing = [s for s in sources if s not in remote_set] if not args.force_images else sources
    print(f"  missing on remote: {len(missing)}")
    if missing:
        print(f"  uploading {len(missing)} images (this may take a while) ...")
        t0 = time.time()
        for i, name in enumerate(missing, 1):
            local_path = LOCAL_FIVEK / name
            if not local_path.exists():
                print(f"  [WARN] local missing: {name}")
                continue
            remote_path = f"{REMOTE_FIVEK}/{name}"
            if args.dry:
                continue
            upload(sftp, local_path, remote_path)
            if i % 25 == 0 or i == len(missing):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(missing) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(missing)}] elapsed {elapsed:.0f}s, rate {rate:.1f}/s, ETA {eta:.0f}s")
    print()

    # 4. scripts
    print("[4/5] Uploading scripts ...")
    for local, remote in UPLOAD_FILES:
        if args.dry:
            print(f"  (dry) {local.name} -> {remote}")
        else:
            upload(sftp, local, remote)
            run(cli, f"chmod +x {remote}")
            print(f"  uploaded {local.name}")
    print()

    # 5. launch
    if args.no_launch or args.dry:
        print("[5/5] Skipping launch (--no-launch or --dry)")
    else:
        print("[5/5] Launching batch in tmux ...")
        # Kill any prior session with the same name to avoid double-launch
        run(cli, f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null || true")
        out, err, rc = run(cli, LAUNCH_CMD)
        if rc != 0:
            print(f"[ERROR] tmux launch rc={rc}\nstdout: {out}\nstderr: {err}")
        else:
            print(f"  tmux session '{TMUX_SESSION}' started")
            # quick status
            time.sleep(2)
            out, _, _ = run(cli, f"tmux ls 2>&1")
            print(f"  tmux ls: {out.strip()}")
            out, _, _ = run(cli, f"ls -la {REMOTE_LOGS}/")
            print(f"  log dir:\n{out}")

    sftp.close()
    cli.close()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
