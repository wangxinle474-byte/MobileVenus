"""Preheat AutoDL for Path X v2 training.

Tasks:
  1. SFTP upload code (training/firered_baseline/, models/* imported by train_lut.py,
     tools/data/data_prep/, scripts/autodl/train_pathX_v2.sh)
  2. SFTP upload v11a backbone ckpt (35.6 MB)
  3. pip install lpips (if missing)
  4. Dry-run train_lut.py --help to verify all imports
"""
import os
import stat
import sys
import time
from pathlib import Path, PurePosixPath

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")
PY = "/root/miniconda3/bin/python"
PIP = "/root/miniconda3/bin/pip"
LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/autodl-tmp/IntelligenceCamera"


# files/dirs to sync (relative to project root)
SYNC_ITEMS = [
    # full dirs (training + tools data prep + entire models/ to keep __init__.py
    # eager imports satisfied)
    ("training/firered_baseline", "dir"),
    ("tools/data/data_prep", "dir"),
    ("models", "dir"),
    # ckpt
    ("checkpoints/lut_v11a_action_gated_context/best.pt", "file"),
]


def ssh_run(c, cmd, t=120):
    _, o, e = c.exec_command(cmd, timeout=t)
    out = o.read().decode(errors="ignore").rstrip()
    err = e.read().decode(errors="ignore").rstrip()
    return out, err


def ssh_print(c, cmd, t=120, label=None):
    if label:
        print(f"--- {label} ---")
    out, err = ssh_run(c, cmd, t)
    if out:
        print(out)
    if err:
        print("[stderr]", err[:400])
    print()


def sftp_mkdir_p(sftp, remote_dir):
    """mkdir -p equivalent."""
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def sftp_put_file(sftp, local_path: Path, remote_path: str, progress=False):
    sftp_mkdir_p(sftp, str(PurePosixPath(remote_path).parent))
    size = local_path.stat().st_size
    start = time.time()
    last_print = [0]

    def cb(transferred, _):
        now = time.time()
        if now - last_print[0] >= 2 or transferred == size:
            pct = transferred / size * 100 if size else 100
            speed = transferred / (now - start) / (1024 ** 2) if now > start else 0
            sys.stdout.write(
                f"\r    {transferred/1024**2:.1f}/{size/1024**2:.1f} MB  {pct:5.1f}%  {speed:.1f} MB/s"
            )
            sys.stdout.flush()
            last_print[0] = now

    if progress and size > 5 * 1024 * 1024:
        sftp.put(str(local_path), remote_path, callback=cb)
        print()
    else:
        sftp.put(str(local_path), remote_path)


def sftp_upload_dir(sftp, local_dir: Path, remote_dir: str):
    """Recursive dir upload (skips __pycache__ and .pyc)."""
    n_files = 0
    total_bytes = 0
    for p in local_dir.rglob("*"):
        if p.is_dir():
            continue
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        rel = p.relative_to(local_dir).as_posix()
        target = f"{remote_dir}/{rel}"
        sftp_put_file(sftp, p, target)
        n_files += 1
        total_bytes += p.stat().st_size
    return n_files, total_bytes


# bash script (Linux equivalent of train_pathX_v2.ps1)
TRAIN_SH = """#!/bin/bash
# Path X v2 retraining on AutoDL (Linux equivalent of train_pathX_v2.ps1)
set -e
cd /root/autodl-tmp/IntelligenceCamera

JSONL_V2="${JSONL_V2:-outputs/inverse_fit_pilot/pathY_2000_master/pseudo_labels.jsonl}"
BACKBONE="${BACKBONE:-checkpoints/lut_v11a_action_gated_context/best.pt}"
OUT_DIR="${OUT_DIR:-checkpoints/lut_v11a_pathX_v2_implicit_head}"
PY=/root/miniconda3/bin/python

if [ ! -f "$JSONL_V2" ]; then
    echo "[ERR] v2 jsonl not found: $JSONL_V2"
    echo "      run: $PY tools/data/data_prep/ingest_pathY_results.py"
    exit 1
fi
if [ ! -f "$BACKBONE" ]; then
    echo "[ERR] backbone ckpt not found: $BACKBONE"
    exit 1
fi

echo "=== v2 jsonl stats ==="
$PY -c "
import json
from collections import Counter
n=0; ac=Counter()
with open('$JSONL_V2', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r=json.loads(line); n+=1; ac[r.get('action','?')] += 1
print(f'total: {n}')
for a,c in sorted(ac.items()):
    print(f'  {a:12s}: {c}')
"

echo
echo "=== Path X v2 training ==="
echo "  jsonl:    $JSONL_V2"
echo "  backbone: $BACKBONE"
echo "  out_dir:  $OUT_DIR"
echo "  expected: ~3h on RTX 5090"
echo

$PY -u training/firered_baseline/train_lut.py \\
    --jsonl "$JSONL_V2" \\
    --out_dir "$OUT_DIR" \\
    --image_size 256 \\
    --batch_size 4 \\
    --epochs 40 \\
    --lr 3e-4 \\
    --lut_lr 1e-3 \\
    --weight_decay 1e-3 \\
    --dropout 0.5 \\
    --val_ratio 0.2 \\
    --seed 42 \\
    --patience 10 \\
    --l1_weight 1.0 \\
    --ssim_weight 0.5 \\
    --smooth_weight 1e-4 \\
    --mono_weight 1e-2 \\
    --coarse_weight 0.5 \\
    --param_weight 0.05 \\
    --named_curves \\
    --nc_n_colors 3 \\
    --nc_n_control_points 7 \\
    --nc_use_7d_anchor \\
    --nc_use_context \\
    --nc_action_gated_context \\
    --use_implicit_head \\
    --implicit_head_base_ch 32 \\
    --implicit_head_gate_init 1.0 \\
    --implicit_head_lr 5e-4 \\
    --backbone_ckpt "$BACKBONE" \\
    --freeze_backbone

echo
echo "=== Training complete ==="
echo "  ckpt: $OUT_DIR/best.pt"
"""


def main():
    if not PASS:
        sys.exit("AUTODL_PASS env var not set")

    print(f"[CONNECT] {USER}@{HOST}:{PORT}")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=20)
    sftp = c.open_sftp()

    # ===== [1] Sync code =====
    print("\n=== [1] Syncing code/ckpt ===")
    t0 = time.time()
    total_files = 0
    total_bytes = 0
    for rel, kind in SYNC_ITEMS:
        local = LOCAL_ROOT / rel
        remote = f"{REMOTE_ROOT}/{rel.replace(os.sep, '/')}"
        if not local.exists():
            print(f"  [SKIP] {rel} (missing locally)")
            continue
        if kind == "dir":
            print(f"  [DIR ] {rel}")
            n, b = sftp_upload_dir(sftp, local, remote)
            print(f"         {n} files, {b/1024**2:.1f} MB")
            total_files += n
            total_bytes += b
        else:
            size_mb = local.stat().st_size / 1024**2
            tag = "BIG " if size_mb > 5 else "FILE"
            print(f"  [{tag}] {rel} ({size_mb:.1f} MB)")
            sftp_put_file(sftp, local, remote, progress=True)
            total_files += 1
            total_bytes += local.stat().st_size
    print(
        f"  total: {total_files} files, {total_bytes/1024**2:.1f} MB in {time.time()-t0:.1f}s"
    )

    # ===== [2] Upload train_pathX_v2.sh =====
    print("\n=== [2] Writing train_pathX_v2.sh on AutoDL ===")
    remote_sh = f"{REMOTE_ROOT}/scripts/autodl/train_pathX_v2.sh"
    sftp_mkdir_p(sftp, str(Path(remote_sh).parent))
    with sftp.open(remote_sh, "w") as f:
        f.write(TRAIN_SH)
    sftp.chmod(remote_sh, 0o755)
    print(f"  wrote {remote_sh}")

    # ===== [3] pip install missing deps =====
    print("\n=== [3] pip install lpips (if missing) ===")
    out, _ = ssh_run(c, f"{PY} -c 'import lpips; print(lpips.__version__)'")
    if "ModuleNotFoundError" in out or not out:
        print("  installing lpips...")
        ssh_print(c, f"{PIP} install --quiet lpips 2>&1 | tail -5", t=180)
    else:
        print(f"  lpips already installed: {out}")

    # ===== [4] Verify train_lut.py imports =====
    print("\n=== [4] Verify train_lut.py --help (catches missing deps) ===")
    ssh_print(
        c,
        f"cd {REMOTE_ROOT} && {PY} training/firered_baseline/train_lut.py --help 2>&1 | head -30",
        label="train_lut.py --help (head)",
    )

    # ===== [5] Verify backbone loadable =====
    print("=== [5] Verify backbone ckpt loadable ===")
    ssh_print(
        c,
        f"{PY} -c \"import torch; ck=torch.load('{REMOTE_ROOT}/checkpoints/lut_v11a_action_gated_context/best.pt', map_location='cpu', weights_only=False); print('keys:', sorted(ck.keys())[:8]); print('val_loss:', ck.get('val_loss')); print('epoch:', ck.get('epoch')); print('state_dict keys:', len(ck['model_state_dict']))\"",
    )

    # ===== [6] Final state =====
    print("=== [6] Final state ===")
    ssh_print(c, f"ls {REMOTE_ROOT}/training/firered_baseline/ | head -20", label="firered_baseline/")
    ssh_print(
        c,
        f"ls -lh {REMOTE_ROOT}/checkpoints/lut_v11a_action_gated_context/",
        label="backbone dir",
    )
    ssh_print(c, "nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader", label="GPU")
    ssh_print(c, "df -h /root/autodl-tmp | tail -1", label="disk")

    sftp.close()
    c.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
