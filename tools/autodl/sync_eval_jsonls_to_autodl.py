r"""Sync firered_pseudo (fivek_500_master) eval jsonl to AutoDL with path rewriting.

Rewrites:
  - orig_path:   E:\Data\dataset\fivek_jpeg\X.jpg  -> /root/autodl-tmp/fivek_jpeg/X.jpg
  - target_path: outputs\teacher_edits\...\NNNN.png -> outputs/teacher_edits/.../NNNN.png
                 (relative paths get forward slashes; train_lut.py prepends PROJECT_ROOT)

Pre-flight: verifies target dirs exist on AutoDL (forward-slash path) and
counts how many target PNGs would resolve. Then SFTP uploads the rewritten
jsonl to {REPO}/outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl.

Also handles clean_expertC and mmart_real_lr if --include flag is given,
but defaults to firered_pseudo only (the most useful for WB comparison).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path, PurePosixPath

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
REPO = "/root/autodl-tmp/IntelligenceCamera"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIVEK_AUTODL = "/root/autodl-tmp/fivek_jpeg"
FIVEK_EXPERT_C_AUTODL = "/root/autodl-tmp/fivek_expert_c"

EVAL_JSONLS = {
    "firered_pseudo": {
        "local": PROJECT_ROOT / "outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl",
        "remote": f"{REPO}/outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl",
    },
    "clean_expertC": {
        "local": PROJECT_ROOT / "outputs/fivek_expert_c_master/pseudo_labels.jsonl",
        "remote": f"{REPO}/outputs/fivek_expert_c_master/pseudo_labels.jsonl",
    },
    "mmart_real_lr": {
        "local": PROJECT_ROOT / "outputs/mmart_pseudo_labels/v1_250/pseudo_labels.jsonl",
        "remote": f"{REPO}/outputs/mmart_pseudo_labels/v1_250/pseudo_labels.jsonl",
    },
}


def rewrite_record(r: dict) -> dict:
    """Rewrite orig_path + target_path for AutoDL compatibility."""
    op = r.get("orig_path", "").replace("\\", "/")
    # E:/Data/dataset/fivek_jpeg/X.jpg -> /root/autodl-tmp/fivek_jpeg/X.jpg
    if "fivek_jpeg" in op:
        fname = op.rsplit("/", 1)[-1]
        op = f"{FIVEK_AUTODL}/{fname}"
    # E:/MMArt_PPR10k/global/X.jpg -> /root/autodl-tmp/MMArt_PPR10k/global/X.jpg
    elif "MMArt_PPR10k" in op:
        rel = op.split("MMArt_PPR10k", 1)[1].lstrip("/")
        op = f"/root/autodl-tmp/MMArt_PPR10k/{rel}"
    r["orig_path"] = op

    tp = r.get("target_path", "").replace("\\", "/")
    if "MMArt_PPR10k" in tp:
        rel = tp.split("MMArt_PPR10k", 1)[1].lstrip("/")
        tp = f"/root/autodl-tmp/MMArt_PPR10k/{rel}"
    r["target_path"] = tp
    return r


def rewrite_jsonl(local: Path) -> Path:
    """Return a tmp file with rewritten records."""
    tmp = Path(tempfile.mkstemp(suffix=".jsonl", prefix="rewrite_")[1])
    n = 0
    with local.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = rewrite_record(r)
            dst.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"  rewrote {n} records -> {tmp}")
    return tmp


def run(c: paramiko.SSHClient, cmd: str, t: int = 30) -> str:
    _, o, _ = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="ignore").rstrip()


def verify_resolution(c: paramiko.SSHClient, rewritten: Path,
                      n_check: int = 20) -> tuple[int, int]:
    """Check that target_path files resolve on AutoDL."""
    ok = 0
    miss = 0
    samples = []
    with rewritten.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            tp = r["target_path"]
            if not tp.startswith("/"):
                tp = f"{REPO}/{tp}"
            samples.append((i, tp))
            if len(samples) >= n_check:
                break
    for i, tp in samples:
        rc = run(c, f"test -f '{tp}' && echo OK || echo NO")
        if rc == "OK":
            ok += 1
        else:
            miss += 1
            if miss <= 3:
                print(f"    miss [{i}]: {tp}")
    return ok, miss


def sftp_put(c, local: Path, remote: str):
    sftp = c.open_sftp()
    try:
        rdir = str(PurePosixPath(remote).parent)
        run(c, f"mkdir -p {rdir}")
        sftp.put(str(local), remote)
        print(f"  uploaded -> {remote}")
    finally:
        sftp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", nargs="+", default=["firered_pseudo"],
                    choices=list(EVAL_JSONLS.keys()) + ["all"])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if "all" in args.include:
        args.include = list(EVAL_JSONLS.keys())

    pw = os.environ.get("AUTODL_PASS", "")
    if not pw:
        sys.exit("AUTODL_PASS not set")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, pw, timeout=15)

    print(f"=== AutoDL: fivek_jpeg ===")
    print(f"  count: {run(c, f'ls {FIVEK_AUTODL} | wc -l')}")
    print(f"=== AutoDL: fivek_per_action target dirs ===")
    for sub in ("wb", "contrast", "shadows", "saturation", "highlights"):
        d = f"{REPO}/outputs/teacher_edits/fivek_per_action/{sub}"
        cnt = run(c, f"ls {d} 2>/dev/null | wc -l")
        print(f"  {sub}: {cnt} files")

    for label in args.include:
        info = EVAL_JSONLS[label]
        local = info["local"]
        remote = info["remote"]
        if not local.exists():
            print(f"\n[{label}] LOCAL MISSING: {local}")
            continue
        print(f"\n=== [{label}] {local.name} ===")
        rewritten = rewrite_jsonl(local)
        ok, miss = verify_resolution(c, rewritten, n_check=20)
        print(f"  AutoDL target_path resolve check: {ok}/{ok+miss} present")
        if args.dry_run:
            print(f"  (dry_run) skip upload")
        elif miss > ok:
            print(f"  WARN majority missing — skipping upload to avoid bad jsonl")
        else:
            sftp_put(c, rewritten, remote)
        try:
            rewritten.unlink()
        except OSError:
            pass

    c.close()


if __name__ == "__main__":
    main()
