"""Rerun inverse_fit_batch for wb + wb_cooler with full stderr visible.

The ingest_pathY_results.py wrapper uses subprocess.run(capture_output=True)
which silently swallows per-sample errors (since inverse_fit_batch's per-sample
errors are caught with except Exception, and the wrapping returncode is still 0).

This script:
  1. Deletes stale pathY_wb / pathY_wb_cooler jsonls
  2. Runs inverse_fit_batch directly for each, with stdout/stderr → log file
  3. Reports counts after each run
  4. Optionally chains the merge step by invoking ingest with --skip_fit
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INV_FIT = ROOT / "tools" / "data" / "data_prep" / "inverse_fit_batch.py"
INGEST = ROOT / "tools" / "data" / "data_prep" / "ingest_pathY_results.py"
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def run_fit(action: str, max_size: int = 512, maxiter: int = 60) -> int:
    captions = ROOT / "data" / "pathY_captions" / f"pathY_{action}.json"
    target_dir = ROOT / "outputs" / "teacher_edits" / "pathY_outputs" / action
    out_dir = ROOT / "outputs" / "inverse_fit_pilot" / f"pathY_{action}"
    log_path = LOGS_DIR / f"inverse_fit_{action}.log"

    print(f"\n=== [{action}] inverse_fit ===")
    print(f"  captions: {captions}")
    print(f"  PNGs:     {target_dir}")
    print(f"  out:      {out_dir}")
    print(f"  log:      {log_path}")

    # Wipe stale jsonl so inverse_fit_batch doesn't see appendable state
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "pseudo_labels.jsonl"
    if jsonl.exists():
        size = jsonl.stat().st_size
        print(f"  WIPING stale jsonl: {jsonl} ({size} bytes)")
        jsonl.unlink()

    cmd = [
        sys.executable, "-u",
        str(INV_FIT),
        "--captions", str(captions),
        "--target_dir", str(target_dir),
        "--out_dir", str(out_dir),
        "--device", "cuda",
        "--max_size", str(max_size),
        "--n_restarts", "1",
        "--maxiter", str(maxiter),
        "--ssim_weight", "0.3",
    ]
    # Stream both stdout and stderr to log file via shell redirect equivalent
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
        rc = proc.wait()

    # Quick stats
    n_records = 0
    if jsonl.exists():
        with open(jsonl, encoding="utf-8") as f:
            n_records = sum(1 for line in f if line.strip())

    # Tail of log
    print(f"  rc={rc}, records={n_records}")
    print(f"  --- log tail (last 30 lines) ---")
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-30:]:
            print(f"  {line}")
    print(f"  ---")
    return n_records


def run_merge():
    """Invoke ingest_pathY_results.py --skip_fit to merge per-action jsonls."""
    print(f"\n=== merge step (ingest --skip_fit) ===")
    cmd = [sys.executable, "-u", str(INGEST), "--skip_fit"]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    for line in proc.stdout:
        sys.stdout.write(line.decode("utf-8", errors="replace"))
    proc.wait()
    print(f"  merge rc={proc.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", default="wb,wb_cooler",
                    help="Comma-separated actions to fit")
    ap.add_argument("--skip_merge", action="store_true",
                    help="Skip the final ingest merge step")
    ap.add_argument("--max_size", type=int, default=512)
    ap.add_argument("--maxiter", type=int, default=60)
    args = ap.parse_args()

    actions = [a.strip() for a in args.actions.split(",") if a.strip()]
    print(f"[plan] actions={actions}, max_size={args.max_size}, maxiter={args.maxiter}")

    summary = {}
    for action in actions:
        n = run_fit(action, args.max_size, args.maxiter)
        summary[action] = n

    print("\n=== fits done ===")
    for a, n in summary.items():
        print(f"  {a:12s}: {n} records")

    if not args.skip_merge:
        run_merge()


if __name__ == "__main__":
    main()
