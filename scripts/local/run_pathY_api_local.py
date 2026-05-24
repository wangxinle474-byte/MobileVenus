"""Path Y local launcher: 5 actions × 300 samples via FireRed-Image-Edit-1.1 API.

Sequentially runs run_firered_online.py for each action. Resume-aware: skips
samples already saved. Outputs to outputs/teacher_edits/pathY_outputs/<action>/.

Estimated runtime: ~14.6 s/sample × 1500 = ~6 h total.
WB-only (most critical): ~73 min.

Usage:
    python scripts/local/run_pathY_api_local.py [--actions wb,highlights,...] [--dry]
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('pathY_local')

PROJECT = Path(__file__).resolve().parents[2]
CAPTIONS_DIR = PROJECT / "data" / "pathY_captions"
OUT_ROOT = PROJECT / "outputs" / "teacher_edits" / "pathY_outputs"
LOG_DIR = OUT_ROOT / "_logs"
INPUT_DIR = Path(r"E:\Data\dataset\fivek_jpeg")
SCRIPT = PROJECT / "tools" / "data" / "editor_models" / "run_firered_online.py"

ACTIONS_DEFAULT = ["wb", "highlights", "saturation", "shadows", "contrast"]


def count_done(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    return len(list(out_dir.glob("*.png")))


def filter_remaining(captions_path: Path, out_dir: Path,
                     tmp_path: Path) -> tuple[Path, int, int]:
    """Build a filtered caption JSON containing only NOT-YET-GENERATED samples.

    Returns (filtered_path, n_remaining, n_total).
    """
    with open(captions_path, encoding="utf-8") as f:
        cfg = json.load(f)
    all_samples = cfg["samples"]
    done_names = {p.stem for p in out_dir.glob("*.png")} if out_dir.exists() else set()
    remaining = [s for s in all_samples if f"{s['idx']:04d}" not in done_names]
    cfg_filtered = {
        "metadata": {
            **cfg.get("metadata", {}),
            "filtered_from": str(captions_path),
            "n_remaining": len(remaining),
            "n_total": len(all_samples),
        },
        "samples": remaining,
    }
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cfg_filtered, f, ensure_ascii=False, indent=2)
    return tmp_path, len(remaining), len(all_samples)


def run_action(action: str, dry: bool) -> dict:
    captions = CAPTIONS_DIR / f"pathY_{action}.json"
    out_dir = OUT_ROOT / action
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not captions.exists():
        logger.error(f"  caption file missing: {captions}")
        return {"action": action, "status": "no_caption"}

    tmp_filtered = LOG_DIR / f"_filtered_{action}.json"
    filtered, n_rem, n_tot = filter_remaining(captions, out_dir, tmp_filtered)
    logger.info(f"  [{action}] {n_tot - n_rem}/{n_tot} done, {n_rem} remaining")
    if n_rem == 0:
        logger.info(f"  [{action}] all done, skip")
        return {"action": action, "status": "complete", "n_total": n_tot}

    if dry:
        logger.info(f"  [{action}] (dry) would call: {SCRIPT} --captions {filtered}")
        return {"action": action, "status": "dry"}

    log_path = LOG_DIR / f"{action}.log"
    cmd = [
        sys.executable, "-u", str(SCRIPT),
        "--captions", str(filtered),
        "--input_dir", str(INPUT_DIR),
        "--out_dir", str(out_dir),
        "--lora", "Lightning",
        "--seed", "42",
    ]
    t0 = time.time()
    logger.info(f"  [{action}] launching ({n_rem} samples, est ~{n_rem * 15 / 60:.1f} min)")
    logger.info(f"  [{action}] log -> {log_path}")
    # Stream to log file and to console
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1,
        )
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            sys.stdout.write(f"  [{action}] {line}")
            sys.stdout.flush()
        proc.wait()
    rc = proc.returncode
    dt = time.time() - t0
    n_done_after = count_done(out_dir)
    logger.info(
        f"  [{action}] DONE rc={rc} in {dt/60:.1f}m, "
        f"saved {n_done_after}/{n_tot}"
    )
    return {
        "action": action, "status": "ok" if rc == 0 else f"rc{rc}",
        "n_total": n_tot, "n_done": n_done_after,
        "runtime_sec": dt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", default=",".join(ACTIONS_DEFAULT),
                    help="comma-separated action order")
    ap.add_argument("--dry", action="store_true",
                    help="don't run, just plan")
    args = ap.parse_args()
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]

    logger.info(f"PROJECT: {PROJECT}")
    logger.info(f"OUT_ROOT: {OUT_ROOT}")
    logger.info(f"INPUT_DIR: {INPUT_DIR}")
    logger.info(f"actions: {actions}")
    logger.info("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    t_overall = time.time()
    for action in actions:
        logger.info(f"=== action: {action} ===")
        results.append(run_action(action, args.dry))
        logger.info("")
    total = time.time() - t_overall

    logger.info("=" * 60)
    logger.info(f"ALL DONE in {total/60:.1f} min")
    for r in results:
        logger.info(f"  {r}")

    # Write summary
    summary_path = LOG_DIR / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "total_sec": total,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
