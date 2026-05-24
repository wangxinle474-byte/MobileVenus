"""Audit AutoDL for eval data: FiveK JPG dir + target PNG dirs per eval set.

Probes for the specific paths referenced by the three TRACK2_EVAL_SETS so we
can decide what to sync (and rewrite) before running eval.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
REPO = "/root/autodl-tmp/IntelligenceCamera"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_JSONLS = {
    "clean_expertC": PROJECT_ROOT / "outputs/fivek_expert_c_master/pseudo_labels.jsonl",
    "firered_pseudo": PROJECT_ROOT / "outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl",
    "mmart_real_lr": PROJECT_ROOT / "outputs/mmart_pseudo_labels/v1_250/pseudo_labels.jsonl",
}


def run(c, cmd, t=30):
    _, o, _ = c.exec_command(cmd, timeout=t)
    return o.read().decode(errors="ignore").rstrip()


def collect_target_dirs(jsonl_path: Path, n_check: int = 50) -> tuple[Counter, set, set]:
    """Return (target_dir_counter, orig_dirs, target_dirs_sampled)."""
    target_dirs = Counter()
    orig_dirs: set[str] = set()
    n = 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            tp = r.get("target_path", "").replace("\\", "/")
            op = r.get("orig_path", "").replace("\\", "/")
            if tp:
                target_dirs[str(Path(tp).parent)] += 1
            if op:
                orig_dirs.add(str(Path(op).parent))
            n += 1
            if n >= 500:
                break
    return target_dirs, orig_dirs, set()


def main():
    pw = os.environ.get("AUTODL_PASS", "")
    if not pw:
        sys.exit("AUTODL_PASS not set")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, pw, timeout=15)

    print(f"=== AutoDL: known data dirs under /root/autodl-tmp ===")
    print(run(c, "ls -d /root/autodl-tmp/*/ 2>/dev/null"))
    print()
    print(f"=== fivek_jpeg dir? ===")
    print(run(c, "ls -d /root/autodl-tmp/fivek_jpeg /root/autodl-tmp/IntelligenceCamera/fivek_jpeg "
                "/root/autodl-tmp/dataset 2>/dev/null && "
                "find /root/autodl-tmp -maxdepth 3 -name 'fivek_jpeg' -type d 2>/dev/null"))
    print()
    print(f"=== fivek_jpeg sample count ===")
    print(run(c, "find /root/autodl-tmp -maxdepth 4 -name 'a0*.jpg' 2>/dev/null | head -3 ; "
                "echo '---'; find /root/autodl-tmp -maxdepth 4 -name 'a0*.jpg' 2>/dev/null | wc -l"))

    for label, jsonl in EVAL_JSONLS.items():
        if not jsonl.exists():
            print(f"\n[{label}] LOCAL MISSING: {jsonl}")
            continue
        td, od, _ = collect_target_dirs(jsonl)
        print(f"\n=== [{label}] {jsonl.name} ===")
        print(f"  unique target dirs (top 5):")
        for d, n in td.most_common(5):
            print(f"    {n:>4}  {d}")
            rc = run(c, f"test -d {REPO}/{d} && echo 'PRESENT' || echo 'MISSING'")
            ls = run(c, f"ls {REPO}/{d} 2>/dev/null | head -3")
            cnt = run(c, f"ls {REPO}/{d} 2>/dev/null | wc -l")
            print(f"          AutoDL: {rc.strip()} ({cnt.strip()} files; sample: {ls.strip().replace(chr(10), ', ')})")
        print(f"  unique orig dirs (top 3):")
        for d in list(od)[:3]:
            print(f"    {d}")

    c.close()


if __name__ == "__main__":
    main()
