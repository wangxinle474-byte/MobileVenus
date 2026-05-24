"""One-shot Path X v2 eval orchestrator.

Run after AutoDL boot to:
  1. Sync patched eval_track2.py to AutoDL
  2. Pre-flight check (ckpt + eval jsonls)
  3. Launch eval_track2.py via nohup with logging
  4. Poll log until completion
  5. Download track2_eval.json locally
  6. Pretty-print per-action PSNR table

Usage (env AUTODL_PASS=...):
  python tools/autodl_eval_v2.py
  python tools/autodl_eval_v2.py --baseline_ckpt checkpoints/lut_v11a_action_gated_context/best.pt

The baseline_ckpt mode runs eval on TWO ckpts and emits a diff table to
expose the WB hypothesis: did Path Y data move per-action WB PSNR?
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import paramiko

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
REPO_REMOTE = "/root/autodl-tmp/IntelligenceCamera"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_CKPT = "checkpoints/lut_v11a_pathX_v2_implicit_head/best.pt"
EVAL_SCRIPT_LOCAL = PROJECT_ROOT / "tools" / "eval_track2.py"


def ssh_connect(timeout: int = 15) -> paramiko.SSHClient:
    pw = os.environ.get("AUTODL_PASS", "")
    if not pw:
        sys.exit("AUTODL_PASS env var not set")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, pw, timeout=timeout)
    return c


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors="ignore")
    err = e.read().decode(errors="ignore")
    rc = o.channel.recv_exit_status()
    return rc, out, err


def sftp_upload(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    sftp.put(str(local), remote)
    print(f"  uploaded {local.name} -> {remote}")


def preflight(c: paramiko.SSHClient, ckpt_remote: str, label: str) -> dict:
    """Verify ckpt + eval jsonls + patched eval_track2.py on remote."""
    print(f"\n=== preflight [{label}] ===")
    info: dict = {}

    rc, out, _ = run(c, f"ls -la {REPO_REMOTE}/{ckpt_remote} 2>&1")
    print(f"  ckpt: {out.strip()}")
    if rc != 0:
        sys.exit(f"ckpt missing: {ckpt_remote}")
    info["ckpt_size_mb"] = int(out.split()[4]) / 1024 / 1024 if out else 0

    rc, out, _ = run(c, f"ls -la {REPO_REMOTE}/tools/eval_track2.py")
    print(f"  eval_track2.py: {out.strip()}")
    if rc != 0:
        sys.exit("eval_track2.py missing on remote")

    eval_jsonls = [
        "outputs/fivek_expert_c_master/pseudo_labels.jsonl",
        "outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl",
        "outputs/mmart_pseudo_labels/v1_250/pseudo_labels.jsonl",
    ]
    info["eval_sets_present"] = []
    info["eval_sets_missing"] = []
    for j in eval_jsonls:
        rc, out, _ = run(c, f"test -f {REPO_REMOTE}/{j} && wc -l {REPO_REMOTE}/{j}")
        if rc == 0:
            print(f"  [OK]   {j}: {out.strip().split()[0]} lines")
            info["eval_sets_present"].append(j)
        else:
            print(f"  [MISS] {j}")
            info["eval_sets_missing"].append(j)

    rc, out, _ = run(c, "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader")
    print(f"  GPU: {out.strip()}")
    return info


def patch_eval_script(c: paramiko.SSHClient) -> None:
    """Re-upload local eval_track2.py to ensure implicit_head fix is on remote."""
    print("\n=== syncing eval_track2.py (with implicit_head fix) ===")
    sftp = c.open_sftp()
    try:
        sftp_upload(sftp, EVAL_SCRIPT_LOCAL, f"{REPO_REMOTE}/tools/eval_track2.py")
    finally:
        sftp.close()


def launch_eval(c: paramiko.SSHClient, ckpt_remote: str, out_dir: str,
                log_path: str, batch_size: int = 4) -> None:
    """Launch eval_track2.py via nohup. Returns immediately."""
    print(f"\n=== launching eval ===")
    print(f"  ckpt:    {ckpt_remote}")
    print(f"  out:     {out_dir}")
    print(f"  log:     {log_path}")
    cmd = (
        f"cd {REPO_REMOTE} && "
        f"mkdir -p {out_dir} && "
        f"nohup python -u tools/eval_track2.py "
        f"--ckpt {ckpt_remote} --out_dir {out_dir} --batch_size {batch_size} "
        f"> {log_path} 2>&1 &"
    )
    rc, out, err = run(c, cmd, timeout=20)
    print(f"  rc={rc}")
    if err.strip():
        print(f"  stderr: {err.strip()}")
    time.sleep(3)
    rc, pid, _ = run(c, "pgrep -af 'python.*eval_track2.py' | head -1")
    print(f"  pid: {pid.strip() or '(not found)'}")


def poll_until_done(c: paramiko.SSHClient, log_path: str,
                    poll_sec: int = 15, max_min: int = 30) -> tuple[bool, str]:
    """Tail log until '[ok] summary written' or error / timeout."""
    print(f"\n=== polling log (every {poll_sec}s, max {max_min} min) ===")
    deadline = time.time() + max_min * 60
    last_lines = 0
    while time.time() < deadline:
        rc, _, _ = run(c, f"pgrep -f 'python.*eval_track2.py' >/dev/null")
        running = (rc == 0)
        rc, tail, _ = run(c, f"wc -l {log_path} 2>/dev/null && tail -n 6 {log_path}")
        cur_lines = int(tail.split()[0]) if tail.strip() else 0
        new_lines = cur_lines - last_lines
        last_lines = cur_lines
        body = "\n".join(tail.splitlines()[1:]) if tail else ""
        print(f"  [{time.strftime('%H:%M:%S')}] running={running} "
              f"lines={cur_lines} (+{new_lines})")
        for line in body.splitlines():
            print(f"    | {line.rstrip()}")
        rc, _, _ = run(c, f"grep -q 'summary written' {log_path}")
        if rc == 0:
            print("  -> DONE")
            return True, ""
        rc, err, _ = run(c, f"grep -E 'Traceback|Error' {log_path} | head -5")
        if err.strip() and not running:
            print("  -> FAILED (process gone + error in log)")
            return False, err.strip()
        if not running:
            time.sleep(2)
            rc, _, _ = run(c, f"grep -q 'summary written' {log_path}")
            if rc == 0:
                print("  -> DONE (post check)")
                return True, ""
            return False, "process gone, no completion marker"
        time.sleep(poll_sec)
    return False, "timeout"


def fetch_results(c: paramiko.SSHClient, out_dir_remote: str,
                  out_dir_local: Path, log_path_remote: str) -> Path:
    out_dir_local.mkdir(parents=True, exist_ok=True)
    sftp = c.open_sftp()
    try:
        for name in ("track2_eval.json",):
            r = f"{out_dir_remote}/{name}"
            local = out_dir_local / name
            sftp.get(r, str(local))
            print(f"  downloaded {r} -> {local}")
        log_local = out_dir_local / "eval.log"
        sftp.get(log_path_remote, str(log_local))
        print(f"  downloaded {log_path_remote} -> {log_local}")
    finally:
        sftp.close()
    return out_dir_local / "track2_eval.json"


def print_table(json_path: Path, label: str) -> dict:
    print(f"\n{'='*78}\nRESULT [{label}]  {json_path}\n{'='*78}")
    d = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"  ckpt val_psnr={d.get('best_val_psnr_at_ckpt'):.2f} dB @ Ep{d.get('epoch_at_ckpt')}")
    print(f"  training_jsonl: {Path(d.get('training_jsonl','')).name}")
    print(f"  flags: implicit_head={d.get('flags',{}).get('use_implicit_head', '?')}")
    print()
    print(f"  {'eval set':<22} {'n_lf':>5} {'leak-free':>10} {'full':>8}")
    print(f"  {'-'*22} {'-'*5} {'-'*10} {'-'*8}")
    summary: dict = {}
    for name, res in d.get("eval_sets", {}).items():
        if not isinstance(res, dict) or "subsets" not in res:
            print(f"  {name:<22} (skipped or error)")
            continue
        lf = res["subsets"]["leak_free"].get("overall")
        full = res["subsets"]["full"].get("overall")
        n_lf = res.get("records_leak_free", 0)
        lf_s = f"{lf:.2f}" if lf else "n/a"
        full_s = f"{full:.2f}" if full else "n/a"
        print(f"  {name:<22} {n_lf:>5} {lf_s:>10} {full_s:>8}")
        # capture per-action breakdown for diff
        pa = res["subsets"]["leak_free"].get("per_action", {})
        summary[name] = {"leak_free_overall": lf, "per_action": pa,
                         "n_lf": n_lf}
    # per-action PSNR (key for WB hypothesis)
    print(f"\n  Per-action leak-free PSNR (firered_pseudo set):")
    pa = summary.get("firered_pseudo", {}).get("per_action", {})
    for a, s in pa.items():
        if isinstance(s, dict):
            print(f"    {a:<12} n={s.get('n','?'):>3}  mean={s.get('mean',0):.2f} dB")
    return summary


def diff_tables(s_v2: dict, s_baseline: dict) -> None:
    print(f"\n{'='*78}\nDIFF  v2 - baseline  (positive = v2 better)\n{'='*78}")
    sets = sorted(set(s_v2.keys()) | set(s_baseline.keys()))
    print(f"  {'eval set':<22} {'baseline':>9} {'v2':>9} {'diff':>9}")
    for name in sets:
        b = s_baseline.get(name, {}).get("leak_free_overall")
        v = s_v2.get(name, {}).get("leak_free_overall")
        if b is None or v is None:
            continue
        print(f"  {name:<22} {b:>9.2f} {v:>9.2f} {(v-b):>+9.2f}")
    print(f"\n  Per-action diff (firered_pseudo, leak-free):")
    pa_b = s_baseline.get("firered_pseudo", {}).get("per_action", {})
    pa_v = s_v2.get("firered_pseudo", {}).get("per_action", {})
    actions = sorted(set(pa_b.keys()) | set(pa_v.keys()))
    print(f"    {'action':<12} {'baseline':>9} {'v2':>9} {'diff':>9}")
    for a in actions:
        b = pa_b.get(a, {}).get("mean") if isinstance(pa_b.get(a), dict) else None
        v = pa_v.get(a, {}).get("mean") if isinstance(pa_v.get(a), dict) else None
        if b is None or v is None:
            continue
        marker = " <-- WB" if a == "wb" else ""
        print(f"    {a:<12} {b:>9.2f} {v:>9.2f} {(v-b):>+9.2f}{marker}")


def run_one(c: paramiko.SSHClient, ckpt_remote: str, label: str,
            out_local_root: Path, batch_size: int, max_min: int) -> dict:
    out_dir_remote = f"outputs/eval_track2/{label}"
    log_remote = f"{REPO_REMOTE}/{out_dir_remote}/eval.log"
    out_local = out_local_root / label

    preflight(c, ckpt_remote, label)
    launch_eval(c, ckpt_remote, out_dir_remote, log_remote, batch_size=batch_size)
    ok, err = poll_until_done(c, log_remote, max_min=max_min)
    if not ok:
        print(f"\n[FAIL] {label}: {err}")
        return {}
    json_path = fetch_results(c, f"{REPO_REMOTE}/{out_dir_remote}",
                               out_local, log_remote)
    return print_table(json_path, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_V2_CKPT,
                    help="v2 ckpt (relative to REPO_REMOTE)")
    ap.add_argument("--label", default="v2",
                    help="folder under outputs/eval_track2/")
    ap.add_argument("--baseline_ckpt", default=None,
                    help="optional baseline ckpt for diff (e.g. "
                         "checkpoints/lut_v11a_action_gated_context/best.pt)")
    ap.add_argument("--baseline_label", default="baseline")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_minutes", type=int, default=30)
    ap.add_argument("--out_local", default="outputs/eval_track2")
    ap.add_argument("--skip_patch", action="store_true",
                    help="don't re-upload eval_track2.py")
    args = ap.parse_args()

    out_local_root = PROJECT_ROOT / args.out_local
    out_local_root.mkdir(parents=True, exist_ok=True)

    c = ssh_connect()
    try:
        if not args.skip_patch:
            patch_eval_script(c)
        s_v2 = run_one(c, args.ckpt, args.label, out_local_root,
                       args.batch_size, args.max_minutes)
        if args.baseline_ckpt:
            s_b = run_one(c, args.baseline_ckpt, args.baseline_label,
                          out_local_root, args.batch_size, args.max_minutes)
            if s_v2 and s_b:
                diff_tables(s_v2, s_b)
    finally:
        c.close()
    print("\n[done]")


if __name__ == "__main__":
    main()
