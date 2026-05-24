"""Sync v2 master jsonl + required PNG dirs to AutoDL, with path rewrite.

Path rewrite rules:
  target_path:
    E:/智能相机/Venus_CVPR2026-main/IntelligenceCamera/outputs/... → outputs/... (relative, train_lut prepends PROJECT_ROOT)
    outputs/... (already relative) → keep
  orig_path:
    E:/Data/dataset/fivek_jpeg/aXXXX-*.jpg → /root/autodl-tmp/fivek_jpeg/aXXXX-*.jpg

Steps:
  1. Read pathY_2000_master/pseudo_labels.jsonl
  2. Rewrite all paths, write to pseudo_labels_autodl.jsonl (local)
  3. Collect unique target_path files needed
  4. SFTP upload PNG dirs (skipping files already on AutoDL with matching size)
  5. SFTP upload rewritten jsonl
  6. Verify all paths resolve on AutoDL (sample check)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path, PurePosixPath

import paramiko

ROOT = Path(__file__).resolve().parents[1]

HOST = "connect.bjb2.seetacloud.com"
PORT = 35345
USER = "root"
PASS = os.environ.get("AUTODL_PASS", "")
REPO = "/root/autodl-tmp/IntelligenceCamera"

# Path rewrite prefixes
LOCAL_REPO_PREFIX = str(ROOT).replace("\\", "/") + "/"
LOCAL_FIVEK_PREFIX_VARIANTS = [
    "E:/Data/dataset/fivek_jpeg/",
    "E:\\Data\\dataset\\fivek_jpeg\\",
]
AUTODL_FIVEK_PREFIX = "/root/autodl-tmp/fivek_jpeg/"


def rewrite_path(p: str, kind: str) -> str:
    """Rewrite a path for AutoDL.

    kind: 'target' or 'orig'
    """
    if not p:
        return p
    p = p.replace("\\", "/")

    if kind == "target":
        # Strip local repo prefix → relative path under REPO
        if p.startswith(LOCAL_REPO_PREFIX):
            return p[len(LOCAL_REPO_PREFIX):]
        # Otherwise assume already relative or some other absolute (rare)
        return p
    elif kind == "orig":
        # Map fivek dataset prefix
        for variant in LOCAL_FIVEK_PREFIX_VARIANTS:
            v = variant.replace("\\", "/")
            if p.startswith(v):
                return AUTODL_FIVEK_PREFIX + p[len(v):]
        return p
    return p


def step1_rewrite_jsonl(in_path: Path, out_path: Path) -> tuple[list[dict], set]:
    """Rewrite paths and return (records, unique target relative paths)."""
    print(f"=== [1] Rewrite jsonl ===")
    print(f"  input:  {in_path}")
    print(f"  output: {out_path}")
    records = [json.loads(l) for l in open(in_path, encoding="utf-8") if l.strip()]
    target_rels = set()
    n_target_changed = 0
    n_orig_changed = 0
    for r in records:
        tp_old = r.get("target_path", "")
        op_old = r.get("orig_path", "")
        tp_new = rewrite_path(tp_old, "target")
        op_new = rewrite_path(op_old, "orig")
        if tp_new != tp_old:
            n_target_changed += 1
        if op_new != op_old:
            n_orig_changed += 1
        r["target_path"] = tp_new
        r["orig_path"] = op_new
        if tp_new and not tp_new.startswith("/"):
            target_rels.add(tp_new)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  records: {len(records)}, target rewrites: {n_target_changed}, orig rewrites: {n_orig_changed}")
    print(f"  unique target rel paths: {len(target_rels)}")
    return records, target_rels


def sftp_mkdir_p(sftp, remote_dir: str):
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def step2_sync_pngs(sftp, target_rels: set):
    """Upload all PNGs referenced in jsonl. Skip if remote exists with same size."""
    print(f"\n=== [2] Sync PNGs ({len(target_rels)} files) ===")
    n_uploaded = 0
    n_skipped = 0
    n_missing_local = 0
    total_bytes = 0
    t0 = time.time()
    for i, rel in enumerate(sorted(target_rels), 1):
        local_p = ROOT / rel
        if not local_p.exists():
            n_missing_local += 1
            if n_missing_local <= 5:
                print(f"  MISSING LOCAL: {rel}")
            continue
        remote_p = f"{REPO}/{rel}"
        try:
            st = sftp.stat(remote_p)
            if st.st_size == local_p.stat().st_size:
                n_skipped += 1
                continue
        except FileNotFoundError:
            pass
        sftp_mkdir_p(sftp, str(PurePosixPath(remote_p).parent))
        sftp.put(str(local_p), remote_p)
        n_uploaded += 1
        total_bytes += local_p.stat().st_size
        if n_uploaded % 50 == 0:
            print(f"  [{i}/{len(target_rels)}] uploaded {n_uploaded}, skipped {n_skipped}, "
                  f"{total_bytes/1024/1024:.1f} MB, dt={time.time()-t0:.0f}s")
    print(f"  done: uploaded={n_uploaded}, skipped={n_skipped} (already there), "
          f"missing_local={n_missing_local}, {total_bytes/1024/1024:.1f} MB, "
          f"dt={time.time()-t0:.1f}s")


def step3_upload_jsonl(sftp, local_jsonl: Path):
    print(f"\n=== [3] Upload rewritten jsonl ===")
    remote = f"{REPO}/outputs/inverse_fit_pilot/pathY_2000_master/pseudo_labels.jsonl"
    sftp_mkdir_p(sftp, str(PurePosixPath(remote).parent))
    sftp.put(str(local_jsonl), remote)
    sz = local_jsonl.stat().st_size
    print(f"  uploaded {sz} bytes → {remote}")


def step4_verify(c, records: list[dict], n_sample: int = 8):
    """Verify a sample of records' paths actually exist on AutoDL."""
    print(f"\n=== [4] Verify (sample {n_sample} records) ===")
    sftp = c.open_sftp()
    # Pick stratified sample: first record per (action, caption_variant), and last
    seen_keys = {}
    for i, r in enumerate(records):
        k = (r.get("action", "?"), r.get("caption_variant", "-"))
        if k not in seen_keys:
            seen_keys[k] = i
    sample_idxs = list(seen_keys.values())[:n_sample]
    if len(records) - 1 not in sample_idxs:
        sample_idxs.append(len(records) - 1)

    n_ok = 0
    n_missing = 0
    for i in sample_idxs:
        r = records[i]
        tp = r.get("target_path", "")
        op = r.get("orig_path", "")
        # Resolve target
        tp_full = tp if tp.startswith("/") else f"{REPO}/{tp}"
        try:
            sftp.stat(tp_full)
            tp_ok = True
        except FileNotFoundError:
            tp_ok = False
        try:
            sftp.stat(op)
            op_ok = True
        except FileNotFoundError:
            op_ok = False
        status = "OK" if (tp_ok and op_ok) else "FAIL"
        print(f"  [{i:4d}] {status} action={r.get('action','?'):10s} "
              f"variant={r.get('caption_variant','-'):8s} "
              f"target_ok={tp_ok} orig_ok={op_ok}")
        if not tp_ok:
            print(f"        target: {tp_full}")
        if not op_ok:
            print(f"        orig:   {op}")
        if tp_ok and op_ok:
            n_ok += 1
        else:
            n_missing += 1
    sftp.close()
    print(f"  verify: {n_ok} OK, {n_missing} FAIL")
    return n_missing == 0


def main():
    if not PASS:
        sys.exit("AUTODL_PASS not set")
    in_path = ROOT / "outputs/inverse_fit_pilot/pathY_2000_master/pseudo_labels.jsonl"
    out_path = ROOT / "outputs/inverse_fit_pilot/pathY_2000_master/pseudo_labels_autodl.jsonl"

    # Step 1 (no SSH needed)
    records, target_rels = step1_rewrite_jsonl(in_path, out_path)

    # Open SSH
    print(f"\n[ssh] connecting to {HOST}:{PORT}...")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, USER, PASS, timeout=15)
    sftp = c.open_sftp()

    # Steps 2, 3
    step2_sync_pngs(sftp, target_rels)
    step3_upload_jsonl(sftp, out_path)

    sftp.close()

    # Step 4
    ok = step4_verify(c, records)

    c.close()
    print(f"\n[{'DONE' if ok else 'FAIL'}] sync complete; ready={ok}")


if __name__ == "__main__":
    main()
