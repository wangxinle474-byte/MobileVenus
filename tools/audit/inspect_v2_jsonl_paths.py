"""Inspect v2 master jsonl path patterns to plan AutoDL sync + rewrite."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "outputs/inverse_fit_pilot/pathY_2000_master/pseudo_labels.jsonl"


def main():
    recs = [json.loads(l) for l in open(MASTER, encoding="utf-8") if l.strip()]
    print(f"Total: {len(recs)} records\n")

    prefixes = Counter()
    for r in recs:
        for key in ("target_path", "orig_path"):
            p = r.get(key, "")
            if not p:
                prefixes[f"(empty {key})"] += 1
            elif p.startswith("E:/") or p.startswith("E:\\"):
                prefixes["E:/  (windows-abs)"] += 1
            elif p.startswith("/root/"):
                prefixes["/root  (autodl-abs)"] += 1
            elif p.startswith("outputs/") or p.startswith("data/"):
                prefixes["relative-from-root"] += 1
            elif p.startswith("/"):
                prefixes[f"other-abs: {p[:35]}"] += 1
            else:
                prefixes[f"other: {p[:30]}"] += 1

    print("Path prefix distribution (target+orig):")
    for k, v in prefixes.most_common(15):
        print(f"  {v:5d}  {k}")
    print()

    print("=== sample records (first per action+variant) ===")
    seen = set()
    for r in recs:
        key = (r.get("action", "?"), r.get("caption_variant", "-"))
        if key in seen:
            continue
        seen.add(key)
        print(f"[{key[0]} variant={key[1]}]")
        print(f"  target_path: {r.get('target_path', '?')}")
        print(f"  orig_path:   {r.get('orig_path', '?')}")
        print(f"  tier:        {r.get('quality_tier', '?')}")
        print()

    # Verify which target_paths actually exist locally vs. are syncable to AutoDL
    target_paths = [r.get("target_path", "") for r in recs]
    orig_paths = [r.get("orig_path", "") for r in recs]

    # Check if files referenced by paths exist locally
    n_target_exist = sum(1 for p in target_paths if p and Path(p).exists())
    n_orig_exist = sum(1 for p in orig_paths if p and Path(p).exists())
    print(f"target_path files existing locally: {n_target_exist}/{len(target_paths)}")
    print(f"orig_path files existing locally:   {n_orig_exist}/{len(orig_paths)}")

    # Show which records have non-existent local files
    missing_targets = [(i, p) for i, p in enumerate(target_paths) if p and not Path(p).exists()]
    if missing_targets:
        print(f"\nFirst 5 missing target_paths locally:")
        for i, p in missing_targets[:5]:
            print(f"  rec[{i}]: {p}")

    # If we'll rewrite to AutoDL, what's the path mapping?
    # Local: E:/智能相机/Venus_CVPR2026-main/IntelligenceCamera/<rest>
    # AutoDL: /root/autodl-tmp/IntelligenceCamera/<rest>
    print()
    print("=== rewrite strategy ===")
    LOCAL_PREFIX = str(ROOT).replace("\\", "/") + "/"
    AUTODL_PREFIX = "/root/autodl-tmp/IntelligenceCamera/"
    print(f"  LOCAL_PREFIX:  {LOCAL_PREFIX}")
    print(f"  AUTODL_PREFIX: {AUTODL_PREFIX}")
    print()

    # Sanity: check what relative paths emerge after rewrite
    n_starts_with_local = sum(1 for p in target_paths
                              if p.replace("\\", "/").startswith(LOCAL_PREFIX))
    n_relative = sum(1 for p in target_paths
                     if p.startswith("outputs/") or p.startswith("data/"))
    n_autodl = sum(1 for p in target_paths if p.startswith("/root/"))
    print(f"target_path startswith LOCAL_PREFIX: {n_starts_with_local}")
    print(f"target_path is relative:             {n_relative}")
    print(f"target_path is /root (autodl):       {n_autodl}")


if __name__ == "__main__":
    main()
