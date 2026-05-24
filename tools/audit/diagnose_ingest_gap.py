"""Diagnose why ingest produces fewer records than PNG count."""
import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]


def main():
    for action in ["wb", "wb_cooler"]:
        cap_path = ROOT / "data" / "pathY_captions" / f"pathY_{action}.json"
        png_dir = ROOT / "outputs" / "teacher_edits" / "pathY_outputs" / action
        jsonl_path = ROOT / "outputs" / "inverse_fit_pilot" / f"pathY_{action}" / "pseudo_labels.jsonl"

        print(f"=== {action} ===")
        cap = json.load(open(cap_path, encoding="utf-8"))
        samples = cap.get("samples", [])
        print(f"  captions samples: {len(samples)}")
        if samples:
            s0 = samples[0]
            print(f"  sample[0] keys: {list(s0.keys())}")
            for k in ("idx", "id", "fivek_id", "image_id", "filename"):
                if k in s0:
                    print(f"  sample[0].{k} = {s0[k]!r}")

        pngs = sorted(png_dir.glob("*.png"))
        print(f"  PNGs: {len(pngs)}")
        if pngs:
            print(f"  PNG[0:3] names: {[p.name for p in pngs[:3]]}")

        # Try matching: PNG name vs sample.idx
        # Extract numeric from PNG name
        png_idxs = []
        for p in pngs:
            m = re.match(r"^(\d+)\.png$", p.name)
            if m:
                png_idxs.append(int(m.group(1)))
        png_idx_set = set(png_idxs)

        sample_idxs = []
        for s in samples:
            for k in ("idx", "id", "fivek_id"):
                if k in s:
                    v = s[k]
                    if isinstance(v, int):
                        sample_idxs.append(v)
                    elif isinstance(v, str) and v.isdigit():
                        sample_idxs.append(int(v))
                    elif isinstance(v, str):
                        m = re.search(r"(\d+)", v)
                        if m:
                            sample_idxs.append(int(m.group(1)))
                    break
        sample_idx_set = set(sample_idxs)

        print(f"  sample idx range: {min(sample_idxs) if sample_idxs else '?'} ~ {max(sample_idxs) if sample_idxs else '?'}")
        print(f"  png idx range:    {min(png_idxs) if png_idxs else '?'} ~ {max(png_idxs) if png_idxs else '?'}")
        print(f"  intersection: {len(png_idx_set & sample_idx_set)} (both)")
        print(f"  png-only (no caption): {len(png_idx_set - sample_idx_set)}")
        print(f"  caption-only (no png): {len(sample_idx_set - png_idx_set)}")

        if jsonl_path.exists():
            lines = [json.loads(l) for l in open(jsonl_path, encoding="utf-8") if l.strip()]
            print(f"  existing jsonl records: {len(lines)}")
            if lines:
                rec_idxs = []
                for r in lines:
                    tp = r.get("target_path", "")
                    m = re.search(r"(\d+)\.png", tp)
                    if m:
                        rec_idxs.append(int(m.group(1)))
                print(f"  jsonl idx range: {min(rec_idxs) if rec_idxs else '?'} ~ {max(rec_idxs) if rec_idxs else '?'}")
                print(f"  records covered out of intersection: {len(set(rec_idxs) & png_idx_set & sample_idx_set)}")

        print()


if __name__ == "__main__":
    main()
