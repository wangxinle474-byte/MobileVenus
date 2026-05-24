"""Generate an HTML viewer for Path Y before/after pairs.

Scans outputs/teacher_edits/pathY_outputs/<action>/*.png and pairs each
edited PNG with its FiveK source JPG, rendering a scrollable side-by-side
viewer with captions.

Usage:
    python tools/visualize_pathY_pairs.py [--action wb] [--max 100]
    # outputs/teacher_edits/pathY_viewer.html
    # then open in browser
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CAPTIONS_DIR = PROJECT / "data" / "pathY_captions"
OUT_BASE = PROJECT / "outputs" / "teacher_edits" / "pathY_outputs"
FIVEK = Path(r"E:\Data\dataset\fivek_jpeg")

ACTIONS = ["wb", "highlights", "saturation", "shadows", "contrast"]


def load_samples(action: str) -> dict[int, dict]:
    captions = CAPTIONS_DIR / f"pathY_{action}.json"
    with open(captions, encoding="utf-8") as f:
        cfg = json.load(f)
    return {s["idx"]: s for s in cfg["samples"]}


def render_html(pairs: list[dict], out_path: Path):
    style = """
    body { font-family: -apple-system, system-ui, sans-serif; margin: 0; padding: 16px; background: #1e1e1e; color: #ddd; }
    h1 { color: #fff; }
    .meta { color: #aaa; margin-bottom: 20px; }
    .row { display: flex; gap: 12px; margin-bottom: 24px; padding: 12px; background: #2a2a2a; border-radius: 8px; }
    .col { flex: 1; min-width: 0; }
    .col img { width: 100%; max-height: 600px; object-fit: contain; background: #000; border-radius: 4px; }
    .col .label { color: #888; font-size: 12px; margin-top: 4px; font-family: monospace; }
    .info { width: 320px; flex-shrink: 0; color: #ccc; font-size: 13px; line-height: 1.5; }
    .info .idx { color: #4fc3f7; font-weight: bold; font-size: 16px; }
    .info .action { color: #ff9800; font-family: monospace; }
    .info .caption { color: #b0bec5; margin-top: 8px; font-style: italic; }
    .stats { background: #333; padding: 12px; border-radius: 8px; margin-bottom: 16px; }
    """
    rows = []
    for p in pairs:
        idx = p["idx"]
        action = p["action"]
        src = p["src"]
        edit = p["edit"]
        cap = p["caption"]
        rows.append(f"""
        <div class="row">
          <div class="col">
            <img src="file:///{src}" loading="lazy">
            <div class="label">orig: {Path(src).name}</div>
          </div>
          <div class="col">
            <img src="file:///{edit}" loading="lazy">
            <div class="label">edit: {Path(edit).name}</div>
          </div>
          <div class="info">
            <div class="idx">#{idx}</div>
            <div class="action">action={action}</div>
            <div class="caption">"{cap}"</div>
          </div>
        </div>
        """)

    stats_per_action = {}
    for p in pairs:
        stats_per_action[p["action"]] = stats_per_action.get(p["action"], 0) + 1
    stats_html = "  |  ".join(f"<b>{a}</b>: {c}" for a, c in sorted(stats_per_action.items()))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Path Y Pairs Viewer</title>
<style>{style}</style>
</head>
<body>
<h1>Path Y 生成结果对比</h1>
<div class="stats">{stats_html} | total pairs: {len(pairs)}</div>
<div class="meta">左：FiveK 原图 | 右：FireRed-1.1 Lightning 编辑后 | 浏览器渲染本地文件路径</div>
{''.join(rows)}
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] viewer -> {out_path}")
    print(f"     open in browser: file:///{out_path.as_posix()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", default=",".join(ACTIONS),
                    help="comma-separated, e.g. 'wb' or 'wb,highlights'")
    ap.add_argument("--max", type=int, default=100,
                    help="max pairs per action (defaults 100)")
    ap.add_argument("--out", default=str(PROJECT / "outputs" / "teacher_edits" / "pathY_viewer.html"))
    args = ap.parse_args()
    actions = [a.strip() for a in args.actions.split(",") if a.strip()]

    pairs = []
    for action in actions:
        out_dir = OUT_BASE / action
        if not out_dir.exists():
            print(f"[skip] {action}: no output dir")
            continue
        samples_by_idx = load_samples(action)
        edited = sorted(out_dir.glob("*.png"))[: args.max]
        for png in edited:
            idx = int(png.stem)
            s = samples_by_idx.get(idx)
            if not s:
                continue
            src_path = FIVEK / s["source_image"]
            if not src_path.exists():
                continue
            pairs.append({
                "idx": idx,
                "action": action,
                "src": str(src_path).replace("\\", "/"),
                "edit": str(png).replace("\\", "/"),
                "caption": s.get("new_caption", ""),
            })
        print(f"  {action}: {len(edited)} edited PNGs found")

    if not pairs:
        print("[ERROR] no pairs found - has any Path Y run completed?")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_html(pairs, out_path)


if __name__ == "__main__":
    main()
