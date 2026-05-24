"""Build HTML viewer for v2 master JSONL (Path Y + Path X v1 merged).

Renders orig | edit | meta(action, caption_variant, tier, pixel_l1, P_inferred).
Groups by action; within action sorted by tier (A first) then pixel_l1 asc.
Uses file:/// URLs so it opens straight from disk.

Usage:
  python tools/build_pathY_v2_viewer.py \
      --jsonl outputs/inverse_fit_pilot/pathY_2000_master/pseudo_labels.jsonl \
      --out outputs/teacher_edits/pathY_v2_viewer.html \
      --tiers A,B,C \
      --max_per_action 80
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TIER_ORDER = {
    "A excellent": 0,
    "B good": 1,
    "C acceptable": 2,
    "D dirty": 3,
    "D fail": 4,
}

TIER_COLOR = {
    "A excellent": "#51cf66",
    "B good": "#94d82d",
    "C acceptable": "#ffd43b",
    "D dirty": "#ff922b",
    "D fail": "#ff6b6b",
}


def file_url(p: str) -> str:
    """Convert local path to file:/// URL the browser can resolve."""
    if not p:
        return ""
    p = p.replace("\\", "/")
    if p.startswith("outputs/"):
        p = str((PROJECT_ROOT / p).as_posix())
    if len(p) >= 2 and p[1] == ":":
        return "file:///" + p
    return "file://" + p


def load_records(jsonl_path: Path) -> list[dict]:
    out = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def stats_table(records: list[dict]) -> str:
    by_action = defaultdict(Counter)
    for r in records:
        by_action[r.get("action", "?")][r.get("quality_tier", "?")] += 1
    actions = sorted(by_action.keys())
    tiers = sorted({t for c in by_action.values() for t in c}, key=lambda x: TIER_ORDER.get(x, 99))
    rows = ["<tr><th>action</th>" + "".join(f"<th>{t}</th>" for t in tiers) + "<th>total</th></tr>"]
    grand = Counter()
    for a in actions:
        c = by_action[a]
        cells = "".join(f"<td>{c[t]}</td>" for t in tiers)
        tot = sum(c.values())
        grand += c
        rows.append(f"<tr><td><b>{a}</b></td>{cells}<td><b>{tot}</b></td></tr>")
    cells = "".join(f"<td>{grand[t]}</td>" for t in tiers)
    rows.append(
        f"<tr style='border-top:2px solid #555'><td><b>TOTAL</b></td>{cells}"
        f"<td><b>{sum(grand.values())}</b></td></tr>"
    )
    return "<table class='stats'>" + "".join(rows) + "</table>"


def wb_split_table(records: list[dict]) -> str:
    wb = [r for r in records if r.get("action") == "wb"]
    by_var = Counter(r.get("caption_variant", "(empty)") for r in wb)
    head = "<tr>" + "".join(f"<th>{k}</th>" for k in by_var.keys()) + "<th>total</th></tr>"
    body = "<tr>" + "".join(f"<td>{v}</td>" for v in by_var.values()) + f"<td><b>{sum(by_var.values())}</b></td></tr>"
    return "<h3>WB caption_variant breakdown</h3><table class='stats'>" + head + body + "</table>"


def format_params(p: dict) -> str:
    if not isinstance(p, dict):
        return ""
    keys = ["white_balance", "brightness", "contrast", "shadows", "highlights", "saturation", "clarity"]
    parts = []
    for k in keys:
        if k in p:
            v = p[k]
            if k == "white_balance":
                parts.append(f"{k}={v:.0f}")
            else:
                parts.append(f"{k}={v:+.1f}")
    return " ".join(parts)


def render_row(r: dict) -> str:
    idx = r.get("idx", "?")
    action = r.get("action", "?")
    variant = r.get("caption_variant", "") or ""
    tier = r.get("quality_tier", "?")
    tier_col = TIER_COLOR.get(tier, "#888")
    cap = r.get("caption", "")
    pixel_l1 = r.get("pixel_l1")
    pixel_l1_str = f"{pixel_l1:.4f}" if isinstance(pixel_l1, (int, float)) else "?"
    orig_url = file_url(r.get("orig_path", ""))
    edit_url = file_url(r.get("target_path", ""))
    params_str = format_params(r.get("P_inferred", {}))
    variant_html = f' <span class="variant">[{variant}]</span>' if variant else ""

    return f"""
<div class="row">
  <div class="col"><img src="{orig_url}" loading="lazy"><div class="label">orig idx={idx}</div></div>
  <div class="col"><img src="{edit_url}" loading="lazy"><div class="label">edit idx={idx}</div></div>
  <div class="info">
    <div class="action">action=<b>{action}</b>{variant_html}
      <span class="tier" style="background:{tier_col}">{tier}</span>
    </div>
    <div class="caption">{cap}</div>
    <div class="metric">pixel_l1 = <b>{pixel_l1_str}</b></div>
    <div class="params">{params_str}</div>
  </div>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", default="A,B,C", help="comma list of tier prefixes to include")
    ap.add_argument("--max_per_action", type=int, default=80, help="cap rows per action (0=no cap)")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(jsonl_path)
    print(f"loaded {len(records)} records from {jsonl_path}")

    tier_prefixes = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    filtered = [r for r in records if r.get("quality_tier", "").startswith(tier_prefixes)]
    print(f"after tier filter {tier_prefixes}: {len(filtered)} records")

    # group by action, sort, cap
    by_action: dict[str, list[dict]] = defaultdict(list)
    for r in filtered:
        by_action[r.get("action", "?")].append(r)
    for a in by_action:
        by_action[a].sort(
            key=lambda r: (
                TIER_ORDER.get(r.get("quality_tier", ""), 99),
                r.get("caption_variant", "") or "~",
                r.get("pixel_l1", 1.0),
            )
        )
        if args.max_per_action > 0:
            by_action[a] = by_action[a][: args.max_per_action]

    sections = []
    actions = sorted(by_action.keys())
    for a in actions:
        rows_html = "\n".join(render_row(r) for r in by_action[a])
        sections.append(
            f'<h2 id="{a}">{a} <span class="count">({len(by_action[a])} shown)</span></h2>'
            + rows_html
        )

    stats_html = stats_table(records)
    wb_html = wb_split_table(records)
    nav_html = " | ".join(f'<a href="#{a}">{a}</a>' for a in actions)

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Path Y v2 viewer ({len(filtered)} rows)</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin:0; padding:16px;
          background:#1a1a1a; color:#ddd; }}
  h1 {{ color:#4a9eff; margin:0 0 12px; }}
  h2 {{ color:#ffd43b; margin:24px 0 12px; border-bottom:1px solid #555; padding-bottom:6px; }}
  h3 {{ color:#ccc; margin:16px 0 8px; }}
  .summary {{ color:#aaa; font-size:13px; margin-bottom:14px; }}
  .nav {{ font-size:13px; margin-bottom:18px; padding:8px; background:#252525; border-radius:6px; }}
  .nav a {{ color:#4a9eff; text-decoration:none; margin-right:8px; }}
  table.stats {{ border-collapse:collapse; margin-bottom:14px; font-size:12px; }}
  table.stats th, table.stats td {{ border:1px solid #444; padding:4px 10px; text-align:center; }}
  table.stats th {{ background:#333; color:#fff; }}
  .row {{ display:flex; gap:12px; margin-bottom:18px; padding:10px; background:#2a2a2a; border-radius:8px; }}
  .col {{ flex:1; min-width:0; }}
  .col img {{ width:100%; max-height:520px; object-fit:contain; background:#000; border-radius:4px; }}
  .label {{ color:#888; font-size:11px; margin-top:4px; font-family:monospace; }}
  .info {{ width:340px; flex-shrink:0; font-size:13px; line-height:1.5; }}
  .action {{ color:#ff9800; font-family:monospace; margin-bottom:6px; }}
  .variant {{ color:#4fc3f7; font-size:11px; }}
  .tier {{ color:#000; padding:2px 6px; border-radius:3px; font-size:11px; margin-left:6px; font-weight:600; }}
  .caption {{ color:#b0bec5; font-style:italic; margin:6px 0; padding:6px; background:#1f1f1f; border-radius:4px; font-size:12px; }}
  .metric {{ color:#69db7c; font-family:monospace; font-size:12px; }}
  .params {{ color:#9aa0a6; font-family:monospace; font-size:11px; margin-top:4px; line-height:1.4; }}
  .count {{ color:#888; font-size:14px; font-weight:normal; }}
</style></head><body>
<h1>Path Y v2 viewer &nbsp; <span style="font-size:14px;color:#888">({len(records)} total, {len(filtered)} after tier filter)</span></h1>
<div class="summary">
  <b>source:</b> {jsonl_path}<br>
  <b>tier filter:</b> {','.join(tier_prefixes)} &nbsp; <b>max per action:</b> {args.max_per_action}<br>
  <b>columns:</b> orig (FiveK input) | edit (FireRed-1.1 output) | meta (action / tier / inverse-fit params)
</div>
<h3>Records by action × tier</h3>
{stats_html}
{wb_html}
<div class="nav"><b>Jump:</b> {nav_html}</div>
{''.join(sections)}
</body></html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"[DONE] {out_path} ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
