"""Audit Path Y master JSONL: per-action / per-tier / per-verdict counts."""
import json
from collections import Counter, defaultdict
from pathlib import Path

JSONLS = [
    'outputs/inverse_fit_pilot/pathY_2000_master/pseudo_labels.jsonl',
    'outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl',
]

for jp in JSONLS:
    p = Path(jp)
    if not p.exists():
        print(f"\n{jp}: MISSING")
        continue
    rows = [json.loads(l) for l in open(p, encoding='utf-8')]
    print(f"\n=== {jp} ===")
    print(f"  Total records: {len(rows)}")

    # Per-action verdict
    pa = defaultdict(Counter)
    for r in rows:
        pa[r['action']][r.get('verdict', '?')] += 1
    print(f"\n  Per-action verdict:")
    print(f"  {'action':<12} {'OK':>4} {'WARN':>5} {'FAIL':>5} {'total':>6}")
    for a in sorted(pa):
        c = pa[a]
        tot = sum(c.values())
        print(f"  {a:<12} {c.get('OK',0):>4} {c.get('WARN',0):>5} "
              f"{c.get('FAIL',0):>5} {tot:>6}")

    # Per-action tier
    pt = defaultdict(Counter)
    for r in rows:
        pt[r['action']][r.get('quality_tier', '?')] += 1
    print(f"\n  Per-action tier:")
    print(f"  {'action':<12} {'A':>4} {'B':>4} {'C':>4} {'D':>4} {'total':>6}")
    for a in sorted(pt):
        c = pt[a]
        tot = sum(c.values())
        a_n = c.get('A excellent', 0)
        b_n = c.get('B good', 0)
        c_n = c.get('C acceptable', 0)
        d_n = c.get('D poor', 0)
        print(f"  {a:<12} {a_n:>4} {b_n:>4} {c_n:>4} {d_n:>4} {tot:>6}")

    # Usable count (A+B+C, used in training)
    usable = sum(1 for r in rows
                 if r.get('quality_tier') in ('A excellent', 'B good', 'C acceptable'))
    print(f"\n  Usable (A+B+C): {usable}")
