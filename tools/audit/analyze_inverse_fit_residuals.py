"""Analyze inverse_fit pipeline residuals (LBFGS pseudo-label fitting quality).

The inverse_fit pipeline (tools/inverse_fit_*.py) takes Expert C target images
and fits 7D ISP parameters via LBFGS to reproduce them. This tool aggregates
the per-action fit residuals and produces a clean diagnostic report:

  - Overall L1 / pixel-residual distribution
  - Per-action breakdown with verdict counts (OK / WARN / FAIL)
  - Quality tier distribution (A excellent / B good / C acceptable / D fail)
  - Top-N worst samples per action (for manual inspection)

Output: stdout + markdown report.

Usage:
  python tools/analyze_inverse_fit_residuals.py \
    --jsonl outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl \
    --summary outputs/inverse_fit_pilot/fivek_500_master/summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ACTIONS_ORDER = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']
TIER_ORDER = ['A excellent', 'B good', 'C acceptable', 'D fail']
VERDICT_ORDER = ['OK', 'WARN', 'FAIL']


def percentile(values, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return float(s[k])


def fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return '0.0%'
    return f'{n / total * 100:.1f}%'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', type=str, required=True)
    ap.add_argument('--summary', type=str, default=None,
                    help='Optional summary.json (auto-located if not given)')
    ap.add_argument('--top_n_worst', type=int, default=5,
                    help='List N worst samples per action (default 5)')
    ap.add_argument('--out_md', type=str, default=None,
                    help='Output markdown path (default: alongside jsonl)')
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    summary_path = (Path(args.summary) if args.summary
                    else jsonl_path.parent / 'summary.json')
    out_md = (Path(args.out_md) if args.out_md
              else jsonl_path.parent / 'residual_analysis.md')

    # Load all records
    records = []
    with jsonl_path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    n_total = len(records)

    # Load summary if exists
    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    overall_l1 = [r.get('pixel_l1', 0.0) for r in records
                  if 'pixel_l1' in r]
    overall_delta = [r.get('delta_target_orig', 0.0) for r in records
                     if 'delta_target_orig' in r]

    by_action = defaultdict(list)
    for r in records:
        action = r.get('action', 'unknown')
        by_action[action].append(r)

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    lines = []
    lines.append(f'# Inverse-fit Residual Analysis — `{jsonl_path.name}`')
    lines.append('')
    lines.append(f'> Source: `{jsonl_path}`')
    if summary_path.exists():
        lines.append(f'> Summary JSON: `{summary_path}`')
    lines.append(f'> Total records: **{n_total}**')
    lines.append('')

    # ------------------------------------------------------------------
    # Section 1: Overall stats
    # ------------------------------------------------------------------
    lines.append('## 1. Overall Statistics')
    lines.append('')
    lines.append('| Metric | Value |')
    lines.append('|--------|------:|')
    lines.append(f'| Total samples | {n_total} |')
    if overall_l1:
        lines.append(f'| Pixel L1 mean | {sum(overall_l1)/len(overall_l1):.4f} |')
        lines.append(f'| Pixel L1 median (p50) | {percentile(overall_l1, 0.5):.4f} |')
        lines.append(f'| Pixel L1 p90 | {percentile(overall_l1, 0.9):.4f} |')
        lines.append(f'| Pixel L1 max | {max(overall_l1):.4f} |')
    if overall_delta:
        lines.append(f'| Δ(target, orig) mean | '
                     f'{sum(overall_delta)/len(overall_delta):.4f} |')
        lines.append('| Δ(target, orig) — measures how far Expert C drifted from input ||')
    lines.append('')

    # ------------------------------------------------------------------
    # Section 2: Verdict distribution
    # ------------------------------------------------------------------
    verdict_counter = Counter(r.get('verdict', 'unknown') for r in records)
    lines.append('## 2. Verdict Distribution (LBFGS quality flag)')
    lines.append('')
    lines.append('| Verdict | n | % |')
    lines.append('|---------|--:|--:|')
    for v in VERDICT_ORDER:
        n = verdict_counter.get(v, 0)
        lines.append(f'| {v} | {n} | {fmt_pct(n, n_total)} |')
    other = sum(verdict_counter.values()) - sum(
        verdict_counter.get(v, 0) for v in VERDICT_ORDER)
    if other > 0:
        lines.append(f'| (other) | {other} | {fmt_pct(other, n_total)} |')
    lines.append('')

    # ------------------------------------------------------------------
    # Section 3: Quality tier distribution (filter for training)
    # ------------------------------------------------------------------
    tier_counter = Counter(r.get('quality_tier', 'unknown') for r in records)
    lines.append('## 3. Quality Tier Distribution')
    lines.append('')
    lines.append(
        '> A excellent + B good + C acceptable are used as training data; '
        'D fail are discarded.')
    lines.append('')
    lines.append('| Tier | n | % |')
    lines.append('|------|--:|--:|')
    for t in TIER_ORDER:
        n = tier_counter.get(t, 0)
        lines.append(f'| {t} | {n} | {fmt_pct(n, n_total)} |')
    lines.append('')
    usable = sum(tier_counter.get(t, 0) for t in TIER_ORDER[:3])
    fail = tier_counter.get('D fail', 0)
    lines.append(f'**Usable for training**: {usable}/{n_total} '
                 f'({fmt_pct(usable, n_total)})')
    lines.append(f'**Discarded (D fail)**: {fail}/{n_total} '
                 f'({fmt_pct(fail, n_total)})')
    lines.append('')

    # ------------------------------------------------------------------
    # Section 4: Per-action breakdown
    # ------------------------------------------------------------------
    lines.append('## 4. Per-action Breakdown')
    lines.append('')
    lines.append('| Action | n | L1 mean | L1 p50 | L1 p90 | Δ mean | '
                 'OK | WARN | FAIL | Usable |')
    lines.append('|--------|--:|--------:|-------:|-------:|-------:|'
                 '---:|----:|----:|-------:|')
    for action in ACTIONS_ORDER:
        subs = by_action.get(action, [])
        if not subs:
            continue
        n = len(subs)
        l1s = [r.get('pixel_l1', 0.0) for r in subs if 'pixel_l1' in r]
        deltas = [r.get('delta_target_orig', 0.0) for r in subs
                  if 'delta_target_orig' in r]
        l1_mean = sum(l1s) / max(len(l1s), 1)
        l1_p50 = percentile(l1s, 0.5)
        l1_p90 = percentile(l1s, 0.9)
        d_mean = sum(deltas) / max(len(deltas), 1)
        verdicts = Counter(r.get('verdict', '') for r in subs)
        tiers = Counter(r.get('quality_tier', '') for r in subs)
        usable_n = sum(tiers.get(t, 0) for t in TIER_ORDER[:3])
        lines.append(
            f'| **{action}** | {n} | '
            f'{l1_mean:.4f} | {l1_p50:.4f} | {l1_p90:.4f} | '
            f'{d_mean:.4f} | '
            f'{verdicts.get("OK", 0)} | '
            f'{verdicts.get("WARN", 0)} | '
            f'{verdicts.get("FAIL", 0)} | '
            f'{usable_n} ({fmt_pct(usable_n, n)}) |'
        )
    lines.append('')
    lines.append('> **Reading**: higher `L1 mean` / `FAIL` = LBFGS could not '
                 'fit the Expert C target with a 7-parameter ISP; this is '
                 'often due to non-ISP-expressible edits (hue rotation, '
                 'tone mapping, masked local edits, etc.).')
    lines.append('')

    # ------------------------------------------------------------------
    # Section 5: Top-N worst samples per action
    # ------------------------------------------------------------------
    lines.append(f'## 5. Top-{args.top_n_worst} Worst Samples per Action')
    lines.append('')
    lines.append('> Highest pixel L1 = LBFGS most struggled to fit. Useful for '
                 'identifying systematic failure modes.')
    lines.append('')
    for action in ACTIONS_ORDER:
        subs = by_action.get(action, [])
        if not subs:
            continue
        sorted_subs = sorted(
            subs, key=lambda r: r.get('pixel_l1', 0.0), reverse=True)
        lines.append(f'### {action}')
        lines.append('')
        lines.append('| rank | source_image | L1 | verdict | tier | '
                     'P_inferred (key) |')
        lines.append('|----:|--------------|---:|---------|------|------|')
        for i, r in enumerate(sorted_subs[:args.top_n_worst]):
            P = r.get('P_inferred', {})
            # Show the most-relevant param for this action
            key_param = {
                'wb': 'white_balance',
                'contrast': 'contrast',
                'saturation': 'saturation',
                'shadows': 'shadows',
                'highlights': 'highlights',
            }.get(action, 'contrast')
            key_val = P.get(key_param, 0.0)
            lines.append(
                f'| {i+1} | `{r.get("source_image", "?")}` | '
                f'{r.get("pixel_l1", 0):.4f} | '
                f'{r.get("verdict", "?")} | '
                f'{r.get("quality_tier", "?")} | '
                f'{key_param}={key_val:.1f} |'
            )
        lines.append('')

    # ------------------------------------------------------------------
    # Section 6: Why wb fails most (interpretive section)
    # ------------------------------------------------------------------
    wb_subs = by_action.get('wb', [])
    if wb_subs:
        wb_fails = [r for r in wb_subs
                    if r.get('quality_tier', '') == 'D fail']
        if wb_fails:
            lines.append('## 6. Why does WB inverse-fit fail so often?')
            lines.append('')
            lines.append(
                f'Of the {len(wb_subs)} wb samples, {len(wb_fails)} '
                f'({fmt_pct(len(wb_fails), len(wb_subs))}) were classified '
                f'D fail. Likely structural reasons:')
            lines.append('')
            lines.append('1. **Planckian locus constraint**: 1D color-temperature '
                         'parameter cannot express off-Planckian color casts '
                         '(magenta/green) that Expert C edits often introduce '
                         '(see [R1] CST-MLP).')
            lines.append('2. **Per-channel curves not modeled**: Expert C may '
                         'apply per-channel R/G/B curves, not pure global gain.')
            lines.append('3. **Local masked edits**: Expert C can selectively '
                         'warm/cool foreground vs background; not '
                         'representable by a global WB scalar.')
            lines.append('')
            # Show wb param distribution
            wb_temps = [r.get('P_inferred', {}).get('white_balance', 0)
                        for r in wb_subs]
            wb_temps = [t for t in wb_temps if t > 0]
            if wb_temps:
                lines.append('### WB inferred temperature distribution')
                lines.append('')
                lines.append('| Stat | Value (K) |')
                lines.append('|------|----------:|')
                lines.append(f'| min | {min(wb_temps):.0f} |')
                lines.append(f'| p10 | {percentile(wb_temps, 0.1):.0f} |')
                lines.append(f'| p50 | {percentile(wb_temps, 0.5):.0f} |')
                lines.append(f'| p90 | {percentile(wb_temps, 0.9):.0f} |')
                lines.append(f'| max | {max(wb_temps):.0f} |')
                lines.append(f'| mean | {sum(wb_temps)/len(wb_temps):.0f} |')
                lines.append('')

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    out_md.write_text('\n'.join(lines), encoding='utf-8')

    # Also print to stdout
    print('\n'.join(lines))
    print(f'\n[saved] markdown report → {out_md}')


if __name__ == '__main__':
    main()
