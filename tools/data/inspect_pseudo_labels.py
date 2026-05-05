"""\u672c\u5730\u8d28\u91cf\u68c0\u9a8c: \u7edf\u8ba1 pseudo_labels.jsonl \u7684\u53ef\u89e3\u6790\u7387 + sample.

\u7528\u6cd5:
    python tools/data/inspect_pseudo_labels.py \\
        --jsonl E:/AesExpert_HF/pseudo_labels.jsonl \\
        [--show_samples 5] [--show_fails 3]

\u8f93\u51fa:
  - parsed_ok \u7387
  - \u5206\u8bed\u8a00 (CN/EN) \u7edf\u8ba1
  - \u5e38\u89c1 parse_issues \u539f\u56e0 top10
  - parsed_tool_call key \u5206\u5e03 (\u5206\u7b56\u6570)
  - think \u957f\u5ea6\u5206\u5e03
  - \u968f\u673a sample (\u4fa7\u9762 OK + \u5931\u8d25)
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True,
                    help='pseudo_labels.jsonl \u8def\u5f84')
    ap.add_argument('--show_samples', type=int, default=3,
                    help='\u63a0\u5c55 N \u4e2a\u968f\u673a OK \u6837\u672c')
    ap.add_argument('--show_fails', type=int, default=3,
                    help='\u63a0\u5c55 N \u4e2a\u968f\u673a\u5931\u8d25\u6837\u672c')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        raise SystemExit(f'not found: {path}')

    records = []
    bad_json = 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                bad_json += 1

    n = len(records)
    print(f'=== file ===')
    print(f'  path:      {path}')
    print(f'  size:      {path.stat().st_size / 1024**2:.2f} MB')
    print(f'  records:   {n}  (bad_json_lines={bad_json})')

    # parsed_ok \u7387
    ok = [r for r in records if r.get('parsed_ok')]
    errs = [r for r in records if 'error' in r]
    no_ok = [r for r in records
             if not r.get('parsed_ok') and 'error' not in r]
    print(f'\n=== parsed_ok ===')
    print(f'  ok:        {len(ok):4d}  ({len(ok) / max(n, 1) * 100:5.1f}%)')
    print(f'  no_ok:     {len(no_ok):4d}  ({len(no_ok) / max(n, 1) * 100:5.1f}%)')
    print(f'  errors:    {len(errs):4d}  ({len(errs) / max(n, 1) * 100:5.1f}%)')

    # \u6309\u8bed\u8a00\u7edf\u8ba1
    print(f'\n=== by lang ===')
    by_lang_total: Counter[str] = Counter()
    by_lang_ok: Counter[str] = Counter()
    for r in records:
        lang = r.get('lang', '?')
        by_lang_total[lang] += 1
        if r.get('parsed_ok'):
            by_lang_ok[lang] += 1
    for lang in sorted(by_lang_total):
        tot = by_lang_total[lang]
        ok_n = by_lang_ok[lang]
        print(f'  {lang:4s}  total={tot:4d}  ok={ok_n:4d}  '
              f'({ok_n / max(tot, 1) * 100:5.1f}%)')

    # parse_issues top
    issue_counter: Counter[str] = Counter()
    for r in records:
        for iss in (r.get('parse_issues') or []):
            # \u5f52\u4e00\u5316: \u622a\u53d6\u9996\u7b26
            key = iss.split(':')[0].strip()[:40]
            issue_counter[key] += 1
    print(f'\n=== top parse issues ===')
    for iss, c in issue_counter.most_common(10):
        print(f'  {c:4d}  {iss}')

    # tool_call key \u5206\u5e03
    all_keys: Counter[str] = Counter()
    key_count_per_sample: Counter[int] = Counter()
    for r in ok:
        ptc = r.get('parsed_tool_call') or {}
        for k in ptc:
            all_keys[k] += 1
        key_count_per_sample[len(ptc)] += 1
    print(f'\n=== parsed_tool_call keys (in {len(ok)} OK samples) ===')
    for k, c in all_keys.most_common():
        bar = '#' * int(c / max(len(ok), 1) * 40)
        print(f'  {k:12s}  {c:4d}  {c / max(len(ok), 1) * 100:5.1f}%  {bar}')
    print(f'\n=== # keys per sample ===')
    for nk in sorted(key_count_per_sample):
        print(f'  {nk} keys:  {key_count_per_sample[nk]:4d}')

    # think \u957f\u5ea6
    think_lens = [len(r.get('think') or '') for r in ok]
    if think_lens:
        import statistics
        print(f'\n=== think length (chars) ===')
        print(f'  min={min(think_lens)}  '
              f'p50={int(statistics.median(think_lens))}  '
              f'mean={int(statistics.mean(think_lens))}  '
              f'max={max(think_lens)}')

    # \u968f\u673a sample
    random.seed(args.seed)
    if ok and args.show_samples > 0:
        print(f'\n=== {args.show_samples} OK samples ===')
        for r in random.sample(ok, min(args.show_samples, len(ok))):
            print(f'\n  [{r.get("sample_id", "?")}] lang={r.get("lang", "?")}')
            uw = r.get('user_want', '')
            print(f'  user_want: {uw[:150]}')
            print(f'  think:     {(r.get("think") or "")[:200]}')
            print(f'  parsed:    {json.dumps(r.get("parsed_tool_call") or {}, ensure_ascii=False)}')

    if no_ok and args.show_fails > 0:
        print(f'\n=== {args.show_fails} FAILED samples ===')
        for r in random.sample(no_ok, min(args.show_fails, len(no_ok))):
            print(f'\n  [{r.get("sample_id", "?")}] lang={r.get("lang", "?")}')
            print(f'  issues:    {r.get("parse_issues")}')
            print(f'  tool_call: {(r.get("tool_call") or "")[:200]!r}')
            print(f'  parsed:    {r.get("parsed_tool_call")}')


if __name__ == '__main__':
    main()
