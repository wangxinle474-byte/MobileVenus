"""分析 infer_lora.py 输出的 eval_full.json:
- 总体 parse 率 + 按语言分组 (CN/EN)
- key 频次分布 (模型偏爱用哪些 Lightroom slider)
- 每个 key 的 value 统计 (mean / std / range)
- pred vs GT 的 L1 误差 (按 key 计算 MAE, 仅在 pred 和 GT 都包含该 key 时)
- 输出 best / worst 样本对照
- 生成 docs/lora_v1/EVAL_REPORT.md
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_json', default='outputs/eval_full.json')
    ap.add_argument('--out_md', default='docs/lora_v1/EVAL_REPORT.md')
    ap.add_argument('--top_k', type=int, default=5,
                    help='best/worst 各打印多少个')
    args = ap.parse_args()

    with open(args.in_json, encoding='utf-8') as f:
        blob = json.load(f)
    meta = blob['meta']
    results = blob['results']
    n = len(results)

    # 总体 parse 率
    n_pred_ok = sum(1 for r in results if r['pred']['parsed_ok'])
    n_gt_ok = sum(1 for r in results if r['gt']['parsed_ok'])

    # 按语言分组
    by_lang = defaultdict(list)
    for r in results:
        by_lang[r['lang']].append(r)
    lang_stats = {
        lang: {
            'n': len(rs),
            'pred_ok': sum(1 for r in rs if r['pred']['parsed_ok']),
        }
        for lang, rs in by_lang.items()
    }

    # key 频次 (pred vs gt)
    pred_keys = Counter()
    gt_keys = Counter()
    for r in results:
        if r['pred']['tool_call']:
            pred_keys.update(r['pred']['tool_call'].keys())
        if r['gt']['tool_call']:
            gt_keys.update(r['gt']['tool_call'].keys())

    # 每个 key 的 value 分布 + pred vs gt MAE
    pred_vals = defaultdict(list)
    gt_vals = defaultdict(list)
    pair_diffs = defaultdict(list)  # key -> [|pred - gt|, ...] 仅当两边都有该 key
    for r in results:
        p = r['pred']['tool_call'] or {}
        g = r['gt']['tool_call'] or {}
        for k, v in p.items():
            pred_vals[k].append(v)
        for k, v in g.items():
            gt_vals[k].append(v)
        for k in set(p.keys()) & set(g.keys()):
            pair_diffs[k].append(abs(p[k] - g[k]))

    # 每个样本的总 L1 距离 (用 union of keys, 缺失视作 0)
    sample_l1 = []
    for r in results:
        p = r['pred']['tool_call'] or {}
        g = r['gt']['tool_call'] or {}
        all_keys = set(p.keys()) | set(g.keys())
        if not all_keys:
            l1 = float('inf')
        else:
            l1 = sum(abs(p.get(k, 0) - g.get(k, 0)) for k in all_keys)
        sample_l1.append((l1, r))
    sample_l1.sort(key=lambda x: x[0])

    # ----- 打印到终端 -----
    def fmt_pct(x, total):
        return f'{x}/{total} ({x/max(1,total)*100:.1f}%)'

    print('=' * 80)
    print(f'LoRA eval analysis  -  source: {args.in_json}')
    print('=' * 80)
    print(f'Total samples: {n}')
    print(f'Pred parsed_ok: {fmt_pct(n_pred_ok, n)}')
    print(f'GT   parsed_ok: {fmt_pct(n_gt_ok, n)}')
    print()
    print('Per-language:')
    for lang, st in sorted(lang_stats.items()):
        print(f'  {lang}: {fmt_pct(st["pred_ok"], st["n"])}')
    print()

    print('Key frequency (pred vs gt):')
    print(f'  {"key":12s}  {"pred":>8s}  {"gt":>8s}')
    all_keys = sorted(set(pred_keys) | set(gt_keys),
                      key=lambda k: -pred_keys.get(k, 0))
    for k in all_keys:
        print(f'  {k:12s}  {pred_keys.get(k, 0):>8d}  {gt_keys.get(k, 0):>8d}')
    print()

    print('Per-key value stats (pred):')
    print(f'  {"key":12s}  {"n":>5s}  {"mean":>8s}  {"std":>8s}  {"min":>6s}  {"max":>6s}')
    for k in all_keys:
        vs = pred_vals[k]
        if not vs:
            continue
        print(f'  {k:12s}  {len(vs):>5d}  {mean(vs):>8.2f}  '
              f'{pstdev(vs) if len(vs) > 1 else 0:>8.2f}  '
              f'{min(vs):>6.0f}  {max(vs):>6.0f}')
    print()

    print('Pred vs GT MAE (when both have same key):')
    print(f'  {"key":12s}  {"n_paired":>10s}  {"MAE":>8s}  {"median":>8s}')
    for k in all_keys:
        diffs = pair_diffs[k]
        if not diffs:
            continue
        print(f'  {k:12s}  {len(diffs):>10d}  {mean(diffs):>8.2f}  {median(diffs):>8.2f}')
    print()

    print(f'Best {args.top_k} samples (lowest L1 to GT):')
    for l1, r in sample_l1[:args.top_k]:
        print(f'  sid={r["sample_id"]:5s} {r["lang"]} L1={l1:7.2f}  pred={r["pred"]["tool_call"]}')
        print(f'                            gt  ={r["gt"]["tool_call"]}')

    print()
    print(f'Worst {args.top_k} samples (highest L1 to GT):')
    for l1, r in sample_l1[-args.top_k:][::-1]:
        print(f'  sid={r["sample_id"]:5s} {r["lang"]} L1={l1:7.2f}  pred={r["pred"]["tool_call"]}')
        print(f'                            gt  ={r["gt"]["tool_call"]}')

    # ----- 写 Markdown 报告 -----
    md = []
    md.append('# LoRA v1 Eval Report\n')
    md.append(f'- **Base model**: `{meta["base_model"]}`')
    md.append(f'- **Adapter**: `{meta["adapter"]}`')
    md.append(f'- **Val set**: `{meta["val_json"]}`')
    md.append(f'- **Samples**: {n}')
    md.append(f'- **Runtime**: {meta["runtime_sec"]:.1f}s ({meta["runtime_sec"]/n:.1f}s/sample)\n')

    md.append('## 1. Format compliance\n')
    md.append('| Metric | Value |')
    md.append('|---|---|')
    md.append(f'| Pred parsed_ok | **{fmt_pct(n_pred_ok, n)}** |')
    md.append(f'| GT parsed_ok | {fmt_pct(n_gt_ok, n)} |')
    for lang, st in sorted(lang_stats.items()):
        md.append(f'| {lang} parsed_ok | {fmt_pct(st["pred_ok"], st["n"])} |')

    md.append('\n## 2. Key usage frequency (pred vs GT)\n')
    md.append('| Key | Pred count | GT count | Pred/GT ratio |')
    md.append('|---|---:|---:|---:|')
    for k in all_keys:
        p_n = pred_keys.get(k, 0)
        g_n = gt_keys.get(k, 0)
        ratio = f'{p_n/g_n:.2f}' if g_n > 0 else '-'
        md.append(f'| `{k}` | {p_n} | {g_n} | {ratio} |')

    md.append('\n## 3. Per-key value statistics (pred)\n')
    md.append('| Key | n | mean | std | min | max |')
    md.append('|---|---:|---:|---:|---:|---:|')
    for k in all_keys:
        vs = pred_vals[k]
        if not vs:
            continue
        md.append(f'| `{k}` | {len(vs)} | {mean(vs):.2f} | '
                  f'{pstdev(vs) if len(vs) > 1 else 0:.2f} | '
                  f'{min(vs):.0f} | {max(vs):.0f} |')

    md.append('\n## 4. Pred vs GT MAE (paired keys only)\n')
    md.append('| Key | n_paired | MAE | Median |')
    md.append('|---|---:|---:|---:|')
    for k in all_keys:
        diffs = pair_diffs[k]
        if not diffs:
            continue
        md.append(f'| `{k}` | {len(diffs)} | {mean(diffs):.2f} | {median(diffs):.2f} |')

    md.append(f'\n## 5. Best {args.top_k} samples (lowest L1 to GT)\n')
    md.append('| sid | lang | L1 | pred | gt |')
    md.append('|---|---|---:|---|---|')
    for l1, r in sample_l1[:args.top_k]:
        md.append(f'| {r["sample_id"]} | {r["lang"]} | {l1:.2f} | '
                  f'`{json.dumps(r["pred"]["tool_call"], ensure_ascii=False)}` | '
                  f'`{json.dumps(r["gt"]["tool_call"], ensure_ascii=False)}` |')

    md.append(f'\n## 6. Worst {args.top_k} samples (highest L1 to GT)\n')
    md.append('| sid | lang | L1 | pred | gt |')
    md.append('|---|---|---:|---|---|')
    for l1, r in sample_l1[-args.top_k:][::-1]:
        md.append(f'| {r["sample_id"]} | {r["lang"]} | {l1:.2f} | '
                  f'`{json.dumps(r["pred"]["tool_call"], ensure_ascii=False)}` | '
                  f'`{json.dumps(r["gt"]["tool_call"], ensure_ascii=False)}` |')

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text('\n'.join(md), encoding='utf-8')
    print(f'\n[OUT] {out_md}')


if __name__ == '__main__':
    main()
