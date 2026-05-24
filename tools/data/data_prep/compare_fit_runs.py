"""对比两组 inverse_fit batch 结果 (e.g., 5-action vs 1-action caption).

读两个 pseudo_labels.jsonl, 按 idx 配对, 输出:
  - 每张图的 L1 / delta / fit_improvement_ratio 对比表
  - L1 分布直方图 (text)
  - tier 分布对比 (excellent/good/acceptable/fail)
  - 回答关键问题: 软化 caption 是否提升 fit 率?
"""
import argparse
import json
from pathlib import Path
from collections import Counter


def load_jsonl(path: Path) -> dict:
    """读 pseudo_labels.jsonl, 返回 {idx: record}."""
    out = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            out[r['idx']] = r
    return out


def tier(l1: float) -> str:
    if l1 < 0.02: return 'A excellent'
    if l1 < 0.05: return 'B good'
    if l1 < 0.10: return 'C acceptable'
    return 'D fail'


def fmt_dist(values: list[float], buckets: list[float] = None) -> str:
    """Text histogram for L1 distribution."""
    if not values:
        return '(empty)'
    if buckets is None:
        buckets = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
    hist = [0] * (len(buckets))
    labels = []
    prev = 0.0
    for i, b in enumerate(buckets):
        labels.append(f'<{b:.2f}')
        hist[i] = sum(1 for v in values if prev <= v < b)
        prev = b
    out = []
    for lbl, n in zip(labels, hist):
        bar = '█' * n
        out.append(f'  L1 {lbl}: {bar} ({n})')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='第一组 jsonl (e.g., 5-action 老版本)')
    ap.add_argument('--b', required=True, help='第二组 jsonl (e.g., 1-action 新版本)')
    ap.add_argument('--name_a', default='A')
    ap.add_argument('--name_b', default='B')
    ap.add_argument('--out', default=None, help='存输出 .md 报表')
    args = ap.parse_args()

    a = load_jsonl(Path(args.a))
    b = load_jsonl(Path(args.b))

    print(f'[INFO] {args.name_a}: {len(a)} records  |  {args.name_b}: {len(b)} records')
    common = sorted(set(a.keys()) & set(b.keys()))
    print(f'[COMMON] {len(common)} idx 对应')

    # 每张对比表
    rows = []
    rows.append('| idx  | delta_A | L1_A   | tier_A | delta_B | L1_B   | tier_B | ΔL1    |')
    rows.append('|------|---------|--------|--------|---------|--------|--------|--------|')
    deltas_a, deltas_b, l1s_a, l1s_b = [], [], [], []
    diffs = []
    tiers_a, tiers_b = [], []

    for idx in common:
        ra, rb = a[idx], b[idx]
        da = ra['delta_target_orig']; db = rb['delta_target_orig']
        la = ra['pixel_l1']; lb = rb['pixel_l1']
        ta = tier(la); tb = tier(lb)
        diff = lb - la
        rows.append(f'| {idx:<4} | {da:.4f}  | {la:.4f} | {ta:<6} | '
                    f'{db:.4f}  | {lb:.4f} | {tb:<6} | {diff:+.4f} |')
        deltas_a.append(da); deltas_b.append(db)
        l1s_a.append(la); l1s_b.append(lb)
        tiers_a.append(ta); tiers_b.append(tb)
        diffs.append(diff)

    table_str = '\n'.join(rows)
    print('\n=== Per-image comparison ===')
    print(table_str)

    # 汇总
    n = len(common)
    median_la = sorted(l1s_a)[n // 2]
    median_lb = sorted(l1s_b)[n // 2]
    mean_la = sum(l1s_a) / n
    mean_lb = sum(l1s_b) / n
    mean_da = sum(deltas_a) / n
    mean_db = sum(deltas_b) / n

    print(f'\n=== Summary ===')
    print(f'             {args.name_a:<20s}  {args.name_b:<20s}')
    print(f'  delta mean: {mean_da:.4f}              {mean_db:.4f}')
    print(f'  L1 mean:    {mean_la:.4f}              {mean_lb:.4f}')
    print(f'  L1 median:  {median_la:.4f}              {median_lb:.4f}')

    # tier counts
    cnt_a = Counter(tiers_a); cnt_b = Counter(tiers_b)
    print(f'\n=== Tier distribution ===')
    for t in ['A excellent', 'B good', 'C acceptable', 'D fail']:
        print(f'  {t:<14s}: {args.name_a}={cnt_a.get(t, 0):2d}  {args.name_b}={cnt_b.get(t, 0):2d}')

    usable_a = sum(1 for t in tiers_a if t != 'D fail')
    usable_b = sum(1 for t in tiers_b if t != 'D fail')
    print(f'\n  usable (L1<0.10): {args.name_a}={usable_a}/{n} ({100*usable_a/n:.0f}%)  '
          f'{args.name_b}={usable_b}/{n} ({100*usable_b/n:.0f}%)')

    # 直方图
    print(f'\n=== L1 distribution ({args.name_a}) ===')
    print(fmt_dist(l1s_a))
    print(f'\n=== L1 distribution ({args.name_b}) ===')
    print(fmt_dist(l1s_b))

    # verdict
    print(f'\n=== Verdict ===')
    if usable_b > usable_a + 2:
        print(f'  ✓ {args.name_b} 显著优于 {args.name_a} (+{usable_b - usable_a} 张可用)')
        print(f'    → 软化 caption 有效, 推荐用 {args.name_b}')
    elif usable_b < usable_a - 2:
        print(f'  ✗ {args.name_a} 反而优于 {args.name_b} ({args.name_b} -{usable_a - usable_b} 张)')
        print(f'    → caption 越复杂 fit 越易, 推荐保留 {args.name_a}')
    else:
        print(f'  ≈ 两组差异不显著 ({args.name_a}={usable_a}, {args.name_b}={usable_b})')
        print(f'    → caption 复杂度对 fit 率影响有限, 选业务需要的')

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(f'# inverse_fit batch 对比: {args.name_a} vs {args.name_b}\n\n')
            f.write(f'## Summary\n')
            f.write(f'|             | {args.name_a} | {args.name_b} |\n|---|---|---|\n')
            f.write(f'| delta mean  | {mean_da:.4f} | {mean_db:.4f} |\n')
            f.write(f'| L1 mean     | {mean_la:.4f} | {mean_lb:.4f} |\n')
            f.write(f'| L1 median   | {median_la:.4f} | {median_lb:.4f} |\n')
            f.write(f'| usable rate | {usable_a}/{n} ({100*usable_a/n:.0f}%) | {usable_b}/{n} ({100*usable_b/n:.0f}%) |\n\n')
            f.write(f'## Per-image\n\n{table_str}\n')
        print(f'\n[SAVED] {args.out}')


if __name__ == '__main__':
    main()
