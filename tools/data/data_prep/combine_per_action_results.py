"""合并 5 个 per-action inverse_fit 结果为 master pseudo_labels + 生成报表.

支持从多个 per-action 根目录合并 (例如合并原 100 + 扩充 400 成 500 master).
"""
import argparse
import json
from pathlib import Path
from collections import Counter
import statistics as stats

ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']
DEFAULT_ROOT = Path('outputs/inverse_fit_pilot/per_action')


def tier(l1: float) -> str:
    if l1 < 0.02: return 'A excellent'
    if l1 < 0.05: return 'B good'
    if l1 < 0.10: return 'C acceptable'
    return 'D fail'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+',
                    default=[str(DEFAULT_ROOT)],
                    help='1+ per-action 根目录, 每个下有 {action}/pseudo_labels.jsonl')
    ap.add_argument('--out_dir',
                    default='outputs/inverse_fit_pilot/fivek_per_action_master',
                    help='master 输出目录')
    args = ap.parse_args()

    master_out = Path(args.out_dir)
    master_out.mkdir(parents=True, exist_ok=True)

    all_records = []
    per_action_stats = {}

    for a in ACTIONS:
        records = []
        for src_root in args.sources:
            jsonl = Path(src_root) / a / 'pseudo_labels.jsonl'
            if not jsonl.exists():
                print(f'[WARN] skip missing: {jsonl}')
                continue
            with open(jsonl, encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    r['action'] = a
                    r['_source'] = str(src_root)
                    records.append(r)
        # dedup by (action, idx) keeping last occurrence
        seen = {}
        for r in records:
            key = (r['action'], r.get('idx'))
            seen[key] = r
        records = list(seen.values())
        all_records.extend(records)

        l1s = [r['pixel_l1'] for r in records]
        deltas = [r['delta_target_orig'] for r in records]
        tiers = [tier(l) for l in l1s]
        cnt = Counter(tiers)
        per_action_stats[a] = {
            'n': len(records),
            'l1_mean': sum(l1s) / len(l1s),
            'l1_median': stats.median(l1s),
            'l1_min': min(l1s),
            'l1_max': max(l1s),
            'delta_mean': sum(deltas) / len(deltas),
            'usable_count': sum(1 for t in tiers if t != 'D fail'),
            'good_count': sum(1 for t in tiers if t in ('A excellent', 'B good')),
            'tier_counts': dict(cnt),
        }

    # 写 master pseudo_labels.jsonl
    master_jsonl = master_out / 'pseudo_labels.jsonl'
    with open(master_jsonl, 'w', encoding='utf-8') as f:
        for r in all_records:
            r['quality_tier'] = tier(r['pixel_l1'])
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[SAVED] {master_jsonl}  ({len(all_records)} records)')

    # 整体统计
    all_l1 = [r['pixel_l1'] for r in all_records]
    all_deltas = [r['delta_target_orig'] for r in all_records]
    all_tiers = [tier(l) for l in all_l1]
    overall = {
        'n_total': len(all_records),
        'l1_mean': sum(all_l1) / len(all_l1),
        'l1_median': stats.median(all_l1),
        'l1_min': min(all_l1),
        'l1_max': max(all_l1),
        'delta_mean': sum(all_deltas) / len(all_deltas),
        'usable_count': sum(1 for t in all_tiers if t != 'D fail'),
        'good_count': sum(1 for t in all_tiers if t in ('A excellent', 'B good')),
        'tier_counts': dict(Counter(all_tiers)),
        'per_action': per_action_stats,
    }
    overall_path = master_out / 'summary.json'
    with open(overall_path, 'w', encoding='utf-8') as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)
    print(f'[SAVED] {overall_path}')

    # ──────────── 终端报表 ────────────
    print('\n' + '=' * 78)
    print('per-action breakdown')
    print('=' * 78)
    print(f'{"action":<11s} {"n":>3s} {"L1 mean":>8s} {"L1 med":>8s} {"L1 max":>8s} '
          f'{"delta":>8s} {"usable":>7s} {"good":>5s}  tier_counts')
    for a in ACTIONS:
        s = per_action_stats[a]
        tc = s['tier_counts']
        tc_str = '  '.join(f'{k.split()[0]}={v}' for k, v in
                           sorted(tc.items(), key=lambda x: x[0]))
        print(f'{a:<11s} {s["n"]:>3d} {s["l1_mean"]:>8.4f} {s["l1_median"]:>8.4f} '
              f'{s["l1_max"]:>8.4f} {s["delta_mean"]:>8.4f} '
              f'{s["usable_count"]:>3d}/{s["n"]:<3d} {s["good_count"]:>5d}  {tc_str}')

    print(f'\n{"OVERALL":<11s} {overall["n_total"]:>3d} {overall["l1_mean"]:>8.4f} '
          f'{overall["l1_median"]:>8.4f} {overall["l1_max"]:>8.4f} '
          f'{overall["delta_mean"]:>8.4f} '
          f'{overall["usable_count"]:>3d}/{overall["n_total"]:<3d} '
          f'{overall["good_count"]:>5d}')
    tc = overall['tier_counts']
    print(f'\n  Tier counts: A_excellent={tc.get("A excellent", 0)}, '
          f'B_good={tc.get("B good", 0)}, '
          f'C_acceptable={tc.get("C acceptable", 0)}, '
          f'D_fail={tc.get("D fail", 0)}')

    # 关键发现
    print('\n' + '=' * 78)
    print('Key findings')
    print('=' * 78)
    by_usable = sorted(ACTIONS, key=lambda a: -per_action_stats[a]['usable_count'])
    n0 = per_action_stats[by_usable[0]]['n']
    print(f'  Best 2 actions  : {by_usable[0]} ({per_action_stats[by_usable[0]]["usable_count"]}/{n0})  '
          f'{by_usable[1]} ({per_action_stats[by_usable[1]]["usable_count"]}/{per_action_stats[by_usable[1]]["n"]})')
    print(f'  Worst 2 actions : {by_usable[-1]} ({per_action_stats[by_usable[-1]]["usable_count"]}/{per_action_stats[by_usable[-1]]["n"]})  '
          f'{by_usable[-2]} ({per_action_stats[by_usable[-2]]["usable_count"]}/{per_action_stats[by_usable[-2]]["n"]})')
    print(f'  Overall usable  : {overall["usable_count"]}/{overall["n_total"]} '
          f'({100*overall["usable_count"]/overall["n_total"]:.0f}%)')


if __name__ == '__main__':
    main()
