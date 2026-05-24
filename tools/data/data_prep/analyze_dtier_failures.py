"""分析 D-tier (L1≥0.10) 失败模式: 按 action 分组, 列出 source_image + venus_suggestion 截断.

帮助决策: D-tier 失败是否集中在特定场景类型, 用于改进 selector / caption.
"""
import json
from pathlib import Path
from collections import defaultdict

MASTER = Path('outputs/inverse_fit_pilot/fivek_per_action_master/pseudo_labels.jsonl')


def main():
    by_action = defaultdict(list)
    with open(MASTER, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['pixel_l1'] >= 0.10:
                by_action[r['action']].append(r)

    total = sum(len(v) for v in by_action.values())
    print(f'D-tier total: {total}/100\n')
    print(f'{"action":<11s} {"count":>5s}  source_image (sorted by L1 desc)')
    print('-' * 80)

    for action in ['contrast', 'saturation', 'shadows', 'highlights', 'wb']:
        records = sorted(by_action.get(action, []),
                         key=lambda x: -x['pixel_l1'])
        if not records:
            continue
        print(f'\n=== {action} ({len(records)} D-tier) ===')
        for r in records:
            l1 = r['pixel_l1']
            delta = r['delta_target_orig']
            src = r['source_image']
            # 失败比例: L1 / delta (越接近 1 越接近 identity fit)
            stall = l1 / delta if delta > 0 else 0
            sugg = r.get('caption', '')[:80]
            print(f'  L1={l1:.4f}  delta={delta:.4f}  stall={stall:.2f}  '
                  f'idx={r["idx"]:<5d} {src}')

    # 模式分析: stall ratio 高 = optimizer 没动 = sigmoid 饱和
    print('\n' + '=' * 80)
    print('Stall analysis (stall = L1/delta, 越接近 1 越说明 fit 几乎没动)')
    print('=' * 80)
    all_d = [r for v in by_action.values() for r in v]
    high_stall = sum(1 for r in all_d
                     if r['delta_target_orig'] > 0
                     and r['pixel_l1'] / r['delta_target_orig'] > 0.85)
    print(f'  high stall (L1/delta > 0.85): {high_stall}/{len(all_d)}  '
          f'→ optimizer 卡住, FireRed 编辑可能合理但 ISP 完全表达不了')
    medium_stall = sum(1 for r in all_d
                       if r['delta_target_orig'] > 0
                       and 0.5 < r['pixel_l1'] / r['delta_target_orig'] <= 0.85)
    print(f'  medium stall (0.5-0.85)     : {medium_stall}/{len(all_d)}  '
          f'→ 部分拟合, ISP 抓住一些 tone 但漏了细节')
    low_stall = sum(1 for r in all_d
                    if r['delta_target_orig'] > 0
                    and r['pixel_l1'] / r['delta_target_orig'] <= 0.5)
    print(f'  low stall (<0.5)            : {low_stall}/{len(all_d)}  '
          f'→ ISP 已极力拟合但 target 太远')

    # 推荐: 哪些图加入 black-list
    print('\n' + '=' * 80)
    print('推荐 selector black-list (high stall, L1/delta > 0.85)')
    print('=' * 80)
    for r in sorted(all_d, key=lambda x:
                    -(x['pixel_l1'] / max(x['delta_target_orig'], 1e-6))):
        stall = r['pixel_l1'] / r['delta_target_orig'] if r['delta_target_orig'] > 0 else 0
        if stall > 0.85:
            print(f'  [{r["action"]:<10s}] idx={r["idx"]:<5d} '
                  f'L1={r["pixel_l1"]:.4f} stall={stall:.2f} {r["source_image"]}')


if __name__ == '__main__':
    main()
