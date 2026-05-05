"""重新解析 pair_scores.json, 把 AesExpert 的定性输出映射到数值。

原解析只处理 'N)' 格式数字, 对 'This image looks quite beautiful.' 等返回 None。
这里加上定性短语 → 数值映射, 再重算 delta 分布。
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


# AesExpert 定性 → 数值映射 (基于 AesMMIT 数据分布经验)
QUALITATIVE_MAP = [
    # (匹配子串 lower, score)
    ('stunning',             9.0),
    ('very beautiful',       8.5),
    ('quite beautiful',      7.5),
    ('very attractive',      7.5),
    ('beautiful',            7.0),
    ('attractive',           6.5),
    ('nice',                 6.0),
    ('good',                 6.0),
    ('decent',               5.5),
    ('average',              5.0),
    ('ordinary',             4.5),
    ('mediocre',             4.0),
    ('subpar',               3.5),
    ('not so good',          3.5),
    ('poor',                 2.5),
    ('very unattractive',    1.5),
    ('unattractive',         2.5),
    ('ugly',                 2.0),
    ('terrible',             1.5),
    ('awful',                1.0),
]


def parse_score(text):
    """先抽数字, 再 fallback 到定性短语."""
    if not text:
        return None, 'empty'
    s = text.strip()

    # 1. 'N)' 或 'N.' 或纯 N 开头
    m = re.match(r'^\s*([0-9]+\.?[0-9]*)\s*(?:\)|\.|/|$)', s)
    if m:
        v = float(m.group(1))
        if 1 <= v <= 10:
            return v, 'numeric'

    # 2. 'X/10' 格式
    m = re.search(r'(\d+\.?\d*)\s*(?:/\s*10|out of 10)', s)
    if m:
        v = float(m.group(1))
        if 1 <= v <= 10:
            return v, 'numeric'

    # 3. 'score: X' 格式
    m = re.search(r'(?:score|rating|rate)[:\s]*(\d+\.?\d*)', s, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        if 1 <= v <= 10:
            return v, 'numeric'

    # 4. 定性短语匹配
    low = s.lower()
    for phrase, score in QUALITATIVE_MAP:
        if phrase in low:
            return score, f'qual:{phrase}'

    # 5. 最后 fallback: 任何 1-10 的数字
    for m in re.finditer(r'\b(\d+\.?\d*)\b', s):
        v = float(m.group(1))
        if 1 <= v <= 10:
            return v, 'numeric-any'

    return None, f'unparsed:{s[:40]!r}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='outputs/aug_ip2p_pilot_v1.json')
    parser.add_argument('--output', default='outputs/aug_ip2p_pilot_v1_reparsed.json')
    parser.add_argument('--delta_thresh', type=float, default=0.5)
    args = parser.parse_args()

    d = json.load(open(args.input, encoding='utf-8'))
    rs = d['results']

    unparsed_reasons = Counter()
    n_both = 0
    deltas = []

    for r in rs:
        s_o, r_o = parse_score(r['score_raw_orig'])
        s_e, r_e = parse_score(r['score_raw_edit'])
        r['score_orig_v2'] = s_o
        r['score_edit_v2'] = s_e
        r['parse_method_orig'] = r_o
        r['parse_method_edit'] = r_e
        if s_o is None:
            unparsed_reasons['orig:' + r_o] += 1
        if s_e is None:
            unparsed_reasons['edit:' + r_e] += 1
        if s_o is not None and s_e is not None:
            n_both += 1
            delta = s_e - s_o
            r['delta_v2'] = delta
            deltas.append((r['idx'], r['image'], s_o, s_e, delta, r['edit_prompt']))

    print('=' * 70)
    print(f'Total pairs: {len(rs)}')
    print(f'Both parseable: {n_both}  ({n_both / len(rs) * 100:.0f}%)')

    if deltas:
        ds = [d[4] for d in deltas]
        up = sum(1 for x in ds if x >= args.delta_thresh)
        neutral = sum(1 for x in ds if -args.delta_thresh < x < args.delta_thresh)
        down = sum(1 for x in ds if x <= -args.delta_thresh)
        print(f'\nDelta distribution (thresh={args.delta_thresh}):')
        print(f'  improved (>= +{args.delta_thresh}): {up}')
        print(f'  neutral (|d|<{args.delta_thresh}):  {neutral}')
        print(f'  degraded (<= -{args.delta_thresh}): {down}')
        print(f'  mean delta: {sum(ds) / len(ds):+.2f}')

    if unparsed_reasons:
        print('\n=== unparsed samples ===')
        for k, v in unparsed_reasons.most_common(10):
            print(f'  {v:3d}  {k}')

    # Top improved pairs
    deltas.sort(key=lambda x: -x[4])
    print('\n=== Top 10 improved pairs ===')
    for idx, img, so, se, dl, pr in deltas[:10]:
        print(f'  [{idx:3d}] {img}  {so:.1f} -> {se:.1f}  (+{dl:.1f})')
        print(f'       prompt: {pr[:90]}')

    # Top degraded pairs
    print('\n=== Top 10 degraded pairs ===')
    for idx, img, so, se, dl, pr in deltas[-10:][::-1]:
        print(f'  [{idx:3d}] {img}  {so:.1f} -> {se:.1f}  ({dl:+.1f})')
        print(f'       prompt: {pr[:90]}')

    # Save reparsed
    d['delta_threshold_v2'] = args.delta_thresh
    d['num_both_parseable_v2'] = n_both
    d['num_improved_v2'] = sum(1 for _, _, _, _, dl, _ in deltas if dl >= args.delta_thresh)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
