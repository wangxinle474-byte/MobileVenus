"""快速查 score≥4 候选池规模 + 已用图像 + 剩余可用数."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from select_fivek_teacher_candidates import (
    score_caption, has_negative, has_suggestion, is_too_positive,
)

ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']


def main():
    with open('data/fivek_venus_labels.json', encoding='utf-8') as f:
        d = json.load(f)
    labels = d['labels']
    print(f'[INFO] loaded {len(labels)} venus labels')

    # 已用图像
    used = set()
    for a in ACTIONS:
        p = Path(f'data/teacher_edits_fivek_{a}_20.json')
        if p.exists():
            with open(p, encoding='utf-8') as f:
                j = json.load(f)
            for s in j['samples']:
                used.add(s['source_image'])
    print(f'[INFO] existing teacher_edits: {len(used)} unique images used')

    # 统计各 score 桶
    buckets = {2: 0, 3: 0, 4: 0, 5: 0}
    filtered_pool = {4: [], 5: []}
    for lbl in labels:
        sugg = lbl.get('raw_response', '')
        if not sugg or len(sugg) < 50:
            continue
        if is_too_positive(sugg) or not has_suggestion(sugg) or has_negative(sugg):
            continue
        score, hits = score_caption(sugg)
        if score >= 2:
            buckets[min(score, 5)] = buckets.get(min(score, 5), 0) + 1
        if score >= 4 and lbl['image'] not in used:
            filtered_pool[4].append(lbl['image'])
        if score >= 5 and lbl['image'] not in used:
            filtered_pool[5].append(lbl['image'])

    print('\n[SCORE BUCKETS] (after suggestion/positive/negative filtering)')
    for s, n in sorted(buckets.items()):
        print(f'  score≥{s}: {n}')

    print(f'\n[AVAILABLE POOL after excluding {len(used)} used imgs]')
    print(f'  score≥4: {len(filtered_pool[4])} images')
    print(f'  score≥5: {len(filtered_pool[5])} images')

    print('\n[SCALE OPTIONS]')
    pool4 = len(filtered_pool[4])
    for per_action in [50, 80, 100, 200]:
        total = per_action * 5
        if total <= pool4:
            print(f'  N={total} (per_action={per_action:3d}) ✓ (pool=score≥4 has {pool4})')
        else:
            print(f'  N={total} (per_action={per_action:3d}) ✗ 池不足 ({pool4})')


if __name__ == '__main__':
    main()
