"""为 5 个单 tone action 各选 N 张 FiveK 候选, 输出 5 个 caption JSON.

逻辑:
  1. 复用 select_fivek_teacher_candidates 的过滤 (score≥4 = 至少 4 个 tone 类别)
  2. 取 top N_total (默认 100) 按 image_name 排序去重
  3. seed=42 deterministic shuffle 后分 5 组 (每组 20)
  4. 每组分配一个 action: contrast / saturation / shadows / highlights / wb
  5. 写 5 个 JSON 到 data/teacher_edits_fivek_<action>_20.json

输出文件结构与 run_firered_online.py 兼容.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from select_fivek_teacher_candidates import (
    score_caption, has_negative, has_suggestion, is_too_positive,
)

ACTIONS = [
    ('contrast',   'Increase contrast. Keep the original composition and subject unchanged.'),
    ('saturation', 'Enhance saturation. Keep the original composition and subject unchanged.'),
    ('shadows',    'Lift shadows. Keep the original composition and subject unchanged.'),
    ('highlights', 'Recover highlights. Keep the original composition and subject unchanged.'),
    ('wb',         'Apply warmer white balance. Keep the original composition and subject unchanged.'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='data/fivek_venus_labels.json')
    ap.add_argument('--per_action', type=int, default=20)
    ap.add_argument('--out_dir', default='data')
    ap.add_argument('--exclude_used', nargs='*',
                    default=['a0006-IMG_2787.jpg'])
    ap.add_argument('--exclude_from_teacher_jsons', nargs='*', default=[],
                    help='从这些已有 teacher JSON 中提取 source_image 加入排除')
    ap.add_argument('--out_suffix', default=None,
                    help='输出文件后缀 (默认用 per_action 数字)')
    ap.add_argument('--min_score', type=int, default=4,
                    help='至少命中几个 tone 类别 (默认 4, 高于 base selector 的 2)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    n_total = args.per_action * len(ACTIONS)
    suffix = args.out_suffix if args.out_suffix else str(args.per_action)

    with open(args.source, encoding='utf-8') as f:
        d = json.load(f)
    labels = d['labels']
    print(f'[INFO] loaded {len(labels)} fivek venus labels')

    excluded = set(args.exclude_used)
    for jpath in args.exclude_from_teacher_jsons:
        jp = Path(jpath)
        if not jp.exists():
            print(f'[WARN] teacher JSON not found: {jp}')
            continue
        with open(jp, encoding='utf-8') as f:
            jd = json.load(f)
        for s in jd.get('samples', []):
            img = s.get('source_image')
            if img:
                excluded.add(img)
        print(f'[INFO] excluded +{len(jd.get("samples", []))} from {jp.name}')
    print(f'[INFO] total excluded images: {len(excluded)}')

    candidates = []
    for lbl in labels:
        suggestion = lbl.get('raw_response', '')
        if not suggestion or len(suggestion) < 50:
            continue
        if is_too_positive(suggestion):
            continue
        if not has_suggestion(suggestion):
            continue
        if has_negative(suggestion):
            continue
        if lbl['image'] in excluded:
            continue
        score, hits = score_caption(suggestion)
        if score < args.min_score:
            continue
        candidates.append({
            'image': lbl['image'],
            'raw_response': suggestion,
            'score': score,
            'hits': hits,
        })

    print(f'[FILTER] {len(candidates)} candidates with score≥{args.min_score}')
    if len(candidates) < n_total:
        print(f'[ERR] 不足 {n_total} 候选; 试降低 --min_score')
        return 1

    candidates.sort(key=lambda x: (-x['score'], x['image']))
    top = candidates[:n_total]

    rng = random.Random(args.seed)
    rng.shuffle(top)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_picked_imgs = set()
    summary = []

    for i, (action_name, caption) in enumerate(ACTIONS):
        group = top[i * args.per_action: (i + 1) * args.per_action]
        samples = []
        for rank, c in enumerate(group, start=1):
            m = re.match(r'a0*(\d+)-', c['image'])
            idx = int(m.group(1)) if m else rank
            samples.append({
                'rank': rank,
                'idx': idx,
                'source_image': c['image'],
                'orig_path': f'E:/Data/dataset/fivek_jpeg/{c["image"]}',
                'venus_suggestion': c['raw_response'][:300],
                'new_caption': caption,
                'tone_target': action_name,
                'score': c['score'],
            })
            all_picked_imgs.add(c['image'])

        out_path = out_dir / f'teacher_edits_fivek_{action_name}_{suffix}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'purpose': f'FiveK teacher edit single-action: {action_name}',
                    'action': action_name,
                    'caption': caption,
                    'n_samples': len(samples),
                    'source': args.source,
                    'seed': args.seed,
                    'group_index': i,
                },
                'samples': samples,
            }, f, indent=2, ensure_ascii=False)

        idxs = [s['idx'] for s in samples]
        summary.append((action_name, out_path.name, len(samples),
                        min(idxs), max(idxs)))
        print(f'[OUT] {action_name:<10s} {out_path.name} '
              f'({len(samples)} samples, idx {min(idxs)}-{max(idxs)})')

    print(f'\n[DONE] {len(ACTIONS)} JSONs in {out_dir}/, '
          f'total {n_total} samples, {len(all_picked_imgs)} unique images')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
