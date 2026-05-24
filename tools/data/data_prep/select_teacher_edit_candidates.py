"""从 aug_ip2p_full.json 按 score_delta 排序选 top N 作为 FireRed/LongCat 教师推理候选。

输出 (与 compare_5_captions_edit.json 同格式)：
  data/teacher_edits_top{N}.json            — caption JSON
  outputs/teacher_edits/originals/<idx>.png — 原图复制 (去掉 _orig 后缀, 4 位 zero-padded)

排除 compare_5 已用过的 5 个 idx (71, 448, 808, 110, 194)。

用法:
  python tools/data/data_prep/select_teacher_edit_candidates.py --top 20
  python tools/data/data_prep/select_teacher_edit_candidates.py --top 50 --min_delta 2.0
"""
import argparse
import json
import shutil
from pathlib import Path


COMPARE_5_USED_IDX = {71, 448, 808, 110, 194}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='data/aug_ip2p_full.json',
                    help='aug_ip2p_full.json 路径')
    ap.add_argument('--top', type=int, default=20,
                    help='选 top N (按 score_delta 降序)')
    ap.add_argument('--min_delta', type=float, default=0.0,
                    help='最小 score_delta 阈值 (默认 0 = 不过滤)')
    ap.add_argument('--out_json', default=None,
                    help='caption 输出 JSON (默认 data/teacher_edits_top{N}.json)')
    ap.add_argument('--out_origs_dir', default='outputs/teacher_edits/originals',
                    help='原图输出目录')
    ap.add_argument('--ip2p_pilot_dir', default='outputs/ip2p_pilot_100',
                    help='ip2p pilot 原图所在目录 (用于备用查找)')
    ap.add_argument('--exclude_used', action='store_true', default=True,
                    help='排除 compare_5 已用过的 5 个 idx')
    args = ap.parse_args()

    out_json = args.out_json or f'data/teacher_edits_top{args.top}.json'
    out_origs_dir = Path(args.out_origs_dir)
    out_origs_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载并排序候选
    with open(args.source, encoding='utf-8') as f:
        d = json.load(f)
    samples = d.get('samples', [])
    print(f'[INFO] loaded {len(samples)} samples from {args.source}')

    pool = [s for s in samples
            if s.get('score_delta', 0) >= args.min_delta
            and (not args.exclude_used or s.get('idx') not in COMPARE_5_USED_IDX)]
    pool.sort(key=lambda x: x.get('score_delta', 0), reverse=True)

    if not pool:
        print(f'[ERR] no candidate matches min_delta={args.min_delta} '
              f'(exclude_used={args.exclude_used})')
        return 1

    top = pool[:args.top]
    if len(top) < args.top:
        print(f'[WARN] only {len(top)} candidates available (asked {args.top})')

    # 2. 复制原图 + 构 caption
    out_samples = []
    n_orig_ok = n_orig_missing = 0
    for rank, s in enumerate(top, start=1):
        idx = s['idx']
        idx_str = f'{idx:04d}'
        orig_src = Path(s.get('orig_path', ''))
        # 备用：直接拼 ip2p_pilot_dir
        if not orig_src.exists():
            orig_src = Path(args.ip2p_pilot_dir) / f'{idx_str}_orig.png'

        if not orig_src.exists():
            print(f'  [{rank:2d}] idx={idx} MISSING orig: {orig_src}')
            n_orig_missing += 1
            continue

        # 复制到 outputs/teacher_edits/originals/<idx>.png  (去掉 _orig 后缀)
        out_orig = out_origs_dir / f'{idx_str}.png'
        if not out_orig.exists():
            shutil.copy(orig_src, out_orig)
        n_orig_ok += 1

        # 拼 caption sample (compare_5_captions_edit.json 风格)
        out_samples.append({
            'rank': rank,
            'idx': idx,
            'source_image': s.get('source_image', f'{idx_str}.png'),
            'orig_path': str(orig_src).replace('\\', '/'),
            'new_caption': s.get('ip2p_prompt', ''),
            # 备份字段供 sceneA 风格使用
            'venus_description': s.get('venus_description', ''),
            'venus_suggestion': s.get('venus_suggestion', ''),
            'aesexpert_desc_edit': s.get('aesexpert_desc_edit', ''),
            # 评分信息 (用于后续分析)
            'score_orig': s.get('score_orig'),
            'score_edit': s.get('score_edit'),
            'score_delta': s.get('score_delta'),
            'orig_bucket': s.get('orig_bucket'),
            'edit_bucket': s.get('edit_bucket'),
            # 原 IP2P 反推参数 (作为初始猜测, 不是 GT)
            'ip2p_inverse_params': s.get('params'),
        })

    # 3. 写 caption JSON
    out_dict = {
        'metadata': {
            'purpose': 'Teacher-edit candidates: high score_delta from aug_ip2p_full',
            'source_file': args.source,
            'top_n': args.top,
            'n_actual': len(out_samples),
            'min_score_delta': args.min_delta,
            'exclude_compare5_idx': sorted(COMPARE_5_USED_IDX) if args.exclude_used else [],
            'caption_style': 'editB (imperative edit instruction from ip2p_prompt)',
            'notes': (
                'Use `new_caption` field for FireRed/LongCat. '
                'Backup `venus_suggestion` available for sceneA-style if needed. '
                'After model inference, this file pairs with outputs/teacher_edits/{firered,longcat}/<idx>.png.'
            ),
        },
        'samples': out_samples,
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, indent=2, ensure_ascii=False)

    print(f'\n[DONE] {len(out_samples)} samples')
    print(f'  origs copied: {n_orig_ok}  (missing: {n_orig_missing})')
    print(f'  -> caption JSON: {out_json}')
    print(f'  -> originals:    {out_origs_dir}/')
    if out_samples:
        deltas = [s['score_delta'] for s in out_samples]
        print(f'  score_delta range: [{min(deltas):.2f}, {max(deltas):.2f}]  '
              f'median={sorted(deltas)[len(deltas)//2]:.2f}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
