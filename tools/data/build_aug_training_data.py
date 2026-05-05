"""合并 pilot 所有数据源 → 最终训练数据 aug_ip2p_pilot.json.

输入:
  - outputs/aug_ip2p_pilot_v1_reparsed.json   (AesExpert 评分 + delta)
  - outputs/aug_ip2p_pilot_with_params.json    (pixel 参数)
  - outputs/aug_ip2p_with_vlm_params.json      (AesExpert 对 edit 图的描述)
  - data/venus_edit_prompts.json               (Venus 原始描述/建议)
  - outputs/ip2p_pilot_100/{idx:04d}_{orig,edit}.png

输出: data/aug_ip2p_pilot.json
    - meta: pipeline/版本/阈值
    - samples: 每条含 orig_path/edit_path/params/description/reason/scores
过滤: delta >= 0.5 (21 对改善)
"""
import argparse
import json
from pathlib import Path
from datetime import datetime


def load(path):
    return json.load(open(path, encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pixel_json',
                        default='outputs/aug_ip2p_pilot_with_params.json')
    parser.add_argument('--vlm_json',
                        default='outputs/aug_ip2p_with_vlm_params.json')
    parser.add_argument('--venus_prompts',
                        default='data/venus_edit_prompts.json')
    parser.add_argument('--image_dir',
                        default='outputs/ip2p_pilot_100')
    parser.add_argument('--output',
                        default='data/aug_ip2p_pilot.json')
    parser.add_argument('--min_edit_score', type=float, default=6.0,
                        help='绝对分档过滤: edit_score >= 此值才保留 (默认 6 = good+)')
    parser.add_argument('--max_edit_score', type=float, default=10.01)
    parser.add_argument('--require_orig_lower', action='store_true',
                        help='额外要求 edit > orig (质量提升)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='[旧] delta 过滤阈值，设了才启用 (宝留兼容)')
    args = parser.parse_args()

    # 载入
    pixel_d = load(args.pixel_json)
    try:
        vlm_d = load(args.vlm_json)
    except Exception as e:
        print(f'[warn] vlm_json load failed: {e}')
        vlm_d = {'results': []}
    venus_d = load(args.venus_prompts)

    # 建 idx → data 索引
    pixel_by_idx = {r['idx']: r for r in pixel_d['results']}
    vlm_by_idx = {r['idx']: r for r in vlm_d.get('results', [])}
    venus_by_idx = {i: p for i, p in enumerate(venus_d['prompts'])}

    image_dir = Path(args.image_dir).as_posix()

    # 过滤并合并
    samples = []
    skipped = {'no_score': 0, 'edit_below_min': 0, 'edit_above_max': 0,
               'orig_higher': 0, 'delta_below': 0}
    for idx, pr in sorted(pixel_by_idx.items()):
        es = pr.get('score_edit_v2')
        os_ = pr.get('score_orig_v2')
        delta = pr.get('delta_v2')
        if es is None:
            skipped['no_score'] += 1
            continue
        # 主过滤: edit_score 绝对档位
        if es < args.min_edit_score:
            skipped['edit_below_min'] += 1
            continue
        if es >= args.max_edit_score:
            skipped['edit_above_max'] += 1
            continue
        # 可选: 要求 edit > orig
        if args.require_orig_lower and (os_ is None or es <= os_):
            skipped['orig_higher'] += 1
            continue
        # 可选: 旧 delta 阈值
        if args.threshold is not None and (delta is None or delta < args.threshold):
            skipped['delta_below'] += 1
            continue
        v = venus_by_idx.get(idx, {})
        vlm = vlm_by_idx.get(idx, {})

        # 档位标签
        def bucket(score):
            if score is None: return None
            for lo, hi, lab in [(0,2,'0-2'),(2,4,'2-4'),(4,6,'4-6'),(6,8,'6-8'),(8,10.01,'8-10')]:
                if lo <= score < hi: return lab
            return None

        sample = {
            'idx': idx,
            'source_image': v.get('image', f'{idx:04d}.png'),
            'orig_path': f'{image_dir}/{idx:04d}_orig.png',
            'edit_path': f'{image_dir}/{idx:04d}_edit.png',
            'venus_description': v.get('description_full', ''),
            'venus_suggestion': v.get('suggestion_full', ''),
            'ip2p_prompt': v.get('edit_prompt', ''),
            # 评分
            'score_orig': os_,
            'score_edit': es,
            'score_delta': round(delta, 2) if delta is not None else None,
            'orig_bucket': bucket(os_),
            'edit_bucket': bucket(es),
            # 参数 (pixel 推导)
            'params': {k: v for k, v in (pr.get('pixel_params') or {}).items()
                       if not k.startswith('_debug_')},
            # AesExpert 对 edit 图的定性描述
            'aesexpert_desc_edit': vlm.get('vlm_output_full', '').strip(),
        }
        samples.append(sample)

    # 按 edit_score 降序 → 同分再按 delta 降序
    samples.sort(key=lambda s: (-s['score_edit'], -(s.get('score_delta') or 0)))

    out = {
        'meta': {
            'pipeline': 'Venus-suggestion -> IP2P -> AesExpert-filter -> pixel-params',
            'source': 'Benchmark_AesGuide',
            'editor': 'timbrooks/instruct-pix2pix',
            'editor_config': {'steps': 20, 'guidance_scale': 7.5, 'image_guidance_scale': 1.5},
            'scorer': 'qyuan/AesMMIT_LLaVA_v1.5_7b_240325',
            'score_parser': 'numeric + qualitative mapping',
            'params_source': 'pixel-level derivation (orig vs edit RGB stats)',
            'filter': {
                'min_edit_score': args.min_edit_score,
                'max_edit_score': args.max_edit_score,
                'require_orig_lower': args.require_orig_lower,
                'delta_threshold': args.threshold,
            },
            'num_samples': len(samples),
            'skipped': skipped,
            'created_at': datetime.now().isoformat(),
        },
        'samples': samples,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f'Samples kept: {len(samples)}')
    print(f'Skipped: {skipped}')
    print(f'Saved: {args.output}')
    print()
    # 档位分布
    from collections import Counter
    bc = Counter(s['edit_bucket'] for s in samples)
    print('edit_bucket 分布:')
    for b in ['0-2', '2-4', '4-6', '6-8', '8-10']:
        print(f'  {b:6s}  {bc.get(b, 0)}')
    print()
    print('=== param stats on filtered samples ===')
    for k in ['ev_compensation', 'white_balance', 'contrast',
              'shadows', 'highlights', 'saturation']:
        vals = [s['params'].get(k) for s in samples if k in s.get('params', {})]
        if vals:
            print(f'  {k:20s}  n={len(vals):3d}  '
                  f'mean={sum(vals) / len(vals):+8.2f}  '
                  f'min={min(vals):+8.1f}  max={max(vals):+8.1f}')


if __name__ == '__main__':
    main()
