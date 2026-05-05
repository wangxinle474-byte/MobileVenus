"""\u6309 AesExpert \u5206\u6863\u4f4d\u8fc7\u6ee4 FiveK \u53c2\u6570\u8bad\u7ec3\u96c6\u3002

\u8f93\u5165:
  - data/aug_fivek_params.json (46167 \u6837\u672c, 5000 \u552f\u4e00\u56fe)
  - outputs/fivek_aesexpert_scores.json (5120 \u5f20\u539f\u56fe \u7684\u8bc4\u5206)

\u8f93\u51fa: data/aug_fivek_params_filtered.json (\u4ec5\u4fdd\u7559\u539f\u56fe\u8bc4\u5206 >= min_score \u7684\u6837\u672c)
"""
import argparse
import json
from pathlib import Path
from collections import Counter
from datetime import datetime


def bucket_of(score):
    if score is None:
        return None
    for lo, hi, lab in [(0, 2, '0-2'), (2, 4, '2-4'), (4, 6, '4-6'),
                         (6, 8, '6-8'), (8, 10.01, '8-10')]:
        if lo <= score < hi:
            return lab
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--params_json', default='data/aug_fivek_params.json')
    ap.add_argument('--scores_json', default='outputs/fivek_aesexpert_scores.json')
    ap.add_argument('--output', default='data/aug_fivek_params_filtered.json')
    ap.add_argument('--min_score', type=float, default=6.0)
    ap.add_argument('--max_score', type=float, default=10.01)
    args = ap.parse_args()

    print(f'Loading {args.params_json}...')
    pd_ = json.load(open(args.params_json, encoding='utf-8'))
    samples = pd_['samples']
    print(f'  {len(samples)} param samples')

    print(f'Loading {args.scores_json}...')
    sd = json.load(open(args.scores_json, encoding='utf-8'))
    scores = sd['scores']  # {filename: {score, parse_method, raw}}
    print(f'  {len(scores)} scored images')

    # \u6309\u539f\u56fe\u8bc4\u5206\u8fc7\u6ee4
    filtered = []
    skipped = {'no_score': 0, 'below_min': 0, 'above_max': 0}
    bucket_counter_in = Counter()
    bucket_counter_out = Counter()

    for s in samples:
        img_name = s['image_name']
        score_entry = scores.get(img_name)
        if not score_entry or score_entry.get('score') is None:
            skipped['no_score'] += 1
            continue
        score = float(score_entry['score'])
        bucket_counter_in[bucket_of(score)] += 1

        if score < args.min_score:
            skipped['below_min'] += 1
            continue
        if score >= args.max_score:
            skipped['above_max'] += 1
            continue

        s_out = dict(s)
        s_out['orig_score'] = score
        s_out['orig_bucket'] = bucket_of(score)
        filtered.append(s_out)
        bucket_counter_out[bucket_of(score)] += 1

    print(f'\nKept: {len(filtered)} / {len(samples)}')
    print(f'Skipped: {skipped}')
    print('\nbucket distribution (\u8f93\u5165\u4e2d\u542b\u5206\u6837\u672c):')
    for b in ['0-2', '2-4', '4-6', '6-8', '8-10']:
        n = bucket_counter_in.get(b, 0)
        print(f'  {b:6s}  {n:5d}')
    print('bucket distribution (\u8fc7\u6ee4\u540e):')
    for b in ['0-2', '2-4', '4-6', '6-8', '8-10']:
        n = bucket_counter_out.get(b, 0)
        print(f'  {b:6s}  {n:5d}')

    # \u53ef\u552f\u4e00\u56fe\u7edf\u8ba1
    unique_imgs_in = set(s['image_name'] for s in samples if scores.get(s['image_name'], {}).get('score') is not None)
    unique_imgs_out = set(s['image_name'] for s in filtered)
    print(f'\nunique imgs (input w/ score): {len(unique_imgs_in)}')
    print(f'unique imgs (output): {len(unique_imgs_out)}')

    out = {
        'meta': {
            **pd_.get('meta', {}),
            'aesexpert_filter': {
                'scores_source': args.scores_json,
                'min_score': args.min_score,
                'max_score': args.max_score,
            },
            'num_samples_input': len(samples),
            'num_samples_output': len(filtered),
            'num_unique_imgs_output': len(unique_imgs_out),
            'filtered_at': datetime.now().isoformat(),
        },
        'samples': filtered,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
