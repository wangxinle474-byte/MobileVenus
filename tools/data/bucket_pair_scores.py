"""\u6309\u7edd\u5bf9\u5206\u6863 (0-2/2-4/4-6/6-8/8-10) \u5206\u6790\u5e76\u8fc7\u6ee4 IP2P \u6837\u672c.

\u8f93\u5165: outputs/aug_ip2p_full_v1_reparsed.json (\u5305\u542b orig_score / edit_score \u6570\u503c)
\u8f93\u51fa: \u6839\u636e --min_edit_score \u8fc7\u6ee4 \u2192 outputs/aug_ip2p_filtered.json
"""
import argparse
import json
from pathlib import Path
from collections import Counter


BUCKETS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10.01)]
BUCKET_LABELS = ['0-2', '2-4', '4-6', '6-8', '8-10']


def bucket_of(score):
    """\u5c06\u6570\u503c\u5206\u5230 5 \u4e2a\u6863."""
    if score is None:
        return None
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= score < hi:
            return BUCKET_LABELS[i]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='outputs/aug_ip2p_full_v1_reparsed.json')
    ap.add_argument('--output', default='outputs/aug_ip2p_filtered.json')
    ap.add_argument('--min_edit_score', type=float, default=6.0,
                    help='\u53ea\u4fdd\u7559 edit_score >= \u6b64\u503c\u7684\u6837\u672c (\u9ed8\u8ba4 6 = "good"+ )')
    ap.add_argument('--max_edit_score', type=float, default=10.01,
                    help='\u4e0a\u9650 (\u9ed8\u8ba4\u4e0d\u9650)')
    ap.add_argument('--require_orig_lower', action='store_true',
                    help='\u8981\u6c42 edit_score > orig_score (delta > 0)')
    args = ap.parse_args()

    d = json.load(open(args.input, encoding='utf-8'))
    # \u517c\u5bb9 reparsed (results) / \u539f\u59cb (pairs) / \u901a\u7528 samples
    samples = d.get('results') or d.get('samples') or d.get('pairs') or d
    if isinstance(samples, dict):
        samples = list(samples.values())

    print(f'Total samples: {len(samples)}')

    # \u83b7\u53d6\u5206\u6570 (\u4f18\u5148 v2 = reparsed)
    def score_of(s, key):
        for k in [f'score_{key}_v2', f'{key}_score_v2', f'score_{key}', f'{key}_score']:
            if k in s and s[k] is not None:
                try:
                    return float(s[k])
                except (TypeError, ValueError):
                    pass
        return None

    if samples:
        print('sample[0] keys:', list(samples[0].keys())[:15])

    # \u53cc\u5411\u5206\u6863\u7edf\u8ba1
    counter_orig = Counter()
    counter_edit = Counter()
    cross = Counter()  # (orig_bucket, edit_bucket)

    enriched = []
    for s in samples:
        os_ = score_of(s, 'orig')
        es = score_of(s, 'edit')
        ob = bucket_of(os_)
        eb = bucket_of(es)
        counter_orig[ob] += 1
        counter_edit[eb] += 1
        if ob and eb:
            cross[(ob, eb)] += 1

        s_out = dict(s)
        s_out['orig_score'] = os_
        s_out['edit_score'] = es
        s_out['orig_bucket'] = ob
        s_out['edit_bucket'] = eb
        if os_ is not None and es is not None:
            s_out['delta'] = es - os_
        enriched.append(s_out)

    # \u6253\u5370\u5206\u5e03
    print('\n=== orig_score \u5206\u6863 ===')
    for b in BUCKET_LABELS + [None]:
        n = counter_orig.get(b, 0)
        print(f'  {str(b):8s}  {n:5d}  {n/len(samples)*100:5.1f}%')

    print('\n=== edit_score \u5206\u6863 ===')
    for b in BUCKET_LABELS + [None]:
        n = counter_edit.get(b, 0)
        print(f'  {str(b):8s}  {n:5d}  {n/len(samples)*100:5.1f}%')

    # \u4ea4\u53c9\u8868: \u884c=orig, \u5217=edit
    print('\n=== \u4ea4\u53c9\u8868 (\u884c=orig \u6863, \u5217=edit \u6863) ===')
    print(f'{"orig\\edit":12s}', end='')
    for b in BUCKET_LABELS:
        print(f'{b:>8s}', end='')
    print()
    for ob in BUCKET_LABELS:
        print(f'{ob:12s}', end='')
        for eb in BUCKET_LABELS:
            print(f'{cross.get((ob, eb), 0):>8d}', end='')
        print()

    # \u8fc7\u6ee4
    filtered = []
    for s in enriched:
        es = s.get('edit_score')
        if es is None:
            continue
        if not (args.min_edit_score <= es < args.max_edit_score):
            continue
        if args.require_orig_lower:
            os_ = s.get('orig_score')
            if os_ is None or es <= os_:
                continue
        filtered.append(s)

    print(f'\n=== \u8fc7\u6ee4\u540e (edit_score \u2208 [{args.min_edit_score}, {args.max_edit_score}))'
          f'{" + edit > orig" if args.require_orig_lower else ""} ===')
    print(f'samples: {len(filtered)}')

    # \u8fc7\u6ee4\u540e\u7684 edit_bucket \u5206\u5e03
    print('\nedit_bucket \u5206\u5e03 (\u8fc7\u6ee4\u540e):')
    bc = Counter(s['edit_bucket'] for s in filtered)
    for b in BUCKET_LABELS:
        print(f'  {b:8s}  {bc.get(b, 0):5d}')

    out = {
        'meta': d.get('meta', {}),
        'filter': {
            'min_edit_score': args.min_edit_score,
            'max_edit_score': args.max_edit_score,
            'require_orig_lower': args.require_orig_lower,
            'num_total': len(samples),
            'num_kept': len(filtered),
        },
        'bucket_distribution': {
            'orig': dict(counter_orig),
            'edit': dict(counter_edit),
            'cross': {f'{ob}-{eb}': v for (ob, eb), v in cross.items()},
        },
        'samples': filtered,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
