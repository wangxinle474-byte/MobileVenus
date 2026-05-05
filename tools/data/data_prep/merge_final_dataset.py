"""\u5408\u5e76 IP2P + FiveK \u4e3a\u7edf\u4e00\u8bad\u7ec3\u6570\u636e\u96c6\u3002

\u8f93\u5165:
  - data/aug_ip2p_full.json (IP2P 633, edit_score >= 6)
  - data/aug_fivek_params_filtered.json (FiveK 26145, orig_score >= 6)

\u8f93\u51fa: data/aug_final.json
  \u7edf\u4e00 schema \u7684\u8bad\u7ec3\u6837\u672c\u96c6\uff0c\u5305\u542b source/params/scores/text\u3002
"""
import argparse
import json
from pathlib import Path
from collections import Counter
from datetime import datetime


def normalize_ip2p_sample(s):
    """IP2P \u6837\u672c \u2192 \u7edf\u4e00 schema."""
    return {
        'id': f'ip2p_{s["idx"]:04d}',
        'source': 'ip2p',
        'input_path': s['orig_path'],
        'target_path': s['edit_path'],
        'params': s.get('params') or {},
        'params_source': 'pixel-derived',
        'input_score': s.get('score_orig'),
        'input_bucket': s.get('orig_bucket'),
        'target_score': s.get('score_edit'),
        'target_bucket': s.get('edit_bucket'),
        'score_delta': s.get('score_delta'),
        'text_prompt': s.get('ip2p_prompt', ''),
        'venus_suggestion': s.get('venus_suggestion', ''),
        'aesexpert_desc': s.get('aesexpert_desc_edit', ''),
        'meta': {
            'source_image': s.get('source_image'),
            'venus_description': s.get('venus_description', ''),
        },
    }


def normalize_fivek_sample(s):
    """FiveK \u6837\u672c \u2192 \u7edf\u4e00 schema."""
    return {
        'id': s['id'],
        'source': 'fivek',
        'input_path': s['image_path'],
        'target_path': None,  # \u672c\u5730\u672a\u62c9 Expert C GT\uff1b\u53ef\u540e\u7eed\u8865\u5145
        'params': s.get('params') or {},
        'params_source': f'lightroom_{s.get("expert", "default")}',
        'input_score': s.get('orig_score'),
        'input_bucket': s.get('orig_bucket'),
        'target_score': None,
        'target_bucket': None,
        'score_delta': None,
        'text_prompt': '',
        'venus_suggestion': '',
        'aesexpert_desc': '',
        'meta': {
            'expert': s.get('expert'),
            'expert_id_raw': s.get('expert_id_raw'),
            'image_id': s.get('image_id'),
            'image_name': s.get('image_name'),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip2p_json', default='data/aug_ip2p_full.json')
    ap.add_argument('--fivek_json', default='data/aug_fivek_params_filtered.json')
    ap.add_argument('--output', default='data/aug_final.json')
    args = ap.parse_args()

    print('=== loading sources ===')
    ip2p = json.load(open(args.ip2p_json, encoding='utf-8'))
    fivek = json.load(open(args.fivek_json, encoding='utf-8'))
    ip2p_samples = ip2p.get('samples', [])
    fivek_samples = fivek.get('samples', [])
    print(f'  IP2P:  {len(ip2p_samples)}')
    print(f'  FiveK: {len(fivek_samples)}')

    merged = []
    for s in ip2p_samples:
        merged.append(normalize_ip2p_sample(s))
    for s in fivek_samples:
        merged.append(normalize_fivek_sample(s))

    print(f'\n=== merged: {len(merged)} samples ===')

    # \u7edf\u8ba1
    by_source = Counter(s['source'] for s in merged)
    has_target = sum(1 for s in merged if s['target_path'])
    has_text = sum(1 for s in merged if s.get('text_prompt'))
    has_input_score = sum(1 for s in merged if s.get('input_score') is not None)
    print(f'by source:        {dict(by_source)}')
    print(f'has target image: {has_target}  ({has_target / len(merged) * 100:.1f}%)')
    print(f'has text prompt:  {has_text}   ({has_text / len(merged) * 100:.1f}%)')
    print(f'has input score:  {has_input_score}  ({has_input_score / len(merged) * 100:.1f}%)')

    # \u6863\u4f4d\u5206\u5e03
    print('\ninput_bucket \u5206\u5e03:')
    bc = Counter(s.get('input_bucket') for s in merged)
    for b in ['0-2', '2-4', '4-6', '6-8', '8-10', None]:
        n = bc.get(b, 0)
        print(f'  {str(b):6s}  {n:6d}')

    # \u53c2\u6570\u7edf\u8ba1
    print('\nparam means (\u4ec5\u6709 params \u7684\u6837\u672c):')
    for k in ['ev_compensation', 'white_balance', 'contrast',
              'shadows', 'highlights', 'saturation']:
        vals = [s['params'].get(k) for s in merged
                if k in s.get('params', {}) and s['params'].get(k) is not None]
        if vals:
            print(f'  {k:20s}  n={len(vals):6d}  '
                  f'mean={sum(vals) / len(vals):+8.2f}  '
                  f'min={min(vals):+8.1f}  max={max(vals):+8.1f}')

    out = {
        'meta': {
            'pipeline': 'IP2P (Venus->IP2P->AesExpert) + FiveK (Lightroom params)',
            'filter_strategy': 'absolute score bucket >= 6 (good or better)',
            'sources': {
                'ip2p': {
                    'file': args.ip2p_json,
                    'count': len(ip2p_samples),
                    'description': 'IP2P edits with edit_score >= 6, params via pixel-level derivation',
                },
                'fivek': {
                    'file': args.fivek_json,
                    'count': len(fivek_samples),
                    'description': 'FiveK input images with AesExpert score >= 6, params from Lightroom expert settings',
                },
            },
            'num_samples': len(merged),
            'created_at': datetime.now().isoformat(),
        },
        'samples': merged,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\nSaved: {args.output}')


if __name__ == '__main__':
    main()
