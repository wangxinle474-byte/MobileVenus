"""\u4ece FiveK Expert \u8bbe\u5b9a\u6784\u5efa (input_jpg, params, expert) \u8bad\u7ec3\u6570\u636e.

\u8f93\u5165:
  - E:\\Data\\dataset\\fivek_expert\\fivek_expert_settings.json (60K \u53c2\u6570\u8bb0\u5f55)
  - E:\\Data\\dataset\\fivek_jpeg\\*.jpg (5120 \u5f20\u539f\u56fe)

\u8f93\u51fa: data/aug_fivek_params.json

\u7b56\u7565:
  1. \u53ea\u4fdd\u7559 JPEG \u5b58\u5728\u7684\u6837\u672c
  2. \u6309 expert \u5206\u7c7b\u6c47\u603b\uff1areal_expert (2200) \u4e0e default (57800)
  3. \u6620\u5c04 6 \u7ef4\u53c2\u6570\uff1aev/wb/contrast/shadows/highlights/saturation
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict


# ID \u2192 \u53ef\u8bfb\u540d
EXPERT_NAME_MAP = {
    'default': 'default',
    '42962A54-F9BA-11DB-B851-000D93313A24': 'expert_a',
    'ED7AD140-FA03-11DB-AB5E-00145166C8C8': 'expert_b',
}


SENTINEL = -999999


def _safe(val, default):
    """\u8fc7\u6ee4 sentinel/\u7a7a\u503c."""
    if val is None or val == SENTINEL:
        return default
    return val


def map_params(targets, raw_settings):
    """6 \u7ef4\u8bad\u7ec3\u53c2\u6570 (\u4e0e IP2P \u63d0\u53d6\u51fa\u6765\u7684 schema \u4e00\u81f4)."""
    return {
        'ev_compensation': float(_safe(targets.get('ev_compensation'), 0.0)),
        'white_balance': int(_safe(targets.get('white_balance'), 5500)),
        'contrast': float(_safe(raw_settings.get('contrast'), 0)),
        'shadows': float(_safe(raw_settings.get('shadows'), 0)),
        'highlights': float(_safe(raw_settings.get('highlights'), 0)),
        'saturation': float(_safe(raw_settings.get('saturation'), 0)),
    }


def has_full_params(targets, raw_settings):
    """\u4e3b\u8981 5 \u4e2a raw_settings \u53c2\u6570\u90fd\u4e0d\u662f sentinel \u624d\u4fdd\u7559."""
    keys = ['contrast', 'shadows', 'highlights', 'saturation']
    return all(raw_settings.get(k, SENTINEL) != SENTINEL for k in keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings_json',
                        default=r'E:\Data\dataset\fivek_expert\fivek_expert_settings.json')
    parser.add_argument('--jpeg_dir',
                        default=r'E:\Data\dataset\fivek_jpeg')
    parser.add_argument('--output', default='data/aug_fivek_params.json')
    parser.add_argument('--mode', choices=['default', 'real', 'all'], default='all',
                        help='default = \u53ea\u4fdd\u7559 default; real = \u53ea\u4fdd\u7559\u771f\u4eba expert; all = \u5168\u90e8')
    parser.add_argument('--require_full_params', action='store_true', default=True,
                        help='\u8981\u6c42 contrast/shadows/highlights/saturation \u90fd\u4e0d\u662f sentinel')
    args = parser.parse_args()

    print(f'Loading {args.settings_json}...')
    d = json.load(open(args.settings_json, encoding='utf-8'))
    samples_in = d['samples']
    print(f'  {len(samples_in)} input records')

    jpeg_dir = Path(args.jpeg_dir)
    jpeg_files = {p.name: p for p in jpeg_dir.glob('*.jpg')}
    print(f'  {len(jpeg_files)} JPEG files in {jpeg_dir}')

    samples_out = []
    stats = {
        'matched': 0,
        'unmatched_jpeg': 0,
        'expert_counter': Counter(),
        'images_per_expert': defaultdict(set),
    }

    stats['skipped_mode'] = 0
    stats['skipped_incomplete'] = 0

    for s in samples_in:
        # \u6309 mode \u8fc7\u6ee4
        is_default = (s['expert'] == 'default')
        if args.mode == 'default' and not is_default:
            stats['skipped_mode'] += 1
            continue
        if args.mode == 'real' and is_default:
            stats['skipped_mode'] += 1
            continue

        # \u53c2\u6570\u5b8c\u5907\u6027\u8fc7\u6ee4
        if args.require_full_params and not has_full_params(s['targets'], s['raw_settings']):
            stats['skipped_incomplete'] += 1
            continue

        # DNG \u540d \u2192 JPG \u540d
        dng_name = s['image_name']
        jpg_name = dng_name.replace('.dng', '.jpg').replace('.DNG', '.jpg')
        jpg_path = jpeg_files.get(jpg_name)
        if jpg_path is None:
            stats['unmatched_jpeg'] += 1
            continue

        params = map_params(s['targets'], s['raw_settings'])
        expert_name = EXPERT_NAME_MAP.get(s['expert'], s['expert'])

        sample_out = {
            'id': s['id'],
            'image_name': jpg_name,
            'image_path': str(jpg_path).replace('\\', '/'),
            'expert': expert_name,
            'expert_id_raw': s['expert'],
            'image_id': s.get('image_id'),
            'params': params,
        }
        samples_out.append(sample_out)
        stats['matched'] += 1
        stats['expert_counter'][expert_name] += 1
        stats['images_per_expert'][expert_name].add(jpg_name)

    print()
    print(f'matched: {stats["matched"]}  '
          f'skipped_mode: {stats["skipped_mode"]}  '
          f'skipped_incomplete: {stats["skipped_incomplete"]}  '
          f'unmatched_jpeg: {stats["unmatched_jpeg"]}')
    print('expert distribution (records):')
    for k, v in stats['expert_counter'].most_common():
        n_imgs = len(stats['images_per_expert'][k])
        print(f'  {k:20s}  records={v:6d}  unique_imgs={n_imgs}')

    # \u53c2\u6570\u7edf\u8ba1
    if samples_out:
        print('\n=== param stats ===')
        for k in ['ev_compensation', 'white_balance', 'contrast',
                 'shadows', 'highlights', 'saturation']:
            vals = [s['params'][k] for s in samples_out]
            print(f'  {k:20s}  n={len(vals)}  '
                  f'mean={sum(vals) / len(vals):+8.2f}  '
                  f'min={min(vals):+8.1f}  max={max(vals):+8.1f}')

    out = {
        'meta': {
            'pipeline': 'FiveK Expert Settings (params-only)',
            'source_json': args.settings_json,
            'source_imgs': args.jpeg_dir,
            'mode': args.mode,
            'require_full_params': args.require_full_params,
            'num_samples': len(samples_out),
            'created_at': datetime.now().isoformat(),
        },
        'samples': samples_out,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\nSaved: {args.output}  ({len(samples_out)} samples)')


if __name__ == '__main__':
    main()
