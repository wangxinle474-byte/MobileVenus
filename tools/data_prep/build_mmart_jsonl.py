#!/usr/bin/env python
"""Download MMArt-PPR10k from HuggingFace mirror, parse XMP, build jsonl
in our v11a-compatible per-action format with full 7D targets from real
Lightroom params.

Schema (one record per sample):
    source_image: short id like 'mmart_1000_1.jpg'
    orig_path:    absolute path to before.jpg
    target_path:  absolute path to processed.jpg (real LR render)
    action:       dominant action by |normalized deviation| of 7D params
    tone_target:  same as action
    P_inferred:   {white_balance, brightness, contrast, shadows,
                   highlights, saturation, clarity} mapped from XMP
    quality_tier: 'A excellent'  (curated by expert artist preset selection)
    verdict:      'OK'
    _source:      'mmart_ppr10k'
    _mmart_sample_id: '1000_1'
    _xmp_raw:     dict of raw XMP fields (for inspection)
    _dominant_dev: float, magnitude of dominant normalized deviation

Usage:
    # Download + build full jsonl (~5 GB, all 4055 samples)
    python tools/build_mmart_jsonl.py \\
        --local_dir e:/MMArt_PPR10k \\
        --out_jsonl outputs/mmart_pseudo_labels/v1/pseudo_labels.jsonl

    # Small sample for validation
    python tools/build_mmart_jsonl.py --max_samples 100 \\
        --local_dir e:/MMArt_PPR10k_sample100 \\
        --out_jsonl outputs/mmart_pseudo_labels/v1_sample100/pseudo_labels.jsonl
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Use HF mirror by default (faster from CN)
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# Longer timeout for flaky network (default 10 s is too short for slow CN links).
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')

from huggingface_hub import snapshot_download  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────── XMP parsing ───────────────────────────
NS = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'crs': 'http://ns.adobe.com/camera-raw-settings/1.0/',
}


def _get_crs(desc: ET.Element, tag: str) -> str | None:
    """Get crs:tag attribute value from rdf:Description element."""
    full = f'{{{NS["crs"]}}}{tag}'
    return desc.attrib.get(full)


def _to_float(v: str | None, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        # LR uses "+5", "-79", "0.5" etc.
        return float(v.replace('+', ''))
    except ValueError:
        return default


def parse_xmp(xmp_path: Path) -> dict | None:
    """Extract camera-raw-settings (crs) fields from XMP sidecar file.

    Returns None if no valid crs:Description found.
    """
    try:
        tree = ET.parse(str(xmp_path))
    except ET.ParseError as e:
        logger.warning(f'XMP parse fail {xmp_path}: {e}')
        return None
    root = tree.getroot()
    for desc in root.iter(f'{{{NS["rdf"]}}}Description'):
        temp = _get_crs(desc, 'Temperature')
        if temp is None:
            continue
        return {
            'Temperature': _to_float(temp, 5500.0),
            'Tint': _to_float(_get_crs(desc, 'Tint')),
            'Exposure2012': _to_float(_get_crs(desc, 'Exposure2012')),
            'Contrast2012': _to_float(_get_crs(desc, 'Contrast2012')),
            'Shadows2012': _to_float(_get_crs(desc, 'Shadows2012')),
            'Highlights2012': _to_float(_get_crs(desc, 'Highlights2012')),
            'Whites2012': _to_float(_get_crs(desc, 'Whites2012')),
            'Blacks2012': _to_float(_get_crs(desc, 'Blacks2012')),
            'Saturation': _to_float(_get_crs(desc, 'Saturation')),
            'Vibrance': _to_float(_get_crs(desc, 'Vibrance')),
            'Clarity2012': _to_float(_get_crs(desc, 'Clarity2012')),
            'Dehaze': _to_float(_get_crs(desc, 'Dehaze')),
            'Texture': _to_float(_get_crs(desc, 'Texture')),
        }
    return None


# ─────────────────────────── 7D mapping ───────────────────────────
# (center, scale) per train_lut.py PARAM_NORM_7D
PARAM_NORM = {
    'white_balance': (6000.0, 4000.0),
    'brightness':    (0.0, 100.0),
    'contrast':      (0.0, 100.0),
    'shadows':       (0.0, 100.0),
    'highlights':    (0.0, 100.0),
    'saturation':    (0.0, 100.0),
    'clarity':       (0.0, 100.0),
}
PARAM_CLIP = {
    'white_balance': (2000.0, 10000.0),
    'brightness':    (-150.0, 150.0),
    'contrast':      (-100.0, 100.0),
    'shadows':       (-100.0, 100.0),
    'highlights':    (-100.0, 100.0),
    'saturation':    (-100.0, 100.0),
    'clarity':       (-100.0, 100.0),
}

# Per-action ↔ 7D param mapping for dominant-action selection
ACTION_TO_PARAM = {
    'contrast': 'contrast',
    'saturation': 'saturation',
    'shadows': 'shadows',
    'highlights': 'highlights',
    'wb': 'white_balance',
}

# Empirical per-action "typical edit magnitude" (deviation from neutral).
# Derived from observing 149 MMArt-PPR10k samples. LR users edit
# highlights very aggressively (p50≈-66), so a uniform normalization (/100)
# would classify ~88% of samples as highlights-dominant. Using these
# empirical scales rebalances the dominant-action distribution.
DOMINANT_ACTION_SCALES = {
    'wb': 700.0,          # K, typical ±700 from 6000
    'contrast': 12.0,      # ±12 typical |deviation|
    'saturation': 12.0,    # ±12 typical |deviation|
    'shadows': 30.0,       # ±30 typical |deviation|
    'highlights': 50.0,    # ±50 typical |deviation|; larger than others because
                           # LR users routinely apply -60 to -90 highlights cuts.
}


def _clip(name: str, v: float) -> float:
    lo, hi = PARAM_CLIP[name]
    return max(lo, min(hi, v))


def xmp_to_7d(xmp: dict, exposure_to_brightness_scale: float = 30.0) -> dict:
    """Map raw XMP crs fields to our 7D ISP parameter dict.

    Notes:
        - LR Exposure2012 is in EV stops [-5,+5]. Our `brightness` is on a
          [-150,150] scale. We multiply EV by 30 so 1 EV ≈ 30 in our scale
          (rough match against FireRed/Qwen pseudo brightness ranges).
        - LR `Saturation` and `Vibrance` are separate but operate on similar
          color richness. We sum them and clip to [-100,100] to fit our
          single `saturation` slot.
    """
    return {
        'white_balance': _clip('white_balance', xmp['Temperature']),
        'brightness':    _clip('brightness',
                               xmp['Exposure2012'] * exposure_to_brightness_scale),
        'contrast':      _clip('contrast', xmp['Contrast2012']),
        'shadows':       _clip('shadows', xmp['Shadows2012']),
        'highlights':    _clip('highlights', xmp['Highlights2012']),
        'saturation':    _clip('saturation',
                               xmp['Vibrance'] + xmp['Saturation']),
        'clarity':       _clip('clarity', xmp['Clarity2012']),
    }


def dominant_action(p7d: dict) -> tuple[str, float]:
    """Pick the ACTION whose deviation from neutral is largest, after dividing
    by per-action empirical edit magnitude. This gives a more balanced
    label distribution than uniform /100 normalization.

    Returns (action_name, deviation_magnitude_in_typical_units).
    """
    best_action, best_dev = 'contrast', -1.0
    for action, pname in ACTION_TO_PARAM.items():
        center, _ = PARAM_NORM[pname]      # neutral center
        scale = DOMINANT_ACTION_SCALES[action]
        dev = abs(p7d[pname] - center) / scale
        if dev > best_dev:
            best_dev = dev
            best_action = action
    return best_action, best_dev


# ─────────────────────────── Build pipeline ───────────────────────────
def _list_sample_ids(repo_id: str) -> list[str]:
    """Query HF dataset for sorted list of unique sample IDs under global/."""
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.dataset_info(repo_id)
    ids = set()
    for s in info.siblings:
        parts = s.rfilename.split('/')
        if parts and parts[0] == 'global' and len(parts) >= 3:
            ids.add(parts[1])
    return sorted(ids)


def _download_one(repo_id: str, rel: str, local_dir: Path,
                  max_retries: int = 4) -> tuple[str, bool, str]:
    """Download a single file with retry + 429-aware backoff."""
    from huggingface_hub import hf_hub_download
    import time
    import random
    # Skip if already present (resume).
    target = local_dir / rel
    if target.exists() and target.stat().st_size > 0:
        return (rel, True, 'cached')
    last_err = ''
    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(
                repo_id=repo_id, filename=rel, repo_type='dataset',
                local_dir=str(local_dir),
                cache_dir=str(local_dir / '.cache'),
                etag_timeout=180,
            )
            return (rel, True, f'attempt={attempt}')
        except Exception as e:
            last_err = repr(e)
            # 429 = rate limit; back off much longer + jitter.
            is_429 = '429' in last_err or 'TooManyRequests' in last_err
            if is_429:
                wait = 30 + 15 * attempt + random.uniform(0, 10)
            else:
                wait = min(30, 3 * attempt) + random.uniform(0, 3)
            time.sleep(wait)
    return (rel, False, last_err)


def download_mmart(local_dir: Path, repo_id: str = 'JarvisArt/MMArt-PPR10k',
                   max_samples: int | None = None,
                   max_retries: int = 4,
                   max_workers: int = 4,
                   which: str = 'all') -> None:
    """Download files via per-file hf_hub_download in parallel.

    Args:
        which: 'all' | 'xmp_only' | 'images_only'. Splitting phases is useful
            on slow networks: XMPs are 10 KB (fast); JPGs are 200-500 KB and
            may time out, so user can run XMP phase first to validate the
            parser, then images later.

    `snapshot_download` internally hits huggingface.co tree-listing endpoint
    which does NOT respect HF_ENDPOINT mirror (causes connect timeouts in CN).
    Direct hf_hub_download per file uses the mirror correctly.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    local_dir.mkdir(parents=True, exist_ok=True)

    file_groups = {
        'all': ('before.jpg', 'processed.jpg', 'config.xmp'),
        'xmp_only': ('config.xmp',),
        'images_only': ('before.jpg', 'processed.jpg'),
    }
    if which not in file_groups:
        raise ValueError(f'which must be one of {list(file_groups)}, got {which}')
    target_files = file_groups[which]

    # Build target rel-path list from pre-listed IDs.
    ids = _list_sample_ids(repo_id)
    if max_samples is not None:
        ids = ids[:max_samples]
    rels = []
    for sid in ids:
        for fname in target_files:
            rels.append(f'global/{sid}/{fname}')
    logger.info(f'Downloading {len(rels)} files ({len(ids)} samples × '
                f'{len(target_files)} files, which={which}, '
                f'workers={max_workers}, retries={max_retries}) → {local_dir}')

    n_ok, n_fail, n_cached = 0, 0, 0
    fail_examples = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_download_one, repo_id, rel, local_dir, max_retries): rel
            for rel in rels
        }
        for i, fut in enumerate(as_completed(futures)):
            rel, ok, msg = fut.result()
            if ok:
                if msg == 'cached':
                    n_cached += 1
                else:
                    n_ok += 1
            else:
                n_fail += 1
                if len(fail_examples) < 3:
                    fail_examples.append((rel, msg))
            if (i + 1) % 50 == 0 or (i + 1) == len(rels):
                logger.info(f'  progress {i+1}/{len(rels)}  '
                            f'ok={n_ok} cached={n_cached} fail={n_fail}')

    logger.info(f'Download done: ok={n_ok} cached={n_cached} fail={n_fail}')
    if n_fail > 0:
        logger.warning(f'First failures: {fail_examples}')
        logger.warning('Re-run the same command to retry remaining failures.')


def walk_samples(local_dir: Path, require_images: bool = True) -> list[dict]:
    """Walk local_dir/global/<id>/ and return list of available sample dicts.

    If require_images=False, only the XMP file must be present (paths to
    before/processed are still recorded for later use once they're downloaded).
    """
    samples = []
    global_dir = local_dir / 'global'
    if not global_dir.exists():
        logger.error(f'{global_dir} does not exist')
        return samples
    n_xmp_only = 0
    for sample_dir in sorted(global_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        before = sample_dir / 'before.jpg'
        processed = sample_dir / 'processed.jpg'
        xmp = sample_dir / 'config.xmp'
        if not xmp.exists():
            continue
        if not (before.exists() and processed.exists()):
            if require_images:
                continue
            n_xmp_only += 1
        samples.append({
            'sample_id': sample_dir.name,
            'before': before,
            'processed': processed,
            'xmp': xmp,
        })
    msg = f'Found {len(samples)} samples in {global_dir}'
    if not require_images:
        msg += f' ({n_xmp_only} xmp-only, images pending download)'
    logger.info(msg)
    return samples


def build_records(samples: list[dict], rank_start: int = 1) -> list[dict]:
    """Parse each XMP, map to 7D, compute dominant action, build jsonl record."""
    records = []
    fails = 0
    action_counts = {a: 0 for a in ACTION_TO_PARAM.keys()}
    for i, s in enumerate(samples):
        xmp_raw = parse_xmp(s['xmp'])
        if xmp_raw is None:
            fails += 1
            continue
        p7d = xmp_to_7d(xmp_raw)
        action, dev = dominant_action(p7d)
        action_counts[action] += 1
        rec = {
            'rank': rank_start + i,
            'idx': i,
            'source_image': f"mmart_{s['sample_id']}.jpg",
            'orig_path': str(s['before'].resolve()),
            'target_path': str(s['processed'].resolve()),
            'caption': f"Apply Lightroom edit (dominant: {action}).",
            'tone_target': action,
            'action': action,
            'P_inferred': p7d,
            'verdict': 'OK',
            '_source': 'mmart_ppr10k',
            '_mmart_sample_id': s['sample_id'],
            '_xmp_raw': xmp_raw,
            '_dominant_dev': float(dev),
            'quality_tier': 'A excellent',
        }
        records.append(rec)
    logger.info(f'Built {len(records)} records, {fails} XMP parse fails')
    logger.info(f'Dominant action distribution: {action_counts}')
    return records


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    logger.info(f'Wrote {len(records)} records → {out_path}')


def summarize(records: list[dict]) -> None:
    """Print summary stats of the 7D parameter ranges."""
    import statistics as stat
    keys = ['white_balance', 'brightness', 'contrast', 'shadows',
            'highlights', 'saturation', 'clarity']
    print('\n=== 7D parameter distribution ===')
    print(f"{'param':>15} {'min':>10} {'p10':>10} {'p50':>10} {'p90':>10} {'max':>10} {'mean':>10}")
    for k in keys:
        vals = [r['P_inferred'][k] for r in records]
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        if n == 0:
            continue
        p10 = vals_sorted[int(n * 0.10)]
        p50 = vals_sorted[int(n * 0.50)]
        p90 = vals_sorted[int(n * 0.90)]
        print(f"{k:>15} {min(vals):>10.2f} {p10:>10.2f} {p50:>10.2f} "
              f"{p90:>10.2f} {max(vals):>10.2f} {stat.mean(vals):>10.2f}")

    # Dominant action distribution
    action_counts = {}
    for r in records:
        action_counts[r['action']] = action_counts.get(r['action'], 0) + 1
    print('\n=== dominant action distribution ===')
    total = len(records)
    for a in sorted(action_counts, key=lambda x: -action_counts[x]):
        print(f"  {a:>15}: {action_counts[a]:>5} ({100*action_counts[a]/total:5.1f}%)")


# ─────────────────────────── CLI ───────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo_id', default='JarvisArt/MMArt-PPR10k')
    p.add_argument('--local_dir', type=Path, default=Path('e:/MMArt_PPR10k'),
                   help='Where to download the dataset triplets')
    p.add_argument('--out_jsonl', type=Path,
                   default=Path('outputs/mmart_pseudo_labels/v1/pseudo_labels.jsonl'),
                   help='Output jsonl path')
    p.add_argument('--max_samples', type=int, default=None,
                   help='If set, only download/process this many samples')
    p.add_argument('--which', choices=['all', 'xmp_only', 'images_only'],
                   default='all',
                   help='What to download. xmp_only is fast (~40MB total) and '
                        'lets you validate the parser before pulling images.')
    p.add_argument('--max_workers', type=int, default=4,
                   help='Parallel HF download workers (lower=more stable on '
                        'slow links).')
    p.add_argument('--max_retries', type=int, default=4)
    p.add_argument('--download_only', action='store_true',
                   help='Skip jsonl build, only download')
    p.add_argument('--parse_only', action='store_true',
                   help='Skip download, only parse local files')
    p.add_argument('--allow_missing_images', action='store_true',
                   help='Build records using XMP even if before.jpg/processed.jpg '
                        'are missing locally (orig_path / target_path will still '
                        'be filled with absolute paths pointing to where the '
                        'images SHOULD be once downloaded).')
    args = p.parse_args()

    if not args.parse_only:
        download_mmart(args.local_dir, args.repo_id,
                       max_samples=args.max_samples,
                       max_workers=args.max_workers,
                       max_retries=args.max_retries,
                       which=args.which)
    if args.download_only:
        logger.info('download_only=True, exiting before parse')
        return

    samples = walk_samples(args.local_dir,
                          require_images=not args.allow_missing_images)
    if args.max_samples:
        samples = samples[:args.max_samples]
    records = build_records(samples)
    write_jsonl(records, args.out_jsonl)
    summarize(records)


if __name__ == '__main__':
    main()
