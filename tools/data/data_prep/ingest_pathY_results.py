"""Path Y: ingest AutoDL FireRed batch results into pseudo_labels.jsonl v2.

Pipeline:
  For each action in [contrast, saturation, shadows, highlights, wb]:
    1. Run inverse_fit_batch on per-action caption JSON + per-action PNG dir
    2. Append per-action pseudo_labels.jsonl
  Merge all 5 per-action files + add action + quality_tier
  Concat with existing 500-master jsonl → v2 master jsonl

Inputs (after AutoDL sync-down):
  data/pathY_captions/pathY_<action>.json     (5 caption files)
  outputs/teacher_edits/pathY_outputs/<action>/<idx>.png  (5 × 300 PNGs)

Outputs:
  outputs/inverse_fit_pilot/pathY_<action>/pseudo_labels.jsonl  (5 files)
  outputs/inverse_fit_pilot/pathY_2000_master/pseudo_labels.jsonl  (merged v2)

Usage:
    python tools/data/data_prep/ingest_pathY_results.py \\
        --pngs_root outputs/teacher_edits/pathY_outputs \\
        --skip_fit  # if you already ran inverse_fit_batch
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('ingest_pathY')

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Base 5 actions + wb_cooler direction-diversity supplement.
# wb_cooler records are tagged action='wb' with caption_variant='cooler' so
# Path X v11a WB head sees them as wb training samples.
ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb', 'wb_cooler']
WB_COOLER_ALIAS = {'wb_cooler': 'wb'}

# Quality tier thresholds (matches merge_wb_clean_to_master.py)
TIER_OK = 0.05      # < 0.05 = B good
TIER_WARN = 0.10    # < 0.10 = C acceptable
TIER_FAIL = 0.15    # >= 0.15 = D dirty (filtered out)


def assign_tier(pixel_l1: float) -> str:
    if pixel_l1 < TIER_OK:
        return 'B good'
    elif pixel_l1 < TIER_WARN:
        return 'C acceptable'
    elif pixel_l1 < TIER_FAIL:
        return 'D dirty'
    else:
        return 'D dirty'


def assign_verdict(pixel_l1: float) -> str:
    if pixel_l1 < 0.04:
        return 'OK'
    elif pixel_l1 < TIER_WARN:
        return 'WARN'
    else:
        return 'FAIL'


def run_inverse_fit(captions: Path, pngs_dir: Path, out_dir: Path,
                    device: str = 'cuda') -> Path:
    """Invoke inverse_fit_batch.py on a per-action caption + PNG dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / 'pseudo_labels.jsonl'
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        logger.info(f'[skip-fit] {jsonl_path} already exists '
                    f'({jsonl_path.stat().st_size} bytes)')
        return jsonl_path

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'tools/data/data_prep/inverse_fit_batch.py'),
        '--captions', str(captions),
        '--target_dir', str(pngs_dir),
        '--out_dir', str(out_dir),
        '--device', device,
        '--max_size', '512',
        '--n_restarts', '1',
        '--maxiter', '60',
        '--ssim_weight', '0.3',
    ]
    logger.info('  running inverse_fit_batch...')
    logger.info(f'  cmd: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f'  inverse_fit failed: {result.stderr[-500:]}')
        raise RuntimeError(f'inverse_fit failed for {captions.name}')
    logger.info(f'  produced: {jsonl_path}')
    return jsonl_path


def patch_record(rec: dict, action: str, source_dir: str) -> dict:
    """Add action, quality_tier, _source fields.

    For wb_cooler: normalize to action='wb' with caption_variant='cooler' so the
    v11a WB head sees them as wb training samples.
    """
    rec = dict(rec)
    if action == 'wb_cooler':
        rec['action'] = 'wb'
        rec['caption_variant'] = 'cooler'
    else:
        rec['action'] = action
        if action == 'wb':
            rec['caption_variant'] = 'warmer'
    pixel_l1 = rec.get('pixel_l1', 1.0)
    rec['quality_tier'] = assign_tier(pixel_l1)
    rec['verdict'] = assign_verdict(pixel_l1)
    rec['_source'] = source_dir
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--captions_dir',
                    default=str(PROJECT_ROOT / 'data/pathY_captions'))
    ap.add_argument('--pngs_root',
                    default=str(PROJECT_ROOT / 'outputs/teacher_edits/pathY_outputs'),
                    help='Root containing <action>/<idx>.png subdirs')
    ap.add_argument('--out_root',
                    default=str(PROJECT_ROOT / 'outputs/inverse_fit_pilot'),
                    help='Where per-action and merged jsonl go')
    ap.add_argument('--existing_master',
                    default=str(PROJECT_ROOT / 'outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl'),
                    help='Existing 500-master to concatenate with new ingest')
    ap.add_argument('--out_name', default='pathY_2000_master',
                    help='Output dir name for merged v2 master')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--skip_fit', action='store_true',
                    help='Skip inverse_fit (assume per-action jsonls exist)')
    args = ap.parse_args()

    captions_dir = Path(args.captions_dir)
    pngs_root = Path(args.pngs_root)
    out_root = Path(args.out_root)

    # === Step 1: per-action inverse_fit (or skip if already done) ===
    per_action_jsonls = []
    for action in ACTIONS:
        captions = captions_dir / f'pathY_{action}.json'
        pngs_dir = pngs_root / action
        out_dir = out_root / f'pathY_{action}'

        if not captions.exists():
            logger.error(f'[{action}] missing captions: {captions}')
            continue
        if not args.skip_fit and not pngs_dir.exists():
            logger.error(f'[{action}] missing PNG dir: {pngs_dir}')
            continue

        logger.info(f'\n=== [{action}] ===')
        logger.info(f'  captions: {captions}')
        logger.info(f'  pngs:     {pngs_dir}')
        logger.info(f'  out:      {out_dir}')

        if not args.skip_fit:
            jsonl = run_inverse_fit(captions, pngs_dir, out_dir, args.device)
        else:
            jsonl = out_dir / 'pseudo_labels.jsonl'
            if not jsonl.exists():
                logger.error(f'[{action}] --skip_fit but no jsonl: {jsonl}')
                continue
        per_action_jsonls.append((action, jsonl))

    # === Step 2: merge per-action jsonls + patch ===
    logger.info(f'\n=== Merging {len(per_action_jsonls)} per-action files ===')
    new_records = []
    for action, jsonl in per_action_jsonls:
        with open(jsonl, encoding='utf-8') as f:
            recs = [json.loads(l) for l in f if l.strip()]
        for rec in recs:
            rec = patch_record(rec, action, str(jsonl.parent))
            new_records.append(rec)
        logger.info(f'  {action:12s} +{len(recs)} records')
    logger.info(f'  total new : {len(new_records)}')

    # === Step 3: concat with existing master ===
    existing_master = Path(args.existing_master)
    existing_records = []
    if existing_master.exists():
        with open(existing_master, encoding='utf-8') as f:
            existing_records = [json.loads(l) for l in f if l.strip()]
        logger.info(f'  existing  : {len(existing_records)} '
                    f'(from {existing_master.name})')
    else:
        logger.warning(f'  no existing master at {existing_master}')

    merged = existing_records + new_records
    logger.info(f'  merged    : {len(merged)} total records')

    # === Step 4: stats + tier breakdown ===
    tier_counter = Counter(r.get('quality_tier', '?') for r in merged)
    action_counter = Counter(r.get('action', '?') for r in merged)
    new_tier_counter = Counter(r.get('quality_tier', '?') for r in new_records)
    new_action_counter = Counter(r.get('action', '?') for r in new_records)

    logger.info(f'\n  Merged action breakdown:')
    for a in ACTIONS:
        logger.info(f'    {a:12s} {action_counter.get(a, 0):4d} '
                    f'(+{new_action_counter.get(a, 0)} new)')

    logger.info(f'\n  Merged tier breakdown:')
    for t in sorted(tier_counter):
        logger.info(f'    {t:15s} {tier_counter[t]:4d} '
                    f'(+{new_tier_counter.get(t, 0)} new)')

    # === Step 5: write merged v2 master ===
    out_dir = out_root / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / 'pseudo_labels.jsonl'
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    summary = {
        'merged_master': str(out_jsonl),
        'total_records': len(merged),
        'existing_records': len(existing_records),
        'new_records': len(new_records),
        'action_counts': dict(action_counter),
        'tier_counts': dict(tier_counter),
        'new_action_counts': dict(new_action_counter),
        'new_tier_counts': dict(new_tier_counter),
        'usable_count': sum(1 for r in merged
                            if r.get('quality_tier', '').startswith(
                                ('A', 'B', 'C'))),
    }
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f'\n[OK] master v2 → {out_jsonl}')
    logger.info(f'  usable (A/B/C) : {summary["usable_count"]}')
    logger.info(f'  summary        : {summary_path}')


if __name__ == '__main__':
    main()
