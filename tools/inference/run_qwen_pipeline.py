"""One-shot pipeline: inverse_fit Qwen wb data + filter/patch + merge for training.

Mirrors tools/data/data_prep/merge_wb_clean_to_master.py pattern (used by v9a_cleanwb).

Run this after Qwen data generation:
  python tools/run_qwen_pipeline.py

Then start training (uses --extra_jsonl to merge Qwen wb samples on top of master):
  python -m training.firered_baseline.train_lut --tag lut_v9a_qwen \\
    --extra_jsonl outputs/inverse_fit_pilot/qwen_wb_merged/pseudo_labels.jsonl \\
    2>&1 | Tee-Object -FilePath checkpoints/lut_v9a_qwen_log.txt
(Other args inherited from default v9a setup; check v9a_cleanwb training cmd if needed.)
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, **kw):
    print(f'\n[CMD] {" ".join(cmd)}')
    r = subprocess.run(cmd, cwd=ROOT, **kw)
    if r.returncode != 0:
        print(f'[FAIL] exit={r.returncode}')
        sys.exit(r.returncode)


def step1_inverse_fit_warm():
    """Inverse fit Qwen warm samples (25 outputs)."""
    out_dir = ROOT / 'outputs/inverse_fit_pilot/qwen_wb_warm'
    if out_dir.exists() and (out_dir / 'pseudo_labels.jsonl').exists():
        with (out_dir / 'pseudo_labels.jsonl').open(encoding='utf-8') as f:
            n = sum(1 for _ in f)
        print(f'[SKIP] qwen_wb_warm already has {n} pseudo_labels')
        return
    run(['python', '-u', 'tools/data/data_prep/inverse_fit_batch.py',
         '--captions', 'data/teacher_edits_fivek_wb_clean_warm.json',
         '--target_dir', 'outputs/teacher_edits/fivek_wb_clean_qwen_warm',
         '--out_dir', 'outputs/inverse_fit_pilot/qwen_wb_warm'])


def step2_inverse_fit_cool():
    """Inverse fit Qwen cool samples (12 outputs)."""
    out_dir = ROOT / 'outputs/inverse_fit_pilot/qwen_wb_cool'
    if out_dir.exists() and (out_dir / 'pseudo_labels.jsonl').exists():
        with (out_dir / 'pseudo_labels.jsonl').open(encoding='utf-8') as f:
            n = sum(1 for _ in f)
        print(f'[SKIP] qwen_wb_cool already has {n} pseudo_labels')
        return
    run(['python', '-u', 'tools/data/data_prep/inverse_fit_batch.py',
         '--captions', 'data/teacher_edits_fivek_wb_clean_cool.json',
         '--target_dir', 'outputs/teacher_edits/fivek_wb_clean_qwen_cool',
         '--out_dir', 'outputs/inverse_fit_pilot/qwen_wb_cool'])


L1_THRESHOLD = 0.15   # same as merge_wb_clean_to_master.py (v9a_cleanwb)


def _patch_sample(s: dict) -> dict:
    """Add 'action' and 'quality_tier' fields. Match merge_wb_clean_to_master.py."""
    s = dict(s)
    s['action'] = s.get('tone_target', 'wb')
    l1 = s.get('pixel_l1', 1.0)
    if l1 < 0.05:
        s['quality_tier'] = 'B good'
    elif l1 < 0.10:
        s['quality_tier'] = 'C acceptable'
    else:
        s['quality_tier'] = 'D dirty'   # filtered out by tier_filter at training time
    s['verdict'] = ('PASS' if l1 < 0.04
                     else ('WARN' if l1 < 0.10 else 'FAIL'))
    return s


def step3_merge_qwen_only():
    """Filter & patch Qwen pseudo_labels for training as --extra_jsonl.

    Output: outputs/inverse_fit_pilot/qwen_wb_merged/pseudo_labels.jsonl
    Mirrors v9a_cleanwb workflow exactly (drop pixel_l1 > 0.15, set action='wb').
    """
    qwen_warm = ROOT / 'outputs/inverse_fit_pilot/qwen_wb_warm/pseudo_labels.jsonl'
    qwen_cool = ROOT / 'outputs/inverse_fit_pilot/qwen_wb_cool/pseudo_labels.jsonl'
    out_dir = ROOT / 'outputs/inverse_fit_pilot/qwen_wb_merged'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'pseudo_labels.jsonl'

    all_samples = []
    for p in (qwen_warm, qwen_cool):
        if not p.exists():
            print(f'[WARN] missing {p}')
            continue
        with p.open(encoding='utf-8') as f:
            samples = [json.loads(l) for l in f if l.strip()]
        all_samples.extend(samples)
        print(f'  loaded {len(samples)} from {p.name}')

    patched = [_patch_sample(s) for s in all_samples]
    kept = [s for s in patched if s.get('pixel_l1', 1.0) <= L1_THRESHOLD]
    dropped = [s for s in patched if s.get('pixel_l1', 1.0) > L1_THRESHOLD]

    print(f'\n[FILTER] L1 threshold = {L1_THRESHOLD}')
    print(f'  kept   : {len(kept):2d} / {len(patched):2d}')
    print(f'  dropped: {len(dropped):2d} / {len(patched):2d}')

    from collections import Counter
    tier_counts = Counter(s['quality_tier'] for s in kept)
    print(f'  kept tiers: {dict(tier_counts)}')

    with out_file.open('w', encoding='utf-8') as f:
        for s in kept:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'\n[SAVED] {out_file}  ({len(kept)} samples)')

    # Compare with FireRed cleanwb merged for context
    fr_merged = ROOT / 'outputs/inverse_fit_pilot/wb_clean_merged/pseudo_labels.jsonl'
    if fr_merged.exists():
        with fr_merged.open(encoding='utf-8') as f:
            n_fr = sum(1 for l in f if l.strip())
        print(f'\n[CONTEXT] FireRed cleanwb merged has {n_fr} samples '
              f'(this is what v9a_cleanwb used)')


def main():
    print('=' * 70)
    print('Qwen wb data Pipeline')
    print('=' * 70)

    step1_inverse_fit_warm()
    step2_inverse_fit_cool()
    step3_merge_qwen_only()

    print('\n' + '=' * 70)
    print('NEXT STEP - run training:')
    print('  Use the same command as v9a_cleanwb but swap the --extra_jsonl path:')
    print('  (replicate v9a_cleanwb training cmd, with this jsonl)')
    print('  --extra_jsonl outputs/inverse_fit_pilot/qwen_wb_merged/pseudo_labels.jsonl')
    print('  --tag lut_v9a_qwen')
    print('=' * 70)


if __name__ == '__main__':
    main()
