"""v10a Pipeline — Re-run inverse_fit on Qwen wb data with 9D PARAM_SPEC
(adds wb_u, wb_v 2D chromaticity offset, off-Planckian).

Outputs new directories with _v10a suffix so 7D vs 9D can be compared:
  outputs/inverse_fit_pilot/qwen_wb_warm_v10a/
  outputs/inverse_fit_pilot/qwen_wb_cool_v10a/
  outputs/inverse_fit_pilot/qwen_wb_merged_v10a/

After running, compare with 7D baselines in *qwen_wb_warm/, *qwen_wb_cool/.
"""
from __future__ import annotations
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd, **kw):
    print(f'\n[CMD] {" ".join(map(str, cmd))}\n')
    r = subprocess.run(cmd, cwd=ROOT, **kw)
    if r.returncode != 0:
        print(f'[FAIL] exit={r.returncode}')
        sys.exit(r.returncode)


def step1_inverse_fit_warm_v10a():
    """Inverse fit Qwen warm with 9D PARAM_SPEC."""
    out_dir = ROOT / 'outputs/inverse_fit_pilot/qwen_wb_warm_v10a'
    if out_dir.exists() and (out_dir / 'pseudo_labels.jsonl').exists():
        with (out_dir / 'pseudo_labels.jsonl').open(encoding='utf-8') as f:
            n = sum(1 for _ in f)
        print(f'[SKIP] qwen_wb_warm_v10a already has {n} pseudo_labels')
        return
    run(['python', '-u', 'tools/data/data_prep/inverse_fit_batch.py',
         '--captions', 'data/teacher_edits_fivek_wb_clean_warm.json',
         '--target_dir', 'outputs/teacher_edits/fivek_wb_clean_qwen_warm',
         '--out_dir', 'outputs/inverse_fit_pilot/qwen_wb_warm_v10a'])


def step2_inverse_fit_cool_v10a():
    """Inverse fit Qwen cool with 9D PARAM_SPEC."""
    out_dir = ROOT / 'outputs/inverse_fit_pilot/qwen_wb_cool_v10a'
    if out_dir.exists() and (out_dir / 'pseudo_labels.jsonl').exists():
        with (out_dir / 'pseudo_labels.jsonl').open(encoding='utf-8') as f:
            n = sum(1 for _ in f)
        print(f'[SKIP] qwen_wb_cool_v10a already has {n} pseudo_labels')
        return
    run(['python', '-u', 'tools/data/data_prep/inverse_fit_batch.py',
         '--captions', 'data/teacher_edits_fivek_wb_clean_cool.json',
         '--target_dir', 'outputs/teacher_edits/fivek_wb_clean_qwen_cool',
         '--out_dir', 'outputs/inverse_fit_pilot/qwen_wb_cool_v10a'])


L1_THRESHOLD = 0.15


def _patch_sample(s: dict) -> dict:
    s = dict(s)
    s['action'] = s.get('tone_target', 'wb')
    l1 = s.get('pixel_l1', 1.0)
    if l1 < 0.05:
        s['quality_tier'] = 'B good'
    elif l1 < 0.10:
        s['quality_tier'] = 'C acceptable'
    else:
        s['quality_tier'] = 'D dirty'
    s['verdict'] = ('PASS' if l1 < 0.04
                     else ('WARN' if l1 < 0.10 else 'FAIL'))
    return s


def step3_merge_v10a():
    """Merge 9D pseudo_labels for v10a model training."""
    qwen_warm = ROOT / 'outputs/inverse_fit_pilot/qwen_wb_warm_v10a/pseudo_labels.jsonl'
    qwen_cool = ROOT / 'outputs/inverse_fit_pilot/qwen_wb_cool_v10a/pseudo_labels.jsonl'
    out_dir = ROOT / 'outputs/inverse_fit_pilot/qwen_wb_merged_v10a'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'pseudo_labels.jsonl'

    all_samples = []
    for p in (qwen_warm, qwen_cool):
        if not p.exists():
            print(f'[WARN] {p} not found, skipping')
            continue
        with p.open(encoding='utf-8') as f:
            for line in f:
                all_samples.append(json.loads(line))
        print(f'  loaded {len(all_samples)} from {p.name}')

    # Filter
    print(f'\n[FILTER] L1 threshold = {L1_THRESHOLD}')
    kept = [s for s in all_samples if s.get('pixel_l1', 1.0) < L1_THRESHOLD]
    dropped = len(all_samples) - len(kept)
    print(f'  kept   : {len(kept)} / {len(all_samples)}')
    print(f'  dropped: {dropped} / {len(all_samples)}')

    # Patch
    patched = [_patch_sample(s) for s in kept]
    tier_dist = {}
    for s in patched:
        t = s['quality_tier']
        tier_dist[t] = tier_dist.get(t, 0) + 1
    print(f'  kept tiers: {tier_dist}')

    # Save
    with out_file.open('w', encoding='utf-8') as f:
        for s in patched:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'\n[SAVED] {out_file}  ({len(patched)} samples)')

    # wb_u, wb_v stats (v10a-specific)
    if patched:
        wb_us = [s['P_inferred'].get('wb_u', 0) for s in patched]
        wb_vs = [s['P_inferred'].get('wb_v', 0) for s in patched]
        print(f'\n[v10a] wb_u stats: mean={sum(wb_us)/len(wb_us):+.3f}  '
              f'range=[{min(wb_us):+.3f}, {max(wb_us):+.3f}]')
        print(f'[v10a] wb_v stats: mean={sum(wb_vs)/len(wb_vs):+.3f}  '
              f'range=[{min(wb_vs):+.3f}, {max(wb_vs):+.3f}]')


def step4_compare_summaries():
    """Print 7D vs 9D PASS rate / l1 / params_stats comparison."""
    print('\n' + '=' * 70)
    print('7D vs 9D Inverse Fit Comparison')
    print('=' * 70)
    for sub in ('qwen_wb_warm', 'qwen_wb_cool'):
        d7 = ROOT / f'outputs/inverse_fit_pilot/{sub}/summary.json'
        d9 = ROOT / f'outputs/inverse_fit_pilot/{sub}_v10a/summary.json'
        if not d7.exists() or not d9.exists():
            print(f'[SKIP] {sub}: missing summary')
            continue
        s7 = json.loads(d7.read_text(encoding='utf-8'))
        s9 = json.loads(d9.read_text(encoding='utf-8'))
        print(f'\n--- {sub} ---')
        for label, s in (('7D', s7), ('9D', s9)):
            l1 = s['pixel_l1']
            print(f'  {label}: n={s["n_fitted"]:2d}  '
                  f'l1 mean={l1["mean"]:.4f} median={l1["median"]:.4f} '
                  f'p25={l1["p25"]:.4f} p75={l1["p75"]:.4f}  '
                  f'verdicts={s["verdict_counts"]}')


def main():
    print('=' * 70)
    print('v10a Pipeline: 9D Inverse Fit (Off-Planckian, +wb_u, +wb_v)')
    print('=' * 70)
    step1_inverse_fit_warm_v10a()
    step2_inverse_fit_cool_v10a()
    step3_merge_v10a()
    step4_compare_summaries()


if __name__ == '__main__':
    main()
