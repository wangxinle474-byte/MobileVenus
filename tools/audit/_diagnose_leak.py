"""Diagnose the leak detection mismatch in eval_track2.py.

Compares the keys used by `build_data` (train side) vs the canonical stems
emitted by `extract_fivek_stem` (eval side).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
from training.firered_baseline.train_lut import build_data  # noqa: E402
from tools._lib.merge_jsonl_for_joint_training import extract_fivek_stem  # noqa: E402

JSONL = Path('outputs/fivek_expert_c_master/pseudo_labels.jsonl')
TIERS = ('A excellent', 'B good', 'C acceptable')

print('--- raw JSONL records (first 3) ---')
with open(JSONL, encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        r = json.loads(line)
        si = r.get('source_image', '?')
        op = r.get('orig_path', '?')
        tp = r.get('target_path', '?')
        print(f'rec {i}: source_image={si!r}')
        print(f'         orig_path={op[-50:]!r}')
        print(f'         target_path={tp[-50:]!r}')
        print(f'         extract_fivek_stem -> {extract_fivek_stem(r)!r}')

print()
print('--- build_data output ---')
train_s, val_s = build_data(JSONL, 0.2, TIERS, 42)
print(f'train n={len(train_s)}  val n={len(val_s)}')
print(f'first train_s keys: {sorted(list(train_s[0].keys()))[:10]}')
print(f'first train_s source_image: {train_s[0].get("source_image", "?")!r}')
print(f'first train_s extract_fivek_stem: '
      f'{extract_fivek_stem(train_s[0])!r}')

print()
print('--- comparison: train_s "source_image" vs eval extract_fivek_stem ---')
train_si = set(s.get('source_image', '') for s in train_s)
val_si = set(s.get('source_image', '') for s in val_s)
print(f'  unique source_image in train: {len(train_si)}')
print(f'  unique source_image in val:   {len(val_si)}')
print(f'  sample 5 train source_image: {sorted(list(train_si))[:5]}')

# Now compute extract_fivek_stem over the same JSONL (eval side)
with open(JSONL, encoding='utf-8') as f:
    raw = [json.loads(l) for l in f if l.strip()]
raw = [r for r in raw if r.get('quality_tier', '') in TIERS]
canon = set(extract_fivek_stem(r) for r in raw)
print(f'  unique canonical stems (eval): {len(canon)}')
print(f'  sample 5 canonical stems:      {sorted(list(canon))[:5]}')

# Intersection
isect = train_si & canon
print(f'  train_si ∩ canon = {len(isect)} (should be ~12612)')
