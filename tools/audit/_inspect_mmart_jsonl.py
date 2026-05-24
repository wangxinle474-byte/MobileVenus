"""Quick spot-check of MMArt jsonl format + path existence."""
import json, os, sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1
         else 'outputs/mmart_pseudo_labels/v1_250/pseudo_labels.jsonl')
lines = p.read_text(encoding='utf-8').splitlines()
print(f'Total records: {len(lines)}')

# Spot-check first 3
for i in range(min(3, len(lines))):
    r = json.loads(lines[i])
    print(f'\n--- record {i} ---')
    for k in ['source_image', 'orig_path', 'target_path', 'action',
             'quality_tier', '_dominant_dev']:
        print(f'  {k}: {r.get(k)}')
    print('  P_inferred:')
    for pk, pv in r['P_inferred'].items():
        print(f'    {pk}: {pv:.3f}')
    print(f'  orig exists: {os.path.exists(r["orig_path"])}')
    print(f'  target exists: {os.path.exists(r["target_path"])}')

# Tally path existence over all records
n_orig_ok = sum(os.path.exists(json.loads(l)['orig_path']) for l in lines)
n_tgt_ok = sum(os.path.exists(json.loads(l)['target_path']) for l in lines)
print(f'\n--- Path existence ---')
print(f'  orig_path exists: {n_orig_ok}/{len(lines)}')
print(f'  target_path exists: {n_tgt_ok}/{len(lines)}')
