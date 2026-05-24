"""打印 master summary.json 简报."""
import json
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else
         'outputs/inverse_fit_pilot/fivek_500_master/summary.json')
with open(p, encoding='utf-8') as f:
    d = json.load(f)

print(f'\n{"="*78}\nOVERALL  (n={d["n_total"]})\n{"="*78}')
print(f'  L1 mean   : {d["l1_mean"]:.4f}')
print(f'  L1 median : {d["l1_median"]:.4f}')
print(f'  L1 max    : {d["l1_max"]:.4f}')
print(f'  delta mean: {d["delta_mean"]:.4f}')
print(f'  usable    : {d["usable_count"]}/{d["n_total"]} '
      f'({100*d["usable_count"]/d["n_total"]:.1f}%)')
print(f'  good      : {d["good_count"]}/{d["n_total"]} '
      f'({100*d["good_count"]/d["n_total"]:.1f}%)')
tc = d['tier_counts']
print(f'  tiers     : A_excellent={tc.get("A excellent", 0)}  '
      f'B_good={tc.get("B good", 0)}  '
      f'C_acceptable={tc.get("C acceptable", 0)}  '
      f'D_fail={tc.get("D fail", 0)}')

print(f'\n{"="*78}\nPER-ACTION\n{"="*78}')
print(f'{"action":<11s} {"n":>3s} {"L1_mean":>8s} {"L1_med":>8s} '
      f'{"L1_max":>8s} {"usable":>8s} {"good":>5s}  tier_counts')
for a, s in d['per_action'].items():
    tc = s['tier_counts']
    tc_str = '  '.join(f'{k.split()[0]}={v}' for k, v in
                       sorted(tc.items(), key=lambda x: x[0]))
    print(f'{a:<11s} {s["n"]:>3d} {s["l1_mean"]:>8.4f} {s["l1_median"]:>8.4f} '
          f'{s["l1_max"]:>8.4f} {s["usable_count"]:>3d}/{s["n"]:<3d} '
          f'{s["good_count"]:>5d}  {tc_str}')
