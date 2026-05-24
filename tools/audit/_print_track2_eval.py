"""Pretty-print a Track 2 eval JSON summary."""
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
d = json.loads(p.read_text(encoding='utf-8'))

print(f'Checkpoint     : {d.get("ckpt", "?")}')
print(f'best_val_psnr  : {d.get("best_val_psnr_at_ckpt", "?")} @ Ep {d.get("epoch_at_ckpt", "?")}')
print(f'training jsonl : {d.get("training_jsonl", "?")}')
print()

for set_name, info in d.get('eval_sets', {}).items():
    print(f'=== {set_name} ===')
    print(f'  records: total={info.get("records_total", 0)}  '
          f'leaked={info.get("records_leaked", 0)}  '
          f'in_val={info.get("records_in_val", 0)}  '
          f'leak_free={info.get("records_leak_free", 0)}  '
          f'(leak_frac={info.get("leak_fraction", 0):.2%})')
    subsets = info.get('subsets', {})
    for sub, v in subsets.items():
        n = v.get('n', 0)
        ov = v.get('overall')
        if ov is None or n == 0:
            print(f'  {sub:<10} n={n:>5}  (empty)')
        else:
            print(f'  {sub:<10} n={n:>5}  overall={ov:6.2f} dB', end='')
            pa = v.get('per_action', {}) or {}
            if pa:
                parts = [f'{a}={pa[a]["mean"]:.2f}' for a in
                         ('contrast', 'saturation', 'shadows',
                          'highlights', 'wb')
                         if a in pa]
                print('  | ' + '  '.join(parts))
            else:
                print()
    print()
