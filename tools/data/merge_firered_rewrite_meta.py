"""\u5408\u5e76 firered_compare_editB_rewrite/\u4e0b\u524d\u540e\u4e24\u6b21\u8dd1\u7684 run_meta.json.

\u80cc\u666f: idx=448 \u9996\u6b21\u8dd1\u5931\u8d25 (\u7f51\u7edc\u4e2d\u65ad), \u7b2c\u4e8c\u6b21\u5355\u6837\u672c\u8865\u8dd1\u540e run_meta.json
\u4f1a\u88ab\u8986\u76d6\u4e3a\u5355\u6837\u672c, \u9700\u8981\u5408\u56de 5/5 \u7248\u672c\u3002
"""
import json
from pathlib import Path

d = Path(__file__).resolve().parent.parent.parent / 'outputs/firered_compare_editB_rewrite'
meta_old = json.load(open(d / 'run_meta_4of5.json', encoding='utf-8'))
meta_new = json.load(open(d / 'run_meta.json', encoding='utf-8'))

# old \u7684 records \u4e2d idx=448 \u662f fail; new \u7684 records \u662f idx=448 ok. \u5408\u5e76:
old_records = meta_old.get('records', [])
new_records = meta_new.get('records', [])
new_map = {r['idx']: r for r in new_records}
merged_records = []
for r in old_records:
    if r['idx'] in new_map:
        merged_records.append(new_map[r['idx']])  # \u7528\u65b0\u7684\u6210\u529f\u8bb0\u5f55\u66ff\u6362
    else:
        merged_records.append(r)
# \u6309 idx \u6392\u5e8f
merged_records.sort(key=lambda r: r['idx'])

merged = {
    **meta_old,
    'records': merged_records,
    'n_total': len(merged_records),
    'n_ok': sum(1 for r in merged_records if r.get('status') == 'ok'),
    'merge_note': '4of5 initial run + 1 supplemental run for idx=448 (network drop)',
}

out_path = d / 'run_meta.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f'[MERGE] {len(merged_records)} records, {merged["n_ok"]}/{merged["n_total"]} ok')
for r in merged_records:
    print(f'  idx={r["idx"]:>3}  status={r.get("status")}  runtime={r.get("runtime_sec", "?")}')
print(f'[OUT] {out_path}')
