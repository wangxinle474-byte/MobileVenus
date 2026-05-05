"""\u540c\u4e00\u5f20\u56fe\u5728\u6269\u5145\u524d/\u540e\u7684\u6570\u636e\u5bf9\u6bd4\u3002

\u9009 3~5 \u5f20 FiveK \u56fe (100% \u91cd\u53e0), \u5e76\u5c55\u793a 1 \u5f20 IP2P \u65b0\u6837\u672c\u4ee5\u5c55\u73b0\u5168\u65b0\u5b57\u6bb5\u3002
"""
import json
from collections import defaultdict


def load(p):
    return json.load(open(p, encoding='utf-8'))


# ------ \u8f7d\u5165 ------
inst = load('data/instruction_data.json')
aug_fk = load('data/aug_fivek_params_filtered.json')
aug_ip2p = load('data/aug_ip2p_full.json')
fk_scores = load('outputs/fivek_aesexpert_scores.json')

# \u5efa\u7d22\u5f15: image_name \u2192 list of records
inst_by_img = defaultdict(list)
for s in inst['samples']:
    if s.get('source') == 'fivek':
        inst_by_img[s['image_name']].append(s)

aug_fk_by_img = defaultdict(list)
for s in aug_fk['samples']:
    aug_fk_by_img[s['image_name']].append(s)

# \u9009 3 \u5f20\u5177\u4ee3\u8868\u6027\u7684: \u4e00\u5f20\u9ad8\u5206, \u4e00\u5f20\u4e2d\u5206, \u4e00\u5f20 expert \u624b\u8c03\u7684
pick_high = None   # score 8-10
pick_mid = None    # score 6-8
pick_expert = None  # expert_a \u6216 expert_b

for img, rs in aug_fk_by_img.items():
    sc = rs[0].get('orig_score')
    for r in rs:
        exp = r.get('expert', 'default')
        if exp in ('expert_a', 'expert_b') and pick_expert is None and sc and sc >= 6:
            pick_expert = img
    if pick_high is None and sc and sc >= 8:
        pick_high = img
    if pick_mid is None and sc and 6 <= sc < 8:
        pick_mid = img
    if pick_high and pick_mid and pick_expert:
        break

# \u4e0b\u9762\u7ed9\u6bcf\u5f20\u6253\u5370\u5bf9\u6bd4
def print_compare(img):
    print('\n' + '=' * 70)
    print(f'\u56fe: {img}')
    print('=' * 70)

    # --- \u6269\u5145\u524d (instruction_data) ---
    print('\n[\u6269\u5145\u524d \u2014 instruction_data.json]')
    insts = inst_by_img.get(img, [])
    print(f'  \u8be5\u56fe\u5728 instruction_data \u91cc\u7684\u6837\u672c\u6570: {len(insts)}')
    for i, s in enumerate(insts[:2]):  # \u5c55\u793a\u524d 2 \u4e2a
        print(f'\n  \u6837\u672c {i+1}:')
        print(f'    source         : {s.get("source")}')
        print(f'    instruction    : {repr(s.get("instruction", ""))[:80]}')
        bp = s.get('base_params', {})
        tp = s.get('target_params', {})
        dp = s.get('delta_params', {})
        print(f'    base_params    : {dict(bp)}')
        print(f'    target_params  : {dict(tp)}')
        print(f'    delta_params   : {dict(dp)}')

    # --- \u6269\u5145\u540e (aug_fivek_filtered) ---
    print('\n[\u6269\u5145\u540e \u2014 aug_fivek_params_filtered.json]')
    augs = aug_fk_by_img.get(img, [])
    print(f'  \u8be5\u56fe\u5728\u65b0\u6570\u636e\u91cc\u7684\u6837\u672c\u6570: {len(augs)}')
    # \u5206\u6309 expert \u5c55\u793a
    by_expert = defaultdict(list)
    for r in augs:
        by_expert[r.get('expert', '?')].append(r)
    for exp, rs in by_expert.items():
        print(f'\n  expert={exp}  ({len(rs)} \u6837\u672c\u53d8\u4f53)')
        r0 = rs[0]
        print(f'    orig_score     : {r0.get("orig_score")}  [{r0.get("orig_bucket")}]  <== [NEW] AesExpert \u8bc4\u5206')
        print(f'    params         : {r0.get("params")}')

    # --- AesExpert \u539f\u59cb\u54cd\u5e94 ---
    sc_entry = fk_scores.get('scores', {}).get(img)
    if sc_entry:
        print(f'\n  [AesExpert \u65b0\u589e\u5b57\u6bb5]')
        print(f'    raw response   : {repr(sc_entry.get("raw", ""))[:80]}')
        print(f'    parse_method   : {sc_entry.get("parse_method")}')


print('\n####### 3 \u5f20\u6837\u672c FiveK \u56fe\u7684\u5bf9\u6bd4 #######')
if pick_high:
    print_compare(pick_high)
if pick_mid:
    print_compare(pick_mid)
if pick_expert:
    print_compare(pick_expert)


# --- IP2P \u65b0\u751f\u6837\u672c (instruction_data \u5b8c\u5168\u6ca1\u6709) ---
print('\n\n####### IP2P \u65b0\u751f\u6837\u672c (\u6269\u5145\u524d\u4e0d\u5b58\u5728) #######')
s0 = aug_ip2p['samples'][0]
src_img = s0.get('source_image', '?')
print('\n' + '=' * 70)
print(f'\u56fe: {src_img}  (IP2P \u65b0\u751f)')
print('=' * 70)
# instruction_data \u91cc\u627e\u627e\u8fd9\u4e2a
hits = [r for r in inst['samples']
        if src_img in str(r.get('image_name', ''))]
print(f'\n[\u6269\u5145\u524d \u2014 instruction_data]')
print(f'  \u5339\u914d\u7684\u6837\u672c\u6570: {len(hits)}  \u2190 \u4e3a 0 \u8868\u793a IP2P \u7684 Benchmark \u56fe\u4e0d\u5728\u65e7\u8bad\u7ec3\u6570\u636e\u91cc')

print(f'\n[\u6269\u5145\u540e \u2014 aug_ip2p_full.json]')
print(f'  idx            : {s0.get("idx")}')
print(f'  source_image   : {s0.get("source_image")}')
print(f'  orig_path      : {s0.get("orig_path")}')
print(f'  edit_path      : {s0.get("edit_path")}       [NEW] IP2P \u7f16\u8f91\u540e\u7684\u76ee\u6807\u56fe')
print(f'  ip2p_prompt    : {repr(s0.get("ip2p_prompt", ""))[:100]}  [NEW] \u7f16\u8f91\u6307\u4ee4')
print(f'  venus_suggestion   [NEW] Venus \u5efa\u8bae:')
print(f'    {repr(s0.get("venus_suggestion", ""))[:200]}')
print(f'  score_orig     : {s0.get("score_orig")}  [{s0.get("orig_bucket")}]')
print(f'  score_edit     : {s0.get("score_edit")}  [{s0.get("edit_bucket")}]')
print(f'  score_delta    : {s0.get("score_delta")}     [NEW] \u63d0\u5347\u5206')
print(f'  params (\u50cf\u7d20\u53cd\u63a8): {s0.get("params")}')
