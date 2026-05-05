"""\u5c55\u793a Top 5 \u6700\u4f73 IP2P edit \u56fe\u4e0e\u4e2d\u8be5\u5904\u7406."""
import json
from pathlib import Path

d = json.load(open('data/aug_ip2p_full.json', encoding='utf-8'))
# Top 5 \u6309 edit_score \u964d\u5e8f, \u540c\u5206\u6309 delta \u964d\u5e8f
samples = sorted(d['samples'],
                 key=lambda s: (-s['score_edit'], -(s.get('score_delta') or 0)))[:5]

for i, s in enumerate(samples):
    print(f'### #{i+1}  idx={s["idx"]}  source={s["source_image"]}')
    print(f'  orig_score : {s["score_orig"]}  ({s.get("orig_bucket")})')
    print(f'  edit_score : {s["score_edit"]}  ({s.get("edit_bucket")})')
    print(f'  delta      : {s.get("score_delta")}')
    print(f'  orig_path  : {s["orig_path"]}')
    print(f'  edit_path  : {s["edit_path"]}')
    # \u68c0\u67e5\u6587\u4ef6\u5b58\u5728
    op = Path(s['orig_path'])
    ep = Path(s['edit_path'])
    print(f'  orig_exists: {op.exists()}  edit_exists: {ep.exists()}')
    p = (s.get('ip2p_prompt') or '').replace('\n', ' ')[:100]
    print(f'  prompt     : "{p}"')
    print()
