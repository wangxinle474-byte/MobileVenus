"""\u83b7\u53d6 Top 5 \u6700\u4f73\u56fe\u7684 Venus \u63cf\u8ff0\uff0c\u4e3a CSGO/Qwen-Image-CN \u8bbe\u8ba1 'a photo of ...' caption\u3002"""
import json
from pathlib import Path

# Top 5 \u4e0a\u6b21\u8bc6\u522b\u51fa\u7684\u6700\u4f73\u56fe (\u6309 edit_score \u964d\u5e8f, \u540c\u5206\u6309 delta)
TOP5_IDX = [71, 448, 808, 110, 194]

# \u8bfb\u539f\u59cb\u63cf\u8ff0
prompts_data = json.load(open('data/venus_edit_prompts.json', encoding='utf-8'))
img2info = {p['image']: p for p in prompts_data['prompts']}

# \u8bfb\u4e4b\u524d\u8bc4\u5206
score_data = json.load(open('outputs/aug_ip2p_full_v1_reparsed.json', encoding='utf-8'))
idx2score = {r['idx']: r for r in score_data['results']}

print('=' * 80)
print('# Top 5 \u6700\u4f73\u56fe \u539f\u59cb\u4fe1\u606f\uff08\u4e3a\u8bbe\u8ba1\u65b0 caption \u51c6\u5907\uff09')
print('=' * 80)

for rank, idx in enumerate(TOP5_IDX, 1):
    s = idx2score.get(idx)
    if not s:
        print(f'\n#{rank}  idx={idx}  NOT FOUND')
        continue
    info = img2info.get(s['image'])
    print(f'\n--- #{rank}  idx={idx}  {s["image"]} ---')
    print(f'  score: {s["score_orig_v2"]} -> {s["score_edit_v2"]} (delta={s.get("delta_v2"):+.1f})')
    print(f'  orig_path: outputs/ip2p_pilot_100/{idx:04d}_orig.png')
    print(f'  ip2p_edit_path: outputs/ip2p_pilot_100/{idx:04d}_edit.png')
    if info:
        desc = info.get('description_full') or ''
        sugg = info.get('suggestion_full') or ''
        old_p = info.get('edit_prompt') or ''
        print(f'\n  [\u539f\u59cb Venus \u63cf\u8ff0]')
        print(f'    {desc[:400]}{"..." if len(desc) > 400 else ""}')
        print(f'\n  [\u539f\u59cb Venus \u5efa\u8bae]')
        print(f'    {sugg[:300]}{"..." if len(sugg) > 300 else ""}')
        print(f'\n  [\u4e4b\u524d\u7528\u7684 IP2P prompt]')
        print(f'    "{old_p}"')
