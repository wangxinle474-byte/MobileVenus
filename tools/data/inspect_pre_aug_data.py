"""\u68c0\u67e5\u6269\u5145\u524d\u7684\u8bad\u7ec3\u6570\u636e\u96c6\u6587\u4ef6\u3002"""
import json
from pathlib import Path

FILES = [
    'data/instruction_data.json',
    'data/fivek_expert_params.json',
    'data/ppr10k_params.json',
    'data/fivek_expert_consensus.json',
    'data/fivek_venus_labels.json',
    'data/venus_pseudo_labels.json',
    'data/fivek_aesthetic_scores.json',
    'data/venus_edit_prompts.json',
]

LIST_KEYS = ['samples', 'results', 'data', 'pairs', 'prompts',
             'items', 'scores', 'labels', 'instructions']


def inspect(path):
    p = Path(path)
    if not p.exists():
        return f'{path}: NOT FOUND'
    sz = p.stat().st_size / 1024 / 1024
    try:
        d = json.load(open(path, encoding='utf-8'))
    except Exception as e:
        return f'{path}: PARSE ERR {e}'

    out = [f'{path}  ({sz:.2f} MB)']
    if isinstance(d, list):
        out.append(f'  type=list, len={len(d)}')
        if d:
            out.append(f'  first keys: {list(d[0].keys())[:10]}')
    elif isinstance(d, dict):
        out.append(f'  type=dict, top_keys: {list(d.keys())[:10]}')
        for k in LIST_KEYS:
            if k in d and isinstance(d[k], list):
                out.append(f'  d["{k}"]=list({len(d[k])})')
                if d[k] and isinstance(d[k][0], dict):
                    out.append(f'    sample keys: {list(d[k][0].keys())[:10]}')
                break
        # \u68c0\u67e5\u662f\u5426\u662f\u591a\u4e2a\u9876\u5c42 list
        for k, v in d.items():
            if isinstance(v, list) and k not in LIST_KEYS and len(v) > 5:
                out.append(f'  d["{k}"]=list({len(v)})')
    return '\n'.join(out)


for f in FILES:
    print(inspect(f))
    print()
