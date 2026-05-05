"""\u68c0\u67e5\u6240\u6709\u8bc4\u5206\u6e90\u7684\u7ed3\u6784."""
import json
from pathlib import Path


def inspect(path, sample_keys=False):
    p = Path(path)
    if not p.exists():
        print(f'\n{path}: NOT FOUND')
        return
    sz = p.stat().st_size / 1024 / 1024
    d = json.load(open(path, encoding='utf-8'))
    print(f'\n=== {path} ({sz:.2f} MB) ===')
    if isinstance(d, dict):
        print(f'top keys: {list(d.keys())}')
        for k, v in d.items():
            if isinstance(v, list):
                print(f'  {k}: list({len(v)})')
                if v and sample_keys:
                    if isinstance(v[0], dict):
                        print(f'    sample[0] keys: {list(v[0].keys())[:10]}')
                        print(f'    sample[0]: {json.dumps(v[0], ensure_ascii=False)[:300]}')
            elif isinstance(v, dict):
                print(f'  {k}: dict({len(v)} keys)')
                # \u5c55\u793a\u524d 2 \u4e2a\u5b50\u5b57\u6bb5
                for kk, vv in list(v.items())[:2]:
                    vs = json.dumps(vv, ensure_ascii=False)[:150]
                    print(f'    {kk}: {vs}')
            else:
                print(f'  {k}: {repr(v)[:100]}')
    elif isinstance(d, list):
        print(f'list({len(d)})')
        if d and sample_keys:
            print(f'sample[0]: {json.dumps(d[0], ensure_ascii=False)[:300]}')


# --- Venus \u76f8\u5173 ---
print('#' * 60)
print('# Venus \u8bc4\u5206\u6e90')
print('#' * 60)
inspect('data/venus_eval_results_all.json', sample_keys=True)
inspect('data/venus_pseudo_labels.json', sample_keys=True)
inspect('data/fivek_venus_labels.json', sample_keys=True)

# --- AesExpert \u76f8\u5173 ---
print('\n\n')
print('#' * 60)
print('# AesExpert \u8bc4\u5206\u6e90')
print('#' * 60)
inspect('data/fivek_aesthetic_scores.json', sample_keys=True)
inspect('data/fivek_expert_consensus.json', sample_keys=True)
inspect('outputs/fivek_aesexpert_scores.json', sample_keys=True)

# --- IP2P \u76f8\u5173 ---
print('\n\n')
print('#' * 60)
print('# IP2P \u4ea7\u7ebf\u8bc4\u5206')
print('#' * 60)
inspect('outputs/aug_ip2p_full_v1_reparsed.json', sample_keys=True)
