"""3-\u6837\u672c smoke test \u9a8c\u8bc1 \u4fee\u6b63\u540e\u7684 SYSTEM_PROMPT \u662f\u5426\u4ea7\u751f\u5408\u6cd5\u7684 <tool_call>{...JSON...}</tool_call>.

\u5728\u91cd\u8dd1\u5168\u91cf 800 \u4e4b\u524d\u8dd1\u8fd9\u4e2a, \u907f\u514d\u4e4b\u524d "tool_call=block" \u7684\u574d\u5821.

\u9a8c\u8bc1\u9879:
  1. <think>\u4e0d\u4e3a\u7a7a + \u6709\u610f\u4e49 (>= 30 \u5b57)
  2. <tool_call> \u5fc5\u987b\u662f\u53ef\u89e3\u6790\u7684 JSON
  3. JSON \u91cc\u5fc5\u987b\u542b\u81f3\u5c11 1 \u4e2a Lightroom \u53c2\u6570\u540d (\u5982 temp/exposure/contrast)
  4. \u4e0d\u80fd\u662f\u5b57\u9762 "block"

\u7528\u6cd5 (\u5728 AutoDL):
  python tools/data/smoke_test_labeling.py
"""
import json
import sys
from pathlib import Path

# 复用主脚本的 prompt + label_one + parse_tool_call (唯一来源)
sys.path.insert(0, str(Path(__file__).parent))
from generate_cot_pseudo_labels import (  # type: ignore
    ALLOWED_LR_KEYS,
    SYSTEM_PROMPT_SHORT,
    label_one,
    parse_tool_call,
)

from openai import OpenAI


def main():
    root = Path('/root/autodl-tmp/datasets/ArtEdit-Bench/ArtEdit-Bench-Lr')

    # 选 3 个样本
    candidates = []
    for lang in ['CN', 'EN']:
        lang_dir = root / lang
        if not lang_dir.exists():
            continue
        for d in sorted(lang_dir.iterdir())[:5]:
            if d.is_dir():
                candidates.append(d)
    samples = candidates[:3]

    print(f'[INFO] testing {len(samples)} samples')
    print(f'[INFO] system prompt has "block" placeholder? '
          f'{"YES (BUG!)" if "block</" in SYSTEM_PROMPT_SHORT else "no (good)"}')
    print(f'[INFO] ALLOWED_LR_KEYS = {sorted(ALLOWED_LR_KEYS)}')

    client = OpenAI(api_key='0', base_url='http://localhost:8086/v1', timeout=180)

    n_ok = 0
    for i, d in enumerate(samples, 1):
        print(f'\n{"="*60}\n[{i}/{len(samples)}] {d}\n{"="*60}')
        result = label_one(client, d, task_type='lightroom', model='jarvisevo', timeout=180)
        if not result or 'error' in result:
            print(f'[ERR] {result.get("error", "empty result")}')
            continue

        print('--- USER WANT ---')
        print((d / 'user_want.txt').read_text(encoding='utf-8').strip()[:200])
        print('--- RAW RESPONSE ---')
        print(result.get('response', '')[:1500])
        print('--- PARSED tool_call ---')
        parsed = result.get('parsed_tool_call') or {}
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        print('--- ISSUES ---')
        issues = result.get('parse_issues') or []
        if issues:
            for iss in issues:
                print(f'  - {iss}')
        else:
            print('  (none)')

        print('--- VERDICT ---')
        if result.get('parsed_ok'):
            n_ok += 1
            print(f'[OK] {len(parsed)} keys: {list(parsed.keys())}')
        else:
            print(f'[FAIL] parsed_ok=False, only {len(parsed)} valid keys')

    print(f'\n{"="*60}')
    print(f'[SUMMARY] {n_ok} / {len(samples)} smoke samples pass validation')
    print(f'{"="*60}')
    if n_ok == len(samples):
        print('OK prompt fix verified, safe to launch full 800-sample batch')
    elif n_ok > 0:
        print('PARTIAL mixed: some bugs remain, inspect failing case')
    else:
        print('FAIL all failed; do NOT launch full batch yet')


if __name__ == '__main__':
    main()
