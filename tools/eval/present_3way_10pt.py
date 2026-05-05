"""\u6253\u5370 N \u7ec4 1-10 \u8bc4\u5206\u7ed3\u679c\u7684\u9010\u56fe\u660e\u7ec6 + \u603b\u5747\u503c\u3002

\u9ed8\u8ba4 4 \u7ec4: LongCat sceneA / LongCat editB / FireRed editB no-rewrite / FireRed editB rewrite
\u4e0d\u5b58\u5728\u7684 JSON \u4f1a\u88ab\u8df3\u8fc7\u5e76\u7ed9\u63d0\u793a\u3002
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
GROUPS = [
    ('LongCat sceneA (\u573a\u666f\u63cf\u8ff0)',
     ROOT / 'outputs/longcat_score_sceneA_10pt.json'),
    ('LongCat editB (\u7f16\u8f91\u6307\u4ee4)',
     ROOT / 'outputs/longcat_score_editB_10pt.json'),
    ('FireRed editB (\u7f16\u8f91\u6307\u4ee4, no rewrite)',
     ROOT / 'outputs/firered_score_editB_10pt.json'),
    ('FireRed editB (\u7f16\u8f91\u6307\u4ee4, with rewrite)',
     ROOT / 'outputs/firered_score_editB_rewrite_10pt.json'),
    ('LongCat editB (\u91cd\u5199 prompt)',
     ROOT / 'outputs/longcat_score_editB_rewritten_10pt.json'),
]


def main():
    summary_rows = []
    for label, path in GROUPS:
        if not path.exists():
            print(f'\n[\u8df3\u8fc7] {label}  (\u672a\u751f\u6210: {path.name})')
            continue
        data = json.load(open(path, encoding='utf-8'))
        means = data['meta']['means']
        print(f'\n{"=" * 70}')
        print(f'  {label}')
        print(f'  \u5747\u503c: {means}')
        print(f'{"=" * 70}')
        for r in data['results']:
            s = r['pred']['scores']
            print(f'  idx={r["idx"]:>4}  IF={s["instruction_follow"]:>2}  '
                  f'Aes={s["aesthetic"]:>2}  IdP={s["identity_preserve"]:>2}  '
                  f'Real={s["realism"]:>2}  Overall={s["overall"]:>2}')
            print(f'           reason: {s["reason"]}')
        summary_rows.append((label, means))

    # \u603b\u8868
    if summary_rows:
        print(f'\n\n{"=" * 70}')
        print('  \u603b\u5747\u503c\u4e00\u89c8 (1-10)')
        print(f'{"=" * 70}')
        print(f'  {"Group":<48}  IF  Aes  IdP  Real  Overall')
        print(f'  {"-" * 48}  --  ---  ---  ----  -------')
        for label, m in summary_rows:
            print(f'  {label:<48}  {m["instruction_follow"]:>3}  '
                  f'{m["aesthetic"]:>3}  {m["identity_preserve"]:>3}  '
                  f'{m["realism"]:>4}  {m["overall"]:>7}')


if __name__ == '__main__':
    main()
