"""\u751f\u6210\u672c\u5730 HTML \u56fe\u5bf9\u6bd4\u9875.

\u7ed3\u679c: outputs/compare_firered_rewrite.html
5 \u884c (5 \u5f20\u6837\u672c) x 4 \u680f (orig / longcat-editB / firered-no-rewrite / firered-rewrite).
\u6bcf\u884c\u4e0a\u5934\u5199\u51fa\u7f16\u8f91\u6307\u4ee4, \u8ba9\u4f60\u80c9\u773c\u5224 rewrite \u662f\u5426\u63d0\u5347.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_HTML = ROOT / 'outputs/compare_firered_rewrite.html'

CAPTIONS = ROOT / 'data/compare_5_captions_edit.json'
COL_DIRS = [
    ('Original',            'firered_compare_editB',         '_orig.png'),
    ('LongCat editB',       'longcat_compare_editB',         '_longcat.png'),
    ('FireRed NO rewrite',  'firered_compare_editB',         '_firered.png'),
    ('FireRed WITH rewrite','firered_compare_editB_rewrite', '_firered.png'),
]


def main():
    cfg = json.load(open(CAPTIONS, encoding='utf-8'))
    samples = cfg['samples']

    html = ['<!doctype html><html><head><meta charset="utf-8">',
            '<title>FireRed rewrite vs no-rewrite</title>',
            '<style>',
            'body { font-family: -apple-system, "Segoe UI", sans-serif; '
            'background:#1a1a1a; color:#eee; margin:24px; }',
            'h1 { color:#fff; }',
            '.sample { margin-bottom:40px; border-bottom:1px solid #333; padding-bottom:20px; }',
            '.caption { font-size:14px; color:#9ee; margin-bottom:8px; }',
            '.row { display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; }',
            '.cell { background:#222; padding:8px; border-radius:6px; }',
            '.cell img { width:100%; height:auto; border-radius:4px; display:block; }',
            '.label { font-size:13px; color:#aaa; margin-bottom:6px; font-weight:600; }',
            '.missing { color:#f88; padding:20px; text-align:center; '
            'background:#331; border-radius:4px; }',
            '</style></head><body>',
            '<h1>FireRed Lightning: rewrite ON vs OFF  (with LongCat editB reference)</h1>',
            f'<p>Captions: {CAPTIONS.name} &middot; {len(samples)} samples &middot; '
            '4 columns per row &middot; Qwen3-VL scores (1-10): '
            'LongCat-editB=8.8 / FireRed-no-rewrite=9.0 / FireRed-rewrite=pending.</p>']

    for s in samples:
        idx = s['idx']
        html.append(f'<div class="sample">')
        html.append(f'<div class="caption"><b>idx={idx}</b> &middot; '
                    f'<code>{s["source_image"]}</code> &middot; '
                    f'{s["new_caption"]}</div>')
        html.append('<div class="row">')
        for label, subdir, suffix in COL_DIRS:
            fname = f'{idx:04d}{suffix}'
            rel_path = f'{subdir}/{fname}'
            abs_path = ROOT / 'outputs' / rel_path
            html.append('<div class="cell">')
            html.append(f'<div class="label">{label}</div>')
            if abs_path.exists():
                html.append(f'<img src="{rel_path}" alt="{label}" />')
            else:
                html.append(f'<div class="missing">missing:<br>{rel_path}</div>')
            html.append('</div>')
        html.append('</div>')
        html.append('</div>')

    html.append('</body></html>')

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text('\n'.join(html), encoding='utf-8')
    print(f'[OK] {OUT_HTML}')
    print(f'     open in browser: file:///{OUT_HTML.as_posix()}')


if __name__ == '__main__':
    main()
