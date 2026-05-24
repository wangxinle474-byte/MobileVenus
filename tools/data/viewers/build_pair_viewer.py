"""生成原图+编辑图对比 viewer.html。

用法:
  python tools/data/viewers/build_pair_viewer.py \
      --caption_json data/teacher_edits_firered_top20_styleC.json \
      --orig_rel ../originals \
      --edit_rel . \
      --title "FireRed Lightning top 20 (style C)" \
      --out outputs/teacher_edits/firered_top20/viewer.html

可选:
  --extra_dir 第三列对比 (相对 out_html 的路径)，比如对比 LongCat 输出
  --extra_label 第三列列名

字段约定:
  caption_json 必须有 samples 数组, 每条 sample 含 idx (int), new_caption (str)
  可选: rank, source_image, venus_scene, aesthetic_target, score_delta
"""
import argparse
import json
from pathlib import Path


HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 20px; }
  h1 { font-size: 18px; margin: 0 0 12px; color: #4a9eff; }
  .summary { font-size: 13px; color: #aaa; margin-bottom: 18px; line-height: 1.6; }
  .row { display: grid; grid-template-columns: %COLS%; gap: 12px; background: #2a2a2a; padding: 10px; border-radius: 8px; margin-bottom: 10px; align-items: start; }
  .row img { width: 100%; height: auto; display: block; border-radius: 4px; background: #000; }
  .meta { font-size: 12px; line-height: 1.45; color: #ccc; }
  .meta .idx { color: #ffa94d; font-weight: 600; font-size: 13px; }
  .meta .delta { color: #51cf66; }
  .meta .scene { color: #f783ac; }
  .meta .cap { color: #ddd; margin-top: 6px; padding: 6px 8px; background: #1f1f1f; border-radius: 4px; font-family: ui-monospace, "Cascadia Code", "Consolas", monospace; font-size: 11.5px; }
  .col-label { font-size: 11px; color: #888; text-align: center; margin-bottom: 4px; }
  .col-label.orig { color: #888; }
  .col-label.edit { color: #4a9eff; }
  .col-label.extra { color: #ffd43b; }
</style>
</head>
<body>
<h1>{title}</h1>
<div class="summary">{summary}</div>
"""

HTML_TAIL = """</body>
</html>
"""


def render_row(s, orig_rel, edit_rel, extra_rel=None, extra_label=None):
    idx = s['idx']
    fname = f'{idx:04d}.png'
    rank = s.get('rank', '')
    rank_str = f' rank #{rank}' if rank else ''
    score_delta = s.get('score_delta')
    delta_str = f' Δ={score_delta:+.1f}' if score_delta is not None else ''
    scene = s.get('venus_scene') or s.get('aesthetic_target') or ''
    cap = s.get('new_caption', '')

    cols = [
        f'<div><div class="col-label orig">ORIGINAL</div><img src="{orig_rel}/{fname}" alt="orig {idx}"></div>',
        f'<div><div class="col-label edit">EDIT</div><img src="{edit_rel}/{fname}" alt="edit {idx}"></div>',
    ]
    if extra_rel:
        label = extra_label or 'EXTRA'
        cols.append(
            f'<div><div class="col-label extra">{label}</div><img src="{extra_rel}/{fname}" alt="extra {idx}"></div>'
        )

    meta = (
        f'<div class="meta">'
        f'<div class="idx">idx={idx}{rank_str}{delta_str}</div>'
        + (f'<div class="scene">scene: {scene}</div>' if scene else '')
        + f'<div class="cap">{cap}</div>'
        + '</div>'
    )
    cols.append(meta)
    return '<div class="row">' + ''.join(cols) + '</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--caption_json', required=True)
    ap.add_argument('--orig_rel', required=True, help='原图目录相对 out html 的路径')
    ap.add_argument('--edit_rel', required=True, help='编辑图目录相对 out html 的路径')
    ap.add_argument('--extra_rel', default=None, help='可选第三列目录(相对路径)')
    ap.add_argument('--extra_label', default=None, help='可选第三列列名')
    ap.add_argument('--title', default='Image edit pair viewer')
    ap.add_argument('--out', required=True, help='输出 html 路径')
    args = ap.parse_args()

    with open(args.caption_json, encoding='utf-8') as f:
        d = json.load(f)
    samples = d.get('samples', [])
    md = d.get('metadata', {})

    n_cols = 3 + (1 if args.extra_rel else 0)
    col_template = '1fr 1fr 2fr' if n_cols == 3 else '1fr 1fr 1fr 2fr'

    summary_lines = []
    for k in ['purpose', 'caption_style', 'n_samples', 'caption_design_notes']:
        v = md.get(k)
        if v:
            summary_lines.append(f'<b>{k}:</b> {v}')
    summary_lines.append(f'<b>caption_json:</b> {args.caption_json}')
    summary_lines.append(f'<b>n_rows:</b> {len(samples)}')
    summary = '<br>'.join(summary_lines)

    head = (
        HTML_HEAD
        .replace('%COLS%', col_template)
        .replace('{title}', args.title)
        .replace('{summary}', summary)
    )

    rows = '\n'.join(
        render_row(s, args.orig_rel, args.edit_rel, args.extra_rel, args.extra_label)
        for s in samples
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(head + rows + HTML_TAIL, encoding='utf-8')
    print(f'[DONE] {len(samples)} rows -> {out_path}')


if __name__ == '__main__':
    main()
