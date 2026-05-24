"""为 per-action pseudo-labels 生成单页 HTML viewer (支持 N=100 / 500 等任意规模).

输出 <master_dir>/viewer.html, 包含:
  - 顶部 summary 卡片 (overall + per-action)
  - 所有样本 4-panel 缩略图 (orig | target | rendered_back | diff_x5)
  - 每张展示: action / idx / L1 / delta / tier badge
  - 过滤: action × 5 checkbox, tier × 4 checkbox, sort by L1 asc/desc
"""
import argparse
import json
from pathlib import Path

ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']


def tier(l1: float) -> str:
    if l1 < 0.02: return 'A'
    if l1 < 0.05: return 'B'
    if l1 < 0.10: return 'C'
    return 'D'


def tier_color(t: str) -> str:
    return {'A': '#10b981', 'B': '#22c55e', 'C': '#f59e0b', 'D': '#ef4444'}[t]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--master_dir',
                    default='outputs/inverse_fit_pilot/fivek_per_action_master',
                    help='含 pseudo_labels.jsonl + summary.json 的 master 目录')
    ap.add_argument('--per_action_roots', nargs='+',
                    default=['outputs/inverse_fit_pilot/per_action'],
                    help='per-action 根目录 (可多个), 用于查找 comparison 图路径')
    ap.add_argument('--title', default=None,
                    help='页面标题')
    args = ap.parse_args()

    master = Path(args.master_dir)
    records = []
    with open(master / 'pseudo_labels.jsonl', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            records.append(r)
    with open(master / 'summary.json', encoding='utf-8') as f:
        summary = json.load(f)

    # 预计算每个 (action, idx) 在哪个 per_action_root 下的 comparison 可用
    def find_img_rel(action: str, idx: int) -> str:
        for root in args.per_action_roots:
            p = Path(root) / action / 'comparison' / f'{idx:04d}.png'
            if p.exists():
                try:
                    rel = p.resolve().relative_to(master.resolve(), walk_up=True)
                except (ValueError, TypeError):
                    # Py<3.12 或跨卷: 手工计算相对路径
                    import os
                    rel = Path(os.path.relpath(p.resolve(), master.resolve()))
                return rel.as_posix()
        return f'../per_action/{action}/comparison/{idx:04d}.png'

    items_json = []
    for r in records:
        action = r['action']
        idx = r['idx']
        img_rel = find_img_rel(action, idx)
        t = tier(r['pixel_l1'])
        items_json.append({
            'action': action,
            'idx': idx,
            'l1': round(r['pixel_l1'], 4),
            'delta': round(r['delta_target_orig'], 4),
            'tier': t,
            'src': r.get('source_image', ''),
            'cap': r.get('caption', ''),
            'img': img_rel,
            'p': r.get('P_inferred', {}),
        })

    # 按 L1 升序排
    items_json.sort(key=lambda x: x['l1'])

    overall = summary
    per_a = summary['per_action']

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{args.title or f'FiveK per-action pseudo-labels viewer (N={len(records)})'}</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 16px;
         background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 18px; margin: 0 0 12px; }}
  .summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
  .card {{ background: #1e293b; padding: 10px 16px; border-radius: 6px;
           border: 1px solid #334155; min-width: 130px; }}
  .card .v {{ font-size: 22px; font-weight: bold; color: #38bdf8; }}
  .card .l {{ font-size: 12px; color: #94a3b8; }}
  .pa-table {{ background: #1e293b; border-collapse: collapse;
               font-size: 13px; margin-bottom: 16px; }}
  .pa-table th, .pa-table td {{ padding: 6px 10px; border: 1px solid #334155; }}
  .pa-table th {{ background: #334155; color: #fbbf24; }}
  .filters {{ background: #1e293b; padding: 12px 16px; border-radius: 6px;
              border: 1px solid #334155; margin-bottom: 16px; }}
  .filters label {{ margin-right: 12px; cursor: pointer; }}
  .filters input {{ margin-right: 4px; }}
  .filters .group {{ display: inline-block; margin-right: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
  .item {{ background: #1e293b; border-radius: 6px; overflow: hidden;
           border: 1px solid #334155; }}
  .item .imgwrap {{ position: relative; }}
  .item img {{ width: 100%; display: block; }}
  .col-headers {{ display: grid; grid-template-columns: repeat(4, 1fr);
                  font-size: 11px; color: #cbd5e1; background: #0f172a;
                  padding: 4px 0; text-align: center; border-bottom: 1px solid #334155; }}
  .col-headers div {{ border-right: 1px solid #334155; padding: 2px 4px; }}
  .col-headers div:last-child {{ border-right: none; }}
  .col-headers .h-orig {{ color: #94a3b8; }}
  .col-headers .h-tgt  {{ color: #fbbf24; }}
  .col-headers .h-back {{ color: #38bdf8; }}
  .col-headers .h-diff {{ color: #ef4444; }}
  .meta {{ padding: 8px 12px; font-size: 12px; line-height: 1.5; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px;
            color: #fff; font-weight: bold; margin-right: 6px; }}
  .row {{ display: flex; justify-content: space-between; align-items: center;
          margin-bottom: 4px; }}
  .row .l {{ color: #94a3b8; }}
  .cap {{ color: #cbd5e1; font-style: italic; max-width: 100%;
          text-overflow: ellipsis; overflow: hidden; white-space: nowrap; }}
  .pp {{ font-size: 11px; color: #94a3b8; }}
  .hide {{ display: none; }}
</style>
</head>
<body>
<h1>FiveK per-action pseudo-labels (N=100) · Adam solver · ssim_w=0.5</h1>
<div style="font-size: 12px; color: #94a3b8; margin-bottom: 12px;">
  每个对比图 4 列: <b style="color:#94a3b8">orig (FiveK 默认 raw 渲染)</b> ·
  <b style="color:#fbbf24">target (FireRed 编辑)</b> ·
  <b style="color:#38bdf8">rendered_back (P_inferred 反推后 ISP 渲染)</b> ·
  <b style="color:#ef4444">diff×5</b>
  <br>
  <b>参数现状</b>: orig 是 <code>use_camera_wb=True</code> 的默认 DNG→JPEG 渲染 (无人工编辑);
  target 是 FireRed 1.1 基于 caption 的生成式编辑 (无 ISP 参数);
  <b>P_inferred 是 inverse_fit 的估计参数</b>, 非 GT.
  <br>
  (注: FiveK Expert C 真人修图 JPEG 在 <code>fivek_expert_c/</code> 目录, 有对应 Lightroom 参数, 但本实验未使用.)
</div>

<div class="summary">
  <div class="card"><div class="v">{overall["n_total"]}</div><div class="l">total</div></div>
  <div class="card"><div class="v">{overall["usable_count"]}</div><div class="l">usable (L1&lt;0.10)</div></div>
  <div class="card"><div class="v">{overall["good_count"]}</div><div class="l">good (L1&lt;0.05)</div></div>
  <div class="card"><div class="v">{overall["l1_median"]:.4f}</div><div class="l">L1 median</div></div>
  <div class="card"><div class="v">{overall["l1_mean"]:.4f}</div><div class="l">L1 mean</div></div>
  <div class="card"><div class="v">{overall["delta_mean"]:.4f}</div><div class="l">delta mean</div></div>
</div>

<table class="pa-table">
  <tr><th>action</th><th>n</th><th>L1 mean</th><th>L1 median</th><th>delta mean</th>
      <th>usable</th><th>good</th><th>tier counts</th></tr>'''
    for a in ACTIONS:
        s = per_a[a]
        tc = s['tier_counts']
        tc_str = ', '.join(f'{k.split()[0]}={v}' for k, v in
                           sorted(tc.items(), key=lambda x: x[0]))
        html += f'''
  <tr><td>{a}</td><td>{s["n"]}</td><td>{s["l1_mean"]:.4f}</td>
      <td>{s["l1_median"]:.4f}</td><td>{s["delta_mean"]:.4f}</td>
      <td>{s["usable_count"]}/{s["n"]}</td>
      <td>{s["good_count"]}</td><td>{tc_str}</td></tr>'''
    html += '''
</table>

<div class="filters">
  <div class="group"><b>action:</b>
'''
    for a in ACTIONS:
        html += f'    <label><input type="checkbox" class="f-action" value="{a}" checked>{a}</label>\n'
    html += '''
  </div>
  <div class="group"><b>tier:</b>
'''
    for t in ['A', 'B', 'C', 'D']:
        c = tier_color(t)
        html += f'    <label><input type="checkbox" class="f-tier" value="{t}" checked><span class="badge" style="background:{c}">{t}</span></label>\n'
    html += '''
  </div>
  <div class="group"><b>sort:</b>
    <label><input type="radio" name="sort" value="l1-asc" checked>L1 asc</label>
    <label><input type="radio" name="sort" value="l1-desc">L1 desc</label>
    <label><input type="radio" name="sort" value="action">by action</label>
  </div>
</div>

<div class="grid" id="grid"></div>

<script>
const ITEMS = '''
    html += json.dumps(items_json, ensure_ascii=False)
    html += ''';
const TIER_COLOR = {A: '#10b981', B: '#22c55e', C: '#f59e0b', D: '#ef4444'};

function render() {
  const actions = [...document.querySelectorAll('.f-action:checked')].map(x => x.value);
  const tiers = [...document.querySelectorAll('.f-tier:checked')].map(x => x.value);
  const sort = document.querySelector('input[name="sort"]:checked').value;

  let items = ITEMS.filter(it => actions.includes(it.action) && tiers.includes(it.tier));
  if (sort === 'l1-asc') items.sort((a, b) => a.l1 - b.l1);
  else if (sort === 'l1-desc') items.sort((a, b) => b.l1 - a.l1);
  else items.sort((a, b) => a.action.localeCompare(b.action) || a.l1 - b.l1);

  const grid = document.getElementById('grid');
  grid.innerHTML = items.map(it => {
    const ppStr = Object.entries(it.p).map(([k, v]) => `${k}=${Number(v).toFixed(1)}`).join('  ');
    return `<div class="item">
      <div class="col-headers">
        <div class="h-orig">1. orig<br><span style="font-size:9px">(默认 raw 渲染)</span></div>
        <div class="h-tgt">2. target<br><span style="font-size:9px">(FireRed edit)</span></div>
        <div class="h-back">3. rendered_back<br><span style="font-size:9px">(ISP fit, P_inferred)</span></div>
        <div class="h-diff">4. diff ×5<br><span style="font-size:9px">|2-3|×5</span></div>
      </div>
      <img loading="lazy" src="${it.img}" alt="${it.idx}">
      <div class="meta">
        <div class="row">
          <span><span class="badge" style="background:${TIER_COLOR[it.tier]}">${it.tier}</span>
                <b>${it.action}</b> · idx=${it.idx}</span>
          <span class="l">L1=${it.l1}  delta=${it.delta}</span>
        </div>
        <div class="row"><span class="cap">${it.cap}</span></div>
        <div class="pp">${ppStr}</div>
      </div>
    </div>`;
  }).join('');
}

document.querySelectorAll('.f-action, .f-tier, input[name="sort"]').forEach(el =>
  el.addEventListener('change', render));
render();
</script>
</body>
</html>'''
    out = master / 'viewer.html'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'[SAVED] {out}  ({len(items_json)} items)')


if __name__ == '__main__':
    main()
