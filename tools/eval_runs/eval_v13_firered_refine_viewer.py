"""v13 FireRed residual refinement — visual comparison viewer.

5-column layout:
  ① 原图 (FiveK input)
  ② v11-d base output  (PSNR vs target)
  ③ v13 refined output (PSNR vs target)
  ④ FireRed target
  ⑤ |refined - target| ×5 error heatmap

Per-action table reports v11-d base PSNR and v13 refined PSNR side by side
so the residual contribution is visible per action.

Loads:
  - v13 best.pt (refiner state + args incl. base_ckpt path)
  - v11-d best.pt (frozen NamedCurves base)

Usage:
  python tools/eval_v13_firered_refine_viewer.py \\
    --ckpt checkpoints/v13a_firered_refine_7actions/best.pt \\
    --jsonl outputs/firered_v11_existing_7actions/pseudo_labels.jsonl \\
    --out_dir outputs/v13a_firered_refine_viewer \\
    --max_samples 80 --tag "v13a 7-action refine"
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.firered_residual_refiner import FireRedResidualRefiner  # noqa: E402
from training.firered_baseline.train_lut import (  # noqa: E402
    LUTDataset, build_data, ACTIONS, set_actions,
)
from training.firered_baseline.train_v13_firered_refine import (  # noqa: E402
    load_v11d_base,
)
from tools._lib.eval_named_curves_viewer import (  # noqa: E402
    add_label, compute_psnr_single, make_error_heatmap, tensor_to_pil,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


ACTION_CN = {
    'contrast': '对比度', 'saturation': '饱和度',
    'shadows': '暗部提亮', 'highlights': '高光压暗', 'wb': '白平衡',
    'brightness': '亮度', 'clarity': '清晰度',
}

# FireRed 1.1 教师生成 pseudo-labels 时使用的 prompt (per-action)
# 与 scripts/local/run_firered_all_actions.py 中的 captions 保持一致
ACTION_PROMPT = {
    'contrast':   'Increase contrast. Keep the original composition and subject unchanged.',
    'saturation': 'Enhance saturation. Keep the original composition and subject unchanged.',
    'shadows':    'Lift shadows. Keep the original composition and subject unchanged.',
    'highlights': 'Recover highlights. Keep the original composition and subject unchanged.',
    'wb':         'Apply warmer white balance. Keep the original composition and subject unchanged.',
    'brightness': 'Increase brightness. Keep the original composition and subject unchanged.',
    'clarity':    'Enhance clarity and sharpness. Keep the original composition and subject unchanged.',
}


def _resolve_base_ckpt(v13_ckpt_path: Path, v13_args: dict,
                       cli_base_ckpt: str | None) -> Path:
    """Pick base_ckpt: CLI override > v13 args > sibling search."""
    if cli_base_ckpt:
        return Path(cli_base_ckpt)
    saved = v13_args.get('base_ckpt')
    if saved:
        p = Path(saved)
        if p.exists():
            return p
    # Fall back to base_args path stored at training time
    base_args = v13_args.get('base_args', {}) or {}
    saved2 = base_args.get('base_ckpt') or base_args.get('ckpt')
    if saved2:
        p = Path(saved2)
        if p.exists():
            return p
    raise RuntimeError(
        f'Could not locate v11-d base checkpoint for {v13_ckpt_path}. '
        f'Pass --base_ckpt explicitly.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True,
                    help='v13 refiner checkpoint (best.pt)')
    ap.add_argument('--base_ckpt', default=None,
                    help='Override v11-d base ckpt path; default reads from '
                         'v13 ckpt args')
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--image_size', type=int, default=None)
    ap.add_argument('--val_ratio', type=float, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split_seed', type=int, default=None)
    ap.add_argument('--max_samples', type=int, default=80)
    ap.add_argument('--tag', default='v13 FireRed refine')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load v13 refiner ckpt ──
    v13_ckpt = torch.load(Path(args.ckpt), map_location=device,
                          weights_only=False)
    v13_args = v13_ckpt['args']
    # v13 saved 'actions' from base; ensure global ACTIONS matches before
    # constructing dataset / refiner.
    ckpt_actions = v13_args.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)

    base_ckpt_path = _resolve_base_ckpt(Path(args.ckpt), v13_args,
                                        args.base_ckpt)
    base_model, base_args = load_v11d_base(base_ckpt_path, device)
    # load_v11d_base re-runs set_actions to match base; fine because v13 was
    # trained with the same actions.

    image_size = args.image_size or v13_args.get(
        'image_size', base_args.get('image_size', 256))
    val_ratio = args.val_ratio if args.val_ratio is not None \
        else v13_args.get('val_ratio',
                          base_args.get('val_ratio', 0.2))
    split_seed = args.split_seed if args.split_seed is not None \
        else v13_args.get('split_seed',
                          base_args.get('split_seed',
                                        base_args.get('seed', args.seed)))
    logger.info(f'v13 ckpt: {args.ckpt} val_psnr='
                f'{v13_ckpt.get("val_psnr", 0):.2f}dB '
                f'(base {v13_ckpt.get("val_psnr_base", 0):.2f}dB) '
                f'@ Ep{v13_ckpt.get("epoch", "?")}')
    logger.info(f'image_size={image_size} val_ratio={val_ratio} '
                f'split_seed={split_seed} actions={ACTIONS}')

    # ── Refiner ──
    refiner = FireRedResidualRefiner(
        base_ch=v13_args.get('base_ch', 32),
        n_actions=len(ACTIONS),
        delta_scale=v13_args.get('delta_scale', 0.5),
    ).to(device)
    refiner.load_state_dict(v13_ckpt['model_state_dict'], strict=True)
    refiner.eval()

    # ── Data (val split aligned with base ckpt) ──
    train_s, val_s = build_data(
        Path(args.jsonl), val_ratio,
        ('A excellent', 'B good', 'C acceptable'),
        split_seed, action_filter=ACTIONS)
    val_ds = LUTDataset(val_s, image_size, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    logger.info(f'val samples: {len(val_ds)}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / 'imgs'
    img_dir.mkdir(exist_ok=True)

    COL_ORIG = (100, 100, 100)
    COL_BASE = (124, 58, 237)     # purple — v11-d
    COL_REF  = (56, 189, 248)     # cyan — v13
    COL_TGT  = (251, 191, 36)
    COL_ERR  = (239, 68, 68)

    results = []
    per_action_base = {a: [] for a in ACTIONS}
    per_action_ref = {a: [] for a in ACTIONS}

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            enc_in = batch['enc_input'].to(device)
            orig = batch['orig'].to(device)
            target = batch['target'].to(device)
            a_oh = batch['action_onehot'].to(device)

            base_out, _, _, _ = base_model(enc_in, orig, a_oh)
            refined, _ = refiner(orig, base_out, a_oh)

            psnr_base = compute_psnr_single(base_out[0], target[0])
            psnr_ref = compute_psnr_single(refined[0], target[0])
            l1_ref = (refined[0] - target[0]).abs().mean().item()

            action = (batch['action'][0] if 'action' in batch
                      else ACTIONS[batch['action_idx'][0].item()])
            per_action_base[action].append(psnr_base)
            per_action_ref[action].append(psnr_ref)

            if idx < args.max_samples:
                orig_pil = tensor_to_pil(orig[0])
                base_pil = tensor_to_pil(base_out[0])
                ref_pil = tensor_to_pil(refined[0])
                tgt_pil = tensor_to_pil(target[0])
                err_pil = make_error_heatmap(refined[0], target[0])

                orig_lab = add_label(orig_pil, '① 原图 (FiveK input)',
                                     COL_ORIG)
                base_lab = add_label(base_pil,
                                     f'② v11-d base  PSNR={psnr_base:.1f}dB',
                                     COL_BASE)
                ref_lab = add_label(ref_pil,
                                    f'③ v13 refined  PSNR={psnr_ref:.1f}dB',
                                    COL_REF, (0, 0, 0))
                tgt_lab = add_label(tgt_pil, '④ FireRed 目标',
                                    COL_TGT, (0, 0, 0))
                err_lab = add_label(err_pil, '⑤ 像素误差 ×5', COL_ERR)

                W, H = orig_lab.size
                strip = Image.new('RGB', (W * 5 + 8, H), (15, 23, 42))
                strip.paste(orig_lab, (0, 0))
                strip.paste(base_lab, (W + 2, 0))
                strip.paste(ref_lab, (W * 2 + 4, 0))
                strip.paste(tgt_lab, (W * 3 + 6, 0))
                strip.paste(err_lab, (W * 4 + 8, 0))

                fname = f'{idx:03d}_{action}.jpg'
                strip.save(img_dir / fname, quality=88)

                results.append({
                    'idx': idx,
                    'action': action,
                    'action_cn': ACTION_CN.get(action, action),
                    'source': batch['source_image'][0],
                    'psnr_base': round(psnr_base, 2),
                    'psnr_ref': round(psnr_ref, 2),
                    'delta_psnr': round(psnr_ref - psnr_base, 2),
                    'l1': round(l1_ref, 4),
                    'img': f'imgs/{fname}',
                })

    overall_base = float(np.mean(
        [p for ps in per_action_base.values() for p in ps]))
    overall_ref = float(np.mean(
        [p for ps in per_action_ref.values() for p in ps]))

    pa_rows = ''
    for a in ACTIONS:
        b = per_action_base[a]
        r = per_action_ref[a]
        if not b:
            continue
        pb = float(np.mean(b))
        pr = float(np.mean(r))
        d = pr - pb
        d_color = ('#22c55e' if d > 0.1 else '#fbbf24' if d > -0.1
                   else '#ef4444')
        pa_rows += (
            f'<tr><td>{ACTION_CN.get(a, a)} ({a})</td>'
            f'<td>{len(b)}</td>'
            f'<td>{pb:.2f}</td>'
            f'<td style="color:{d_color};font-weight:bold">{pr:.2f}</td>'
            f'<td style="color:{d_color};font-weight:bold">{d:+.2f}</td>'
            f'</tr>')

    # ── FireRed teacher prompt table ──
    prompt_rows = ''
    for a in ACTIONS:
        prompt = ACTION_PROMPT.get(a, '(missing)')
        prompt_rows += (
            f'<tr><td>{ACTION_CN.get(a, a)} ({a})</td>'
            f'<td class="prompt-cell">{prompt}</td></tr>')

    # Sort sample cards by refined PSNR ascending (worst on top to inspect)
    results.sort(key=lambda r: r['psnr_ref'])
    items_html = ''
    for r in results:
        ref_color = ('#22c55e' if r['psnr_ref'] > 24
                     else '#fbbf24' if r['psnr_ref'] > 22
                     else '#ef4444')
        d_color = ('#22c55e' if r['delta_psnr'] > 0.1
                   else '#fbbf24' if r['delta_psnr'] > -0.1
                   else '#ef4444')
        prompt_text = ACTION_PROMPT.get(r['action'], '')
        items_html += f'''
        <div class="item" data-action="{r['action']}">
          <div class="imgwrap"><img src="{r['img']}" loading="lazy"></div>
          <div class="meta">
            <div class="row">
              <span class="badge">{r['action_cn']}</span>
              <span style="color:{ref_color};font-weight:bold;font-size:14px">
                v13 {r['psnr_ref']:.2f} dB
              </span>
              <span style="color:#a78bfa;font-size:12px">
                base {r['psnr_base']:.2f} dB
              </span>
              <span style="color:{d_color};font-weight:bold;font-size:13px">
                Δ{r['delta_psnr']:+.2f}
              </span>
            </div>
            <div class="row"><span class="l">prompt</span>
              <span class="prompt-inline">{prompt_text}</span></div>
            <div class="row"><span class="l">文件</span>
              <span class="pp">{r['source']}</span></div>
          </div>
        </div>'''

    action_filters = ''.join(
        f'<label><input type="checkbox" class="f-action" '
        f'value="{a}" checked> {ACTION_CN.get(a, a)}</label>'
        for a in ACTIONS
    )

    delta_overall = overall_ref - overall_base
    delta_color = ('#22c55e' if delta_overall > 0.05
                   else '#ef4444' if delta_overall < -0.05
                   else '#fbbf24')

    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>v13 FireRed Refine — {args.tag}</title>
<style>
  body {{ font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif; margin: 0;
         padding: 16px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 20px; margin: 0 0 6px; }}
  .subtitle {{ font-size: 13px; color: #94a3b8; margin-bottom: 16px; }}
  .summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
  .card {{ background: #1e293b; padding: 10px 16px; border-radius: 6px;
           border: 1px solid #334155; min-width: 140px; }}
  .card .v {{ font-size: 24px; font-weight: bold; color: #38bdf8; }}
  .card .l {{ font-size: 12px; color: #94a3b8; }}
  .pa-table {{ background: #1e293b; border-collapse: collapse;
               font-size: 13px; width: 100%; max-width: 720px;
               margin-bottom: 16px; }}
  .pa-table th, .pa-table td {{ padding: 6px 14px; border: 1px solid #334155;
                                 text-align: center; }}
  .pa-table th {{ background: #334155; color: #fbbf24; }}
  .pa-table td:first-child {{ text-align: left; }}
  .prompt-table {{ background: #1e293b; border-collapse: collapse;
                   font-size: 13px; width: 100%; max-width: 1100px;
                   margin-bottom: 16px; }}
  .prompt-table th, .prompt-table td {{ padding: 6px 14px;
                                          border: 1px solid #334155;
                                          text-align: left; }}
  .prompt-table th {{ background: #334155; color: #fbbf24;
                       text-align: center; }}
  .prompt-table td:first-child {{ width: 140px; color: #38bdf8;
                                    font-weight: bold; }}
  .prompt-table .prompt-cell {{ color: #e2e8f0; font-family:
    'Consolas', 'Courier New', monospace; font-size: 12px; }}
  .row .prompt-inline {{ color: #94a3b8; font-size: 11px;
                          font-family: 'Consolas', 'Courier New', monospace;
                          font-style: italic; }}
  .legend {{ background: #1e293b; padding: 12px 16px; border-radius: 6px;
             border: 1px solid #334155; margin-bottom: 16px; font-size: 13px;
             line-height: 1.8; }}
  .legend span {{ margin-right: 16px; }}
  .dot {{ display: inline-block; width: 14px; height: 14px;
          border-radius: 3px; vertical-align: middle; margin-right: 4px; }}
  .filters {{ background: #1e293b; padding: 12px 16px; border-radius: 6px;
              border: 1px solid #334155; margin-bottom: 16px; }}
  .filters label {{ margin-right: 14px; cursor: pointer; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px;
           max-width: 1500px; }}
  .item {{ background: #1e293b; border-radius: 6px; overflow: hidden;
           border: 1px solid #334155; }}
  .item img {{ width: 100%; display: block; }}
  .meta {{ padding: 8px 12px; font-size: 12px; line-height: 1.6; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px;
            color: #fff; font-weight: bold; margin-right: 8px;
            background: #7c3aed; }}
  .row {{ display: flex; justify-content: flex-start; align-items: center;
          margin-bottom: 2px; gap: 12px; flex-wrap: wrap; }}
  .row .l {{ color: #94a3b8; min-width: 70px; }}
  .pp {{ font-size: 11px; color: #94a3b8; }}
  .hide {{ display: none; }}
</style>
</head>
<body>
<h1>v13 FireRed 残差精修 — 视觉对比</h1>
<p class="subtitle">{args.tag} &nbsp;|&nbsp; refiner ckpt: {args.ckpt}
  &nbsp;|&nbsp; base ckpt: {base_ckpt_path}
  &nbsp;|&nbsp; {len(results)} val samples
  &nbsp;|&nbsp; v13 val_psnr={v13_ckpt.get("val_psnr", 0):.2f}dB</p>

<div class="summary">
  <div class="card"><div class="v" style="color:#a78bfa">
    {overall_base:.2f}</div>
    <div class="l">v11-d base PSNR (dB)</div></div>
  <div class="card"><div class="v">{overall_ref:.2f}</div>
    <div class="l">v13 refined PSNR (dB)</div></div>
  <div class="card"><div class="v" style="color:{delta_color}">
    {delta_overall:+.2f}</div>
    <div class="l">refine gain (dB)</div></div>
  <div class="card"><div class="v">{len(results)}</div>
    <div class="l">val 样本数</div></div>
</div>

<h3 style="margin: 4px 0 8px; font-size: 14px;">Per-action PSNR (base vs v13)</h3>
<table class="pa-table">
  <thead><tr><th>动作</th><th>n</th><th>base</th><th>v13</th><th>Δ</th></tr></thead>
  <tbody>{pa_rows}</tbody>
</table>

<h3 style="margin: 4px 0 8px; font-size: 14px;">FireRed 1.1 教师使用的 prompt 模板</h3>
<table class="prompt-table">
  <thead><tr><th style="width:140px">动作</th><th>Prompt (生成 pseudo-label 时送给 FireRed)</th></tr></thead>
  <tbody>{prompt_rows}</tbody>
</table>

<div class="legend">
  <b>5 列对照说明:</b><br>
  <span><span class="dot" style="background:rgb(100,100,100)"></span>
    ① 原图</span>
  <span><span class="dot" style="background:rgb(124,58,237)"></span>
    ② v11-d base 输出</span>
  <span><span class="dot" style="background:rgb(56,189,248)"></span>
    ③ v13 残差精修输出</span>
  <span><span class="dot" style="background:rgb(251,191,36)"></span>
    ④ FireRed 目标</span>
  <span><span class="dot" style="background:rgb(239,68,68)"></span>
    ⑤ 像素误差热图 (×5)</span>
</div>

<div class="filters">
  <b>按动作过滤:</b> {action_filters}
</div>

<p class="subtitle">默认排序: v13 PSNR 从低到高 (差的在前)</p>
<div class="grid" id="grid">{items_html}</div>

<script>
const items = document.querySelectorAll('.item');
const checkboxes = document.querySelectorAll('.f-action');
function update() {{
  const active = new Set(Array.from(checkboxes)
    .filter(c => c.checked).map(c => c.value));
  items.forEach(it => it.classList.toggle('hide',
    !active.has(it.dataset.action)));
}}
checkboxes.forEach(c => c.addEventListener('change', update));
</script>
</body>
</html>'''

    viewer_path = out_dir / 'viewer.html'
    viewer_path.write_text(html, encoding='utf-8')
    logger.info(f'Viewer: {viewer_path}')
    logger.info(f'Overall base PSNR:  {overall_base:.2f} dB')
    logger.info(f'Overall v13  PSNR:  {overall_ref:.2f} dB')
    logger.info(f'Refine gain:        {delta_overall:+.2f} dB')


if __name__ == '__main__':
    main()
