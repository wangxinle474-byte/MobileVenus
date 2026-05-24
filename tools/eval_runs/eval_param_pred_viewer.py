"""HTML viewer focusing on 7D parameter prediction quality.

For each val sample, shows:
  - Image strip: original | CN map | refined output | target | error heatmap
  - Per-sample 7D parameter table: name | GT (physical) | Pred (physical) |
    error (with the ACTIVE parameter highlighted)
  - Aggregate header with per-action MAE on the active parameter

Adapted from `tools/eval_named_curves_viewer.py` but param-prediction focused.

Usage:
  python tools/eval_param_pred_viewer.py \
    --ckpt checkpoints/lut_v11a_seed42/best.pt \
    --jsonl outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl \
    --out_dir outputs/lut_v11a_param_viewer \
    --max_samples 80
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.firered_baseline.train_lut import (  # noqa: E402
    ACTIONS,
    ACTION_TO_PARAM_7D_INDICES,
    LUTDataset,
    NamedCurvesPredictor,
    PARAM_NAMES_7D,
    PARAM_NORM_7D,
    build_data,
    set_actions,
)
from training.firered_baseline.color_naming import (  # noqa: E402
    compute_color_naming_maps,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

CN_LABELS_FULL6 = ['red', 'green', 'blue', 'yel-org-brn', 'pink-purp', 'achrom']
CN_LABELS_COMPACT3 = ['warm', 'cool', 'neutral']
CN_COLORS_FULL6 = [
    '#ef4444', '#22c55e', '#3b82f6',
    '#f59e0b', '#ec4899', '#94a3b8',
]
CN_COLORS_COMPACT3 = ['#f59e0b', '#3b82f6', '#94a3b8']

ACTION_CN = {
    'contrast': '对比度', 'saturation': '饱和度',
    'shadows': '暗部提亮', 'highlights': '高光压暗', 'wb': '白平衡',
    'brightness': '亮度', 'clarity': '清晰度',
}
PARAM_CN = {
    'white_balance': 'WB(色温)',
    'brightness': '亮度',
    'contrast': '对比度',
    'shadows': '暗部',
    'highlights': '高光',
    'saturation': '饱和度',
    'clarity': '清晰度',
}
PARAM_UNITS = {
    'white_balance': 'K',
}


def compute_psnr_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean()
    return (-10 * torch.log10(mse.clamp(min=1e-10))).item()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_error_heatmap(pred: torch.Tensor, target: torch.Tensor) -> Image.Image:
    err = (pred - target).abs().mean(dim=0).cpu().numpy()
    err5 = np.clip(err * 5.0, 0, 1)
    r = np.clip(err5 * 2 - 0.5, 0, 1)
    g = np.clip(1 - np.abs(err5 - 0.5) * 2, 0, 1)
    b = np.clip(1 - err5, 0, 1) * 0.5
    rgb = np.stack([r, g, b], axis=-1) * 255
    return Image.fromarray(rgb.astype(np.uint8))


def add_label(img: Image.Image, text: str, bg_color, text_color=(255, 255, 255)):
    W, H = img.size
    bar_h = 28
    new = Image.new('RGB', (W, H + bar_h), bg_color)
    draw = ImageDraw.Draw(new)
    try:
        font = ImageFont.truetype("msyh.ttc", 15)
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, 4), text, fill=text_color, font=font)
    new.paste(img, (0, bar_h))
    return new


def visualize_cn_map(cn_maps: torch.Tensor, labels, colors, size=128):
    N, H, W = cn_maps.shape
    argmax = cn_maps.argmax(dim=0).cpu().numpy()
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for i, c_hex in enumerate(colors):
        r = int(c_hex[1:3], 16)
        g = int(c_hex[3:5], 16)
        b = int(c_hex[5:7], 16)
        mask = (argmax == i)
        out[mask] = [r, g, b]
    img = Image.fromarray(out)
    if (H, W) != (size, size):
        img = img.resize((size, size), Image.NEAREST)
    return img


def denormalize_one(name: str, v_norm: float) -> float:
    cfg = PARAM_NORM_7D[name]
    return v_norm * cfg['scale'] + cfg['center']


def fmt_param(name: str, v: float) -> str:
    if name == 'white_balance':
        return f'{v:.0f}'
    return f'{v:+.1f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='outputs/lut_v11a_param_viewer')
    ap.add_argument('--image_size', type=int, default=None)
    ap.add_argument('--val_ratio', type=float, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split_seed', type=int, default=None)
    ap.add_argument('--max_samples', type=int, default=80)
    ap.add_argument('--tag', default='v11 NamedCurves — Param Prediction')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(Path(args.ckpt), map_location=device, weights_only=False)
    ca = ckpt['args']
    ckpt_actions = ca.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)
    image_size = args.image_size or ca.get('image_size', 256)
    val_ratio = args.val_ratio if args.val_ratio is not None \
        else ca.get('val_ratio', 0.2)
    logger.info(f'Loaded {args.ckpt}: '
                f'val_psnr={ckpt.get("val_psnr", 0):.2f}dB '
                f'@ Ep{ckpt.get("epoch", "?")}')

    n_colors = ca.get('nc_n_colors', 3)
    n_control_points = ca.get('nc_n_control_points', 7)
    use_attention = ca.get('nc_use_attention', False)
    per_action_curves = ca.get('nc_per_action_curves', False)
    use_7d_anchor = ca.get('nc_use_7d_anchor', True)
    use_context = ca.get('nc_use_context', False)
    action_gated_context = ca.get('nc_action_gated_context', False)
    use_action_context = ca.get('nc_use_action_context', False)
    use_region_basis = ca.get('nc_use_region_basis', False)
    use_region_param_delta = ca.get('nc_use_region_param_delta', False)
    use_learned_cn = ca.get('nc_use_learned_cn', False)
    use_wb_head = ca.get('nc_use_wb_head', False)
    use_nilut_residual = ca.get('nc_use_nilut_residual', False)
    use_vera_renderer = ca.get('nc_use_vera_renderer', False)

    if not use_7d_anchor:
        raise RuntimeError(
            'This viewer requires use_7d_anchor=True (model must predict 7D '
            'params).')

    cn_labels = CN_LABELS_FULL6 if n_colors == 6 else CN_LABELS_COMPACT3
    cn_colors = CN_COLORS_FULL6 if n_colors == 6 else CN_COLORS_COMPACT3

    split_seed = args.split_seed if args.split_seed is not None \
        else ca.get('split_seed', ca.get('seed', args.seed))
    tier_filter = ('A excellent', 'B good', 'C acceptable')
    train_s, val_s = build_data(
        Path(args.jsonl), val_ratio, tier_filter, split_seed,
        action_filter=ACTIONS)

    val_ds = LUTDataset(val_s, image_size, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = NamedCurvesPredictor(
        n_colors=n_colors,
        n_control_points=n_control_points,
        use_attention=use_attention,
        per_action_curves=per_action_curves,
        use_7d_anchor=use_7d_anchor,
        use_context=use_context,
        action_gated_context=action_gated_context,
        use_action_context=use_action_context,
        use_region_basis=use_region_basis,
        use_region_param_delta=use_region_param_delta,
        use_learned_cn=use_learned_cn,
        use_wb_head=use_wb_head,
        use_nilut_residual=use_nilut_residual,
        use_vera_renderer=use_vera_renderer,
        nilut_hidden=ca.get('nilut_hidden', 32),
        nilut_n_layers=ca.get('nilut_n_layers', 3),
        nilut_n_freq=ca.get('nilut_n_freq', 4),
        nilut_gate_init=ca.get('nilut_gate_init', 1.0),
        # Path X: implicit residual head
        use_implicit_head=ca.get('use_implicit_head', False),
        implicit_head_base_ch=ca.get('implicit_head_base_ch', 32),
        implicit_head_gate_init=ca.get('implicit_head_gate_init', 0.0),
        image_size=image_size,
        n_actions=len(ACTIONS),
        dropout=ca.get('dropout', 0.5),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / 'imgs'
    img_dir.mkdir(exist_ok=True)

    COL_ORIG = (100, 100, 100)
    COL_REFINED = (56, 189, 248)
    COL_TGT = (251, 191, 36)
    COL_ERR = (239, 68, 68)
    COL_CN = (147, 51, 234)

    results = []
    per_action_psnr = {a: [] for a in ACTIONS}
    # active-param errors
    per_action_active_err = {a: [] for a in ACTIONS}
    # per-param errors across all samples
    per_param_err = {p: [] for p in PARAM_NAMES_7D}

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            enc_in = batch['enc_input'].to(device)
            orig = batch['orig'].to(device)
            target = batch['target'].to(device)
            a_oh = batch['action_onehot'].to(device)
            gt_p7 = batch['params_norm_7d'].to(device)  # (1, 7)

            refined, cn_avg, y_b, pred_p7 = model(enc_in, orig, a_oh)

            psnr = compute_psnr_single(refined[0], target[0])
            l1 = (refined[0] - target[0]).abs().mean().item()
            action = (batch['action'][0] if 'action' in batch
                      else ACTIONS[batch['action_idx'][0].item()])
            action_idx = ACTIONS.index(action)
            active_idx = ACTION_TO_PARAM_7D_INDICES[action_idx]

            per_action_psnr[action].append(psnr)

            gt_norm = gt_p7[0].cpu().numpy()
            pred_norm = pred_p7[0].cpu().numpy()
            param_rows = []
            for i, name in enumerate(PARAM_NAMES_7D):
                gt_phys = denormalize_one(name, float(gt_norm[i]))
                pred_phys = denormalize_one(name, float(pred_norm[i]))
                err = pred_phys - gt_phys
                param_rows.append({
                    'name': name,
                    'name_cn': PARAM_CN[name],
                    'gt': gt_phys,
                    'pred': pred_phys,
                    'err': err,
                    'unit': PARAM_UNITS.get(name, ''),
                    'active': (i == active_idx),
                })
                per_param_err[name].append(abs(err))
                if i == active_idx:
                    per_action_active_err[action].append(abs(err))

            # Image strip
            cn_maps = compute_color_naming_maps(
                orig, grouping=('full6' if n_colors == 6 else 'compact3'))

            if idx < args.max_samples:
                orig_pil = tensor_to_pil(orig[0])
                refined_pil = tensor_to_pil(refined[0])
                tgt_pil = tensor_to_pil(target[0])
                err_pil = make_error_heatmap(refined[0], target[0])
                cn_pil = visualize_cn_map(cn_maps[0], cn_labels, cn_colors,
                                           size=image_size)

                orig_lab = add_label(orig_pil, '① 原图', COL_ORIG)
                cn_lab = add_label(cn_pil,
                                    f'② CN ({n_colors}色)',
                                    COL_CN, (255, 255, 255))
                ref_lab = add_label(refined_pil,
                                     f'③ 模型输出  PSNR={psnr:.1f}dB',
                                     COL_REFINED, (0, 0, 0))
                tgt_lab = add_label(tgt_pil, '④ GT 目标 (Expert C)',
                                     COL_TGT, (0, 0, 0))
                err_lab = add_label(err_pil, '⑤ 像素误差 ×5', COL_ERR)

                W, H = orig_lab.size
                strip = Image.new('RGB', (W * 5 + 8, H), (15, 23, 42))
                strip.paste(orig_lab, (0, 0))
                strip.paste(cn_lab, (W + 2, 0))
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
                    'psnr': round(psnr, 2),
                    'l1': round(l1, 4),
                    'param_rows': param_rows,
                    'active_param': PARAM_NAMES_7D[active_idx],
                    'active_err': abs(param_rows[active_idx]['err']),
                    'img': f'imgs/{fname}',
                })

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    overall_psnr = float(np.mean(
        [p for ps in per_action_psnr.values() for p in ps]))

    pa_summary = {}
    for a in ACTIONS:
        ps_list = per_action_psnr[a]
        err_list = per_action_active_err[a]
        if not ps_list:
            continue
        param_idx = ACTION_TO_PARAM_7D_INDICES[ACTIONS.index(a)]
        pa_summary[a] = {
            'n': len(ps_list),
            'psnr': round(float(np.mean(ps_list)), 2),
            'param': PARAM_NAMES_7D[param_idx],
            'mae': round(float(np.mean(err_list)), 2) if err_list else 0.0,
            'p50_err': round(float(np.percentile(err_list, 50)), 2) if err_list else 0.0,
            'p90_err': round(float(np.percentile(err_list, 90)), 2) if err_list else 0.0,
        }

    pp_summary = {}
    for p in PARAM_NAMES_7D:
        errs = per_param_err[p]
        if errs:
            pp_summary[p] = {
                'mae': round(float(np.mean(errs)), 2),
                'p90': round(float(np.percentile(errs, 90)), 2),
            }
        else:
            pp_summary[p] = {'mae': 0.0, 'p90': 0.0}

    # Sort results by active_err desc (worst predictions first)
    results.sort(key=lambda r: -r['active_err'])

    # ------------------------------------------------------------------
    # Build HTML
    # ------------------------------------------------------------------
    pa_rows = ''
    for a in ACTIONS:
        if a not in pa_summary:
            continue
        s = pa_summary[a]
        unit = PARAM_UNITS.get(s['param'], '')
        psnr_color = ('#22c55e' if s['psnr'] > 23.94
                      else '#fbbf24' if s['psnr'] > 22 else '#ef4444')
        pa_rows += (f'<tr><td>{ACTION_CN.get(a, a)} ({a})</td>'
                    f'<td>{s["n"]}</td>'
                    f'<td style="color:{psnr_color};font-weight:bold">'
                    f'{s["psnr"]:.2f}</td>'
                    f'<td>{PARAM_CN[s["param"]]}</td>'
                    f'<td><b>{s["mae"]:.2f}{unit}</b></td>'
                    f'<td>{s["p50_err"]:.2f}{unit}</td>'
                    f'<td>{s["p90_err"]:.2f}{unit}</td></tr>')

    pp_rows = ''
    for p in PARAM_NAMES_7D:
        s = pp_summary[p]
        unit = PARAM_UNITS.get(p, '')
        pp_rows += (f'<tr><td>{PARAM_CN[p]}</td>'
                    f'<td>{s["mae"]:.2f}{unit}</td>'
                    f'<td>{s["p90"]:.2f}{unit}</td></tr>')

    items_html = ''
    for r in results:
        psnr_color = ('#22c55e' if r['psnr'] > 24
                      else '#fbbf24' if r['psnr'] > 22
                      else '#ef4444')
        # Per-sample param table
        ptable_rows = ''
        for p in r['param_rows']:
            cls = 'param-active' if p['active'] else ''
            err_color = ('#22c55e' if abs(p['err']) < 5
                         else '#fbbf24' if abs(p['err']) < 20
                         else '#ef4444')
            ptable_rows += (f'<tr class="{cls}">'
                            f'<td>{p["name_cn"]}</td>'
                            f'<td>{fmt_param(p["name"], p["gt"])}{p["unit"]}</td>'
                            f'<td>{fmt_param(p["name"], p["pred"])}{p["unit"]}</td>'
                            f'<td style="color:{err_color}">'
                            f'{p["err"]:+.1f}{p["unit"]}</td></tr>')

        items_html += f'''
        <div class="item" data-action="{r['action']}">
          <div class="imgwrap"><img src="{r['img']}" loading="lazy"></div>
          <div class="meta">
            <div class="row">
              <span class="badge">{r['action_cn']}</span>
              <span style="color:{psnr_color};font-weight:bold;font-size:14px">
                PSNR {r['psnr']:.2f} dB
              </span>
            </div>
            <div class="row"><span class="l">文件</span>
              <span class="pp">{r['source']}</span></div>
            <div class="row"><span class="l">活动参数</span>
              <span>{PARAM_CN[r['active_param']]}, |err|={r['active_err']:.1f}</span></div>
            <table class="ptab">
              <thead><tr><th>参数</th><th>GT</th><th>Pred</th><th>err</th></tr></thead>
              <tbody>{ptable_rows}</tbody>
            </table>
          </div>
        </div>'''

    action_filters = ''.join(
        f'<label><input type="checkbox" class="f-action" '
        f'value="{a}" checked> {ACTION_CN.get(a, a)}</label>'
        for a in ACTIONS
    )

    flag_summary = (
        f'context={use_context}, action_gated={action_gated_context}, '
        f'action_ctx={use_action_context}, '
        f'region_basis={use_region_basis}, '
        f'region_param_delta={use_region_param_delta}'
    )

    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{args.tag}</title>
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
  .twocol {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }}
  .twocol > div {{ flex: 1; min-width: 320px; }}
  .pa-table, .pp-table {{ background: #1e293b; border-collapse: collapse;
               font-size: 13px; width: 100%; }}
  .pa-table th, .pa-table td, .pp-table th, .pp-table td
    {{ padding: 6px 12px; border: 1px solid #334155; text-align: center; }}
  .pa-table th, .pp-table th {{ background: #334155; color: #fbbf24; }}
  .legend {{ background: #1e293b; padding: 12px 16px; border-radius: 6px;
             border: 1px solid #334155; margin-bottom: 16px; font-size: 13px;
             line-height: 1.8; }}
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
  .row {{ display: flex; justify-content: space-between; align-items: center;
          margin-bottom: 2px; gap: 8px; }}
  .row .l {{ color: #94a3b8; min-width: 70px; }}
  .pp {{ font-size: 11px; color: #94a3b8; }}
  .hide {{ display: none; }}
  .ptab {{ width: 100%; border-collapse: collapse; font-size: 11px;
           margin-top: 6px; }}
  .ptab th, .ptab td {{ padding: 3px 8px; border: 1px solid #334155;
                         text-align: center; }}
  .ptab th {{ background: #334155; color: #fbbf24; }}
  .param-active {{ background: rgba(124, 58, 237, 0.25); font-weight: bold; }}
</style>
</head>
<body>
<h1>{args.tag}</h1>
<p class="subtitle">checkpoint: {args.ckpt}
  &nbsp;|&nbsp; {flag_summary}
  &nbsp;|&nbsp; ckpt val_psnr={ckpt.get("val_psnr", 0):.2f}dB
  &nbsp;|&nbsp; jsonl: {args.jsonl}
  &nbsp;|&nbsp; {len(results)} val samples</p>

<div class="summary">
  <div class="card"><div class="v">{overall_psnr:.2f}</div>
    <div class="l">overall PSNR (dB)</div></div>
  <div class="card"><div class="v" style="color:#fbbf24">23.94</div>
    <div class="l">7D ISP 上限</div></div>
  <div class="card"><div class="v" style="color:{
    "#22c55e" if overall_psnr > 23.94 else "#ef4444"}">
    {overall_psnr - 23.94:+.2f}</div>
    <div class="l">vs 上限 (dB)</div></div>
  <div class="card"><div class="v">{len(results)}</div>
    <div class="l">val 样本数</div></div>
</div>

<div class="twocol">
  <div>
    <h3 style="margin: 4px 0 8px; font-size: 14px;">
      Per-action: PSNR + Active param MAE</h3>
    <table class="pa-table">
      <thead><tr><th>动作</th><th>n</th><th>PSNR (dB)</th>
                  <th>活动参数</th><th>MAE</th>
                  <th>p50_err</th><th>p90_err</th></tr></thead>
      <tbody>{pa_rows}</tbody>
    </table>
  </div>
  <div>
    <h3 style="margin: 4px 0 8px; font-size: 14px;">
      Per-parameter MAE (across all samples)</h3>
    <table class="pp-table">
      <thead><tr><th>参数</th><th>MAE</th><th>p90</th></tr></thead>
      <tbody>{pp_rows}</tbody>
    </table>
    <div style="font-size: 11px; color: #94a3b8; margin-top: 6px;">
      Note: most non-active params have GT=center; this table shows whether
      the model correctly outputs near-center for them.
    </div>
  </div>
</div>

<div class="legend">
  <b>5 列对照说明:</b><br>
  <span><span class="dot" style="background:rgb(100,100,100);
    display:inline-block;width:14px;height:14px;border-radius:3px;
    vertical-align:middle;margin-right:4px"></span>① 原图</span>
  <span><span style="background:rgb(147,51,234);display:inline-block;
    width:14px;height:14px;border-radius:3px;vertical-align:middle;
    margin-right:4px"></span>② Color Naming 分解</span>
  <span><span style="background:rgb(56,189,248);display:inline-block;
    width:14px;height:14px;border-radius:3px;vertical-align:middle;
    margin-right:4px"></span>③ 模型输出 (Bezier + 7D anchor)</span>
  <span><span style="background:rgb(251,191,36);display:inline-block;
    width:14px;height:14px;border-radius:3px;vertical-align:middle;
    margin-right:4px"></span>④ GT 目标</span>
  <span><span style="background:rgb(239,68,68);display:inline-block;
    width:14px;height:14px;border-radius:3px;vertical-align:middle;
    margin-right:4px"></span>⑤ 像素误差 (×5)</span>
  <br><br>
  <b>每张图下表:</b> 显示该样本 7 个 ISP 参数的 GT vs Pred 对比.
  <span style="background:rgba(124,58,237,0.25);padding:0 4px">紫色高亮行</span>
  = 当前 action 对应的活动参数 (其他参数 GT 通常 = center).
</div>

<div class="filters">
  <b>按动作过滤:</b> {action_filters}
</div>

<p class="subtitle">默认排序: 活动参数预测误差从大到小 (差的在前)</p>
<div class="grid" id="grid">{items_html}</div>

<script>
const items = document.querySelectorAll('.item');
const checkboxes = document.querySelectorAll('.f-action');
function update() {{
  const active = new Set(Array.from(checkboxes)
    .filter(c => c.checked).map(c => c.value));
  items.forEach(it => it.classList.toggle('hide', !active.has(it.dataset.action)));
}}
checkboxes.forEach(c => c.addEventListener('change', update));
</script>
</body>
</html>'''

    viewer_path = out_dir / 'viewer.html'
    viewer_path.write_text(html, encoding='utf-8')
    logger.info(f'Viewer: {viewer_path}')
    logger.info(f'Overall PSNR: {overall_psnr:.2f} dB')
    logger.info('Per-action active-param MAE:')
    for a in ACTIONS:
        if a in pa_summary:
            s = pa_summary[a]
            unit = PARAM_UNITS.get(s['param'], '')
            logger.info(f'  {a:<12} {s["param"]:<14} '
                        f'MAE={s["mae"]:.2f}{unit} '
                        f'p90={s["p90_err"]:.2f}{unit}')


if __name__ == '__main__':
    main()
