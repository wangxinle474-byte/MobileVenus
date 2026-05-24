"""v9 (NamedCurvesPredictor) 视觉对比 viewer.

4 列布局: 原图 | Refined (Bezier 曲线 + CN 加权) | FireRed 目标 | 误差热图
每张图附:
  - CN 颜色分布 (饼图风格)
  - Bezier 曲线参数化 (按色域 × RGB 通道)
  - 可选 7D 预测参数 (若 anchor 开启)
"""
from __future__ import annotations

import argparse
import json
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
    NamedCurvesPredictor, LUTDataset, build_data, ACTIONS,
    PARAM_NAMES_7D, PARAM_NORM_7D, set_actions,
)
from training.firered_baseline.color_naming import (  # noqa: E402
    compute_color_naming_maps,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Color name labels (matches color_naming.py groupings)
CN_LABELS_FULL6 = ['red', 'green', 'blue', 'yel-org-brn', 'pink-purp', 'achrom']
CN_LABELS_COMPACT3 = ['warm', 'cool', 'neutral']
CN_COLORS_FULL6 = [
    '#ef4444', '#22c55e', '#3b82f6',
    '#f59e0b', '#ec4899', '#94a3b8',
]
CN_COLORS_COMPACT3 = ['#f59e0b', '#3b82f6', '#94a3b8']


# ============================================================
# Helpers
# ============================================================
def compute_psnr_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean()
    return (-10 * torch.log10(mse.clamp(min=1e-10))).item()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_error_heatmap(pred: torch.Tensor, target: torch.Tensor) -> Image.Image:
    err = (pred - target).abs().mean(dim=0).cpu().numpy()  # (H, W)
    err5 = np.clip(err * 5.0, 0, 1)
    r = np.clip(err5 * 2 - 0.5, 0, 1)
    g = np.clip(1 - np.abs(err5 - 0.5) * 2, 0, 1)
    b = np.clip(1 - err5, 0, 1) * 0.5
    rgb = np.stack([r, g, b], axis=-1) * 255
    return Image.fromarray(rgb.astype(np.uint8))


def make_context_heatmap(ctx_map: torch.Tensor) -> Image.Image:
    """Visualize v9d context map. blue (low) → green (mid) → red (high).

    ctx_map: (1, H, W) tensor in [0, 1]
    """
    c = ctx_map.squeeze(0).cpu().numpy()  # (H, W)
    c = np.clip(c, 0, 1)
    # Blue-Green-Red colormap (viridis-like)
    r = np.clip(c * 2 - 0.5, 0, 1)
    g = np.clip(1 - np.abs(c - 0.5) * 2, 0, 1)
    b = np.clip((1 - c) * 1.5, 0, 1)
    rgb = np.stack([r, g, b], axis=-1) * 255
    return Image.fromarray(rgb.astype(np.uint8))


def add_label(img: Image.Image, text: str, bg_color, text_color=(255, 255, 255)):
    """在图片顶部添加带背景色的标签条."""
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
    """Visualize per-pixel argmax of CN maps as a colored image.

    cn_maps: (N, H, W)
    """
    N, H, W = cn_maps.shape
    argmax = cn_maps.argmax(dim=0).cpu().numpy()  # (H, W)
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


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/lut_v9a/best.pt')
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='outputs/lut_v9a_viewer')
    ap.add_argument('--image_size', type=int, default=None)
    ap.add_argument('--val_ratio', type=float, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split_seed', type=int, default=None)
    ap.add_argument('--max_samples', type=int, default=80)
    ap.add_argument('--tag', default='v9 (NamedCurves)')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(Path(args.ckpt), map_location=device, weights_only=False)
    ckpt_args = ckpt['args']
    ckpt_actions = ckpt_args.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)
    image_size = args.image_size or ckpt_args.get('image_size', 256)
    val_ratio = args.val_ratio if args.val_ratio is not None \
        else ckpt_args.get('val_ratio', 0.2)
    split_seed = args.split_seed if args.split_seed is not None \
        else ckpt_args.get('split_seed', ckpt_args.get('seed', args.seed))
    logger.info(f'Loaded {args.ckpt}: '
                f'val_psnr={ckpt.get("val_psnr", 0):.2f}dB '
                f'@ Ep{ckpt.get("epoch", "?")}')

    if not ckpt_args.get('named_curves', False):
        raise RuntimeError(
            f'Checkpoint is not a v9 (named_curves) model. '
            f'Use eval_lut_viewer_v8.py for v8 or eval_lut_viewer_v2.py for v1-v7.')

    n_colors = ckpt_args.get('nc_n_colors', 6)
    n_control_points = ckpt_args.get('nc_n_control_points', 11)
    use_attention = ckpt_args.get('nc_use_attention', False)
    per_action_curves = ckpt_args.get('nc_per_action_curves', False)
    use_7d_anchor = ckpt_args.get('nc_use_7d_anchor', False)
    use_context = ckpt_args.get('nc_use_context', False)
    action_gated_context = ckpt_args.get('nc_action_gated_context', False)
    use_action_context = ckpt_args.get('nc_use_action_context', False)
    use_region_basis = ckpt_args.get('nc_use_region_basis', False)
    use_region_param_delta = ckpt_args.get('nc_use_region_param_delta', False)
    use_learned_cn = ckpt_args.get('nc_use_learned_cn', False)
    use_wb_head = ckpt_args.get('nc_use_wb_head', False)
    use_nilut_residual = ckpt_args.get('nc_use_nilut_residual', False)
    use_vera_renderer = ckpt_args.get('nc_use_vera_renderer', False)

    cn_labels = CN_LABELS_FULL6 if n_colors == 6 else CN_LABELS_COMPACT3
    cn_colors = CN_COLORS_FULL6 if n_colors == 6 else CN_COLORS_COMPACT3

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
        nilut_hidden=ckpt_args.get('nilut_hidden', 32),
        nilut_n_layers=ckpt_args.get('nilut_n_layers', 3),
        nilut_n_freq=ckpt_args.get('nilut_n_freq', 4),
        nilut_gate_init=ckpt_args.get('nilut_gate_init', 1.0),
        use_implicit_head=ckpt_args.get('use_implicit_head', False),
        implicit_head_base_ch=ckpt_args.get('implicit_head_base_ch', 32),
        implicit_head_gate_init=ckpt_args.get('implicit_head_gate_init', 0.0),
        image_size=image_size,
        n_actions=len(ACTIONS),
        dropout=ckpt_args.get('dropout', 0.5),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / 'imgs'
    img_dir.mkdir(exist_ok=True)

    COL_ORIG    = (100, 100, 100)
    COL_REFINED = (56, 189, 248)
    COL_TGT     = (251, 191, 36)
    COL_ERR     = (239, 68, 68)
    COL_CN      = (147, 51, 234)

    ACTION_CN = {
        'contrast': '对比度', 'saturation': '饱和度',
        'shadows': '暗部提亮', 'highlights': '高光压暗', 'wb': '白平衡',
        'brightness': '亮度', 'clarity': '清晰度',
    }
    PARAM_CN = {
        'white_balance': 'WB', 'brightness': '亮度', 'contrast': '对比度',
        'shadows': '暗部', 'highlights': '高光',
        'saturation': '饱和度', 'clarity': '清晰度',
    }

    results = []
    per_action_psnr = {a: [] for a in ACTIONS}
    cn_dist_avg = np.zeros(n_colors, dtype=np.float64)
    n_cn = 0
    # v9d context map stats
    ctx_means = []        # per-image mean context
    ctx_lum_corr = []     # per-image Pearson r between ctx and luminance
    ctx_global_sum = None
    ctx_global_count = 0

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            enc_in = batch['enc_input'].to(device)
            orig = batch['orig'].to(device)
            target = batch['target'].to(device)
            a_oh = batch['action_onehot'].to(device)

            refined, cn_avg, y_b, pred_p7 = model(enc_in, orig, a_oh)

            psnr = compute_psnr_single(refined[0], target[0])
            l1 = (refined[0] - target[0]).abs().mean().item()

            action = (batch['action'][0] if 'action' in batch
                      else ACTIONS[batch['action_idx'][0].item()])
            per_action_psnr[action].append(psnr)

            # Per-image CN distribution
            cn_maps = compute_color_naming_maps(
                orig, grouping=('full6' if n_colors == 6 else 'compact3'))
            cn_dist = cn_maps.mean(dim=[2, 3])[0].cpu().numpy()
            cn_dist_avg += cn_dist
            n_cn += 1

            # v9d context map stats
            if use_context and hasattr(model, '_last_context_map'):
                ctx_t = model._last_context_map[0, 0]  # (H, W)
                ctx_means.append(float(ctx_t.mean().item()))
                # Luminance from orig (Rec.709)
                lum = (0.299 * orig[0, 0] + 0.587 * orig[0, 1]
                       + 0.114 * orig[0, 2])
                # Pearson correlation
                cf = ctx_t.flatten()
                lf = lum.flatten()
                cf_c = cf - cf.mean()
                lf_c = lf - lf.mean()
                denom = (torch.sqrt((cf_c ** 2).sum())
                         * torch.sqrt((lf_c ** 2).sum()) + 1e-8)
                r = float(((cf_c * lf_c).sum() / denom).item())
                ctx_lum_corr.append(r)
                # Accumulate spatial mean (for global heatmap)
                if ctx_global_sum is None:
                    ctx_global_sum = ctx_t.cpu().numpy().copy()
                else:
                    ctx_global_sum += ctx_t.cpu().numpy()
                ctx_global_count += 1

            if idx < args.max_samples:
                orig_pil = tensor_to_pil(orig[0])
                refined_pil = tensor_to_pil(refined[0])
                tgt_pil = tensor_to_pil(target[0])
                err_pil = make_error_heatmap(refined[0], target[0])
                cn_pil = visualize_cn_map(cn_maps[0], cn_labels, cn_colors,
                                           size=image_size)

                # v9d: extract context map for this sample
                ctx_pil = None
                ctx_mean = 0.0
                if use_context and hasattr(model, '_last_context_map'):
                    ctx_map = model._last_context_map[0]  # (1, H, W)
                    ctx_mean = float(ctx_map.mean().item())
                    ctx_pil = make_context_heatmap(ctx_map)

                COL_CTX = (236, 72, 153)  # pink/magenta for context label

                orig_lab = add_label(orig_pil,
                    '① 原图 (FiveK input)', COL_ORIG)
                cn_lab = add_label(cn_pil,
                    f'② Color Naming ({n_colors} 色)',
                    COL_CN, (255, 255, 255))
                ref_lab = add_label(refined_pil,
                    f'④ NamedCurves 输出  PSNR={psnr:.1f}dB' if ctx_pil
                    else f'③ NamedCurves 输出  PSNR={psnr:.1f}dB',
                    COL_REFINED, (0, 0, 0))
                tgt_lab = add_label(tgt_pil,
                    '⑤ FireRed 目标' if ctx_pil else '④ FireRed 目标',
                    COL_TGT, (0, 0, 0))
                err_lab = add_label(err_pil,
                    f'⑥ 像素误差 ×5' if ctx_pil else f'⑤ 像素误差 ×5',
                    COL_ERR)

                W, H = orig_lab.size
                if ctx_pil is not None:
                    ctx_lab = add_label(ctx_pil,
                        f'③ Context map  mean={ctx_mean:.2f}',
                        COL_CTX, (255, 255, 255))
                    strip = Image.new('RGB', (W * 6 + 10, H), (15, 23, 42))
                    strip.paste(orig_lab,  (0,             0))
                    strip.paste(cn_lab,    (W + 2,         0))
                    strip.paste(ctx_lab,   (W * 2 + 4,     0))
                    strip.paste(ref_lab,   (W * 3 + 6,     0))
                    strip.paste(tgt_lab,   (W * 4 + 8,     0))
                    strip.paste(err_lab,   (W * 5 + 10,    0))
                else:
                    strip = Image.new('RGB', (W * 5 + 8, H), (15, 23, 42))
                    strip.paste(orig_lab,  (0,             0))
                    strip.paste(cn_lab,    (W + 2,         0))
                    strip.paste(ref_lab,   (W * 2 + 4,     0))
                    strip.paste(tgt_lab,   (W * 3 + 6,     0))
                    strip.paste(err_lab,   (W * 4 + 8,     0))

                fname = f'{idx:03d}_{action}.jpg'
                strip.save(img_dir / fname, quality=88)

                # CN distribution as label-value pairs
                cn_rows = []
                for i, lab in enumerate(cn_labels):
                    cn_rows.append({
                        'label': lab,
                        'color': cn_colors[i],
                        'value': round(float(cn_dist[i]), 3),
                    })

                # 7D params (if anchor enabled)
                param_rows = []
                if use_7d_anchor:
                    pred_np = pred_p7[0].cpu().numpy()
                    for i, name in enumerate(PARAM_NAMES_7D):
                        pred_phys = denormalize_one(name, float(pred_np[i]))
                        param_rows.append({
                            'name': PARAM_CN[name],
                            'pred': round(pred_phys, 1),
                        })

                results.append({
                    'idx': idx,
                    'action': action,
                    'action_cn': ACTION_CN.get(action, action),
                    'source': batch['source_image'][0],
                    'psnr': round(psnr, 2),
                    'l1': round(l1, 4),
                    'cn_dist': cn_rows,
                    'params': param_rows,
                    'img': f'imgs/{fname}',
                })

    # Aggregate
    overall_psnr = float(np.mean(
        [p for ps in per_action_psnr.values() for p in ps]))
    pa_summary = {
        a: {
            'n': len(ps),
            'psnr': round(float(np.mean(ps)), 2) if ps else 0,
        }
        for a, ps in per_action_psnr.items()
    }
    cn_dist_avg /= max(n_cn, 1)

    # Sort by PSNR (worst first for review)
    results.sort(key=lambda r: r['psnr'])

    # ----- Per-action table -----
    pa_rows = ''
    for a in ACTIONS:
        s = pa_summary[a]
        cn_label = ACTION_CN.get(a, a)
        ref_color = ('#22c55e' if s['psnr'] > 23.94
                     else '#fbbf24' if s['psnr'] > 22 else '#ef4444')
        pa_rows += (f'<tr><td>{cn_label} ({a})</td><td>{s["n"]}</td>'
                    f'<td style="color:{ref_color};font-weight:bold">'
                    f'{s["psnr"]:.2f}</td></tr>')

    # ----- CN aggregate table -----
    cn_agg_rows = ''
    for i, lab in enumerate(cn_labels):
        pct = float(cn_dist_avg[i]) * 100
        bar_w = pct * 5  # 5 px per percent
        cn_agg_rows += (
            f'<tr><td><span class="dot" '
            f'style="background:{cn_colors[i]}"></span>{lab}</td>'
            f'<td><div class="cn-bar-wrap"><div class="cn-bar" '
            f'style="width:{bar_w:.1f}px;background:{cn_colors[i]}"></div>'
            f'<span class="cn-bar-val">{pct:.1f}%</span></div></td></tr>')

    # ----- Sample cards -----
    items_html = ''
    for r in results:
        psnr_color = ('#22c55e' if r['psnr'] > 24
                      else '#fbbf24' if r['psnr'] > 22
                      else '#ef4444')
        # CN inline bars
        cn_inline = ''
        for cn in r['cn_dist']:
            cn_inline += (
                f'<span class="cn-tag" '
                f'style="background:{cn["color"]};opacity:'
                f'{0.3 + cn["value"]*0.7:.2f}">'
                f'{cn["label"]}: {cn["value"]*100:.0f}%</span>')
        param_html = ''
        if r['params']:
            ptable = ''.join(
                f'<tr><td>{p["name"]}</td><td>{p["pred"]}</td></tr>'
                for p in r['params'])
            param_html = (
                f'<details class="param-det"><summary>7D 参数 (pred)</summary>'
                f'<table class="ptab">'
                f'<thead><tr><th>参数</th><th>pred</th></tr></thead>'
                f'<tbody>{ptable}</tbody></table></details>')

        items_html += f'''
        <div class="item" data-action="{r['action']}">
          <div class="imgwrap"><img src="{r['img']}" loading="lazy"></div>
          <div class="meta">
            <div class="row">
              <span class="badge">{r['action_cn']}</span>
              <span style="color:{psnr_color};font-weight:bold;font-size:14px">
                {r['psnr']:.2f} dB
              </span>
            </div>
            <div class="row"><span class="l">文件</span>
              <span class="pp">{r['source']}</span></div>
            <div class="row" style="flex-wrap:wrap;gap:4px;margin-top:6px">{cn_inline}</div>
            {param_html}
          </div>
        </div>'''

    action_filters = ''.join(
        f'<label><input type="checkbox" class="f-action" '
        f'value="{a}" checked> {ACTION_CN.get(a, a)}</label>'
        for a in ACTIONS
    )

    variant_label = (
        f'colors={n_colors}, ctrl_pts={n_control_points}, '
        f'attn={use_attention}, per_action={per_action_curves}, '
        f'7d_anchor={use_7d_anchor}')

    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>v9 NamedCurves — {args.tag}</title>
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
  .pa-table, .cn-table {{ background: #1e293b; border-collapse: collapse;
               font-size: 13px; width: 100%; }}
  .pa-table th, .pa-table td, .cn-table th, .cn-table td
    {{ padding: 6px 14px; border: 1px solid #334155; text-align: center; }}
  .pa-table th, .cn-table th {{ background: #334155; color: #fbbf24; }}
  .pa-table td:first-child, .cn-table td:first-child {{ text-align: left; }}
  .dot {{ display: inline-block; width: 14px; height: 14px;
          border-radius: 3px; vertical-align: middle; margin-right: 4px; }}
  .cn-bar-wrap {{ display: flex; align-items: center; gap: 8px; }}
  .cn-bar {{ height: 16px; border-radius: 3px; }}
  .cn-bar-val {{ font-size: 11px; color: #94a3b8; }}
  .legend {{ background: #1e293b; padding: 12px 16px; border-radius: 6px;
             border: 1px solid #334155; margin-bottom: 16px; font-size: 13px;
             line-height: 1.8; }}
  .legend span {{ margin-right: 16px; }}
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
  .cn-tag {{ display: inline-block; padding: 2px 6px; border-radius: 3px;
             color: #fff; font-size: 10px; font-weight: bold; }}
  .hide {{ display: none; }}
  .param-det {{ margin-top: 6px; }}
  .param-det summary {{ cursor: pointer; color: #a78bfa; font-size: 12px;
                         padding: 4px 0; }}
  .ptab {{ width: 100%; border-collapse: collapse; font-size: 11px;
           margin-top: 4px; }}
  .ptab th, .ptab td {{ padding: 3px 8px; border: 1px solid #334155;
                         text-align: center; }}
  .ptab th {{ background: #334155; color: #fbbf24; }}
</style>
</head>
<body>
<h1>v9 NamedCurves — 视觉效果对比</h1>
<p class="subtitle">{args.tag} &nbsp;|&nbsp; {variant_label}
  &nbsp;|&nbsp; checkpoint: {args.ckpt}
  &nbsp;|&nbsp; {len(results)} val samples
  &nbsp;|&nbsp; ckpt val_psnr={ckpt.get("val_psnr", 0):.2f}dB</p>

<div class="summary">
  <div class="card"><div class="v">{overall_psnr:.2f}</div>
    <div class="l">overall PSNR (dB)</div></div>
  <div class="card"><div class="v" style="color:#fbbf24">23.94</div>
    <div class="l">7D ISP 上限</div></div>
  <div class="card"><div class="v" style="color:{
    '#22c55e' if overall_psnr > 23.94 else '#ef4444'}">
    {overall_psnr - 23.94:+.2f}</div>
    <div class="l">vs 上限 (dB)</div></div>
  <div class="card"><div class="v">{len(results)}</div>
    <div class="l">val 样本数</div></div>
</div>

<div class="twocol">
  <div>
    <h3 style="margin: 4px 0 8px; font-size: 14px;">Per-action PSNR</h3>
    <table class="pa-table">
      <thead><tr><th>动作</th><th>n</th><th>PSNR (dB)</th></tr></thead>
      <tbody>{pa_rows}</tbody>
    </table>
  </div>
  <div>
    <h3 style="margin: 4px 0 8px; font-size: 14px;">
      Color Naming 分布 (val 平均)</h3>
    <table class="cn-table">
      <thead><tr><th>色名</th><th>占比</th></tr></thead>
      <tbody>{cn_agg_rows}</tbody>
    </table>
  </div>
</div>

<div class="legend">
  <b>5 列对照说明:</b><br>
  <span><span class="dot" style="background:rgb(100,100,100)"></span>
    ① 原图</span>
  <span><span class="dot" style="background:rgb(147,51,234)"></span>
    ② Color Naming 分解 (固定算法, 按色域分组)</span>
  <span><span class="dot" style="background:rgb(56,189,248)"></span>
    ③ NamedCurves 输出 (Bezier 曲线 + CN 加权融合)</span>
  <span><span class="dot" style="background:rgb(251,191,36)"></span>
    ④ FireRed 目标</span>
  <span><span class="dot" style="background:rgb(239,68,68)"></span>
    ⑤ 像素误差热图 (×5)</span>
</div>

<div class="filters">
  <b>按动作过滤:</b> {action_filters}
</div>

<p class="subtitle">默认排序: PSNR 从低到高 (差的在前)</p>
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
    logger.info(f'CN distribution (avg): '
                f'{dict(zip(cn_labels, [round(float(v),3) for v in cn_dist_avg]))}')
    if use_context and ctx_means:
        avg_mean = float(np.mean(ctx_means))
        std_mean = float(np.std(ctx_means))
        avg_corr = float(np.mean(ctx_lum_corr))
        std_corr = float(np.std(ctx_lum_corr))
        logger.info(f'Context map stats (val set, n={len(ctx_means)}):')
        logger.info(f'  per-image mean: avg={avg_mean:.3f}, std={std_mean:.3f}')
        logger.info(f'  Pearson r(ctx, luminance): '
                    f'avg={avg_corr:+.3f}, std={std_corr:.3f}')
        logger.info(f'  |r|>0.3 (strong corr): '
                    f'{sum(1 for r in ctx_lum_corr if abs(r) > 0.3)}/{len(ctx_lum_corr)}')


if __name__ == '__main__':
    main()
