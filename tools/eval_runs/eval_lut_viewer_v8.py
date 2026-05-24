"""v8 (ParamResidualLUTPredictor) 视觉对比 viewer.

5 列布局: 原图 | Coarse (7D ISP 渲染) | Refined (+ 残差 LUT) | FireRed 目标 | 误差热图
每张图附 7D 预测参数 vs P_inferred 真值表, 用于验证可解释性.
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
    ParamResidualLUTPredictor, LUTDataset, build_data, ACTIONS,
    PARAM_NAMES_7D, PARAM_NORM_7D,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


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


def denormalize_one(name: str, v_norm: float) -> float:
    cfg = PARAM_NORM_7D[name]
    return v_norm * cfg['scale'] + cfg['center']


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/lut_v8/best.pt')
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='outputs/lut_v8_viewer')
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_samples', type=int, default=80)
    ap.add_argument('--tag', default='v8 (7D ISP + 残差 LUT, dim=17)')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(Path(args.ckpt), map_location=device, weights_only=False)
    ckpt_args = ckpt['args']
    logger.info(f'Loaded {args.ckpt}: '
                f'val_psnr={ckpt.get("val_psnr", 0):.2f}dB '
                f'@ Ep{ckpt.get("epoch", "?")}')

    if not ckpt_args.get('param_residual_lut', False):
        raise RuntimeError(
            f'Checkpoint is not a v8 (param_residual_lut) model. '
            f'Use eval_lut_viewer_v2.py for v1-v7 checkpoints.')

    tier_filter = ('A excellent', 'B good', 'C acceptable')
    train_s, val_s = build_data(
        Path(args.jsonl), args.val_ratio, tier_filter, args.seed)

    val_ds = LUTDataset(val_s, args.image_size, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = ParamResidualLUTPredictor(
        n_luts=ckpt_args['n_luts'], lut_dim=ckpt_args['lut_dim'],
        image_size=ckpt_args['image_size'], dropout=ckpt_args['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / 'imgs'
    img_dir.mkdir(exist_ok=True)

    # 标签颜色
    COL_ORIG    = (100, 100, 100)   # 灰  — 原图
    COL_COARSE  = (147, 51, 234)    # 紫  — 7D ISP 粗渲染
    COL_REFINED = (56, 189, 248)    # 蓝  — 残差 LUT 后输出
    COL_TGT     = (251, 191, 36)    # 黄  — FireRed 目标
    COL_ERR     = (239, 68, 68)     # 红  — 误差热图

    ACTION_CN = {
        'contrast': '对比度',
        'saturation': '饱和度',
        'shadows': '暗部提亮',
        'highlights': '高光压暗',
        'wb': '白平衡',
    }
    PARAM_CN = {
        'white_balance': 'WB',
        'brightness': '亮度',
        'contrast': '对比度',
        'shadows': '暗部',
        'highlights': '高光',
        'saturation': '饱和度',
        'clarity': '清晰度',
    }

    results = []
    per_action_psnr = {a: [] for a in ACTIONS}
    per_action_coarse_psnr = {a: [] for a in ACTIONS}
    param_errors = {p: [] for p in PARAM_NAMES_7D}  # |pred - gt| (normalized)

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            enc_in = batch['enc_input'].to(device)
            orig = batch['orig'].to(device)
            target = batch['target'].to(device)
            a_oh = batch['action_onehot'].to(device)
            gt_p7 = batch['params_norm_7d']  # (1, 7) on cpu

            refined, weights, coarse, pred_p7 = model(enc_in, orig, a_oh)
            psnr_ref = compute_psnr_single(refined[0], target[0])
            psnr_coarse = compute_psnr_single(coarse[0], target[0])
            l1 = (refined[0] - target[0]).abs().mean().item()

            action = (batch['action'][0] if 'action' in batch
                      else ACTIONS[batch['action_idx'][0].item()])
            per_action_psnr[action].append(psnr_ref)
            per_action_coarse_psnr[action].append(psnr_coarse)

            # Param accuracy (normalized space)
            pred_np = pred_p7[0].cpu().numpy()
            gt_np = gt_p7[0].cpu().numpy()
            for i, name in enumerate(PARAM_NAMES_7D):
                param_errors[name].append(abs(pred_np[i] - gt_np[i]))

            if idx < args.max_samples:
                orig_pil = tensor_to_pil(orig[0])
                coarse_pil = tensor_to_pil(coarse[0])
                refined_pil = tensor_to_pil(refined[0])
                tgt_pil = tensor_to_pil(target[0])
                err_pil = make_error_heatmap(refined[0], target[0])

                orig_lab = add_label(orig_pil,
                    '① 原图 (FiveK input)', COL_ORIG)
                coarse_lab = add_label(coarse_pil,
                    f'② 7D ISP coarse  PSNR={psnr_coarse:.1f}dB',
                    COL_COARSE, (255, 255, 255))
                ref_lab = add_label(refined_pil,
                    f'③ +残差 LUT  PSNR={psnr_ref:.1f}dB',
                    COL_REFINED, (0, 0, 0))
                tgt_lab = add_label(tgt_pil,
                    '④ FireRed 目标', COL_TGT, (0, 0, 0))
                err_lab = add_label(err_pil,
                    f'⑤ 像素误差 ×5', COL_ERR)

                W, H = orig_lab.size
                strip = Image.new('RGB', (W * 5 + 8, H), (15, 23, 42))
                strip.paste(orig_lab,   (0,             0))
                strip.paste(coarse_lab, (W + 2,         0))
                strip.paste(ref_lab,    (W * 2 + 4,     0))
                strip.paste(tgt_lab,    (W * 3 + 6,     0))
                strip.paste(err_lab,    (W * 4 + 8,     0))

                fname = f'{idx:03d}_{action}.jpg'
                strip.save(img_dir / fname, quality=88)

                # Build pred-vs-gt parameter rows (physical values)
                p_rows = []
                for i, name in enumerate(PARAM_NAMES_7D):
                    pred_phys = denormalize_one(name, float(pred_np[i]))
                    gt_phys = denormalize_one(name, float(gt_np[i]))
                    diff = pred_phys - gt_phys
                    p_rows.append({
                        'name': PARAM_CN[name],
                        'pred': round(pred_phys, 1),
                        'gt': round(gt_phys, 1),
                        'diff': round(diff, 1),
                    })

                results.append({
                    'idx': idx,
                    'action': action,
                    'action_cn': ACTION_CN.get(action, action),
                    'source': batch['source_image'][0],
                    'psnr_ref': round(psnr_ref, 2),
                    'psnr_coarse': round(psnr_coarse, 2),
                    'gain': round(psnr_ref - psnr_coarse, 2),
                    'l1': round(l1, 4),
                    'weights': [round(w, 3)
                                for w in weights[0].cpu().numpy().tolist()],
                    'params': p_rows,
                    'img': f'imgs/{fname}',
                })

    # 汇总
    overall_psnr = float(np.mean(
        [p for ps in per_action_psnr.values() for p in ps]))
    overall_coarse_psnr = float(np.mean(
        [p for ps in per_action_coarse_psnr.values() for p in ps]))
    pa_summary = {
        a: {
            'n': len(ps),
            'psnr_ref': round(float(np.mean(ps)), 2) if ps else 0,
            'psnr_coarse': round(float(np.mean(per_action_coarse_psnr[a])), 2)
                if per_action_coarse_psnr[a] else 0,
        }
        for a, ps in per_action_psnr.items()
    }
    param_mae_norm = {p: round(float(np.mean(es)), 3)
                      for p, es in param_errors.items() if es}

    # 排序: 默认 PSNR 从低到高 (差的在前)
    results.sort(key=lambda r: r['psnr_ref'])

    # ----- per-action 表 -----
    pa_rows = ''
    for a in ACTIONS:
        s = pa_summary[a]
        cn = ACTION_CN.get(a, a)
        gain = s['psnr_ref'] - s['psnr_coarse']
        ref_color = ('#22c55e' if s['psnr_ref'] > 23.94
                     else '#fbbf24' if s['psnr_ref'] > 22 else '#ef4444')
        pa_rows += (f'<tr><td>{cn} ({a})</td><td>{s["n"]}</td>'
                    f'<td>{s["psnr_coarse"]:.2f}</td>'
                    f'<td style="color:{ref_color};font-weight:bold">'
                    f'{s["psnr_ref"]:.2f}</td>'
                    f'<td>+{gain:.2f}</td></tr>')

    # ----- 7D 参数 MAE 表 -----
    p_rows_html = ''
    for name in PARAM_NAMES_7D:
        cfg = PARAM_NORM_7D[name]
        mae_norm = param_mae_norm.get(name, 0.0)
        mae_phys = mae_norm * cfg['scale']
        p_rows_html += (f'<tr><td>{PARAM_CN[name]}</td>'
                        f'<td>{mae_norm:.3f}</td>'
                        f'<td>{mae_phys:.1f}</td>'
                        f'<td>±{cfg["scale"]:.0f}</td></tr>')

    # ----- 样本卡 -----
    items_html = ''
    for r in results:
        w_str = ', '.join(f'{w:.2f}' for w in r['weights'])
        psnr_color = ('#22c55e' if r['psnr_ref'] > 24
                      else '#fbbf24' if r['psnr_ref'] > 22
                      else '#ef4444')
        param_table = ''.join(
            f'<tr><td>{p["name"]}</td><td>{p["pred"]}</td>'
            f'<td>{p["gt"]}</td>'
            f'<td style="color:{"#94a3b8" if abs(p["diff"]) < 5 else "#fbbf24"}">'
            f'{p["diff"]:+}</td></tr>'
            for p in r['params']
        )
        items_html += f'''
        <div class="item" data-action="{r['action']}">
          <div class="imgwrap"><img src="{r['img']}" loading="lazy"></div>
          <div class="meta">
            <div class="row">
              <span class="badge">{r['action_cn']}</span>
              <span style="color:{psnr_color};font-weight:bold;font-size:14px">
                {r['psnr_ref']:.2f} dB
              </span>
              <span style="color:#a78bfa;font-size:11px">
                (coarse {r['psnr_coarse']:.2f} → +{r['gain']:.2f})
              </span>
            </div>
            <div class="row"><span class="l">文件</span>
              <span class="pp">{r['source']}</span></div>
            <div class="row"><span class="l">LUT权重</span>
              <span class="pp">[{w_str}]</span></div>
            <details class="param-det"><summary>7D 参数 (pred vs gt)</summary>
              <table class="ptab">
                <thead><tr><th>参数</th><th>pred</th><th>gt</th>
                  <th>Δ</th></tr></thead>
                <tbody>{param_table}</tbody>
              </table>
            </details>
          </div>
        </div>'''

    action_filters = ''.join(
        f'<label><input type="checkbox" class="f-action" '
        f'value="{a}" checked> {ACTION_CN.get(a, a)}</label>'
        for a in ACTIONS
    )

    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>v8 (7D ISP + 残差 LUT) 视觉对比 — {args.tag}</title>
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
  .pa-table, .p-table {{ background: #1e293b; border-collapse: collapse;
               font-size: 13px; width: 100%; }}
  .pa-table th, .pa-table td, .p-table th, .p-table td
    {{ padding: 6px 14px; border: 1px solid #334155; text-align: center; }}
  .pa-table th, .p-table th {{ background: #334155; color: #fbbf24; }}
  .pa-table td:first-child, .p-table td:first-child {{ text-align: left; }}
  .legend {{ background: #1e293b; padding: 12px 16px; border-radius: 6px;
             border: 1px solid #334155; margin-bottom: 16px; font-size: 13px;
             line-height: 1.8; }}
  .legend .dot {{ display: inline-block; width: 14px; height: 14px;
                  border-radius: 3px; vertical-align: middle; margin-right: 4px; }}
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
  .hide {{ display: none; }}
  .param-det {{ margin-top: 6px; }}
  .param-det summary {{ cursor: pointer; color: #a78bfa; font-size: 12px;
                         padding: 4px 0; }}
  .ptab {{ width: 100%; border-collapse: collapse; font-size: 11px;
           margin-top: 4px; }}
  .ptab th, .ptab td {{ padding: 3px 8px; border: 1px solid #334155;
                         text-align: center; }}
  .ptab th {{ background: #334155; color: #fbbf24; }}
  .ptab td:first-child {{ text-align: left; color: #cbd5e1; }}
</style>
</head>
<body>
<h1>v8: 7D ISP + 残差 LUT — 视觉效果对比</h1>
<p class="subtitle">{args.tag} &nbsp;|&nbsp; checkpoint: {args.ckpt}
  &nbsp;|&nbsp; {len(results)} val samples
  &nbsp;|&nbsp; ckpt val_psnr={ckpt.get("val_psnr", 0):.2f}dB</p>

<div class="summary">
  <div class="card"><div class="v">{overall_psnr:.2f}</div>
    <div class="l">refined PSNR (dB)</div></div>
  <div class="card"><div class="v" style="color:#a78bfa">
    {overall_coarse_psnr:.2f}</div>
    <div class="l">coarse PSNR (dB)</div></div>
  <div class="card"><div class="v" style="color:#22c55e">
    +{overall_psnr - overall_coarse_psnr:.2f}</div>
    <div class="l">残差 LUT 增益</div></div>
  <div class="card"><div class="v" style="color:#fbbf24">23.94</div>
    <div class="l">7D ISP 理论上限</div></div>
  <div class="card"><div class="v" style="color:{
    '#22c55e' if overall_psnr > 23.94 else '#ef4444'}">
    {overall_psnr - 23.94:+.2f}</div>
    <div class="l">vs 上限 (dB)</div></div>
</div>

<div class="twocol">
  <div>
    <h3 style="margin: 4px 0 8px; font-size: 14px;">Per-action PSNR</h3>
    <table class="pa-table">
      <thead><tr><th>动作</th><th>n</th><th>coarse</th>
        <th>refined</th><th>增益</th></tr></thead>
      <tbody>{pa_rows}</tbody>
    </table>
  </div>
  <div>
    <h3 style="margin: 4px 0 8px; font-size: 14px;">
      7D 参数预测 MAE (val 集)</h3>
    <table class="p-table">
      <thead><tr><th>参数</th><th>norm MAE</th><th>物理 MAE</th>
        <th>scale</th></tr></thead>
      <tbody>{p_rows_html}</tbody>
    </table>
  </div>
</div>

<div class="legend">
  <b>5 列对照说明:</b><br>
  <span><span class="dot" style="background:rgb(100,100,100)"></span>
    ① 原图 — FiveK 输入</span>
  <span><span class="dot" style="background:rgb(147,51,234)"></span>
    ② Coarse — 7D ISP 参数 (可微 ISP) 渲染</span>
  <span><span class="dot" style="background:rgb(56,189,248)"></span>
    ③ Refined — 加上残差 3D-LUT 后的最终输出</span>
  <span><span class="dot" style="background:rgb(251,191,36)"></span>
    ④ FireRed 目标 (ground truth)</span>
  <span><span class="dot" style="background:rgb(239,68,68)"></span>
    ⑤ 像素误差热图 (×5 放大)</span>
</div>

<div class="filters">
  <b>按动作过滤:</b> {action_filters}
</div>

<p class="subtitle">默认排序: refined PSNR 从低到高 (差的在前, 方便审查)</p>
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
    logger.info(f'Refined PSNR: {overall_psnr:.2f} dB '
                f'| Coarse PSNR: {overall_coarse_psnr:.2f} dB '
                f'| Gain: +{overall_psnr - overall_coarse_psnr:.2f} dB')
    logger.info(f'7D Param MAE (norm): {param_mae_norm}')


if __name__ == '__main__':
    main()
