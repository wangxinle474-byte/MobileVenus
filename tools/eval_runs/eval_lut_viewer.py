"""加载 LUT checkpoint, 对验证集逐张推理, 生成 4 列对比图 + HTML viewer.

输出布局:
  orig | LUT_pred | FireRed_target | error_heatmap (per-pixel L1)

并附:
  - per-action PSNR/L1 表
  - 总体 PSNR
  - 过滤 (action / quality_tier)
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
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.firered_baseline.train_lut import (  # noqa: E402
    LUTPredictor, LUTDataset, build_data, ACTIONS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def compute_psnr_per_image(pred: torch.Tensor, target: torch.Tensor):
    """Per-image PSNR for (B, 3, H, W) [0, 1] tensors. Returns (B,)."""
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])
    return -10 * torch.log10(mse.clamp(min=1e-10))


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(3, H, W) [0, 1] → PIL."""
    arr = (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def error_heatmap(pred: torch.Tensor, target: torch.Tensor) -> Image.Image:
    """生成 per-pixel L1 误差热图 (放大 5×).

    pred, target: (3, H, W) [0, 1]
    """
    err = (pred - target).abs().mean(dim=0)  # (H, W)
    err_amp = (err * 5.0).clamp(0, 1).cpu().numpy()  # 放大让小误差也可见
    # 简单 viridis-like colormap (BGR style)
    r = np.clip(err_amp * 2 - 0.5, 0, 1)
    g = np.clip(1 - (err_amp - 0.5).__abs__() * 2, 0, 1)
    b = np.clip(1 - err_amp, 0, 1) * 0.5
    rgb = np.stack([r, g, b], axis=-1) * 255
    return Image.fromarray(rgb.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/lut_v1/best.pt')
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='outputs/lut_v1_viewer')
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_samples', type=int, default=80,
                    help='最多保存多少 val 样本; 太多会让 HTML 卡')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load ckpt
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt['args']
    logger.info(f'Loaded {ckpt_path}: best_val_psnr={ckpt.get("val_psnr", "?"):.2f}dB')

    # Build same val split as training (need same seed + val_ratio)
    tier_filter = ('A excellent', 'B good', 'C acceptable')
    train_s, val_s = build_data(
        Path(args.jsonl), args.val_ratio, tier_filter, args.seed)
    logger.info(f'Val set: {len(val_s)} samples')

    val_ds = LUTDataset(val_s, args.image_size, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Model
    model = LUTPredictor(
        n_luts=ckpt_args['n_luts'],
        lut_dim=ckpt_args['lut_dim'],
        image_size=ckpt_args['image_size'],
        dropout=ckpt_args['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Output dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / 'imgs'
    img_dir.mkdir(exist_ok=True)

    # Eval all
    results = []
    per_action_psnr = {a: [] for a in ACTIONS}
    per_action_l1 = {a: [] for a in ACTIONS}

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            enc_in = batch['enc_input'].to(device)
            orig = batch['orig'].to(device)
            target = batch['target'].to(device)
            a_oh = batch['action_onehot'].to(device)

            pred, weights = model(enc_in, orig, a_oh)

            psnr = compute_psnr_per_image(pred, target).item()
            l1 = (pred - target).abs().mean().item()

            action = batch['action'][0] if 'action' in batch else ACTIONS[batch['action_idx'][0]]
            per_action_psnr[action].append(psnr)
            per_action_l1[action].append(l1)

            # Save image strip (orig | pred | target | err)
            if idx < args.max_samples:
                orig_pil = tensor_to_pil(orig[0])
                pred_pil = tensor_to_pil(pred[0])
                tgt_pil = tensor_to_pil(target[0])
                err_pil = error_heatmap(pred[0], target[0])

                W, H = orig_pil.size
                strip = Image.new('RGB', (W * 4, H), (15, 23, 42))
                strip.paste(orig_pil, (0, 0))
                strip.paste(pred_pil, (W, 0))
                strip.paste(tgt_pil, (W * 2, 0))
                strip.paste(err_pil, (W * 3, 0))
                strip_path = img_dir / f'{idx:03d}_{action}.jpg'
                strip.save(strip_path, quality=88)

                results.append({
                    'idx': idx,
                    'action': action,
                    'source_image': batch['source_image'][0],
                    'psnr': float(psnr),
                    'l1': float(l1),
                    'weights': [float(w) for w in weights[0].cpu().numpy()],
                    'img': f'imgs/{idx:03d}_{action}.jpg',
                })

    # Summary
    all_psnr = [r['psnr'] for r in results]
    summary = {
        'ckpt': str(ckpt_path),
        'val_psnr_overall': float(np.mean([p for ps in per_action_psnr.values() for p in ps])),
        'val_l1_overall': float(np.mean([l for ls in per_action_l1.values() for l in ls])),
        'per_action': {
            a: {
                'n': len(per_action_psnr[a]),
                'psnr': float(np.mean(per_action_psnr[a])) if per_action_psnr[a] else 0.0,
                'l1': float(np.mean(per_action_l1[a])) if per_action_l1[a] else 0.0,
            }
            for a in ACTIONS
        },
        'n_total': sum(len(v) for v in per_action_psnr.values()),
    }

    logger.info(f'\n=== Overall ===')
    logger.info(f'  N={summary["n_total"]}  val_PSNR={summary["val_psnr_overall"]:.2f} dB  '
                f'val_L1={summary["val_l1_overall"]:.4f}')
    logger.info(f'\n=== Per-action ===')
    for a in ACTIONS:
        s = summary['per_action'][a]
        if s['n'] > 0:
            logger.info(f'  {a:<11s} n={s["n"]:>3d}  PSNR={s["psnr"]:.2f}dB  L1={s["l1"]:.4f}')

    json.dump(summary, open(out_dir / 'summary.json', 'w'), indent=2)
    json.dump(results, open(out_dir / 'results.json', 'w'), indent=2)

    # HTML viewer
    pa_rows = ''
    for a in ACTIONS:
        s = summary['per_action'][a]
        if s['n'] > 0:
            pa_rows += (f'<tr><td>{a}</td><td>{s["n"]}</td>'
                        f'<td>{s["psnr"]:.2f}</td><td>{s["l1"]:.4f}</td></tr>')

    items_html = ''
    for r in results:
        weights_str = ', '.join(f'{w:.2f}' for w in r['weights'])
        items_html += f'''
        <div class="item" data-action="{r['action']}">
          <div class="col-headers">
            <div class="h-orig">orig</div>
            <div class="h-pred">LUT pred</div>
            <div class="h-tgt">FireRed target</div>
            <div class="h-err">err×5</div>
          </div>
          <div class="imgwrap"><img src="{r['img']}" loading="lazy"></div>
          <div class="meta">
            <div class="row">
              <span class="badge" style="background:#7c3aed">{r['action']}</span>
              <span class="pp">{r['source_image']}</span>
            </div>
            <div class="row"><span class="l">PSNR</span><b style="color:#38bdf8">{r['psnr']:.2f} dB</b></div>
            <div class="row"><span class="l">L1</span><span>{r['l1']:.4f}</span></div>
            <div class="row"><span class="l">LUT weights</span><span class="pp">[{weights_str}]</span></div>
          </div>
        </div>'''

    action_filters = ''.join(
        f'<label><input type="checkbox" class="f-action" value="{a}" checked> {a}</label>'
        for a in ACTIONS
    )

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LUT v1 eval viewer — best.pt</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 16px;
         background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 18px; margin: 0 0 12px; }}
  .summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
  .card {{ background: #1e293b; padding: 10px 16px; border-radius: 6px;
           border: 1px solid #334155; min-width: 140px; }}
  .card .v {{ font-size: 22px; font-weight: bold; color: #38bdf8; }}
  .card .l {{ font-size: 12px; color: #94a3b8; }}
  .pa-table {{ background: #1e293b; border-collapse: collapse;
               font-size: 13px; margin-bottom: 16px; }}
  .pa-table th, .pa-table td {{ padding: 6px 10px; border: 1px solid #334155; }}
  .pa-table th {{ background: #334155; color: #fbbf24; }}
  .filters {{ background: #1e293b; padding: 12px 16px; border-radius: 6px;
              border: 1px solid #334155; margin-bottom: 16px; }}
  .filters label {{ margin-right: 12px; cursor: pointer; }}
  .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }}
  .item {{ background: #1e293b; border-radius: 6px; overflow: hidden;
           border: 1px solid #334155; }}
  .item img {{ width: 100%; display: block; }}
  .col-headers {{ display: grid; grid-template-columns: repeat(4, 1fr);
                  font-size: 11px; color: #cbd5e1; background: #0f172a;
                  padding: 4px 0; text-align: center; border-bottom: 1px solid #334155; }}
  .col-headers div {{ border-right: 1px solid #334155; padding: 2px 4px; }}
  .col-headers div:last-child {{ border-right: none; }}
  .col-headers .h-orig {{ color: #94a3b8; }}
  .col-headers .h-pred {{ color: #38bdf8; }}
  .col-headers .h-tgt  {{ color: #fbbf24; }}
  .col-headers .h-err  {{ color: #ef4444; }}
  .meta {{ padding: 8px 12px; font-size: 12px; line-height: 1.5; }}
  .badge {{ display: inline-block; padding: 1px 6px; border-radius: 3px;
            color: #fff; font-weight: bold; margin-right: 6px; }}
  .row {{ display: flex; justify-content: space-between; align-items: center;
          margin-bottom: 3px; }}
  .row .l {{ color: #94a3b8; }}
  .pp {{ font-size: 11px; color: #94a3b8; }}
  .hide {{ display: none; }}
</style>
</head>
<body>
<h1>LUT v1 — 3 basis LUTs (33³), tested on {summary["n_total"]} val samples</h1>

<div class="summary">
  <div class="card"><div class="v">{summary["val_psnr_overall"]:.2f}</div><div class="l">overall PSNR (dB)</div></div>
  <div class="card"><div class="v">{summary["val_l1_overall"]:.4f}</div><div class="l">overall L1</div></div>
  <div class="card"><div class="v" style="color:#fbbf24">23.94</div><div class="l">7D ISP ceiling ref</div></div>
  <div class="card"><div class="v" style="color:{'#22c55e' if summary['val_psnr_overall'] > 23.94 else '#ef4444'}">
    {summary["val_psnr_overall"] - 23.94:+.2f}</div><div class="l">vs 7D ceiling (dB)</div></div>
</div>

<table class="pa-table">
  <thead><tr><th>action</th><th>n</th><th>PSNR (dB)</th><th>L1</th></tr></thead>
  <tbody>{pa_rows}</tbody>
</table>

<div class="filters">
  <b>Action filter:</b> {action_filters}
</div>

<div class="grid">{items_html}</div>

<script>
const items = document.querySelectorAll('.item');
const checkboxes = document.querySelectorAll('.f-action');
function update() {{
  const active = new Set(Array.from(checkboxes).filter(c => c.checked).map(c => c.value));
  items.forEach(it => {{
    it.classList.toggle('hide', !active.has(it.dataset.action));
  }});
}}
checkboxes.forEach(c => c.addEventListener('change', update));
</script>
</body>
</html>'''

    (out_dir / 'viewer.html').write_text(html, encoding='utf-8')
    logger.info(f'\nViewer: {out_dir / "viewer.html"}')
    logger.info(f'Summary: {out_dir / "summary.json"}')


if __name__ == '__main__':
    main()
