"""生成标注清晰的 LUT eval 对比 viewer.

每张图展示 4 列: 原图 | LUT预测 | FireRed目标 | 误差热图
每列上方有醒目的中文标签, 说明该图来源.
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
    LUTPredictor, LUTDataset, build_data, ACTIONS,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


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
        font = ImageFont.truetype("msyh.ttc", 16)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/lut_v1/best.pt')
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='outputs/lut_v1_viewer')
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_samples', type=int, default=80)
    ap.add_argument('--tag', default='3D-LUT v1 (dim=33, 3 basis)')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(Path(args.ckpt), map_location=device, weights_only=False)
    ckpt_args = ckpt['args']
    logger.info(f'Loaded {args.ckpt}: val_psnr={ckpt.get("val_psnr", "?"):.2f}dB')

    tier_filter = ('A excellent', 'B good', 'C acceptable')
    train_s, val_s = build_data(
        Path(args.jsonl), args.val_ratio, tier_filter, args.seed)

    val_ds = LUTDataset(val_s, args.image_size, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = LUTPredictor(
        n_luts=ckpt_args['n_luts'], lut_dim=ckpt_args['lut_dim'],
        image_size=ckpt_args['image_size'], dropout=ckpt_args['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / 'imgs'
    img_dir.mkdir(exist_ok=True)

    # 标签颜色定义
    COL_ORIG = (100, 100, 100)     # 灰色 — 原图
    COL_PRED = (56, 189, 248)      # 蓝色 — LUT 预测
    COL_TGT  = (251, 191, 36)      # 黄色 — FireRed 目标
    COL_ERR  = (239, 68, 68)       # 红色 — 误差热图

    ACTION_CN = {
        'contrast': '对比度',
        'saturation': '饱和度',
        'shadows': '暗部提亮',
        'highlights': '高光压暗',
        'wb': '白平衡',
    }

    results = []
    per_action_psnr = {a: [] for a in ACTIONS}

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            enc_in = batch['enc_input'].to(device)
            orig = batch['orig'].to(device)
            target = batch['target'].to(device)
            a_oh = batch['action_onehot'].to(device)

            pred, weights = model(enc_in, orig, a_oh)
            psnr = compute_psnr_single(pred[0], target[0])
            l1 = (pred[0] - target[0]).abs().mean().item()

            action = batch['action'][0] if 'action' in batch else ACTIONS[batch['action_idx'][0].item()]
            per_action_psnr[action].append(psnr)

            if idx < args.max_samples:
                orig_pil = tensor_to_pil(orig[0])
                pred_pil = tensor_to_pil(pred[0])
                tgt_pil = tensor_to_pil(target[0])
                err_pil = make_error_heatmap(pred[0], target[0])

                # 添加标签
                orig_lab = add_label(orig_pil,
                    '① 原图 (FiveK input)', COL_ORIG)
                pred_lab = add_label(pred_pil,
                    f'② {args.tag} 预测', COL_PRED, (0, 0, 0))
                tgt_lab = add_label(tgt_pil,
                    '③ FireRed 编辑目标 (ground truth)', COL_TGT, (0, 0, 0))
                err_lab = add_label(err_pil,
                    f'④ 像素误差×5  PSNR={psnr:.1f}dB', COL_ERR)

                W, H = orig_lab.size
                strip = Image.new('RGB', (W * 4 + 6, H), (15, 23, 42))
                strip.paste(orig_lab, (0, 0))
                strip.paste(pred_lab, (W + 2, 0))
                strip.paste(tgt_lab, (W * 2 + 4, 0))
                strip.paste(err_lab, (W * 3 + 6, 0))

                fname = f'{idx:03d}_{action}.jpg'
                strip.save(img_dir / fname, quality=90)

                results.append({
                    'idx': idx,
                    'action': action,
                    'action_cn': ACTION_CN.get(action, action),
                    'source': batch['source_image'][0],
                    'psnr': round(psnr, 2),
                    'l1': round(l1, 4),
                    'weights': [round(w, 3) for w in weights[0].cpu().numpy().tolist()],
                    'img': f'imgs/{fname}',
                })

    # 汇总
    overall_psnr = float(np.mean([p for ps in per_action_psnr.values() for p in ps]))
    pa_summary = {a: round(float(np.mean(ps)), 2) if ps else 0 for a, ps in per_action_psnr.items()}

    # 按 PSNR 排序 (从低到高, 差的排前面方便查看)
    results.sort(key=lambda r: r['psnr'])

    # 生成 HTML
    pa_rows = ''
    for a in ACTIONS:
        n = len(per_action_psnr[a])
        p = pa_summary[a]
        cn = ACTION_CN.get(a, a)
        status = '✅ 超 ceiling' if p > 23.94 else '⚠️ 未超'
        pa_rows += f'<tr><td>{cn} ({a})</td><td>{n}</td><td><b>{p:.2f}</b></td><td>{status}</td></tr>'

    items_html = ''
    for r in results:
        w_str = ', '.join(f'{w:.2f}' for w in r['weights'])
        psnr_color = '#22c55e' if r['psnr'] > 23.94 else '#fbbf24' if r['psnr'] > 22 else '#ef4444'
        items_html += f'''
        <div class="item" data-action="{r['action']}">
          <div class="imgwrap"><img src="{r['img']}" loading="lazy"></div>
          <div class="meta">
            <div class="row">
              <span class="badge">{r['action_cn']}</span>
              <span style="color:{psnr_color};font-weight:bold;font-size:14px">{r['psnr']:.2f} dB</span>
            </div>
            <div class="row"><span class="l">文件</span><span class="pp">{r['source']}</span></div>
            <div class="row"><span class="l">LUT权重</span><span class="pp">[{w_str}]</span></div>
          </div>
        </div>'''

    action_filters = ''.join(
        f'<label><input type="checkbox" class="f-action" value="{a}" checked> {ACTION_CN.get(a,a)}</label>'
        for a in ACTIONS
    )

    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>方案A: 3D-LUT 视觉效果对比 — {args.tag}</title>
<style>
  body {{ font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif; margin: 0; padding: 16px;
         background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 20px; margin: 0 0 6px; }}
  .subtitle {{ font-size: 13px; color: #94a3b8; margin-bottom: 16px; }}
  .summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
  .card {{ background: #1e293b; padding: 10px 16px; border-radius: 6px;
           border: 1px solid #334155; min-width: 140px; }}
  .card .v {{ font-size: 24px; font-weight: bold; color: #38bdf8; }}
  .card .l {{ font-size: 12px; color: #94a3b8; }}
  .pa-table {{ background: #1e293b; border-collapse: collapse;
               font-size: 13px; margin-bottom: 16px; width: auto; }}
  .pa-table th, .pa-table td {{ padding: 6px 14px; border: 1px solid #334155; }}
  .pa-table th {{ background: #334155; color: #fbbf24; }}
  .legend {{ background: #1e293b; padding: 12px 16px; border-radius: 6px;
             border: 1px solid #334155; margin-bottom: 16px; font-size: 13px; }}
  .legend .dot {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px;
                  vertical-align: middle; margin-right: 4px; }}
  .legend span {{ margin-right: 20px; }}
  .filters {{ background: #1e293b; padding: 12px 16px; border-radius: 6px;
              border: 1px solid #334155; margin-bottom: 16px; }}
  .filters label {{ margin-right: 14px; cursor: pointer; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; max-width: 1200px; }}
  .item {{ background: #1e293b; border-radius: 6px; overflow: hidden;
           border: 1px solid #334155; }}
  .item img {{ width: 100%; display: block; }}
  .meta {{ padding: 8px 12px; font-size: 12px; line-height: 1.6; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px;
            color: #fff; font-weight: bold; margin-right: 8px; background: #7c3aed; }}
  .row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; }}
  .row .l {{ color: #94a3b8; min-width: 70px; }}
  .pp {{ font-size: 11px; color: #94a3b8; }}
  .hide {{ display: none; }}
  .sort-info {{ font-size: 12px; color: #94a3b8; margin-bottom: 8px; }}
</style>
</head>
<body>
<h1>方案A: Image-Adaptive 3D-LUT 视觉效果对比</h1>
<p class="subtitle">{args.tag} &nbsp;|&nbsp; checkpoint: {args.ckpt} &nbsp;|&nbsp; {len(results)} val samples</p>

<div class="summary">
  <div class="card"><div class="v">{overall_psnr:.2f}</div><div class="l">整体 val PSNR (dB)</div></div>
  <div class="card"><div class="v" style="color:#fbbf24">23.94</div><div class="l">7D ISP 参数天花板</div></div>
  <div class="card"><div class="v" style="color:{'#22c55e' if overall_psnr > 23.94 else '#ef4444'}">
    {overall_psnr - 23.94:+.2f}</div><div class="l">vs 天花板 (dB)</div></div>
  <div class="card"><div class="v">{len(results)}</div><div class="l">验证集样本数</div></div>
</div>

<table class="pa-table">
  <thead><tr><th>动作</th><th>样本数</th><th>PSNR (dB)</th><th>vs 天花板</th></tr></thead>
  <tbody>{pa_rows}</tbody>
</table>

<div class="legend">
  <b>每行图像说明:</b><br>
  <span><span class="dot" style="background:rgb(100,100,100)"></span>① 原图 — FiveK 数据集输入</span>
  <span><span class="dot" style="background:rgb(56,189,248)"></span>② LUT 预测 — 我们的 3D-LUT 模型输出</span>
  <span><span class="dot" style="background:rgb(251,191,36)"></span>③ FireRed 目标 — FireRed-Image-Edit 生成的编辑结果 (ground truth)</span>
  <span><span class="dot" style="background:rgb(239,68,68)"></span>④ 误差热图 — 预测与目标的像素差异 (×5 放大)</span>
</div>

<div class="filters">
  <b>按动作过滤:</b> {action_filters}
  <label style="margin-left:20px;color:#fbbf24"><input type="checkbox" id="sort-toggle"> 按 PSNR 倒序 (好的在前)</label>
</div>

<p class="sort-info">当前排序: PSNR 从低到高 (差的在前, 方便审查问题)</p>
<div class="grid" id="grid">{items_html}</div>

<script>
const items = document.querySelectorAll('.item');
const checkboxes = document.querySelectorAll('.f-action');
function update() {{
  const active = new Set(Array.from(checkboxes).filter(c => c.checked).map(c => c.value));
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


if __name__ == '__main__':
    main()
