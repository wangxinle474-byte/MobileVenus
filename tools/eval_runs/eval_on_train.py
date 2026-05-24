"""Evaluate model on TRAINING images: orig | pred | GT comparison.

Picks N samples per action from the training split and renders:
  - Original image
  - Model output (with matching action)
  - FireRed GT target
  - PSNR overlay

Usage:
  python tools/eval_on_train.py --n_per_action 4
  python tools/eval_on_train.py --split val   # use val instead
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.firered_baseline.train_lut import (  # noqa: E402
    ACTIONS,
    ACTION_TO_IDX,
    ACTION_TO_PARAM_7D_INDICES,
    PARAM_NAMES_7D,
    PARAM_NORM_7D,
    LUTDataset,
    NamedCurvesPredictor,
    build_data,
    set_actions,
)

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

ACTION_CN = {
    'contrast': '对比度增强', 'saturation': '饱和度增强',
    'shadows': '暗部提亮', 'highlights': '高光压暗',
    'wb': '白平衡调整',
    'brightness': '亮度调整', 'clarity': '清晰度调整',
}


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ca = ckpt['args']
    ckpt_actions = ca.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)
    model = NamedCurvesPredictor(
        n_colors=ca.get('nc_n_colors', 3),
        n_control_points=ca.get('nc_n_control_points', 7),
        use_attention=ca.get('nc_use_attention', False),
        per_action_curves=ca.get('nc_per_action_curves', False),
        use_7d_anchor=ca.get('nc_use_7d_anchor', True),
        use_context=ca.get('nc_use_context', False),
        action_gated_context=ca.get('nc_action_gated_context', False),
        use_action_context=ca.get('nc_use_action_context', False),
        use_region_basis=ca.get('nc_use_region_basis', False),
        use_region_param_delta=ca.get('nc_use_region_param_delta', False),
        use_learned_cn=ca.get('nc_use_learned_cn', False),
        use_wb_head=ca.get('nc_use_wb_head', False),
        use_nilut_residual=ca.get('nc_use_nilut_residual', False),
        use_vera_renderer=ca.get('nc_use_vera_renderer', False),
        nilut_hidden=ca.get('nilut_hidden', 32),
        nilut_n_layers=ca.get('nilut_n_layers', 3),
        nilut_n_freq=ca.get('nilut_n_freq', 4),
        nilut_gate_init=ca.get('nilut_gate_init', 1.0),
        use_implicit_head=ca.get('use_implicit_head', False),
        implicit_head_base_ch=ca.get('implicit_head_base_ch', 32),
        implicit_head_gate_init=ca.get('implicit_head_gate_init', 0.0),
        image_size=ca.get('image_size', 256),
        n_actions=len(ACTIONS),
        dropout=ca.get('dropout', 0.5),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    info = {
        'val_psnr': ckpt.get('val_psnr', 0),
        'epoch': ckpt.get('epoch', '?'),
        'use_implicit_head': ca.get('use_implicit_head', False),
        'image_size': ca.get('image_size', 256),
        'split_seed': ca.get('split_seed', ca.get('seed', 42)),
        'val_ratio': ca.get('val_ratio', 0.2),
        'actions': list(ACTIONS),
    }
    return model, info


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean()
    return float(-10 * torch.log10(mse.clamp(min=1e-10)))


def add_label(img: Image.Image, text: str, color=(56, 189, 248)) -> Image.Image:
    W, H = img.size
    bar = 28
    new = Image.new('RGB', (W, H + bar), (15, 23, 42))
    draw = ImageDraw.Draw(new)
    try:
        font = ImageFont.truetype("msyh.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, 5), text, fill=color, font=font)
    new.paste(img, (0, bar))
    return new


def make_error_heatmap(pred: torch.Tensor, target: torch.Tensor) -> Image.Image:
    err = (pred - target).abs().mean(dim=0).cpu().numpy()
    err5 = np.clip(err * 5.0, 0, 1)
    r = np.clip(err5 * 2 - 0.5, 0, 1)
    g = np.clip(1 - np.abs(err5 - 0.5) * 2, 0, 1)
    b = np.clip(1 - err5, 0, 1) * 0.5
    rgb = np.stack([r, g, b], axis=-1) * 255
    return Image.fromarray(rgb.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser(description='Evaluate model on train/val images')
    ap.add_argument('--ckpt',
                    default='checkpoints/lut_v11a_pathX_implicit_head/best.pt')
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--split', choices=['train', 'val', 'both'], default='train')
    ap.add_argument('--n_per_action', type=int, default=4)
    ap.add_argument('--out_dir', default='outputs/eval_on_train')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading: {args.ckpt}")
    model, info = load_model(args.ckpt, device)
    print(f"  val_psnr={info['val_psnr']:.2f}dB @ Ep{info['epoch']}, "
          f"implicit_head={info['use_implicit_head']}")

    # Build data with same split as training
    tier_filter = ('A excellent', 'B good', 'C acceptable')
    split_seed = info['split_seed']
    train_s, val_s = build_data(
        Path(args.jsonl), val_ratio=info['val_ratio'], tier_filter=tier_filter,
        seed=split_seed, action_filter=ACTIONS)
    print(f"  train={len(train_s)}, val={len(val_s)} "
          f"(split_seed={info['split_seed']})")

    if args.split == 'train':
        samples = train_s
    elif args.split == 'val':
        samples = val_s
    else:
        samples = train_s + val_s

    # Pick N samples per action
    rng = np.random.default_rng(args.seed)
    by_action = {a: [] for a in ACTIONS}
    for s in samples:
        if s['action'] in by_action:
            by_action[s['action']].append(s)

    selected = []
    for action in ACTIONS:
        pool = by_action[action]
        if not pool:
            print(f"  WARNING: no samples for {action}")
            continue
        idx = rng.choice(len(pool), min(args.n_per_action, len(pool)), replace=False)
        for i in idx:
            selected.append(pool[i])

    print(f"\nSelected {len(selected)} samples ({args.n_per_action} per action)")

    # Run inference
    image_size = info['image_size']
    ds = LUTDataset(selected, image_size, is_train=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / 'imgs'
    img_dir.mkdir(exist_ok=True)

    results = []
    per_action_psnr = {a: [] for a in ACTIONS}

    print("\nRunning inference...")
    with torch.no_grad():
        for idx in range(len(ds)):
            batch = ds[idx]
            enc_input = batch['enc_input'].unsqueeze(0).to(device)
            orig = batch['orig'].unsqueeze(0).to(device)
            target = batch['target'].unsqueeze(0).to(device)
            action_oh = batch['action_onehot'].unsqueeze(0).to(device)
            action = batch['action']

            refined, _, _, pred_p7 = model(enc_input, orig, action_oh)

            psnr_pred = compute_psnr(refined[0], target[0])
            psnr_orig = compute_psnr(orig[0], target[0])  # 不做编辑的 PSNR
            per_action_psnr[action].append(psnr_pred)

            # Build 4-panel strip: orig | pred | GT | error
            orig_pil = add_label(tensor_to_pil(orig[0]),
                                 f'① 原图', (180, 180, 180))
            pred_pil = add_label(tensor_to_pil(refined[0]),
                                 f'② 模型输出  PSNR={psnr_pred:.1f}dB',
                                 (56, 189, 248))
            gt_pil = add_label(tensor_to_pil(target[0]),
                               '③ FireRed GT',
                               (251, 191, 36))
            err_pil = add_label(make_error_heatmap(refined[0], target[0]),
                                f'④ 像素误差 ×5',
                                (239, 68, 68))

            W, H = orig_pil.size
            strip = Image.new('RGB', (W * 4 + 6, H), (15, 23, 42))
            strip.paste(orig_pil, (0, 0))
            strip.paste(pred_pil, (W + 2, 0))
            strip.paste(gt_pil, (W * 2 + 4, 0))
            strip.paste(err_pil, (W * 3 + 6, 0))

            fname = f'{idx:03d}_{action}_{batch["source_image"].replace(".jpg", "")}.jpg'
            strip.save(img_dir / fname, quality=88)

            # Decode predicted parameters for the active action
            ai = ACTIONS.index(action)
            active_param_idx = ACTION_TO_PARAM_7D_INDICES[ai]
            active_param_name = PARAM_NAMES_7D[active_param_idx]
            active_param_norm = float(pred_p7[0, active_param_idx].cpu())
            cfg = PARAM_NORM_7D[active_param_name]
            active_param_phys = active_param_norm * cfg['scale'] + cfg['center']

            # GT 7D
            gt_phys = batch['params_norm_7d'][active_param_idx].item() * cfg['scale'] + cfg['center']

            results.append({
                'idx': idx,
                'action': action,
                'action_cn': ACTION_CN.get(action, action),
                'source': batch['source_image'],
                'psnr_pred': round(psnr_pred, 2),
                'psnr_orig': round(psnr_orig, 2),
                'gain': round(psnr_pred - psnr_orig, 2),
                'active_param': active_param_name,
                'pred_phys': active_param_phys,
                'gt_phys': gt_phys,
                'img': f'imgs/{fname}',
            })
            print(f"  [{idx+1}/{len(ds)}] {action:<11} {batch['source_image']:<32} "
                  f"PSNR={psnr_pred:.2f}dB (vs orig {psnr_orig:.2f}, +{psnr_pred - psnr_orig:.2f})")

    # Aggregate stats
    overall_psnr = float(np.mean([p for ps in per_action_psnr.values() for p in ps]))

    pa_summary = {}
    for a in ACTIONS:
        if per_action_psnr[a]:
            pa_summary[a] = {
                'n': len(per_action_psnr[a]),
                'mean': round(float(np.mean(per_action_psnr[a])), 2),
                'min': round(float(np.min(per_action_psnr[a])), 2),
                'max': round(float(np.max(per_action_psnr[a])), 2),
            }

    # Build HTML
    pa_rows = ''
    for a in ACTIONS:
        if a not in pa_summary:
            continue
        s = pa_summary[a]
        color = ('#22c55e' if s['mean'] > 25 else
                 '#fbbf24' if s['mean'] > 22 else '#ef4444')
        pa_rows += (f'<tr><td>{ACTION_CN.get(a, a)}</td><td>{s["n"]}</td>'
                    f'<td style="color:{color};font-weight:bold">{s["mean"]:.2f}</td>'
                    f'<td>{s["min"]:.2f}</td><td>{s["max"]:.2f}</td></tr>')

    # Sort results by PSNR ascending (worst first)
    results_sorted = sorted(results, key=lambda r: r['psnr_pred'])

    items_html = ''
    for r in results_sorted:
        psnr_color = ('#22c55e' if r['psnr_pred'] > 25 else
                      '#fbbf24' if r['psnr_pred'] > 22 else '#ef4444')
        unit = 'K' if r['active_param'] == 'white_balance' else ''
        if r['active_param'] == 'white_balance':
            pred_str = f'{r["pred_phys"]:.0f}{unit}'
            gt_str = f'{r["gt_phys"]:.0f}{unit}'
        else:
            pred_str = f'{r["pred_phys"]:+.1f}'
            gt_str = f'{r["gt_phys"]:+.1f}'

        items_html += f'''
        <div class="item" data-action="{r['action']}">
          <img src="{r['img']}" loading="lazy">
          <div class="meta">
            <span class="badge">{r['action_cn']}</span>
            <span style="color:{psnr_color};font-weight:bold;">PSNR {r['psnr_pred']:.2f} dB</span>
            <span class="dim">  (原图基线 {r['psnr_orig']:.2f}, 增益 +{r['gain']:.2f})</span>
            <br>
            <span class="dim">{r['source']}  |  {r['active_param']}: GT={gt_str} → Pred={pred_str}</span>
          </div>
        </div>'''

    action_filters = ''.join(
        f'<label><input type="checkbox" class="f-action" value="{a}" checked> {ACTION_CN.get(a, a)}</label>'
        for a in ACTIONS
    )

    split_label = {'train': '训练集', 'val': '验证集', 'both': '全部'}[args.split]
    overall_color = ('#22c55e' if overall_psnr > 25 else
                     '#fbbf24' if overall_psnr > 22 else '#ef4444')

    html = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Eval on {split_label} — {Path(args.ckpt).stem}</title>
<style>
  body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 0;
         padding: 20px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 22px; margin: 0 0 8px; color: #38bdf8; }}
  .info {{ font-size: 13px; color: #94a3b8; margin-bottom: 16px; }}
  .summary {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
  .card {{ background: #1e293b; padding: 12px 18px; border-radius: 6px;
           border: 1px solid #334155; min-width: 130px; }}
  .card .v {{ font-size: 26px; font-weight: bold; color: {overall_color}; }}
  .card .l {{ font-size: 12px; color: #94a3b8; }}
  table {{ background: #1e293b; border-collapse: collapse;
           font-size: 13px; margin-bottom: 16px; }}
  th, td {{ padding: 6px 12px; border: 1px solid #334155; text-align: center; }}
  th {{ background: #334155; color: #fbbf24; }}
  .filters {{ background: #1e293b; padding: 10px 14px; border-radius: 6px;
              border: 1px solid #334155; margin-bottom: 16px; font-size: 13px; }}
  .filters label {{ margin-right: 14px; cursor: pointer; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 12px;
           max-width: 1500px; }}
  .item {{ background: #1e293b; border-radius: 6px; overflow: hidden;
           border: 1px solid #334155; }}
  .item img {{ width: 100%; display: block; }}
  .meta {{ padding: 8px 12px; font-size: 12px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 3px;
            color: #fff; font-weight: bold; margin-right: 8px;
            background: #7c3aed; }}
  .dim {{ color: #94a3b8; }}
  .hide {{ display: none; }}
</style>
</head>
<body>
<h1>📷 模型在{split_label}图片上的预测效果</h1>
<p class="info">
  Checkpoint: {args.ckpt} | val_psnr={info["val_psnr"]:.2f}dB @ Ep{info["epoch"]}
  | implicit_head={info["use_implicit_head"]} | split_seed={info["split_seed"]}<br>
  数据: {args.jsonl} | {split_label} (n={len(samples)}, 抽样 {len(results)})
</p>

<div class="summary">
  <div class="card"><div class="v">{overall_psnr:.2f}</div>
    <div class="l">overall PSNR (dB)</div></div>
  <div class="card"><div class="v" style="color:#94a3b8">{len(results)}</div>
    <div class="l">样本数</div></div>
</div>

<h3>Per-action PSNR (预测 vs FireRed GT)</h3>
<table>
  <thead><tr><th>动作</th><th>n</th><th>PSNR mean</th><th>min</th><th>max</th></tr></thead>
  <tbody>{pa_rows}</tbody>
</table>

<div class="filters">
  <b>过滤动作:</b> {action_filters}
</div>

<p class="info">默认排序: PSNR 从低到高 (差的在前)。<br>
4 列: ① 原图 | ② 模型输出 (用对应 action 推理) | ③ FireRed GT 目标 | ④ 像素误差 ×5</p>

<div class="grid" id="grid">{items_html}</div>

<script>
const items = document.querySelectorAll('.item');
const cbs = document.querySelectorAll('.f-action');
function update() {{
  const active = new Set(Array.from(cbs).filter(c => c.checked).map(c => c.value));
  items.forEach(it => it.classList.toggle('hide', !active.has(it.dataset.action)));
}}
cbs.forEach(c => c.addEventListener('change', update));
</script>
</body>
</html>'''

    viewer = out_dir / 'viewer.html'
    viewer.write_text(html, encoding='utf-8')

    # Console summary
    print(f"\n{'='*60}")
    print(f"  Per-action PSNR ({split_label})")
    print(f"{'='*60}")
    for a in ACTIONS:
        if a not in pa_summary:
            continue
        s = pa_summary[a]
        print(f"  {a:<12} n={s['n']:<3} mean={s['mean']:.2f}dB "
              f"(min={s['min']:.2f}, max={s['max']:.2f})")
    print(f"{'-'*60}")
    print(f"  Overall PSNR: {overall_psnr:.2f} dB")
    print(f"{'='*60}")
    print(f"\nViewer: {viewer}")

    import os
    os.startfile(str(viewer))


if __name__ == '__main__':
    main()
