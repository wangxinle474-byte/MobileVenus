"""挑选每个模型在 val 集上 L1(model_render, FireRed_edit) 最小的 top picks
(分 action), 渲染为 4-panel viewer.

用法:
  python training/firered_baseline/render_best_picks.py \
      --ckpts checkpoints/firered_v1/best.pt checkpoints/firered_v3/best.pt \
      --names v1 v3 \
      --top_per_action 2 \
      --out_dir outputs/firered_best_picks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.diff_isp import apply_diff_isp  # noqa: E402
from training.firered_baseline.train import (  # noqa: E402
    PARAM_NAMES, ACTIONS, ACTION_TO_IDX, ACTION_PRIMARY_PARAM,
    FireRed7DModel, denormalize_t, build_data,
)


def load_image_tensor(path: Path, size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def make_panel(images: list, labels: list, font_h: int = 22):
    h, w = images[0].size[1], images[0].size[0]
    n = len(images)
    pad = 4
    panel = Image.new('RGB', (w * n + pad * (n - 1), h + font_h + 4),
                      (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype('arial.ttf', font_h - 4)
    except Exception:
        font = ImageFont.load_default()
    for i, (im, lab) in enumerate(zip(images, labels)):
        x = i * (w + pad)
        panel.paste(im, (x, font_h + 4))
        draw.text((x + 4, 2), lab, fill=(20, 20, 20), font=font)
    return panel


def eval_all_val(ckpt_path: str, val_s: list, device, image_size=256,
                 render_size=512):
    """对每张 val 算 L1(model_render, FireRed_edit) + P_pred. 返回 list."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})
    model = FireRed7DModel(image_size=saved_args.get('image_size', image_size),
                           dropout=saved_args.get('dropout', 0.3)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    out = []
    with torch.no_grad():
        for s in val_s:
            try:
                orig_t = load_image_tensor(Path(s['orig_path']),
                                            render_size).to(device)
                fr_t = load_image_tensor(Path(s['target_path']),
                                          render_size).to(device)
            except Exception:
                continue
            orig_pil = Image.open(s['orig_path']).convert('RGB')
            x_in = tf(orig_pil).unsqueeze(0).to(device)
            a_oh = torch.zeros(1, len(ACTIONS), device=device)
            a_oh[0, ACTION_TO_IDX[s['action']]] = 1.0
            pred_norm = model(x_in, a_oh)[0]
            P_pred = {p: float(denormalize_t(p, pred_norm[i].cpu()).item())
                      for i, p in enumerate(PARAM_NAMES)}
            params_t = {p: torch.tensor([P_pred[p]], device=device,
                                         dtype=torch.float32)
                        for p in PARAM_NAMES}
            model_render = apply_diff_isp(orig_t, params_t)
            # NaN guard
            model_render = torch.nan_to_num(model_render, nan=0.5,
                                            posinf=1.0, neginf=0.0)
            L1_mt = F.l1_loss(model_render, fr_t).item()
            L1_ot = F.l1_loss(orig_t, fr_t).item()
            out.append({
                'sample': s,
                'P_pred': P_pred,
                'L1_mt': L1_mt,
                'L1_ot': L1_ot,
                'orig_render': orig_t.cpu(),
                'fr_render': fr_t.cpu(),
                'model_render': model_render.cpu(),
            })
    return out, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+',
                    default=['checkpoints/firered_v1/best.pt',
                             'checkpoints/firered_v3/best.pt',
                             'checkpoints/firered_v4/best.pt'])
    ap.add_argument('--names', nargs='+', default=['v1', 'v3', 'v4'])
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--top_per_action', type=int, default=2)
    ap.add_argument('--render_size', type=int, default=512)
    ap.add_argument('--out_dir', default='outputs/firered_best_picks')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, val_s = build_data(
        Path(args.jsonl), val_ratio=0.2,
        tier_filter=('A excellent', 'B good', 'C acceptable'),
        seed=args.seed)
    print(f'val n={len(val_s)}')

    # 评估每个模型在 val 上的全集 L1
    per_model = {}
    for ckpt, name in zip(args.ckpts, args.names):
        if not Path(ckpt).exists():
            print(f'skip {name}: ckpt not found')
            continue
        print(f'\nEvaluating {name}...')
        per_model[name], _ = eval_all_val(
            ckpt, val_s, device, render_size=args.render_size)

    # 对每个模型 + 每个 action, 取 L1_mt 最小的 top-K
    rows_html = []
    for name, results in per_model.items():
        # group by action
        by_action = {}
        for r in results:
            a = r['sample']['action']
            by_action.setdefault(a, []).append(r)
        for a in ACTIONS:
            if a not in by_action:
                continue
            picks = sorted(by_action[a], key=lambda x: x['L1_mt'])[
                :args.top_per_action]
            for rank, p in enumerate(picks):
                s = p['sample']
                stem = Path(s['orig_path']).stem
                # 4-panel: orig | FR | model | (delta map?)
                orig_pil = to_pil(p['orig_render'])
                fr_pil = to_pil(p['fr_render'])
                model_pil = to_pil(p['model_render'])

                # delta heatmap (model vs FR)
                delta = (p['model_render'] - p['fr_render']).abs().squeeze(0)
                delta_lum = delta.mean(dim=0).clamp(0, 0.3) / 0.3  # 归一化
                delta_rgb = torch.stack([delta_lum, delta_lum * 0.3,
                                          delta_lum * 0.1], dim=0)
                delta_pil = to_pil(delta_rgb.unsqueeze(0))

                tier = s['quality_tier']
                panel = make_panel(
                    [orig_pil, fr_pil, model_pil, delta_pil],
                    [f'orig  L1(o,FR)={p["L1_ot"]:.3f}',
                     f'FireRed edit  ({tier})',
                     f'{name}\u2192ISP\u2192render  L1(m,FR)={p["L1_mt"]:.3f}',
                     f'|model - FR| heatmap'])
                out_png = out_dir / f'{name}_{a}_top{rank+1}_{stem}.png'
                panel.save(out_png)
                print(f'  [{name}] {a} top{rank+1}: {stem}  '
                      f'L1(m,FR)={p["L1_mt"]:.4f}  L1(o,FR)={p["L1_ot"]:.4f}')

                P_pred = p['P_pred']
                P_inv = s['P_inferred']
                rows_html.append({
                    'model': name,
                    'action': a,
                    'rank': rank + 1,
                    'image_id': stem,
                    'tier': tier,
                    'L1_mt': p['L1_mt'],
                    'L1_ot': p['L1_ot'],
                    'png': out_png.name,
                    'P_pred': P_pred,
                    'P_inv': {pn: float(P_inv.get(pn, 0.0))
                              for pn in PARAM_NAMES},
                })

    # build index.html
    html = [
        '<html><head><meta charset=utf-8><title>FireRed Best Picks</title>',
        '<style>body{font-family:sans-serif;background:#222;color:#eee;padding:16px;max-width:1500px;margin:auto}',
        'h2{margin-top:24px;border-bottom:1px solid #555;padding-bottom:4px}',
        'h3{margin-top:18px;color:#9cf}',
        'img{width:100%;border:1px solid #555;margin:6px 0}',
        'table{font-size:12px;background:#333;border-collapse:collapse;margin:4px 0}',
        'th,td{padding:3px 8px;border:1px solid #555;text-align:center}',
        '.r{color:#9cf}.t{color:#fc9}.b{color:#9f9;font-weight:bold}',
        '</style></head><body>',
        '<h1>FireRed Baseline Best Picks (val 中每个 action L1(m,FR) 最小)</h1>',
        '<p>4-panel: <b>orig</b> | <b>FireRed edit</b> | '
        '<b>model→ISP→render</b> | <b>|model−FR| 差异热图</b> (越暗越好)</p>',
        '<p>评判: <b>L1(m,FR)</b> 越接近 L1(o,FR) 的零点越好; 但要看 panel 3 跟 panel 2 视觉相似度</p>',
    ]
    # group by model then action
    cur_model = None
    cur_action = None
    for r in rows_html:
        if r['model'] != cur_model:
            html.append(f'<h2 class=b>{r["model"]}</h2>')
            cur_model = r['model']
            cur_action = None
        if r['action'] != cur_action:
            html.append(f'<h3>action: {r["action"]}</h3>')
            cur_action = r['action']
        html.append(
            f'<p><b>top {r["rank"]}: {r["image_id"]}</b> '
            f'<small>(tier {r["tier"]}, '
            f'L1(o,FR)={r["L1_ot"]:.3f}, '
            f'<span class=b>L1(m,FR)={r["L1_mt"]:.3f}</span>)</small></p>')
        html.append(f'<img src="{r["png"]}">')
        html.append('<table><tr><th>param</th>'
                    '<th class=r>P_pred (model)</th>'
                    '<th class=t>P_inferred (fitted GT)</th>'
                    '<th>dP</th></tr>')
        for pn in PARAM_NAMES:
            v_pred = r['P_pred'][pn]
            v_inv = r['P_inv'][pn]
            d = v_pred - v_inv
            html.append(
                f'<tr><td>{pn}</td>'
                f'<td class=r>{v_pred:+.2f}</td>'
                f'<td class=t>{v_inv:+.2f}</td>'
                f'<td>{d:+.2f}</td></tr>')
        html.append('</table>')
    html.append('</body></html>')
    (out_dir / 'index.html').write_text('\n'.join(html), encoding='utf-8')
    print(f'\nwrote {out_dir/"index.html"}')


if __name__ == '__main__':
    main()
