"""抽 val 集每个 action 1 张, 用 best.pt 模型预测 P_pred, 跑 diff_isp 渲染,
拼出 4-panel 对比: orig | FireRed_edit | model_render | fitted_render.

输出: outputs/firered_v1_val_compare/ (每张一个 PNG + 一个 index.html)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.diff_isp import apply_diff_isp  # noqa: E402
from training.firered_baseline.train import (  # noqa: E402
    PARAM_NAMES, ACTIONS, ACTION_TO_IDX, FireRed7DModel,
    denormalize_t, build_data,
)


def load_image(path: Path, size: int = 512) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def make_panel(images: list, labels: list, font_h: int = 24) -> Image.Image:
    from PIL import ImageDraw, ImageFont
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/firered_v1/best.pt')
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='outputs/firered_v1_val_compare')
    ap.add_argument('--image_size', type=int, default=256,
                    help='模型输入尺寸 (训练时一致)')
    ap.add_argument('--render_size', type=int, default=512,
                    help='diff_isp 渲染尺寸')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 best.pt
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})
    model = FireRed7DModel(
        image_size=saved_args.get('image_size', args.image_size),
        dropout=saved_args.get('dropout', 0.3)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'loaded {args.ckpt} (Ep {ckpt["epoch"]}, val={ckpt["val_loss"]:.4f})')

    # 2. 重建 val split (与训练同 seed)
    train_s, val_s = build_data(
        Path(args.jsonl),
        val_ratio=saved_args.get('val_ratio', 0.2),
        tier_filter=tuple(['A excellent', 'B good', 'C acceptable']),
        seed=saved_args.get('seed', args.seed))

    # 3. 每个 action 找一张 tier 最好的 val 图
    picks = {}
    tier_order = ['A excellent', 'B good', 'C acceptable']
    for a in ACTIONS:
        cand = [s for s in val_s if s['action'] == a]
        if not cand:
            continue
        cand.sort(key=lambda s: tier_order.index(s['quality_tier']))
        picks[a] = cand[0]

    # 4. 模型推断 + 渲染
    model_input_tf = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows_html = []
    for a in ACTIONS:
        if a not in picks:
            continue
        s = picks[a]
        orig_path = Path(s['orig_path'])
        target_path = Path(s['target_path'])

        # model 输入 (256 normalized)
        orig_pil = Image.open(orig_path).convert('RGB')
        x_model = model_input_tf(orig_pil).unsqueeze(0).to(device)
        action_oh = torch.zeros(1, len(ACTIONS), device=device)
        action_oh[0, ACTION_TO_IDX[a]] = 1.0
        with torch.no_grad():
            pred_norm = model(x_model, action_oh)[0]  # (7,) ∈ [-1,1]

        # 反归一化到物理参数
        P_pred = {p: float(denormalize_t(p, pred_norm[i].cpu()).item())
                  for i, p in enumerate(PARAM_NAMES)}

        # diff_isp 渲染 (用 render_size, 不归一化, 直接 [0,1])
        orig_render = load_image(orig_path, args.render_size).to(device)
        # FireRed edit (target)
        try:
            fr_edit = load_image(target_path, args.render_size).to(device)
            fr_edit_pil = to_pil(fr_edit)
        except Exception as e:
            print(f'  WARN: load target {target_path} failed: {e}')
            fr_edit_pil = Image.new('RGB',
                                    (args.render_size, args.render_size),
                                    (200, 200, 200))

        # model 预测渲染
        params_t = {p: torch.tensor([P_pred[p]], device=device,
                                    dtype=torch.float32)
                    for p in PARAM_NAMES}
        with torch.no_grad():
            model_render = apply_diff_isp(orig_render, params_t)
        model_render_pil = to_pil(model_render)

        # inverse_fit 反推参数渲染 (作为模型应该达到的上限)
        P_inv = s['P_inferred']
        params_inv_t = {p: torch.tensor([P_inv.get(p, 0.0)], device=device,
                                        dtype=torch.float32)
                        for p in PARAM_NAMES}
        with torch.no_grad():
            inv_render = apply_diff_isp(orig_render, params_inv_t)
        inv_render_pil = to_pil(inv_render)

        orig_pil_resized = to_pil(orig_render)

        import torch.nn.functional as F
        L1_om = F.l1_loss(orig_render, model_render).item()
        L1_oi = F.l1_loss(orig_render, inv_render).item()
        L1_ot = F.l1_loss(orig_render, fr_edit.to(device)).item() \
            if isinstance(fr_edit, torch.Tensor) else float('nan')
        L1_mt = F.l1_loss(model_render, fr_edit.to(device)).item() \
            if isinstance(fr_edit, torch.Tensor) else float('nan')
        L1_it = F.l1_loss(inv_render, fr_edit.to(device)).item() \
            if isinstance(fr_edit, torch.Tensor) else float('nan')

        # 4-panel
        panel = make_panel(
            [orig_pil_resized, fr_edit_pil, model_render_pil, inv_render_pil],
            ['orig',
             f'FireRed edit  L1(o,FR)={L1_ot:.3f}',
             f'model render  L1(o,m)={L1_om:.3f}  L1(m,FR)={L1_mt:.3f}',
             f'inv_fit render  L1(o,i)={L1_oi:.3f}  L1(i,FR)={L1_it:.3f}'])

        out_png = out_dir / f'{a}_{Path(orig_path).stem}.png'
        panel.save(out_png)
        print(f'  {a}: {Path(orig_path).stem}  → {out_png.name}')
        print(f'    P_pred:     {", ".join(f"{p[:4]}={P_pred[p]:.1f}" for p in PARAM_NAMES)}')
        print(f'    P_inferred: {", ".join(f"{p[:4]}={P_inv.get(p,0.0):.1f}" for p in PARAM_NAMES)}')

        rows_html.append({
            'action': a,
            'png': out_png.name,
            'image_id': Path(orig_path).stem,
            'tier': s['quality_tier'],
            'P_pred': P_pred,
            'P_inv': {p: float(P_inv.get(p, 0.0)) for p in PARAM_NAMES},
        })

    # index.html
    html = ['<html><head><meta charset=utf-8>',
            '<title>FireRed v1 val compare</title>',
            '<style>body{font-family:sans-serif;background:#222;color:#eee;padding:16px}',
            'h2{margin-top:24px}img{width:100%;border:1px solid #555}',
            'table{font-size:12px;background:#333;border-collapse:collapse;margin:8px 0}',
            'th,td{padding:4px 8px;border:1px solid #555}.r{color:#9cf}.t{color:#fc9}'
            '</style></head><body>',
            f'<h1>FireRed v1 (Ep {ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.4f})</h1>',
            '<p>4-panel: <b>orig</b> | <b>FireRed edit</b> | '
            '<b>model→ISP→render</b> | <b>inverse_fit→ISP→render</b></p>']
    for r in rows_html:
        html.append(f'<h2>{r["action"]} - {r["image_id"]} ({r["tier"]})</h2>')
        html.append(f'<img src="{r["png"]}">')
        html.append('<table><tr><th>param</th><th class=r>P_pred (model)</th>'
                    '<th class=t>P_inferred (fitted, GT)</th></tr>')
        for p in PARAM_NAMES:
            html.append(f'<tr><td>{p}</td>'
                        f'<td class=r>{r["P_pred"][p]:+.2f}</td>'
                        f'<td class=t>{r["P_inv"][p]:+.2f}</td></tr>')
        html.append('</table>')
    html.append('</body></html>')
    (out_dir / 'index.html').write_text('\n'.join(html), encoding='utf-8')
    print(f'\nwrote {out_dir/"index.html"}')


if __name__ == '__main__':
    main()
