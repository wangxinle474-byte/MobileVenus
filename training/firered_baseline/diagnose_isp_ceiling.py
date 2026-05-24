"""诊断 7D ISP 表达上限: panel 4 (inv_fit render) 离 panel 2 (FR edit) 多远?

如果 inv_render 都跟 FR 差很远 -> 7D ISP 不够丰富, 需要扩参数 / 换架构
如果 inv_render 几乎等于 FR -> model 训练问题, 加数据 / 优化 loss

输出:
- 全 73 val 样本 PSNR(inv, FR), PSNR(orig, FR), PSNR(model, FR) 三方分布
- best/worst 3 张高清 side-by-side
- 量化 "double gap": model -> inv -> FR
"""
from __future__ import annotations

import argparse
import math
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
    PARAM_NAMES, ACTIONS, ACTION_TO_IDX,
    FireRed7DModel, denormalize_t, build_data,
)


def load_image_tensor(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    return 99.0 if mse < 1e-10 else 10 * math.log10(1.0 / mse)


def make_panel(images, labels, font_h=24):
    h, w = images[0].size[1], images[0].size[0]
    n = len(images)
    pad = 6
    panel = Image.new('RGB', (w * n + pad * (n - 1), h + font_h + 8),
                      (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype('arial.ttf', font_h - 4)
    except Exception:
        font = ImageFont.load_default()
    for i, (im, lab) in enumerate(zip(images, labels)):
        x = i * (w + pad)
        panel.paste(im, (x, font_h + 6))
        draw.text((x + 6, 4), lab, fill=(20, 20, 20), font=font)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/firered_v1/best.pt',
                    help='用于算 model->FR gap 的 ckpt (展示用)')
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--render_size', type=int, default=512)
    ap.add_argument('--out_dir', default='outputs/firered_isp_ceiling')
    ap.add_argument('--top_k', type=int, default=3,
                    help='best/worst 各几张')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型 (只用于 panel 5 对比, 主诊断不依赖)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})
    model = FireRed7DModel(image_size=saved_args.get('image_size', 256),
                           dropout=saved_args.get('dropout', 0.3)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _, val_s = build_data(
        Path(args.jsonl), 0.2,
        ('A excellent', 'B good', 'C acceptable'), 42)

    tf = T.Compose([
        T.Resize((saved_args.get('image_size', 256),) * 2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    records = []
    print(f'scanning {len(val_s)} val samples...')
    with torch.no_grad():
        for s in val_s:
            try:
                orig_t = load_image_tensor(Path(s['orig_path']),
                                            args.render_size).to(device)
                fr_t = load_image_tensor(Path(s['target_path']),
                                          args.render_size).to(device)
            except Exception:
                continue

            # inv_render (P_inferred -> diff_isp)
            P_inv = s['P_inferred']
            params_inv = {p: torch.tensor([P_inv.get(p, 0.0)], device=device,
                                          dtype=torch.float32)
                          for p in PARAM_NAMES}
            inv_render = apply_diff_isp(orig_t, params_inv)
            inv_render = torch.nan_to_num(inv_render, nan=0.5, posinf=1.0,
                                          neginf=0.0)

            # model_render (P_pred -> diff_isp)
            orig_pil = Image.open(s['orig_path']).convert('RGB')
            x_in = tf(orig_pil).unsqueeze(0).to(device)
            a_oh = torch.zeros(1, len(ACTIONS), device=device)
            a_oh[0, ACTION_TO_IDX[s['action']]] = 1.0
            pred_norm = model(x_in, a_oh)[0]
            P_pred = {p: float(denormalize_t(p, pred_norm[i].cpu()).item())
                      for i, p in enumerate(PARAM_NAMES)}
            params_pred = {p: torch.tensor([P_pred[p]], device=device,
                                            dtype=torch.float32)
                           for p in PARAM_NAMES}
            model_render = apply_diff_isp(orig_t, params_pred)
            model_render = torch.nan_to_num(model_render, nan=0.5, posinf=1.0,
                                            neginf=0.0)

            psnr_o_fr = psnr(orig_t, fr_t)         # 原图离 FR (基线)
            psnr_inv_fr = psnr(inv_render, fr_t)   # inv 离 FR (7D 上限)
            psnr_mod_fr = psnr(model_render, fr_t)  # model 离 FR (模型)
            psnr_mod_inv = psnr(model_render, inv_render)  # model 离 inv

            records.append({
                'sample': s,
                'P_pred': P_pred,
                'P_inv': {pn: float(P_inv.get(pn, 0.0)) for pn in PARAM_NAMES},
                'psnr_o_fr': psnr_o_fr,
                'psnr_inv_fr': psnr_inv_fr,
                'psnr_mod_fr': psnr_mod_fr,
                'psnr_mod_inv': psnr_mod_inv,
                'orig': orig_t.cpu(),
                'fr': fr_t.cpu(),
                'inv': inv_render.cpu(),
                'model': model_render.cpu(),
            })

    # 汇总分布
    arr_o_fr = np.array([r['psnr_o_fr'] for r in records])
    arr_inv_fr = np.array([r['psnr_inv_fr'] for r in records])
    arr_mod_fr = np.array([r['psnr_mod_fr'] for r in records])
    arr_mod_inv = np.array([r['psnr_mod_inv'] for r in records])

    print(f'\n{"="*90}')
    print(f'73 val 样本 PSNR 分布 (dB, 越高越好):')
    print(f'{"="*90}')
    print(f'{"metric":<35s}{"mean":>10s}{"std":>10s}{"min":>10s}{"max":>10s}'
          f'{"median":>10s}')

    def line(label, a):
        return (f'{label:<35s}{a.mean():>10.2f}{a.std():>10.2f}'
                f'{a.min():>10.2f}{a.max():>10.2f}{np.median(a):>10.2f}')

    print(line('PSNR(orig, FR)         <baseline>', arr_o_fr))
    print(line('PSNR(inv_render, FR)   <7D upper>', arr_inv_fr))
    print(line('PSNR(model, FR)        <model>   ', arr_mod_fr))
    print(line('PSNR(model, inv_render)<m vs inv>', arr_mod_inv))

    print(f'\n{"="*90}')
    print('Double gap analysis:')
    print(f'{"="*90}')
    inv_gain = arr_inv_fr - arr_o_fr      # 7D ISP 能拉近多少
    mod_gain = arr_mod_fr - arr_o_fr      # model 实际拉近多少
    mod_vs_inv = arr_mod_fr - arr_inv_fr  # model 离 inv 上限多远
    print(f'7D ISP 理论拉近 (inv - orig) PSNR:  mean={inv_gain.mean():+.2f}dB  '
          f'positive={int((inv_gain>0).sum())}/{len(inv_gain)}')
    print(f'model 实际拉近 (mod - orig) PSNR:   mean={mod_gain.mean():+.2f}dB  '
          f'positive={int((mod_gain>0).sum())}/{len(mod_gain)}')
    print(f'model 落后于 7D 上限 (mod - inv):   mean={mod_vs_inv.mean():+.2f}dB  '
          f'(负值表示 model 离 inv 还有差距)')

    # 关键判断
    print(f'\n{"="*90}')
    print('解读 ("没学到感觉" 根因):')
    print(f'{"="*90}')
    if inv_gain.mean() < 3.0:
        print(f'>>> 7D ISP **表达力不足** <<< inv_render 平均仅比 orig 接近 FR {inv_gain.mean():.1f}dB')
        print('    即使参数预测完美, ISP 也只能复现一小部分 FireRed 风格.')
        print('    根因: 7D Lightroom param 是 global operator, 难复现 FR 的 spatial/non-linear 风格')
    elif inv_gain.mean() < 6.0:
        print(f'>>> 7D ISP **中等表达** <<< inv_render 平均比 orig 接近 FR {inv_gain.mean():.1f}dB')
        print('    7D 能拿到 FR 风格的 50-70%, 剩余靠模型/数据优化也只能逼近 inv 上限')
    else:
        print(f'>>> 7D ISP **足够** <<< inv_render 平均比 orig 接近 FR {inv_gain.mean():.1f}dB')
        print('    瓶颈在 model 训练, 不是 ISP 表达')

    print()
    if mod_vs_inv.mean() < -2.0:
        print(f'>>> model 离 7D 上限还有 {-mod_vs_inv.mean():.1f}dB 差距 <<<')
        print('    优先 G1 扩数据 / G2 双头 / 主线切换 可能有效')
    else:
        print('>>> model 基本贴近 7D 上限, 训练已饱和 <<<')
        print('    继续优化 model 边际效益小, 需扩 ISP 表达力或换路线')

    # 选 best/worst PSNR(inv, FR) 各 top-K, 渲染 5-panel 高清
    sorted_by_inv = sorted(records, key=lambda r: r['psnr_inv_fr'],
                           reverse=True)
    best = sorted_by_inv[:args.top_k]
    worst = sorted_by_inv[-args.top_k:][::-1]

    print(f'\n生成 {len(best)+len(worst)} 张 5-panel 高清对比...')

    rows_html = []
    for group, samples in [('BEST (7D ISP 可表达)', best),
                            ('WORST (7D ISP 难表达)', worst)]:
        for rank, r in enumerate(samples):
            s = r['sample']
            stem = Path(s['orig_path']).stem
            a = s['action']

            # 计算 delta heatmap (inv vs FR, 显示 ISP 上限的缺口)
            delta_inv_fr = (r['inv'] - r['fr']).abs().squeeze(0)
            delta_lum = delta_inv_fr.mean(dim=0).clamp(0, 0.3) / 0.3
            delta_rgb = torch.stack([delta_lum, delta_lum * 0.3,
                                      delta_lum * 0.05], dim=0)

            panel = make_panel(
                [to_pil(r['orig']), to_pil(r['fr']),
                 to_pil(r['inv']), to_pil(r['model']),
                 to_pil(delta_rgb.unsqueeze(0))],
                [f'orig  PSNR(o,FR)={r["psnr_o_fr"]:.1f}dB',
                 f'FireRed_edit  ({a}, {s["quality_tier"]})',
                 f'inv\u2192ISP  PSNR={r["psnr_inv_fr"]:.1f}dB  <7D 上限>',
                 f'model\u2192ISP  PSNR={r["psnr_mod_fr"]:.1f}dB',
                 f'|inv - FR|  (\u202F\u202F7D \u7f3a\u53e3)'])

            tag = group.split()[0].lower()
            out_png = out_dir / f'{tag}_{rank+1}_{a}_{stem}.png'
            panel.save(out_png)
            print(f'  [{group}] {a} {stem}  inv_PSNR={r["psnr_inv_fr"]:.1f}  '
                  f'mod_PSNR={r["psnr_mod_fr"]:.1f}  '
                  f'-> {out_png.name}')

            rows_html.append({
                'group': group, 'rank': rank + 1, 'action': a,
                'image_id': stem, 'png': out_png.name,
                'tier': s['quality_tier'],
                'psnr_o_fr': r['psnr_o_fr'],
                'psnr_inv_fr': r['psnr_inv_fr'],
                'psnr_mod_fr': r['psnr_mod_fr'],
                'P_pred': r['P_pred'], 'P_inv': r['P_inv'],
            })

    # index.html
    html = [
        '<html><head><meta charset=utf-8>',
        '<title>FireRed 7D ISP 表达上限诊断</title>',
        '<style>body{font-family:sans-serif;background:#222;color:#eee;'
        'padding:16px;max-width:1500px;margin:auto}',
        'h2{margin-top:24px;border-bottom:1px solid #555;padding:4px 0}',
        'img{width:100%;border:1px solid #555;margin:6px 0}',
        '.box{background:#333;padding:12px;border-radius:6px;margin:12px 0}',
        'table{font-size:12px;background:#2a2a2a;border-collapse:collapse;'
        'margin:4px 0}',
        'th,td{padding:4px 8px;border:1px solid #555;text-align:center}',
        '.r{color:#9cf}.t{color:#fc9}.k{color:#9f9;font-weight:bold}',
        '</style></head><body>',
        '<h1>FireRed 7D ISP 表达上限诊断</h1>',
    ]
    html.append('<div class=box><h2>诊断摘要</h2><table>')
    html.append('<tr><th>指标</th><th>均值 (dB)</th><th>说明</th></tr>')
    html.append(f'<tr><td>PSNR(orig, FR)</td><td>{arr_o_fr.mean():.2f}</td>'
                '<td>baseline — 原图离 FR 多远</td></tr>')
    html.append(f'<tr><td>PSNR(inv_render, FR)</td>'
                f'<td class=k>{arr_inv_fr.mean():.2f}</td>'
                '<td>7D ISP 理论上限 (P 完美拟合后渲染)</td></tr>')
    html.append(f'<tr><td>PSNR(model, FR)</td><td>{arr_mod_fr.mean():.2f}</td>'
                f'<td>模型实际效果 (用 {args.ckpt} 算)</td></tr>')
    html.append(f'<tr><td>7D 拉近增益 (inv-orig)</td>'
                f'<td class=k>{inv_gain.mean():+.2f}</td>'
                '<td>正值 = ISP 帮助; 越大 ISP 越能表达 FR 风格</td></tr>')
    html.append(f'<tr><td>模型实际增益 (mod-orig)</td>'
                f'<td>{mod_gain.mean():+.2f}</td>'
                '<td>模型实际把图拉近 FR 多少</td></tr>')
    html.append(f'<tr><td>模型距 7D 上限 (mod-inv)</td>'
                f'<td>{mod_vs_inv.mean():+.2f}</td>'
                '<td>0=已达上限, 负=还能优化</td></tr>')
    html.append('</table></div>')

    html.append('<div class=box><h2>关键结论</h2>')
    if inv_gain.mean() < 3.0:
        html.append('<p class=k>7D ISP 表达力不足是主要 bottleneck</p>')
        html.append(f'<p>inv_render 平均仅比 orig 接近 FR {inv_gain.mean():.1f}dB. '
                    '即参数预测<b>完美</b>也只能复现 FR 风格的一小部分.</p>')
        html.append('<p>根因: 7D Lightroom 参数是 global 算子, FireRed 用神经网络做的 '
                    '<b>spatial-varying / non-linear</b> 风格无法被 7D 表达.</p>')
        html.append('<p><b>建议</b>: 扩 ISP 参数 (加 HSL split, tone curves, '
                    'local contrast) 或换 ISP 模型 (spatial-varying / learned)</p>')
    elif inv_gain.mean() < 6.0:
        html.append('<p class=k>7D ISP 部分有效, 模型优化仍有空间</p>')
        html.append(f'<p>inv_render 比 orig 接近 FR {inv_gain.mean():.1f}dB. '
                    '7D 能拿到 FR 风格的 50-70%, 还有空间提升.</p>')
        html.append('<p><b>建议</b>: 1) G1 扩数据让 model 贴近 inv 上限; '
                    '2) 同时考虑加 2-3 个 ISP 参数 (vibrance, HSL)</p>')
    else:
        html.append('<p class=k>7D ISP 足够, 训练问题</p>')
        html.append('<p><b>建议</b>: G1 扩数据 / Qwen3-VL LoRA 主线</p>')
    html.append('</div>')

    cur_group = None
    for r in rows_html:
        if r['group'] != cur_group:
            html.append(f'<h2>{r["group"]}</h2>')
            cur_group = r['group']
        gap_inv_o = r['psnr_inv_fr'] - r['psnr_o_fr']
        gap_mod_o = r['psnr_mod_fr'] - r['psnr_o_fr']
        html.append(f'<h3>{r["action"]} — {r["image_id"]} '
                    f'<small>(tier {r["tier"]}, '
                    f'7D拉近 <span class=k>+{gap_inv_o:.1f}dB</span>, '
                    f'model拉近 +{gap_mod_o:.1f}dB)</small></h3>')
        html.append(f'<img src="{r["png"]}">')

    html.append('</body></html>')
    (out_dir / 'index.html').write_text('\n'.join(html), encoding='utf-8')
    print(f'\nwrote {out_dir/"index.html"}')


if __name__ == '__main__':
    main()
