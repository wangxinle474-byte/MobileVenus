from __future__ import annotations

import argparse
import html
import json
import logging
import random
import re
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
    LUTDataset,
    NamedCurvesPredictor,
    build_data,
    set_actions,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ACTION_CN = {
    'contrast': '对比度',
    'saturation': '饱和度',
    'shadows': '暗部提亮',
    'highlights': '高光压暗',
    'wb': '白平衡',
    'brightness': '亮度',
    'clarity': '清晰度',
}

COLORS = {
    'orig': (100, 100, 100),
    'ctx': (236, 72, 153),
    'refined': (56, 189, 248),
    'target': (251, 191, 36),
    'error': (239, 68, 68),
}


def _get(ca: dict, key: str, default):
    return ca[key] if key in ca else default


def _safe_name(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', text)[:120]


def _font(size: int):
    for name in ('msyh.ttc', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def add_label(img: Image.Image, text: str, bg_color, text_color=(255, 255, 255)):
    w, h = img.size
    bar_h = 28
    out = Image.new('RGB', (w, h + bar_h), bg_color)
    draw = ImageDraw.Draw(out)
    font = _font(14)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((w - tw) // 2, 4), text, fill=text_color, font=font)
    out.paste(img, (0, bar_h))
    return out


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def context_to_pil(ctx: torch.Tensor) -> Image.Image:
    c = ctx.detach().cpu().numpy()
    c = np.clip(c, 0.0, 1.0)
    r = np.clip(c * 2.0 - 0.5, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(c - 0.5) * 2.0, 0.0, 1.0)
    b = np.clip((1.0 - c) * 1.5, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1) * 255
    return Image.fromarray(rgb.astype(np.uint8))


def error_to_pil(pred: torch.Tensor, target: torch.Tensor) -> Image.Image:
    err = (pred - target).abs().mean(dim=0).detach().cpu().numpy()
    e = np.clip(err * 5.0, 0.0, 1.0)
    r = np.clip(e * 2.0 - 0.5, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(e - 0.5) * 2.0, 0.0, 1.0)
    b = np.clip(1.0 - e, 0.0, 1.0) * 0.5
    rgb = np.stack([r, g, b], axis=-1) * 255
    return Image.fromarray(rgb.astype(np.uint8))


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().clamp(min=1e-10)
    return float((-10.0 * torch.log10(mse)).item())


def pearson_ctx_lum(ctx: torch.Tensor, orig: torch.Tensor) -> float:
    lum = 0.299 * orig[0] + 0.587 * orig[1] + 0.114 * orig[2]
    cf = ctx.flatten()
    lf = lum.flatten()
    cf = cf - cf.mean()
    lf = lf - lf.mean()
    denom = torch.sqrt((cf ** 2).sum()) * torch.sqrt((lf ** 2).sum()) + 1e-8
    return float(((cf * lf).sum() / denom).item())


def make_strip(images: list[Image.Image], gap: int = 2) -> Image.Image:
    w, h = images[0].size
    strip = Image.new('RGB', (w * len(images) + gap * (len(images) - 1), h), (15, 23, 42))
    for i, img in enumerate(images):
        strip.paste(img, (i * (w + gap), 0))
    return strip


def onehot_for(action: str, device, dtype=torch.float32) -> torch.Tensor:
    oh = torch.zeros(1, len(ACTIONS), device=device, dtype=dtype)
    oh[0, ACTIONS.index(action)] = 1.0
    return oh


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ca = ckpt['args']
    ckpt_actions = ca.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)
    model = NamedCurvesPredictor(
        n_colors=_get(ca, 'nc_n_colors', 3),
        n_control_points=_get(ca, 'nc_n_control_points', 7),
        use_attention=_get(ca, 'nc_use_attention', False),
        per_action_curves=_get(ca, 'nc_per_action_curves', False),
        use_7d_anchor=_get(ca, 'nc_use_7d_anchor', True),
        use_context=_get(ca, 'nc_use_context', False),
        action_gated_context=_get(ca, 'nc_action_gated_context', False),
        use_action_context=_get(ca, 'nc_use_action_context', False),
        use_region_basis=_get(ca, 'nc_use_region_basis', False),
        use_region_param_delta=_get(ca, 'nc_use_region_param_delta', False),
        use_learned_cn=_get(ca, 'nc_use_learned_cn', False),
        use_wb_head=_get(ca, 'nc_use_wb_head', False),
        use_nilut_residual=_get(ca, 'nc_use_nilut_residual', False),
        use_vera_renderer=_get(ca, 'nc_use_vera_renderer', False),
        use_implicit_head=_get(ca, 'use_implicit_head', False),
        implicit_head_base_ch=_get(ca, 'implicit_head_base_ch', 32),
        implicit_head_gate_init=_get(ca, 'implicit_head_gate_init', 0.0),
        nilut_hidden=_get(ca, 'nilut_hidden', 32),
        nilut_n_layers=_get(ca, 'nilut_n_layers', 3),
        nilut_n_freq=_get(ca, 'nilut_n_freq', 4),
        nilut_gate_init=_get(ca, 'nilut_gate_init', 1.0),
        image_size=_get(ca, 'image_size', 256),
        n_actions=len(ACTIONS),
        dropout=_get(ca, 'dropout', 0.5),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    if not (model.use_context and model.use_action_context):
        raise RuntimeError('This viewer expects a checkpoint with --nc_use_context --nc_use_action_context')
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/lut_v11d_action_context/best.pt')
    ap.add_argument('--jsonl', default='outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='outputs/lut_v11d_action_context_viewer')
    ap.add_argument('--image_size', type=int, default=None)
    ap.add_argument('--max_samples', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split_seed', type=int, default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, ckpt = load_model(Path(args.ckpt), device)
    ca = ckpt['args']
    image_size = args.image_size or _get(ca, 'image_size', 256)
    split_seed = args.split_seed
    if split_seed is None:
        split_seed = _get(ca, 'split_seed', _get(ca, 'seed', 42))

    _, val_s = build_data(
        Path(args.jsonl), _get(ca, 'val_ratio', 0.2),
        ('A excellent', 'B good', 'C acceptable'), split_seed,
        action_filter=ACTIONS)
    ds = LUTDataset(val_s, image_size, is_train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    out_dir = Path(args.out_dir)
    img_dir = out_dir / 'imgs'
    avg_dir = out_dir / 'avg_maps'
    img_dir.mkdir(parents=True, exist_ok=True)
    avg_dir.mkdir(parents=True, exist_ok=True)

    per_action_psnr = {a: [] for a in ACTIONS}
    ctx_stats = {a: {'mean': [], 'std': [], 'corr': []} for a in ACTIONS}
    ctx_sums = {a: None for a in ACTIONS}
    ctx_count = 0
    cards = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            enc_in = batch['enc_input'].to(device)
            orig = batch['orig'].to(device)
            target = batch['target'].to(device)
            a_oh = batch['action_onehot'].to(device)
            action = batch['action'][0]
            source = batch['source_image'][0]

            refined, _, _, _ = model(enc_in, orig, a_oh)
            psnr = compute_psnr(refined[0], target[0])
            per_action_psnr[action].append(psnr)
            current_ctx = model._last_context_map[0, 0].detach().cpu()

            action_ctx_images = []
            for a in ACTIONS:
                _, _, _, _ = model(enc_in, orig, onehot_for(a, device))
                ctx = model._last_context_map[0, 0].detach().cpu()
                ctx_np = ctx.numpy()
                ctx_stats[a]['mean'].append(float(ctx.mean().item()))
                ctx_stats[a]['std'].append(float(ctx.std().item()))
                ctx_stats[a]['corr'].append(pearson_ctx_lum(ctx, orig[0].detach().cpu()))
                ctx_sums[a] = ctx_np.copy() if ctx_sums[a] is None else ctx_sums[a] + ctx_np
                if idx < args.max_samples:
                    ctx_lab = add_label(
                        context_to_pil(ctx),
                        f'{ACTION_CN.get(a, a)} mean={ctx.mean().item():.2f}',
                        COLORS['ctx'])
                    action_ctx_images.append(ctx_lab)
            ctx_count += 1

            if idx < args.max_samples:
                base = _safe_name(f'{idx:03d}_{action}_{Path(source).stem}')
                orig_lab = add_label(tensor_to_pil(orig[0]), '① 原图', COLORS['orig'])
                ctx_lab = add_label(
                    context_to_pil(current_ctx),
                    f'② 当前action ctx mean={current_ctx.mean().item():.2f}',
                    COLORS['ctx'])
                ref_lab = add_label(
                    tensor_to_pil(refined[0]),
                    f'③ v11d输出 PSNR={psnr:.1f}dB',
                    COLORS['refined'], (0, 0, 0))
                tgt_lab = add_label(tensor_to_pil(target[0]), '④ FireRed目标', COLORS['target'], (0, 0, 0))
                err_lab = add_label(error_to_pil(refined[0], target[0]), '⑤ 误差热图×5', COLORS['error'])
                strip = make_strip([orig_lab, ctx_lab, ref_lab, tgt_lab, err_lab])
                grid = make_strip(action_ctx_images)
                strip_path = img_dir / f'{base}_strip.jpg'
                grid_path = img_dir / f'{base}_all_actions.jpg'
                strip.save(strip_path, quality=90)
                grid.save(grid_path, quality=90)
                cards.append({
                    'idx': idx,
                    'action': action,
                    'source': source,
                    'psnr': round(psnr, 2),
                    'strip': strip_path.relative_to(out_dir).as_posix(),
                    'grid': grid_path.relative_to(out_dir).as_posix(),
                })

    avg_images = []
    avg_files = {}
    for a in ACTIONS:
        avg_ctx = ctx_sums[a] / max(ctx_count, 1)
        avg_img = add_label(
            context_to_pil(torch.from_numpy(avg_ctx)),
            f'{ACTION_CN.get(a, a)} avg mean={avg_ctx.mean():.2f}',
            COLORS['ctx'])
        avg_path = avg_dir / f'avg_{a}.jpg'
        avg_img.save(avg_path, quality=92)
        avg_images.append(avg_img)
        avg_files[a] = avg_path.relative_to(out_dir).as_posix()
    avg_grid = make_strip(avg_images)
    avg_grid_path = avg_dir / 'avg_all_actions.jpg'
    avg_grid.save(avg_grid_path, quality=92)

    all_psnr = [v for values in per_action_psnr.values() for v in values]
    summary = {
        'checkpoint': args.ckpt,
        'ckpt_val_psnr': round(float(ckpt.get('val_psnr', 0.0)), 3),
        'ckpt_epoch': ckpt.get('epoch', None),
        'eval_overall': round(float(np.mean(all_psnr)), 3) if all_psnr else 0.0,
        'split_seed': split_seed,
        'per_action_psnr': {
            a: {
                'n': len(values),
                'mean': round(float(np.mean(values)), 3) if values else 0.0,
            }
            for a, values in per_action_psnr.items()
        },
        'context_stats': {
            a: {
                'mean_avg': round(float(np.mean(ctx_stats[a]['mean'])), 4),
                'mean_std': round(float(np.std(ctx_stats[a]['mean'])), 4),
                'pixel_std_avg': round(float(np.mean(ctx_stats[a]['std'])), 4),
                'lum_corr_avg': round(float(np.mean(ctx_stats[a]['corr'])), 4),
                'lum_corr_std': round(float(np.std(ctx_stats[a]['corr'])), 4),
                'avg_map': avg_files[a],
            }
            for a in ACTIONS
        },
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    pa_rows = ''.join(
        f'<tr><td>{html.escape(ACTION_CN.get(a, a))} ({a})</td><td>{summary["per_action_psnr"][a]["n"]}</td>'
        f'<td>{summary["per_action_psnr"][a]["mean"]:.2f}</td></tr>'
        for a in ACTIONS)
    ctx_rows = ''.join(
        f'<tr><td>{html.escape(ACTION_CN.get(a, a))} ({a})</td>'
        f'<td>{summary["context_stats"][a]["mean_avg"]:.3f}</td>'
        f'<td>{summary["context_stats"][a]["pixel_std_avg"]:.3f}</td>'
        f'<td>{summary["context_stats"][a]["lum_corr_avg"]:+.3f}</td></tr>'
        for a in ACTIONS)
    cards.sort(key=lambda x: x['psnr'])
    cards_html = ''.join(
        f'<div class="item" data-action="{html.escape(c["action"])}">'
        f'<h3>{html.escape(ACTION_CN.get(c["action"], c["action"]))} | {c["psnr"]:.2f} dB | {html.escape(c["source"])}</h3>'
        f'<img src="{html.escape(c["strip"])}" loading="lazy">'
        f'<p>同一原图输入 {len(ACTIONS)} 个 action 的 action-conditioned context maps:</p>'
        f'<img src="{html.escape(c["grid"])}" loading="lazy">'
        f'</div>'
        for c in cards)
    filters = ''.join(
        f'<label><input type="checkbox" class="f-action" value="{a}" checked> {html.escape(ACTION_CN.get(a, a))}</label>'
        for a in ACTIONS)

    viewer = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>v11d Action Context Viewer</title>
<style>
body {{ font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif; margin: 0; padding: 16px; background: #0f172a; color: #e2e8f0; }}
h1 {{ margin: 0 0 8px; font-size: 22px; }}
.subtitle {{ color: #94a3b8; font-size: 13px; margin-bottom: 16px; }}
.summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
.card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; min-width: 150px; }}
.card .v {{ font-size: 24px; font-weight: bold; color: #38bdf8; }}
.card .l {{ font-size: 12px; color: #94a3b8; }}
table {{ border-collapse: collapse; background: #1e293b; margin: 8px 0 16px; font-size: 13px; }}
th, td {{ border: 1px solid #334155; padding: 6px 12px; text-align: center; }}
th {{ background: #334155; color: #fbbf24; }}
.twocol {{ display: flex; flex-wrap: wrap; gap: 16px; }}
.twocol > div {{ flex: 1; min-width: 360px; }}
.avg img, .item img {{ width: 100%; display: block; border-radius: 6px; }}
.avg, .item, .filters {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin-bottom: 14px; }}
.item h3 {{ margin: 0 0 8px; font-size: 14px; color: #f8fafc; }}
.item p {{ color: #94a3b8; font-size: 12px; margin: 8px 0; }}
.filters label {{ margin-right: 14px; cursor: pointer; }}
.hide {{ display: none; }}
</style>
</head>
<body>
<h1>v11d Action-conditioned Context Viewer</h1>
<p class="subtitle">checkpoint: {html.escape(args.ckpt)} | ckpt best={summary['ckpt_val_psnr']:.2f}dB @ Ep{summary['ckpt_epoch']} | split_seed={split_seed}</p>
<div class="summary">
  <div class="card"><div class="v">{summary['eval_overall']:.2f}</div><div class="l">eval overall PSNR</div></div>
  <div class="card"><div class="v">{len(ds)}</div><div class="l">val samples</div></div>
  <div class="card"><div class="v">{len(cards)}</div><div class="l">visualized samples</div></div>
</div>
<div class="twocol">
<div><h2>Per-action PSNR</h2><table><thead><tr><th>Action</th><th>n</th><th>PSNR</th></tr></thead><tbody>{pa_rows}</tbody></table></div>
<div><h2>Context stats over all val images</h2><table><thead><tr><th>Action</th><th>mean</th><th>pixel std</th><th>corr(ctx, lum)</th></tr></thead><tbody>{ctx_rows}</tbody></table></div>
</div>
<div class="avg"><h2>Average context maps by action</h2><img src="{avg_grid_path.relative_to(out_dir).as_posix()}"></div>
<div class="filters"><b>按动作过滤:</b> {filters}</div>
<div id="grid">{cards_html}</div>
<script>
const items = document.querySelectorAll('.item');
const boxes = document.querySelectorAll('.f-action');
function update() {{
  const active = new Set(Array.from(boxes).filter(b => b.checked).map(b => b.value));
  items.forEach(it => it.classList.toggle('hide', !active.has(it.dataset.action)));
}}
boxes.forEach(b => b.addEventListener('change', update));
</script>
</body>
</html>'''
    (out_dir / 'viewer.html').write_text(viewer, encoding='utf-8')
    logger.info(f'Viewer: {out_dir / "viewer.html"}')
    logger.info(f'Summary: {out_dir / "summary.json"}')
    logger.info(f'Overall PSNR: {summary["eval_overall"]:.2f} dB')


if __name__ == '__main__':
    main()
