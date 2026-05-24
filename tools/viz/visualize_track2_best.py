"""Visualize Track 2 best checkpoint (Path A VeraRenderer, 42.66 dB) on
clean Expert C val.

For each of the 5 actions, picks top-1 / median-1 / bottom-1 PSNR samples
and renders a 4-panel strip: source / Path A prediction / target / error
heatmap (×5 amplified). Saves JPEGs plus a self-contained HTML viewer.

Default: clean Expert C val (n=3,127, the IN_VAL set behind the 42.66 dB
headline). Other val sets can be selected via --eval_set.

Usage:
  python tools/visualize_track2_best.py \
    --ckpt checkpoints/lut_track2A_vera_clean_seed42/best.pt \
    --out_dir outputs/viz_track2_best
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.firered_baseline.train_lut import (  # noqa: E402
    ACTIONS,
    LUTDataset,
    build_data,
)
from tools._lib.eval_track2 import build_model_from_ckpt  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# Same eval-set registry as eval_track2.py so --eval_set names match.
EVAL_SETS = {
    'clean_expertC': {
        'jsonl': 'outputs/fivek_expert_c_master/pseudo_labels.jsonl',
        'desc': 'Clean Expert C (15,739 noise-free targets, in-domain '
                'for Track 2). Path A scores 42.66 dB on the val split.',
    },
    'firered_pseudo': {
        'jsonl': 'outputs/inverse_fit_pilot/fivek_500_master/'
                 'pseudo_labels.jsonl',
        'desc': 'FireRed pseudo (372 records, val=74 internal split). '
                'Path A transfer = 21.59 dB.',
    },
    'mmart_real_lr': {
        'jsonl': 'outputs/mmart_pseudo_labels/v1_250/pseudo_labels.jsonl',
        'desc': 'MMArt-PPR10k 250 (real LR XMP, 100% leak-free against '
                'FiveK). Path A = 22.80 dB.',
    },
}

TIER_FILTER = ('A excellent', 'B good', 'C acceptable')

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
    'orig': (100, 116, 139),
    'pred': (56, 189, 248),
    'target': (251, 191, 36),
    'error': (239, 68, 68),
}


def _font(size: int):
    for name in ('msyh.ttc', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def add_label(img: Image.Image, text: str, bg_color,
              text_color=(255, 255, 255)) -> Image.Image:
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


def error_to_pil(pred: torch.Tensor, target: torch.Tensor,
                 amplify: float = 5.0) -> Image.Image:
    err = (pred - target).abs().mean(dim=0).detach().cpu().numpy()
    e = np.clip(err * amplify, 0.0, 1.0)
    # red-yellow-blue colormap: blue=low, yellow=mid, red=high
    r = np.clip(e * 2.0 - 0.5, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(e - 0.5) * 2.0, 0.0, 1.0)
    b = np.clip((1.0 - e) * 1.2, 0.0, 1.0) * 0.6
    rgb = np.stack([r, g, b], axis=-1) * 255
    return Image.fromarray(rgb.astype(np.uint8))


def make_strip(images: list, gap: int = 2,
               bg=(15, 23, 42)) -> Image.Image:
    w, h = images[0].size
    strip = Image.new('RGB', (w * len(images) + gap * (len(images) - 1), h),
                      bg)
    for i, img in enumerate(images):
        strip.paste(img, (i * (w + gap), 0))
    return strip


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().clamp(min=1e-10)
    return float((-10.0 * torch.log10(mse)).item())


def _safe_name(text: str) -> str:
    import re
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', text)[:120]


def evaluate_all(model, loader, device) -> list[dict]:
    """Forward pass on every sample; return list of (psnr, action, source,
    enc_in, orig, target, refined)."""
    records = []
    pbar = tqdm(loader, desc='forward', mininterval=1.0,
                dynamic_ncols=True, file=sys.stderr)
    with torch.no_grad():
        for batch in pbar:
            enc_in = batch['enc_input'].to(device)
            orig = batch['orig'].to(device)
            target = batch['target'].to(device)
            a_oh = batch['action_onehot'].to(device)

            refined, _, _, _ = model(enc_in, orig, a_oh)
            B = refined.shape[0]
            for i in range(B):
                psnr = compute_psnr(refined[i], target[i])
                records.append({
                    'psnr': psnr,
                    'action': batch['action'][i],
                    'source': batch['source_image'][i],
                    'orig': orig[i].detach().cpu(),
                    'target': target[i].detach().cpu(),
                    'refined': refined[i].detach().cpu(),
                })
    return records


def pick_representatives(records: list[dict], n_per_action: int = 3,
                         buckets: tuple = ('top', 'median', 'bottom')
                         ) -> dict[str, list[dict]]:
    """Group by action, sort by PSNR, pick top/median/bottom-N from each."""
    by_action: dict[str, list[dict]] = {a: [] for a in ACTIONS}
    for r in records:
        if r['action'] in by_action:
            by_action[r['action']].append(r)
    picks: dict[str, list[dict]] = {a: [] for a in ACTIONS}
    for a, items in by_action.items():
        if not items:
            continue
        items.sort(key=lambda x: x['psnr'])
        n = len(items)
        for bucket in buckets:
            if bucket == 'top':
                chosen = items[-n_per_action:]
                tag = 'top'
            elif bucket == 'bottom':
                chosen = items[:n_per_action]
                tag = 'bot'
            elif bucket == 'median':
                lo = max(0, n // 2 - n_per_action // 2)
                hi = min(n, lo + n_per_action)
                chosen = items[lo:hi]
                tag = 'med'
            else:
                continue
            for c in chosen:
                c = dict(c)
                c['bucket'] = tag
                picks[a].append(c)
    return picks


def render_strip(r: dict, ckpt_label: str) -> Image.Image:
    orig_lab = add_label(tensor_to_pil(r['orig']),
                         '① Source (input)', COLORS['orig'])
    pred_lab = add_label(tensor_to_pil(r['refined']),
                         f'② {ckpt_label}  PSNR={r["psnr"]:.2f} dB',
                         COLORS['pred'], (0, 0, 0))
    tgt_lab = add_label(tensor_to_pil(r['target']),
                        '③ Target (Expert C, clean)', COLORS['target'],
                        (0, 0, 0))
    err_lab = add_label(error_to_pil(r['refined'], r['target']),
                        '④ Error heatmap ×5', COLORS['error'])
    return make_strip([orig_lab, pred_lab, tgt_lab, err_lab])


def write_html(out_dir: Path, ckpt_path: str, ckpt_val_psnr: float,
               ckpt_epoch, eval_set_key: str, eval_set_desc: str,
               n_val: int, all_psnr: list[float],
               per_action_psnr: dict[str, list[float]],
               cards: list[dict]) -> Path:
    overall = float(np.mean(all_psnr)) if all_psnr else 0.0
    pa_rows = ''.join(
        f'<tr><td>{html.escape(ACTION_CN.get(a, a))} ({a})</td>'
        f'<td>{len(per_action_psnr[a])}</td>'
        f'<td>{float(np.mean(per_action_psnr[a])):.2f}</td>'
        f'<td>{float(np.median(per_action_psnr[a])):.2f}</td>'
        f'<td>{float(np.min(per_action_psnr[a])):.2f}</td>'
        f'<td>{float(np.max(per_action_psnr[a])):.2f}</td></tr>'
        for a in ACTIONS if per_action_psnr[a])

    cards_by_action: dict[str, list[dict]] = {a: [] for a in ACTIONS}
    for c in cards:
        cards_by_action[c['action']].append(c)
    sections = []
    for a in ACTIONS:
        if not cards_by_action[a]:
            continue
        cards_by_action[a].sort(key=lambda x: -x['psnr'])
        items_html = ''.join(
            f'<div class="item" data-bucket="{c["bucket"]}">'
            f'<h3>{html.escape(c["bucket"].upper())} | '
            f'{c["psnr"]:.2f} dB | {html.escape(c["source"])}</h3>'
            f'<img src="{html.escape(c["strip"])}" loading="lazy">'
            f'</div>'
            for c in cards_by_action[a])
        sections.append(
            f'<section class="action-section" data-action="{a}">'
            f'<h2>{html.escape(ACTION_CN.get(a, a))} ({a}) — '
            f'mean {float(np.mean(per_action_psnr[a])):.2f} dB '
            f'over n={len(per_action_psnr[a])}</h2>'
            f'<div class="grid">{items_html}</div>'
            f'</section>')

    filters = ''.join(
        f'<label><input type="checkbox" class="f-action" value="{a}" checked>'
        f' {html.escape(ACTION_CN.get(a, a))}</label>'
        for a in ACTIONS if cards_by_action[a])

    viewer = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Track 2 best — Path A viewer</title>
<style>
body {{ font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
        margin: 0; padding: 16px; background: #0f172a; color: #e2e8f0; }}
h1 {{ margin: 0 0 8px; font-size: 22px; }}
.subtitle {{ color: #94a3b8; font-size: 13px; margin-bottom: 16px; }}
.summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
.card {{ background: #1e293b; border: 1px solid #334155;
         border-radius: 8px; padding: 10px 14px; min-width: 150px; }}
.card .v {{ font-size: 24px; font-weight: bold; color: #38bdf8; }}
.card .l {{ font-size: 12px; color: #94a3b8; }}
table {{ border-collapse: collapse; background: #1e293b;
         margin: 8px 0 16px; font-size: 13px; }}
th, td {{ border: 1px solid #334155; padding: 6px 12px; text-align: center; }}
th {{ background: #334155; color: #fbbf24; }}
.action-section {{ background: #1e293b; border: 1px solid #334155;
                   border-radius: 8px; padding: 12px; margin-bottom: 14px; }}
.action-section h2 {{ margin: 0 0 10px; font-size: 16px; color: #f8fafc; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
.item {{ background: #0f172a; border: 1px solid #334155;
          border-radius: 6px; padding: 10px; }}
.item h3 {{ margin: 0 0 6px; font-size: 13px; color: #cbd5e1; }}
.item img {{ width: 100%; display: block; border-radius: 4px; }}
.filters {{ background: #1e293b; border: 1px solid #334155;
            border-radius: 8px; padding: 12px; margin-bottom: 14px; }}
.filters label {{ margin-right: 14px; cursor: pointer; }}
.hide {{ display: none; }}
</style>
</head>
<body>
<h1>Track 2 best — Path A (VeraRenderer) effect viewer</h1>
<p class="subtitle">
  checkpoint: {html.escape(ckpt_path)} |
  ckpt best={ckpt_val_psnr:.2f} dB @ Ep{ckpt_epoch} |
  eval set: <b>{html.escape(eval_set_key)}</b>
</p>
<p class="subtitle">{html.escape(eval_set_desc)}</p>
<div class="summary">
  <div class="card"><div class="v">{overall:.2f}</div>
       <div class="l">overall PSNR (this eval)</div></div>
  <div class="card"><div class="v">{n_val}</div>
       <div class="l">val samples</div></div>
  <div class="card"><div class="v">{len(cards)}</div>
       <div class="l">visualized samples</div></div>
</div>
<h2>Per-action breakdown</h2>
<table>
  <thead><tr><th>Action</th><th>n</th><th>mean</th>
              <th>median</th><th>min</th><th>max</th></tr></thead>
  <tbody>{pa_rows}</tbody>
</table>
<div class="filters">
  <b>Filter by action:</b> {filters} &nbsp;|&nbsp;
  <b>Bucket:</b>
  <label><input type="checkbox" class="f-bucket" value="top" checked> top</label>
  <label><input type="checkbox" class="f-bucket" value="med" checked> median</label>
  <label><input type="checkbox" class="f-bucket" value="bot" checked> bottom</label>
</div>
{''.join(sections)}
<script>
const items = document.querySelectorAll('.item');
const sections = document.querySelectorAll('.action-section');
const aboxes = document.querySelectorAll('.f-action');
const bboxes = document.querySelectorAll('.f-bucket');
function update() {{
  const activeA = new Set(Array.from(aboxes).filter(b => b.checked).map(b => b.value));
  const activeB = new Set(Array.from(bboxes).filter(b => b.checked).map(b => b.value));
  sections.forEach(s => s.classList.toggle('hide', !activeA.has(s.dataset.action)));
  items.forEach(it => it.classList.toggle('hide', !activeB.has(it.dataset.bucket)));
}}
aboxes.forEach(b => b.addEventListener('change', update));
bboxes.forEach(b => b.addEventListener('change', update));
</script>
</body>
</html>'''
    html_path = out_dir / 'viewer.html'
    html_path.write_text(viewer, encoding='utf-8')
    return html_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',
                    default='checkpoints/lut_track2A_vera_clean_seed42/best.pt',
                    help='Track 2 checkpoint (default: Path A)')
    ap.add_argument('--ckpt_label', default='Path A (VeraRenderer)',
                    help='Display name in panel ②')
    ap.add_argument('--eval_set', choices=list(EVAL_SETS.keys()),
                    default='clean_expertC',
                    help='Which val set to visualize')
    ap.add_argument('--out_dir', default='outputs/viz_track2_best')
    ap.add_argument('--n_per_action', type=int, default=3,
                    help='Reps per (action, bucket) — total = 5×3×n')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--image_size', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None,
                    help='Optionally cap forward-pass to N samples (debug)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device = {device}')

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ca = ckpt['args']
    image_size = args.image_size or ca.get('image_size', 256)
    logger.info(f'loaded ckpt: {ckpt_path}  '
                f'best_val_psnr={ckpt.get("val_psnr", 0.0):.3f} dB  '
                f'epoch={ckpt.get("epoch")}  image_size={image_size}')

    model = build_model_from_ckpt(ckpt, image_size, device)

    # Build val split — same recipe as train_lut.build_data so it matches
    # the training-time split exactly (matters for clean_expertC IN_VAL).
    eval_cfg = EVAL_SETS[args.eval_set]
    val_ratio = ca.get('val_ratio') or 0.2
    split_seed = ca.get('split_seed')
    if split_seed is None:
        split_seed = ca.get('seed', 42)
    # If eval_set jsonl matches training jsonl, use training split_seed.
    # Else, the val side of build_data is determined by --eval_set's own
    # internal 0.2 ratio with split_seed=42 (matches train_lut default).
    train_jsonl_str = ca.get('jsonl', '')
    eval_jsonl = Path(eval_cfg['jsonl'])
    if Path(train_jsonl_str).name == eval_jsonl.name:
        logger.info(f'eval_set matches training jsonl; '
                    f'using training split_seed={split_seed}, '
                    f'val_ratio={val_ratio}')
    else:
        split_seed = 42
        val_ratio = 0.2
        logger.info(f'eval_set differs from training jsonl; '
                    f'using default split_seed=42, val_ratio=0.2')

    _, val_s = build_data(eval_jsonl, val_ratio, TIER_FILTER, split_seed,
                          action_filter=ACTIONS)
    if args.limit:
        val_s = val_s[:args.limit]
    logger.info(f'val samples: {len(val_s)}')

    ds = LUTDataset(val_s, image_size, is_train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    records = evaluate_all(model, loader, device)
    all_psnr = [r['psnr'] for r in records]
    per_action_psnr: dict[str, list[float]] = {a: [] for a in ACTIONS}
    for r in records:
        if r['action'] in per_action_psnr:
            per_action_psnr[r['action']].append(r['psnr'])
    overall = float(np.mean(all_psnr)) if all_psnr else 0.0
    logger.info(f'eval overall PSNR = {overall:.3f} dB on n={len(records)}')

    out_dir = Path(args.out_dir) / args.eval_set
    img_dir = out_dir / 'strips'
    img_dir.mkdir(parents=True, exist_ok=True)

    picks = pick_representatives(records, n_per_action=args.n_per_action)
    cards = []
    for a, items in picks.items():
        for c in items:
            base = _safe_name(f'{a}_{c["bucket"]}_{c["psnr"]:.1f}_'
                              f'{Path(c["source"]).stem}')
            strip = render_strip(c, args.ckpt_label)
            strip_path = img_dir / f'{base}.jpg'
            strip.save(strip_path, quality=92)
            cards.append({
                'action': a,
                'bucket': c['bucket'],
                'psnr': c['psnr'],
                'source': c['source'],
                'strip': strip_path.relative_to(out_dir).as_posix(),
            })

    # JSON summary
    summary = {
        'checkpoint': str(ckpt_path),
        'ckpt_val_psnr': float(ckpt.get('val_psnr', 0.0)),
        'ckpt_epoch': ckpt.get('epoch'),
        'eval_set': args.eval_set,
        'eval_set_jsonl': str(eval_jsonl),
        'actions': list(ACTIONS),
        'n_val': len(records),
        'overall_psnr': overall,
        'per_action': {
            a: {
                'n': len(v),
                'mean': float(np.mean(v)) if v else 0.0,
                'median': float(np.median(v)) if v else 0.0,
                'min': float(np.min(v)) if v else 0.0,
                'max': float(np.max(v)) if v else 0.0,
            }
            for a, v in per_action_psnr.items()
        },
        'visualized_samples': len(cards),
    }
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    html_path = write_html(
        out_dir,
        ckpt_path=str(ckpt_path),
        ckpt_val_psnr=float(ckpt.get('val_psnr', 0.0)),
        ckpt_epoch=ckpt.get('epoch'),
        eval_set_key=args.eval_set,
        eval_set_desc=eval_cfg['desc'],
        n_val=len(records),
        all_psnr=all_psnr,
        per_action_psnr=per_action_psnr,
        cards=cards,
    )
    logger.info(f'viewer: {html_path}')
    logger.info(f'summary: {out_dir / "summary.json"}')
    logger.info(f'overall PSNR: {overall:.3f} dB ({len(records)} samples)')


if __name__ == '__main__':
    main()
