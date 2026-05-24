"""Visualize Track 2 Path A predictions side-by-side with FireRed pseudo
targets on the *same images*.

Addresses the observation that Expert C clean targets look visually
subtler than FireRed pseudo targets despite Path A's 42.66 dB on Expert C
val. This is ceiling (iii) in §5.1: Expert C's 7D parameters are typically
near-zero on most slots (professional restraint), while FireRed's pseudo
parameters engage all 7 slots aggressively, producing more dramatic edits.

For each FireRed pseudo record (n=499 on FiveK images), we:
  - Look up the matching Clean Expert C record (same image_stem, same action)
  - Run Path A inference on the FiveK source
  - Render a 5-panel strip:
      ① Source (FiveK input)
      ② FireRed pseudo target (dramatic AI edit, what model trained on
         in Track 1 baseline)
      ③ Clean Expert C target (subtle real Expert C 7D-render, what Path A
         trained on in Track 2)
      ④ Path A prediction (reproduces ③ at 42.66 dB)
      ⑤ Error map (Path A vs Expert C, ×5 amplified)

The 7D parameters of ② and ③ are printed in the panel labels so the
"subtle vs dramatic" contrast is visible at the metadata level too.

Usage:
  python tools/visualize_track2_vs_firered.py \
    --ckpt checkpoints/lut_track2A_vera_clean_seed42/best.pt \
    --out_dir outputs/viz_track2_vs_firered
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
)
from tools._lib.eval_track2 import build_model_from_ckpt  # noqa: E402
from tools._lib.merge_jsonl_for_joint_training import extract_fivek_stem  # noqa: E402

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
    'orig': (100, 116, 139),
    'firered': (217, 70, 239),
    'expertC': (251, 191, 36),
    'pred': (56, 189, 248),
    'error': (239, 68, 68),
}

PARAM_KEYS = ['white_balance', 'brightness', 'contrast', 'shadows',
              'highlights', 'saturation', 'clarity']


def _font(size: int):
    for name in ('msyh.ttc', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def add_label(img: Image.Image, lines: list, bg_color,
              text_color=(255, 255, 255)) -> Image.Image:
    if isinstance(lines, str):
        lines = [lines]
    line_h = 18
    bar_h = 4 + line_h * len(lines)
    w, h = img.size
    out = Image.new('RGB', (w, h + bar_h), bg_color)
    draw = ImageDraw.Draw(out)
    font = _font(13)
    for i, text in enumerate(lines):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((w - tw) // 2, 3 + i * line_h), text, fill=text_color,
                  font=font)
    out.paste(img, (0, bar_h))
    return out


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def error_to_pil(pred: torch.Tensor, target: torch.Tensor,
                 amplify: float = 5.0) -> Image.Image:
    err = (pred - target).abs().mean(dim=0).detach().cpu().numpy()
    e = np.clip(err * amplify, 0.0, 1.0)
    r = np.clip(e * 2.0 - 0.5, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(e - 0.5) * 2.0, 0.0, 1.0)
    b = np.clip((1.0 - e) * 1.2, 0.0, 1.0) * 0.6
    rgb = np.stack([r, g, b], axis=-1) * 255
    return Image.fromarray(rgb.astype(np.uint8))


def make_strip(images: list, gap: int = 2,
               bg=(15, 23, 42)) -> Image.Image:
    """Pad images to a common height before pasting (different label-bar
    heights would otherwise crop the taller panels)."""
    w = images[0].size[0]
    h = max(img.size[1] for img in images)
    strip = Image.new(
        'RGB', (w * len(images) + gap * (len(images) - 1), h), bg)
    for i, img in enumerate(images):
        # Center-align each panel vertically inside the strip's max height.
        y = (h - img.size[1]) // 2
        strip.paste(img, (i * (w + gap), y))
    return strip


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().clamp(min=1e-10)
    return float((-10.0 * torch.log10(mse)).item())


def fmt_params(P: dict) -> str:
    """One-line param summary; non-default-zero values highlighted."""
    parts = []
    for k in PARAM_KEYS:
        v = P.get(k, 0.0) or 0.0
        if k == 'white_balance':
            parts.append(f'WB={v:.0f}')
        else:
            parts.append(f'{k[:3]}={v:+.1f}')
    return ' '.join(parts)


def count_nonzero_params(P: dict, wb_neutral: float = 5500.0,
                         wb_tol: float = 250.0) -> int:
    """How many of the 7 params are *non-default*. WB is non-default if
    it deviates from 5500K by more than ±250K. Others are non-zero."""
    n = 0
    for k in PARAM_KEYS:
        v = P.get(k, 0.0) or 0.0
        if k == 'white_balance':
            if abs(v - wb_neutral) > wb_tol:
                n += 1
        else:
            if abs(v) > 0.5:
                n += 1
    return n


def load_pseudo_target(target_path: str, size: int) -> torch.Tensor:
    """Load a FireRed pseudo target PNG (path relative to project root)."""
    p = Path(target_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return None
    img = Image.open(p).convert('RGB').resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def build_matched_pairs(firered_jsonl: Path, expertC_jsonl: Path
                        ) -> tuple[list, dict]:
    """Index Expert C by (canonical_stem, action) and return list of
    FireRed records whose (stem, action) has a matching Expert C record.
    Each returned item is a dict {'firered': fr_rec, 'expertC': ec_rec}."""
    # Build Expert C index
    ec_index: dict[tuple[str, str], dict] = {}
    with open(expertC_jsonl, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            stem = extract_fivek_stem(r)
            action = r.get('action', '')
            ec_index[(stem, action)] = r
    logger.info(f'Expert C index: {len(ec_index)} (stem, action) entries')

    # Iterate FireRed records, match
    matches = []
    skipped = {'unsupported_action': 0, 'no_match': 0, 'no_firered_target': 0}
    with open(firered_jsonl, encoding='utf-8') as f:
        for line in f:
            fr = json.loads(line)
            stem = extract_fivek_stem(fr)
            action = fr.get('action', '')
            if action not in ACTIONS:
                skipped['unsupported_action'] += 1
                continue
            ec = ec_index.get((stem, action))
            if ec is None:
                skipped['no_match'] += 1
                continue
            ft_path = fr.get('target_path', '')
            if not ft_path:
                skipped['no_firered_target'] += 1
                continue
            ft_full = Path(ft_path)
            if not ft_full.is_absolute():
                ft_full = PROJECT_ROOT / ft_full
            if not ft_full.exists():
                skipped['no_firered_target'] += 1
                continue
            matches.append({
                'firered': fr,
                'expertC': ec,
                'stem': stem,
                'action': action,
            })
    logger.info(f'matched pairs: {len(matches)} | skipped: {skipped}')
    return matches, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',
                    default='checkpoints/lut_track2A_vera_clean_seed42/best.pt')
    ap.add_argument('--ckpt_label', default='Path A (VeraRenderer)')
    ap.add_argument('--firered_jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--expertC_jsonl',
                    default='outputs/fivek_expert_c_master/pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='outputs/viz_track2_vs_firered')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--image_size', type=int, default=None)
    ap.add_argument('--max_samples', type=int, default=120,
                    help='Cap total visualized strips (top by visual delta)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ca = ckpt['args']
    image_size = args.image_size or ca.get('image_size', 256)
    logger.info(f'ckpt {ckpt_path} | val={ckpt.get("val_psnr", 0.0):.3f} '
                f'@ Ep{ckpt.get("epoch")} | image_size={image_size}')

    model = build_model_from_ckpt(ckpt, image_size, device)

    matches, skipped = build_matched_pairs(
        Path(args.firered_jsonl), Path(args.expertC_jsonl))
    if not matches:
        raise RuntimeError('No (FireRed, Expert C) matches found.')

    # LUTDataset on the Expert C records (so target = clean Expert C)
    ec_records = [m['expertC'] for m in matches]
    ds = LUTDataset(ec_records, image_size, is_train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    records: list[dict] = []
    with torch.no_grad():
        idx_global = 0
        pbar = tqdm(loader, desc='forward', mininterval=1.0,
                    dynamic_ncols=True, file=sys.stderr)
        for batch in pbar:
            enc_in = batch['enc_input'].to(device)
            orig = batch['orig'].to(device)
            target = batch['target'].to(device)
            a_oh = batch['action_onehot'].to(device)
            refined, _, _, _ = model(enc_in, orig, a_oh)
            B = refined.shape[0]
            for i in range(B):
                m = matches[idx_global]
                fr_target = load_pseudo_target(
                    m['firered']['target_path'], image_size)
                if fr_target is None:
                    idx_global += 1
                    continue
                # Visual delta: how different FireRed target is from
                # Expert C target on this image (both are "edits" of the
                # same source, so this measures how aggressive FireRed
                # was relative to Expert C).
                vis_delta = float(
                    (fr_target - target[i].cpu()).abs().mean().item())
                psnr_vs_ec = compute_psnr(refined[i], target[i])
                records.append({
                    'idx': idx_global,
                    'stem': m['stem'],
                    'action': m['action'],
                    'psnr_vs_ec': psnr_vs_ec,
                    'firered_vs_ec_delta': vis_delta,
                    'firered_P': m['firered'].get('P_inferred', {}),
                    'expertC_P': m['expertC'].get('P_inferred', {}),
                    'orig': orig[i].detach().cpu(),
                    'target_ec': target[i].detach().cpu(),
                    'target_fr': fr_target,
                    'refined': refined[i].detach().cpu(),
                })
                idx_global += 1

    logger.info(f'forward done | n={len(records)} | '
                f'mean PSNR vs Expert C = '
                f'{float(np.mean([r["psnr_vs_ec"] for r in records])):.3f} dB')

    # Group + sort: take top-N by FireRed-vs-Expert-C visual delta,
    # spread roughly evenly across actions.
    by_action: dict[str, list[dict]] = {a: [] for a in ACTIONS}
    for r in records:
        if r['action'] in by_action:
            by_action[r['action']].append(r)
    for a in ACTIONS:
        by_action[a].sort(key=lambda x: -x['firered_vs_ec_delta'])
    cap_per_action = max(1, args.max_samples // len(ACTIONS))
    picked = []
    for a in ACTIONS:
        picked.extend(by_action[a][:cap_per_action])

    # Render strips
    out_dir = Path(args.out_dir)
    img_dir = out_dir / 'strips'
    img_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    n_ec_nonzero = []
    n_fr_nonzero = []
    for r in records:
        n_ec_nonzero.append(count_nonzero_params(r['expertC_P']))
        n_fr_nonzero.append(count_nonzero_params(r['firered_P']))

    for c in picked:
        ec_n = count_nonzero_params(c['expertC_P'])
        fr_n = count_nonzero_params(c['firered_P'])
        orig_lab = add_label(
            tensor_to_pil(c['orig']),
            ['① Source (FiveK input)'],
            COLORS['orig'])
        fr_lab = add_label(
            tensor_to_pil(c['target_fr']),
            [f'② FireRed pseudo target | {fr_n}/7 params engaged',
             fmt_params(c['firered_P'])],
            COLORS['firered'])
        ec_lab = add_label(
            tensor_to_pil(c['target_ec']),
            [f'③ Clean Expert C (7D ISP render) | {ec_n}/7 params engaged',
             fmt_params(c['expertC_P'])],
            COLORS['expertC'], (0, 0, 0))
        pred_lab = add_label(
            tensor_to_pil(c['refined']),
            [f'④ {args.ckpt_label}  PSNR vs ③ = {c["psnr_vs_ec"]:.2f} dB'],
            COLORS['pred'], (0, 0, 0))
        err_lab = add_label(
            error_to_pil(c['refined'], c['target_ec']),
            ['⑤ Error |④−③| ×5'],
            COLORS['error'])
        strip = make_strip([orig_lab, fr_lab, ec_lab, pred_lab, err_lab])
        base = (f'{c["action"]}_{c["stem"]}_'
                f'frVsEc={c["firered_vs_ec_delta"]:.3f}'
                f'_psnr={c["psnr_vs_ec"]:.1f}')
        base = base.replace('/', '_').replace('\\', '_')
        strip_path = img_dir / f'{base}.jpg'
        strip.save(strip_path, quality=92)
        cards.append({
            'action': c['action'],
            'stem': c['stem'],
            'psnr_vs_ec': c['psnr_vs_ec'],
            'firered_vs_ec_delta': c['firered_vs_ec_delta'],
            'ec_n_nonzero': ec_n,
            'fr_n_nonzero': fr_n,
            'firered_P': c['firered_P'],
            'expertC_P': c['expertC_P'],
            'strip': strip_path.relative_to(out_dir).as_posix(),
        })

    # Summary
    mean_psnr = float(np.mean([r['psnr_vs_ec'] for r in records]))
    ec_n_mean = float(np.mean(n_ec_nonzero))
    fr_n_mean = float(np.mean(n_fr_nonzero))
    summary = {
        'checkpoint': str(ckpt_path),
        'ckpt_val_psnr': float(ckpt.get('val_psnr', 0.0)),
        'ckpt_epoch': ckpt.get('epoch'),
        'actions': list(ACTIONS),
        'firered_records': len(records),
        'matched_pairs': len(records),
        'visualized_pairs': len(cards),
        'overall_psnr_vs_ec': mean_psnr,
        'mean_expertC_params_engaged': ec_n_mean,
        'mean_firered_params_engaged': fr_n_mean,
        'per_action': {
            a: {
                'n': len(by_action[a]),
                'mean_psnr_vs_ec': float(np.mean(
                    [r['psnr_vs_ec'] for r in by_action[a]])) if by_action[a] else 0.0,
                'mean_firered_vs_ec_delta': float(np.mean(
                    [r['firered_vs_ec_delta'] for r in by_action[a]])) if by_action[a] else 0.0,
                'mean_ec_n_nonzero': float(np.mean(
                    [count_nonzero_params(r['expertC_P'])
                     for r in by_action[a]])) if by_action[a] else 0.0,
                'mean_fr_n_nonzero': float(np.mean(
                    [count_nonzero_params(r['firered_P'])
                     for r in by_action[a]])) if by_action[a] else 0.0,
            }
            for a in ACTIONS
        },
        'skipped': skipped,
    }
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    # HTML viewer
    pa_rows = ''.join(
        f'<tr><td>{html.escape(ACTION_CN.get(a, a))} ({a})</td>'
        f'<td>{summary["per_action"][a]["n"]}</td>'
        f'<td>{summary["per_action"][a]["mean_psnr_vs_ec"]:.2f}</td>'
        f'<td>{summary["per_action"][a]["mean_ec_n_nonzero"]:.1f}</td>'
        f'<td>{summary["per_action"][a]["mean_fr_n_nonzero"]:.1f}</td>'
        f'<td>{summary["per_action"][a]["mean_firered_vs_ec_delta"]:.3f}</td></tr>'
        for a in ACTIONS if summary['per_action'][a]['n'] > 0)

    cards_by_action: dict[str, list[dict]] = {a: [] for a in ACTIONS}
    for c in cards:
        cards_by_action[c['action']].append(c)
    for a in ACTIONS:
        cards_by_action[a].sort(key=lambda x: -x['firered_vs_ec_delta'])

    sections = []
    for a in ACTIONS:
        if not cards_by_action[a]:
            continue
        items_html = ''.join(
            f'<div class="item">'
            f'<h3>{html.escape(c["stem"])} | '
            f'<span class="psnr">Path A vs Expert C = {c["psnr_vs_ec"]:.2f} dB</span> | '
            f'<span class="fr">FireRed-vs-ExpertC visual Δ = {c["firered_vs_ec_delta"]:.3f}</span> | '
            f'<span class="params">EC engages {c["ec_n_nonzero"]}/7 params, '
            f'FR engages {c["fr_n_nonzero"]}/7</span></h3>'
            f'<img src="{html.escape(c["strip"])}" loading="lazy">'
            f'</div>'
            for c in cards_by_action[a])
        sections.append(
            f'<section data-action="{a}">'
            f'<h2>{html.escape(ACTION_CN.get(a, a))} ({a}) — '
            f'{len(cards_by_action[a])} samples shown out of '
            f'{summary["per_action"][a]["n"]} matched</h2>'
            f'<div class="grid">{items_html}</div>'
            f'</section>')

    viewer = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Track 2 vs FireRed — visual comparison</title>
<style>
body {{ font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
        margin: 0; padding: 16px; background: #0f172a; color: #e2e8f0; }}
h1 {{ margin: 0 0 8px; font-size: 22px; }}
.subtitle {{ color: #94a3b8; font-size: 13px; margin-bottom: 16px; }}
.banner {{ background: #1e293b; border-left: 4px solid #fbbf24;
           border-radius: 6px; padding: 14px 18px; margin: 14px 0; }}
.banner h2 {{ margin: 0 0 6px; font-size: 16px; color: #fbbf24; }}
.banner p {{ margin: 6px 0; line-height: 1.6; font-size: 13px; }}
.banner code {{ background: #0f172a; padding: 2px 6px; border-radius: 3px;
                 color: #38bdf8; font-size: 12px; }}
.summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }}
.card {{ background: #1e293b; border: 1px solid #334155;
         border-radius: 8px; padding: 10px 14px; min-width: 150px; }}
.card .v {{ font-size: 22px; font-weight: bold; color: #38bdf8; }}
.card .l {{ font-size: 11px; color: #94a3b8; }}
table {{ border-collapse: collapse; background: #1e293b;
         margin: 8px 0 16px; font-size: 13px; }}
th, td {{ border: 1px solid #334155; padding: 6px 12px; text-align: center; }}
th {{ background: #334155; color: #fbbf24; }}
section {{ background: #1e293b; border: 1px solid #334155;
           border-radius: 8px; padding: 12px; margin-bottom: 14px; }}
section h2 {{ margin: 0 0 10px; font-size: 16px; color: #f8fafc; }}
.grid {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
.item {{ background: #0f172a; border: 1px solid #334155;
          border-radius: 6px; padding: 10px; }}
.item h3 {{ margin: 0 0 8px; font-size: 12px; color: #cbd5e1;
            font-weight: normal; line-height: 1.5; }}
.item h3 .psnr {{ color: #38bdf8; }}
.item h3 .fr {{ color: #e879f9; }}
.item h3 .params {{ color: #fbbf24; }}
.item img {{ width: 100%; display: block; border-radius: 4px; }}
.filters {{ background: #1e293b; border: 1px solid #334155;
            border-radius: 8px; padding: 12px; margin-bottom: 14px; }}
.filters label {{ margin-right: 14px; cursor: pointer; }}
.hide {{ display: none; }}
</style>
</head>
<body>
<h1>Track 2 — Path A vs FireRed pseudo: visual contradiction explained</h1>
<p class="subtitle">
  checkpoint: {html.escape(str(ckpt_path))} |
  best_val={ckpt.get("val_psnr", 0.0):.2f} dB @ Ep{ckpt.get("epoch")}
</p>

<div class="banner">
<h2>为什么 Path A 的 42.66 dB 看上去并不"漂亮" — 三句话总结</h2>
<p><b>① Expert C 是 Adobe 摄影师的"克制式"修图</b>，每张 (image, action)
   对里通常只有 <code>{ec_n_mean:.1f} / 7</code> 个 7D 参数是非零的（即多数
   slot 直接走默认值）。FireRed 是生成式 AI 编辑模型，每张对 <code>{fr_n_mean:.1f} / 7</code>
   个参数都激进出手。所以 ② FireRed pseudo 视觉冲击大，③ Expert C 7D-render 视觉冲击小，
   <b>这是监督分布本身的性质，不是 Path A 的失败</b>。</p>
<p><b>② 你看到的"Expert C 效果不好"其实是 ceiling (iii) 在视觉层面的表现</b> —
   论文 §5.1 ceiling (iii)：<i>纯 7D ISP 渲染对真实 Expert C JPEG 的拟合上限 ≈23.94 dB</i>。
   我们这里 ③ 显示的是 7D-render（监督目标），不是 Adobe 真实 Expert C JPEG
   (本机无原图)。即便完美还原 ③，仍是真 Expert C 的一个有损 7D 投影。</p>
<p><b>③ 把这两点连起来 — Path A 在 ③ 上取得 42.66 dB 说明架构内部容量充足</b>，
   ceiling 不在架构而在监督分布：a) Expert C 7D 参数本来就稀疏（克制式审美），
   b) 7D ISP 表征本身存在容量上限。这正是 §5.2 五个 bridging interventions
   全部证伪后得到的 manifold hypothesis 在视觉上的对应物。</p>
</div>

<div class="summary">
  <div class="card"><div class="v">{mean_psnr:.2f}</div>
       <div class="l">Path A PSNR vs Expert C (on FireRed-overlap set)</div></div>
  <div class="card"><div class="v">{len(records)}</div>
       <div class="l">matched FireRed pairs</div></div>
  <div class="card"><div class="v">{ec_n_mean:.1f} / 7</div>
       <div class="l">avg Expert C params engaged</div></div>
  <div class="card"><div class="v">{fr_n_mean:.1f} / 7</div>
       <div class="l">avg FireRed params engaged</div></div>
  <div class="card"><div class="v">{len(cards)}</div>
       <div class="l">strips rendered (top by FR-vs-EC visual Δ)</div></div>
</div>

<h2>Per-action breakdown</h2>
<table>
<thead><tr><th>Action</th><th>n matched</th>
<th>Path A PSNR vs EC</th>
<th>EC params engaged (avg /7)</th>
<th>FR params engaged (avg /7)</th>
<th>FR-vs-EC visual Δ</th></tr></thead>
<tbody>{pa_rows}</tbody>
</table>

<div class="filters">
  <b>5-panel legend:</b>
  <span style="color:#94a3b8">① Source</span> &nbsp;|&nbsp;
  <span style="color:#e879f9">② FireRed pseudo target</span> &nbsp;|&nbsp;
  <span style="color:#fbbf24">③ Clean Expert C target</span> &nbsp;|&nbsp;
  <span style="color:#38bdf8">④ Path A prediction</span> &nbsp;|&nbsp;
  <span style="color:#ef4444">⑤ Error ×5</span>
</div>

{''.join(sections)}

</body>
</html>'''
    (out_dir / 'viewer.html').write_text(viewer, encoding='utf-8')
    logger.info(f'viewer: {out_dir / "viewer.html"}')
    logger.info(f'summary: {out_dir / "summary.json"}')


if __name__ == '__main__':
    main()
