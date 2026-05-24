"""Master comparison viewer: same architecture family under two different
supervisions, on the same images.

For each (image, action) where both FireRed pseudo and clean Expert C
records exist, renders a 5-panel strip:

  ① Source (FiveK input)
  ② v11a (FireRed-trained, baseline 24.46 dB)   prediction
  ③ FireRed pseudo target  (the supervision behind ②)
  ④ Path A (Expert-C-trained, 42.66 dB)         prediction
  ⑤ Clean Expert C 7D-render target  (the supervision behind ④)

So you can read each row as: "what FireRed supervision produces (②) vs
what Expert C supervision produces (④) on the *same* source."

Path A's higher in-domain PSNR is on a *subtle* target distribution;
v11a's lower PSNR is on a *dramatic* target distribution. The visual
gap between ② and ④ is the manifold gap of §5.2 made concrete.

Usage:
  python tools/visualize_supervision_comparison.py
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
    'firered_pred': (192, 38, 211),    # purple — FireRed-model prediction
    'firered_tgt': (217, 70, 239),     # lighter purple — FireRed target
    'pathA_pred': (14, 165, 233),      # cyan — Path A prediction
    'pathA_tgt': (251, 191, 36),       # gold — Expert C target
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


def add_label(img: Image.Image, lines, bg_color,
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


def make_strip(images, gap: int = 2, bg=(15, 23, 42)) -> Image.Image:
    w = images[0].size[0]
    h = max(img.size[1] for img in images)
    strip = Image.new('RGB', (w * len(images) + gap * (len(images) - 1), h),
                      bg)
    for i, img in enumerate(images):
        y = (h - img.size[1]) // 2
        strip.paste(img, (i * (w + gap), y))
    return strip


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = ((pred - target) ** 2).mean().clamp(min=1e-10)
    return float((-10.0 * torch.log10(mse)).item())


def checkpoint_actions(ckpt: dict) -> list[str]:
    return list(ckpt.get('args', {}).get('actions') or ACTIONS)


def fmt_params(P: dict) -> str:
    parts = []
    for k in PARAM_KEYS:
        v = P.get(k, 0.0) or 0.0
        if k == 'white_balance':
            parts.append(f'WB={v:.0f}')
        else:
            parts.append(f'{k[:3]}={v:+.1f}')
    return ' '.join(parts)


def load_pseudo_target(target_path: str, size: int):
    p = Path(target_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return None
    img = Image.open(p).convert('RGB').resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def build_matched_pairs(firered_jsonl: Path, expertC_jsonl: Path
                        ) -> list:
    ec_index: dict[tuple[str, str], dict] = {}
    with open(expertC_jsonl, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            stem = extract_fivek_stem(r)
            action = r.get('action', '')
            ec_index[(stem, action)] = r
    logger.info(f'Expert C index: {len(ec_index)} (stem, action) entries')

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
            ft_full = Path(ft_path) if ft_path else None
            if ft_full and not ft_full.is_absolute():
                ft_full = PROJECT_ROOT / ft_full
            if not ft_full or not ft_full.exists():
                skipped['no_firered_target'] += 1
                continue
            matches.append({
                'firered': fr,
                'expertC': ec,
                'stem': stem,
                'action': action,
            })
    logger.info(f'matched pairs: {len(matches)} | skipped: {skipped}')
    return matches


def forward_on_records(model, records: list, image_size: int,
                       batch_size: int, device: torch.device) -> list:
    """Run inference on a list of records using LUTDataset transforms.
    Returns list of dicts with orig, target, refined tensors per record."""
    ds = LUTDataset(records, image_size, is_train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    results = []
    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='forward', mininterval=1.0,
                          dynamic_ncols=True, file=sys.stderr):
            enc_in = batch['enc_input'].to(device)
            orig = batch['orig'].to(device)
            target = batch['target'].to(device)
            a_oh = batch['action_onehot'].to(device)
            refined, _, _, _ = model(enc_in, orig, a_oh)
            B = refined.shape[0]
            for i in range(B):
                results.append({
                    'orig': orig[i].detach().cpu(),
                    'target': target[i].detach().cpu(),
                    'refined': refined[i].detach().cpu(),
                })
                idx += 1
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_firered',
                    default='checkpoints/lut_v11a_action_gated_context/'
                            'best.pt',
                    help='v11a (FireRed-trained) checkpoint')
    ap.add_argument('--ckpt_expertC',
                    default='checkpoints/lut_track2A_vera_clean_seed42/'
                            'best.pt',
                    help='Path A (Expert-C-trained) checkpoint')
    ap.add_argument('--firered_jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--expertC_jsonl',
                    default='outputs/fivek_expert_c_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--out_dir',
                    default='outputs/viz_supervision_comparison')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--max_samples', type=int, default=100)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load both checkpoints (use the smaller image_size for both forwards)
    ckpt_fr = torch.load(args.ckpt_firered, map_location=device,
                         weights_only=False)
    ckpt_ec = torch.load(args.ckpt_expertC, map_location=device,
                         weights_only=False)
    image_size = min(
        ckpt_fr['args'].get('image_size', 256),
        ckpt_ec['args'].get('image_size', 256))
    actions_fr = checkpoint_actions(ckpt_fr)
    actions_ec = checkpoint_actions(ckpt_ec)
    if actions_fr != actions_ec:
        raise RuntimeError(
            'Compared checkpoints must use the same action list. '
            f'firered={actions_fr}, expertC={actions_ec}')
    logger.info(f'FireRed model: {args.ckpt_firered}  val='
                f'{ckpt_fr.get("val_psnr", 0.0):.3f}  Ep{ckpt_fr.get("epoch")}')
    logger.info(f'Expert C model: {args.ckpt_expertC}  val='
                f'{ckpt_ec.get("val_psnr", 0.0):.3f}  Ep{ckpt_ec.get("epoch")}')
    logger.info(f'image_size = {image_size}')

    model_fr = build_model_from_ckpt(ckpt_fr, image_size, device)
    model_ec = build_model_from_ckpt(ckpt_ec, image_size, device)

    matches = build_matched_pairs(
        Path(args.firered_jsonl), Path(args.expertC_jsonl))
    if not matches:
        raise RuntimeError('No matched pairs.')

    # Forward FireRed model on FireRed records (PSNR vs FireRed target)
    # Forward Expert C model on Expert C records (PSNR vs Expert C target)
    fr_records = [m['firered'] for m in matches]
    ec_records = [m['expertC'] for m in matches]

    logger.info(f'forward v11a on FireRed records ({len(fr_records)}) ...')
    fr_out = forward_on_records(model_fr, fr_records, image_size,
                                args.batch_size, device)
    logger.info(f'forward Path A on Expert C records ({len(ec_records)}) ...')
    ec_out = forward_on_records(model_ec, ec_records, image_size,
                                args.batch_size, device)

    # Aggregate per-pair: pull orig from EC dataset (both share same source
    # image, so orig should be ≈ identical aside from transform ordering).
    pairs = []
    for i, m in enumerate(matches):
        psnr_fr = compute_psnr(fr_out[i]['refined'], fr_out[i]['target'])
        psnr_ec = compute_psnr(ec_out[i]['refined'], ec_out[i]['target'])
        # Visual delta between the two targets — the "manifold gap" magnitude
        gap = float((fr_out[i]['target'] - ec_out[i]['target']).abs().mean().item())
        pairs.append({
            'idx': i,
            'stem': m['stem'],
            'action': m['action'],
            'orig': ec_out[i]['orig'],   # use EC's loader copy
            'pred_fr': fr_out[i]['refined'],
            'target_fr': fr_out[i]['target'],
            'pred_ec': ec_out[i]['refined'],
            'target_ec': ec_out[i]['target'],
            'psnr_fr': psnr_fr,
            'psnr_ec': psnr_ec,
            'manifold_gap': gap,
            'firered_P': m['firered'].get('P_inferred', {}),
            'expertC_P': m['expertC'].get('P_inferred', {}),
        })

    mean_psnr_fr = float(np.mean([p['psnr_fr'] for p in pairs]))
    mean_psnr_ec = float(np.mean([p['psnr_ec'] for p in pairs]))
    mean_gap = float(np.mean([p['manifold_gap'] for p in pairs]))
    logger.info(f'mean PSNR v11a-on-FireRed = {mean_psnr_fr:.3f} dB')
    logger.info(f'mean PSNR PathA-on-ExpertC = {mean_psnr_ec:.3f} dB')
    logger.info(f'mean visual manifold gap (FR target vs EC target) = '
                f'{mean_gap:.4f}')

    # Pick top-N per action by manifold gap (most striking comparisons)
    by_action: dict[str, list] = {a: [] for a in ACTIONS}
    for p in pairs:
        by_action[p['action']].append(p)
    for a in ACTIONS:
        by_action[a].sort(key=lambda x: -x['manifold_gap'])
    cap_per_action = max(1, args.max_samples // len(ACTIONS))
    picked = []
    for a in ACTIONS:
        picked.extend(by_action[a][:cap_per_action])

    out_dir = Path(args.out_dir)
    img_dir = out_dir / 'strips'
    img_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for c in picked:
        orig_lab = add_label(
            tensor_to_pil(c['orig']),
            ['① Source (FiveK input)'],
            COLORS['orig'])
        pred_fr_lab = add_label(
            tensor_to_pil(c['pred_fr']),
            [f'② v11a (FireRed-trained) prediction',
             f'PSNR vs ③ = {c["psnr_fr"]:.2f} dB'],
            COLORS['firered_pred'])
        tgt_fr_lab = add_label(
            tensor_to_pil(c['target_fr']),
            ['③ FireRed pseudo target (dramatic AI edit)',
             fmt_params(c['firered_P'])],
            COLORS['firered_tgt'])
        pred_ec_lab = add_label(
            tensor_to_pil(c['pred_ec']),
            [f'④ Path A (Expert-C-trained) prediction',
             f'PSNR vs ⑤ = {c["psnr_ec"]:.2f} dB'],
            COLORS['pathA_pred'])
        tgt_ec_lab = add_label(
            tensor_to_pil(c['target_ec']),
            ['⑤ Clean Expert C target (subtle 7D render)',
             fmt_params(c['expertC_P'])],
            COLORS['pathA_tgt'], (0, 0, 0))
        strip = make_strip([orig_lab, pred_fr_lab, tgt_fr_lab,
                            pred_ec_lab, tgt_ec_lab])
        base = (f'{c["action"]}_{c["stem"]}_gap={c["manifold_gap"]:.3f}'
                f'_frPSNR={c["psnr_fr"]:.1f}_ecPSNR={c["psnr_ec"]:.1f}')
        base = base.replace('/', '_').replace('\\', '_')
        strip_path = img_dir / f'{base}.jpg'
        strip.save(strip_path, quality=92)
        cards.append({
            'action': c['action'],
            'stem': c['stem'],
            'psnr_fr': c['psnr_fr'],
            'psnr_ec': c['psnr_ec'],
            'manifold_gap': c['manifold_gap'],
            'strip': strip_path.relative_to(out_dir).as_posix(),
        })

    # Summary
    summary = {
        'ckpt_firered': args.ckpt_firered,
        'ckpt_expertC': args.ckpt_expertC,
        'matched_pairs': len(pairs),
        'visualized_strips': len(cards),
        'mean_psnr_v11a_on_firered': mean_psnr_fr,
        'mean_psnr_pathA_on_expertC': mean_psnr_ec,
        'mean_manifold_gap': mean_gap,
        'per_action': {
            a: {
                'n': len(by_action[a]),
                'mean_psnr_fr': float(np.mean(
                    [p['psnr_fr'] for p in by_action[a]])) if by_action[a] else 0.0,
                'mean_psnr_ec': float(np.mean(
                    [p['psnr_ec'] for p in by_action[a]])) if by_action[a] else 0.0,
                'mean_manifold_gap': float(np.mean(
                    [p['manifold_gap'] for p in by_action[a]])) if by_action[a] else 0.0,
            }
            for a in ACTIONS
        },
    }
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    # HTML viewer
    pa_rows = ''.join(
        f'<tr><td>{html.escape(ACTION_CN.get(a, a))} ({a})</td>'
        f'<td>{summary["per_action"][a]["n"]}</td>'
        f'<td style="color:#e879f9">'
        f'{summary["per_action"][a]["mean_psnr_fr"]:.2f}</td>'
        f'<td style="color:#38bdf8">'
        f'{summary["per_action"][a]["mean_psnr_ec"]:.2f}</td>'
        f'<td>{summary["per_action"][a]["mean_manifold_gap"]:.3f}</td></tr>'
        for a in ACTIONS if summary['per_action'][a]['n'] > 0)

    cards_by_action: dict[str, list] = {a: [] for a in ACTIONS}
    for c in cards:
        cards_by_action[c['action']].append(c)
    sections = []
    for a in ACTIONS:
        if not cards_by_action[a]:
            continue
        cards_by_action[a].sort(key=lambda x: -x['manifold_gap'])
        items_html = ''.join(
            f'<div class="item">'
            f'<h3>{html.escape(c["stem"])} | '
            f'<span class="fr">v11a-FireRed = {c["psnr_fr"]:.2f} dB</span> | '
            f'<span class="ec">Path A-ExpertC = {c["psnr_ec"]:.2f} dB</span> | '
            f'<span class="gap">manifold gap = {c["manifold_gap"]:.3f}</span></h3>'
            f'<img src="{html.escape(c["strip"])}" loading="lazy">'
            f'</div>'
            for c in cards_by_action[a])
        sections.append(
            f'<section data-action="{a}">'
            f'<h2>{html.escape(ACTION_CN.get(a, a))} ({a})</h2>'
            f'<div class="grid">{items_html}</div>'
            f'</section>')

    viewer = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Supervision distribution comparison</title>
<style>
body {{ font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
        margin: 0; padding: 16px; background: #0f172a; color: #e2e8f0; }}
h1 {{ margin: 0 0 8px; font-size: 22px; }}
.subtitle {{ color: #94a3b8; font-size: 13px; margin-bottom: 16px; }}
.banner {{ background: #1e293b; border-left: 4px solid #38bdf8;
           border-radius: 6px; padding: 14px 18px; margin: 14px 0; }}
.banner h2 {{ margin: 0 0 6px; font-size: 17px; color: #38bdf8; }}
.banner p {{ margin: 6px 0; line-height: 1.6; font-size: 13px; }}
.banner code {{ background: #0f172a; padding: 2px 6px; border-radius: 3px;
                 color: #fbbf24; font-size: 12px; }}
.banner .fr {{ color: #e879f9; font-weight: bold; }}
.banner .ec {{ color: #38bdf8; font-weight: bold; }}
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
.item h3 .fr {{ color: #e879f9; }}
.item h3 .ec {{ color: #38bdf8; }}
.item h3 .gap {{ color: #fbbf24; }}
.item img {{ width: 100%; display: block; border-radius: 4px; }}
.legend {{ background: #1e293b; border: 1px solid #334155;
           border-radius: 8px; padding: 12px; margin-bottom: 14px;
           font-size: 13px; }}
.legend b {{ color: #fbbf24; }}
.next {{ background: #14532d; border: 1px solid #22c55e;
         border-radius: 8px; padding: 14px 18px; margin: 14px 0; }}
.next h2 {{ margin: 0 0 6px; color: #86efac; font-size: 17px; }}
.next p {{ margin: 6px 0; line-height: 1.6; }}
.next code {{ background: #0f172a; padding: 2px 6px; border-radius: 3px;
              color: #fbbf24; font-size: 12px; }}
</style>
</head>
<body>
<h1>同源 / 同架构家族 / 两种监督分布对比</h1>
<p class="subtitle">
  ②③ v11a (FireRed-trained):
  <code>{html.escape(args.ckpt_firered)}</code> ·
  ④⑤ Path A (Expert-C-trained):
  <code>{html.escape(args.ckpt_expertC)}</code>
</p>

<div class="banner">
<h2>核心发现 — 你看到的"FireRed 更好"是正确的</h2>
<p>同样输入下，<span class="fr">② v11a-FireRed 输出</span>视觉冲击大、像 FireRed 的 AI 修图；
   <span class="ec">④ Path A-ExpertC 输出</span>克制、像专业摄影师的 subtle retouch。
   决定外观的<b>不是架构</b>（Path A 更强），而是<b>监督分布本身</b>。这正是论文
   §5.2 + §5.2b 五个 bridging interventions 全部证伪后得到的 manifold hypothesis 的
   视觉证据。</p>
<p>PSNR 对比看似 PathA-ExpertC (42 dB) 完胜 v11a-FireRed (24.6 dB)，但这是
   <b>在不同分布上算的</b>，本质不可比。视觉质量上 ② 更接近你和大多数人对"修图"
   的预期。</p>
</div>

<div class="next">
<h2>下一步建议 — 把 Path A 的架构红利搬到 FireRed 监督上</h2>
<p>你说"向 FireRed 对齐"的最优路径是 <b>Path A 架构 + FireRed 监督</b>，相当于在
   §5.3 Path 1 的 "per-distribution serving" 框架下，专门为 FireRed 分布训一个
   Path A checkpoint。预期：</p>
<p>· <b>视觉效果</b>：和 ② v11a-FireRed 等价或更好（同分布监督）<br>
   · <b>in-domain PSNR</b>：参考 Track 2 的 +0.78 dB lift，预期 24.46 → 25.0~25.5 dB<br>
   · <b>shadows action</b>：v11a 在这个 action 上卡在 26.35 dB；Path A 用 MLP 渲染器
     在 Expert C 数据上 shadows lift 了 +2.76，FireRed 数据上预期也有类似收益<br>
   · <b>训练时间</b>：80 epoch ≈ 5-6 h (参考 Track 2A run)，与之前完全可比</p>
<p>命令: <code>python training/firered_baseline/train_lut.py --jsonl outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl --epochs 80 --param_weight 0.05 --seed 42 --nc_use_vera_renderer --vera_latent_dim 64 --vera_hidden 128 --vera_n_layers 6 --vera_gate_init 1.0</code></p>
</div>

<div class="summary">
  <div class="card"><div class="v" style="color:#e879f9">
       {mean_psnr_fr:.2f}</div>
       <div class="l">v11a-FireRed avg PSNR (on FireRed targets)</div></div>
  <div class="card"><div class="v" style="color:#38bdf8">
       {mean_psnr_ec:.2f}</div>
       <div class="l">PathA-ExpertC avg PSNR (on Expert C targets)</div></div>
  <div class="card"><div class="v" style="color:#fbbf24">
       {mean_gap:.3f}</div>
       <div class="l">avg visual manifold gap (FR target vs EC target)</div></div>
  <div class="card"><div class="v">{len(pairs)}</div>
       <div class="l">matched pairs</div></div>
  <div class="card"><div class="v">{len(cards)}</div>
       <div class="l">strips rendered (top by manifold gap)</div></div>
</div>

<h2>Per-action breakdown</h2>
<table>
<thead><tr><th>Action</th><th>n matched</th>
<th>v11a-FireRed PSNR</th>
<th>PathA-ExpertC PSNR</th>
<th>manifold gap</th></tr></thead>
<tbody>{pa_rows}</tbody>
</table>

<div class="legend">
<b>5-panel legend</b>: <br>
<span style="color:#94a3b8">① Source</span> &nbsp;|&nbsp;
<span style="color:#c026d3">② v11a-FireRed prediction</span> &nbsp;|&nbsp;
<span style="color:#e879f9">③ FireRed pseudo target</span> &nbsp;|&nbsp;
<span style="color:#0ea5e9">④ Path A-ExpertC prediction</span> &nbsp;|&nbsp;
<span style="color:#fbbf24">⑤ Clean Expert C target</span><br>
<b>Sort key</b>: within each action, samples are ordered by manifold gap
(largest visual disagreement between ③ and ⑤ first).
</div>

{''.join(sections)}

</body>
</html>'''
    (out_dir / 'viewer.html').write_text(viewer, encoding='utf-8')
    logger.info(f'viewer: {out_dir / "viewer.html"}')
    logger.info(f'summary: {out_dir / "summary.json"}')


if __name__ == '__main__':
    main()
