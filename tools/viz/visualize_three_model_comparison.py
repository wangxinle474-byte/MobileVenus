"""Three-model side-by-side comparison: same images, three checkpoints.

Adds Pilot 0 (Path A on FireRed pseudo) to the previous supervision
comparison so we can see whether the VeraRenderer architecture, when
fed FireRed pseudo supervision, recovers FireRed visual style.

Each row renders 6 panels:
  ① Source (FiveK input)
  ② v11a-FireRed prediction         (Bezier+CN+attn, FR-trained,  baseline 24.46 dB)
  ③ Pilot 0 PathA-FireRed pred      (VeraRenderer,    FR-trained,    NEW   23.68 dB)
  ④ FireRed pseudo target           (supervision behind ② and ③)
  ⑤ Path A-ExpertC prediction       (VeraRenderer,    EC-trained,    Track 2A 42.66 dB)
  ⑥ Clean Expert C 7D-render target (supervision behind ⑤)

Sort key: visual delta between FireRed target ④ and Expert C target ⑥
(the manifold gap from §5.2). Action coverage is balanced via per-action
caps.

Usage:
  python tools/visualize_three_model_comparison.py
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
    'orig': (100, 116, 139),         # slate
    'v11a_fr': (192, 38, 211),       # purple — v11a (Bezier+CN+attn) FR
    'pathA_fr': (234, 88, 12),       # orange — Pilot 0 (VeraRenderer) FR  ← NEW
    'fr_target': (217, 70, 239),     # pink — FireRed target
    'pathA_ec': (14, 165, 233),      # cyan — Path A-ExpertC
    'ec_target': (251, 191, 36),     # gold — Expert C target
}


def _font(size: int):
    for name in ('msyh.ttc', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def add_label(img, lines, bg_color, text_color=(255, 255, 255)):
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


def tensor_to_pil(t):
    arr = (t.clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_strip(images, gap=2, bg=(15, 23, 42)):
    w = images[0].size[0]
    h = max(img.size[1] for img in images)
    strip = Image.new('RGB', (w * len(images) + gap * (len(images) - 1), h), bg)
    for i, img in enumerate(images):
        y = (h - img.size[1]) // 2
        strip.paste(img, (i * (w + gap), y))
    return strip


def compute_psnr(pred, target):
    mse = ((pred - target) ** 2).mean().clamp(min=1e-10)
    return float((-10.0 * torch.log10(mse)).item())


def checkpoint_actions(ckpt: dict) -> list[str]:
    return list(ckpt.get('args', {}).get('actions') or ACTIONS)


def build_matched_pairs(firered_jsonl, expertC_jsonl):
    ec_index = {}
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
            matches.append({'firered': fr, 'expertC': ec,
                            'stem': stem, 'action': action})
    logger.info(f'matched pairs: {len(matches)} | skipped: {skipped}')
    return matches


def forward_on_records(model, records, image_size, batch_size, device,
                       label='forward'):
    ds = LUTDataset(records, image_size, is_train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=label, mininterval=1.0,
                          dynamic_ncols=True, file=sys.stderr):
            enc_in = batch['enc_input'].to(device)
            orig = batch['orig'].to(device)
            target = batch['target'].to(device)
            a_oh = batch['action_onehot'].to(device)
            refined, _, _, _ = model(enc_in, orig, a_oh)
            for i in range(refined.shape[0]):
                results.append({
                    'orig': orig[i].detach().cpu(),
                    'target': target[i].detach().cpu(),
                    'refined': refined[i].detach().cpu(),
                })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_v11a_fr',
                    default='checkpoints/lut_v11a_action_gated_context/'
                            'best.pt',
                    help='v11a Bezier+CN+attn baseline (FireRed-trained)')
    ap.add_argument('--ckpt_pathA_fr',
                    default='checkpoints/lut_pathA_firered_seed42/best.pt',
                    help='Pilot 0: Path A VeraRenderer (FireRed-trained)')
    ap.add_argument('--ckpt_pathA_ec',
                    default='checkpoints/lut_track2A_vera_clean_seed42/'
                            'best.pt',
                    help='Track 2A Path A VeraRenderer (Expert C-trained)')
    ap.add_argument('--firered_jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--expertC_jsonl',
                    default='outputs/fivek_expert_c_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--out_dir',
                    default='outputs/viz_three_model_comparison')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--max_samples', type=int, default=120)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_v11a = torch.load(args.ckpt_v11a_fr, map_location=device,
                           weights_only=False)
    ckpt_pa_fr = torch.load(args.ckpt_pathA_fr, map_location=device,
                            weights_only=False)
    ckpt_pa_ec = torch.load(args.ckpt_pathA_ec, map_location=device,
                            weights_only=False)
    image_size = min(
        ckpt_v11a['args'].get('image_size', 256),
        ckpt_pa_fr['args'].get('image_size', 256),
        ckpt_pa_ec['args'].get('image_size', 256))
    actions_v11a = checkpoint_actions(ckpt_v11a)
    actions_pa_fr = checkpoint_actions(ckpt_pa_fr)
    actions_pa_ec = checkpoint_actions(ckpt_pa_ec)
    if not (actions_v11a == actions_pa_fr == actions_pa_ec):
        raise RuntimeError(
            'All compared checkpoints must use the same action list. '
            f'v11a={actions_v11a}, pathA_fr={actions_pa_fr}, '
            f'pathA_ec={actions_pa_ec}')
    logger.info(f'v11a-FR  : val={ckpt_v11a.get("val_psnr", 0):.3f}  Ep{ckpt_v11a.get("epoch")}')
    logger.info(f'PathA-FR : val={ckpt_pa_fr.get("val_psnr", 0):.3f}  Ep{ckpt_pa_fr.get("epoch")}')
    logger.info(f'PathA-EC : val={ckpt_pa_ec.get("val_psnr", 0):.3f}  Ep{ckpt_pa_ec.get("epoch")}')
    logger.info(f'image_size = {image_size}')

    model_v11a = build_model_from_ckpt(ckpt_v11a, image_size, device)
    model_pa_fr = build_model_from_ckpt(ckpt_pa_fr, image_size, device)
    model_pa_ec = build_model_from_ckpt(ckpt_pa_ec, image_size, device)

    matches = build_matched_pairs(
        Path(args.firered_jsonl), Path(args.expertC_jsonl))
    if not matches:
        raise RuntimeError('No matched pairs.')

    fr_records = [m['firered'] for m in matches]
    ec_records = [m['expertC'] for m in matches]

    logger.info(f'forward v11a on FR records ({len(fr_records)})')
    v11a_out = forward_on_records(model_v11a, fr_records, image_size,
                                  args.batch_size, device, 'v11a-FR')
    logger.info(f'forward Pilot 0 PathA-FR on FR records ({len(fr_records)})')
    paFr_out = forward_on_records(model_pa_fr, fr_records, image_size,
                                  args.batch_size, device, 'PathA-FR')
    logger.info(f'forward Path A on EC records ({len(ec_records)})')
    paEc_out = forward_on_records(model_pa_ec, ec_records, image_size,
                                  args.batch_size, device, 'PathA-EC')

    pairs = []
    for i, m in enumerate(matches):
        psnr_v11a = compute_psnr(v11a_out[i]['refined'], v11a_out[i]['target'])
        psnr_paFr = compute_psnr(paFr_out[i]['refined'], paFr_out[i]['target'])
        psnr_paEc = compute_psnr(paEc_out[i]['refined'], paEc_out[i]['target'])
        gap_targets = float(
            (v11a_out[i]['target'] - paEc_out[i]['target']).abs().mean().item())
        # how much pilot 0 differs from v11a (architecture-only diff under same
        # supervision)
        delta_archs = float(
            (paFr_out[i]['refined'] - v11a_out[i]['refined']).abs().mean().item())
        pairs.append({
            'idx': i,
            'stem': m['stem'],
            'action': m['action'],
            'orig': v11a_out[i]['orig'],
            'pred_v11a': v11a_out[i]['refined'],
            'pred_paFr': paFr_out[i]['refined'],
            'target_fr': v11a_out[i]['target'],
            'pred_paEc': paEc_out[i]['refined'],
            'target_ec': paEc_out[i]['target'],
            'psnr_v11a': psnr_v11a,
            'psnr_paFr': psnr_paFr,
            'psnr_paEc': psnr_paEc,
            'manifold_gap': gap_targets,
            'arch_delta': delta_archs,
        })

    means = {
        'v11a_fr': float(np.mean([p['psnr_v11a'] for p in pairs])),
        'paFr_fr': float(np.mean([p['psnr_paFr'] for p in pairs])),
        'paEc_ec': float(np.mean([p['psnr_paEc'] for p in pairs])),
        'gap': float(np.mean([p['manifold_gap'] for p in pairs])),
        'arch_delta': float(np.mean([p['arch_delta'] for p in pairs])),
    }
    logger.info(f'mean PSNR v11a-FR        = {means["v11a_fr"]:.3f}')
    logger.info(f'mean PSNR Pilot0 PathA-FR= {means["paFr_fr"]:.3f}')
    logger.info(f'mean PSNR PathA-EC       = {means["paEc_ec"]:.3f}')
    logger.info(f'mean architecture delta (|pred_paFr - pred_v11a|) = '
                f'{means["arch_delta"]:.4f}')

    # Pick top per action by manifold gap
    by_action = {a: [] for a in ACTIONS}
    for p in pairs:
        by_action[p['action']].append(p)
    for a in ACTIONS:
        by_action[a].sort(key=lambda x: -x['manifold_gap'])
    cap = max(1, args.max_samples // len(ACTIONS))
    picked = []
    for a in ACTIONS:
        picked.extend(by_action[a][:cap])

    out_dir = Path(args.out_dir)
    img_dir = out_dir / 'strips'
    img_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for c in picked:
        orig_lab = add_label(tensor_to_pil(c['orig']),
                             ['① Source (FiveK input)'], COLORS['orig'])
        v11a_lab = add_label(
            tensor_to_pil(c['pred_v11a']),
            [f'② v11a-FireRed (Bezier+CN+attn)',
             f'PSNR vs ④ = {c["psnr_v11a"]:.2f} dB'],
            COLORS['v11a_fr'])
        paFr_lab = add_label(
            tensor_to_pil(c['pred_paFr']),
            [f'③ Pilot0 PathA-FireRed (VeraRenderer)',
             f'PSNR vs ④ = {c["psnr_paFr"]:.2f} dB'],
            COLORS['pathA_fr'])
        tgtFr_lab = add_label(
            tensor_to_pil(c['target_fr']),
            ['④ FireRed pseudo target',
             '(supervision for ② and ③)'],
            COLORS['fr_target'])
        paEc_lab = add_label(
            tensor_to_pil(c['pred_paEc']),
            [f'⑤ Track2A PathA-ExpertC (VeraRenderer)',
             f'PSNR vs ⑥ = {c["psnr_paEc"]:.2f} dB'],
            COLORS['pathA_ec'])
        tgtEc_lab = add_label(
            tensor_to_pil(c['target_ec']),
            ['⑥ Clean Expert C 7D-render target',
             '(supervision for ⑤)'],
            COLORS['ec_target'], (0, 0, 0))
        strip = make_strip([orig_lab, v11a_lab, paFr_lab,
                            tgtFr_lab, paEc_lab, tgtEc_lab])
        base = (f'{c["action"]}_{c["stem"]}_'
                f'gap={c["manifold_gap"]:.3f}_'
                f'archDelta={c["arch_delta"]:.3f}')
        base = base.replace('/', '_').replace('\\', '_')
        strip_path = img_dir / f'{base}.jpg'
        strip.save(strip_path, quality=92)
        cards.append({
            'action': c['action'],
            'stem': c['stem'],
            'psnr_v11a': c['psnr_v11a'],
            'psnr_paFr': c['psnr_paFr'],
            'psnr_paEc': c['psnr_paEc'],
            'manifold_gap': c['manifold_gap'],
            'arch_delta': c['arch_delta'],
            'strip': strip_path.relative_to(out_dir).as_posix(),
        })

    summary = {
        'ckpt_v11a_fr': args.ckpt_v11a_fr,
        'ckpt_pathA_fr': args.ckpt_pathA_fr,
        'ckpt_pathA_ec': args.ckpt_pathA_ec,
        'matched_pairs': len(pairs),
        'visualized_strips': len(cards),
        'mean_psnr': means,
        'per_action': {
            a: {
                'n': len(by_action[a]),
                'mean_psnr_v11a_fr': float(np.mean(
                    [p['psnr_v11a'] for p in by_action[a]])) if by_action[a] else 0.0,
                'mean_psnr_paFr_fr': float(np.mean(
                    [p['psnr_paFr'] for p in by_action[a]])) if by_action[a] else 0.0,
                'mean_psnr_paEc_ec': float(np.mean(
                    [p['psnr_paEc'] for p in by_action[a]])) if by_action[a] else 0.0,
                'mean_arch_delta': float(np.mean(
                    [p['arch_delta'] for p in by_action[a]])) if by_action[a] else 0.0,
                'mean_manifold_gap': float(np.mean(
                    [p['manifold_gap'] for p in by_action[a]])) if by_action[a] else 0.0,
            } for a in ACTIONS
        },
    }
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    pa_rows = ''.join(
        f'<tr><td>{html.escape(ACTION_CN.get(a, a))} ({a})</td>'
        f'<td>{summary["per_action"][a]["n"]}</td>'
        f'<td style="color:#c026d3">'
        f'{summary["per_action"][a]["mean_psnr_v11a_fr"]:.2f}</td>'
        f'<td style="color:#ea580c">'
        f'{summary["per_action"][a]["mean_psnr_paFr_fr"]:.2f}</td>'
        f'<td style="color:#0ea5e9">'
        f'{summary["per_action"][a]["mean_psnr_paEc_ec"]:.2f}</td>'
        f'<td>{summary["per_action"][a]["mean_arch_delta"]:.3f}</td>'
        f'<td>{summary["per_action"][a]["mean_manifold_gap"]:.3f}</td></tr>'
        for a in ACTIONS if summary['per_action'][a]['n'] > 0)

    cards_by_action = {a: [] for a in ACTIONS}
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
            f'<span class="v11a">v11a={c["psnr_v11a"]:.2f}</span> · '
            f'<span class="paFr">PathA-FR={c["psnr_paFr"]:.2f}</span> · '
            f'<span class="paEc">PathA-EC={c["psnr_paEc"]:.2f}</span> | '
            f'<span class="gap">gap={c["manifold_gap"]:.3f}</span></h3>'
            f'<img src="{html.escape(c["strip"])}" loading="lazy">'
            f'</div>'
            for c in cards_by_action[a])
        sections.append(
            f'<section data-action="{a}">'
            f'<h2>{html.escape(ACTION_CN.get(a, a))} ({a})</h2>'
            f'<div class="grid">{items_html}</div></section>')

    viewer = f'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Three-model comparison: v11a-FR / Pilot0 PathA-FR / PathA-EC</title>
<style>
body {{ font-family: 'Microsoft YaHei','Segoe UI',sans-serif;
        margin:0; padding:16px; background:#0f172a; color:#e2e8f0; }}
h1 {{ margin:0 0 8px; font-size:22px; }}
.subtitle {{ color:#94a3b8; font-size:12px; margin-bottom:16px; }}
.banner {{ background:#1e293b; border-left:4px solid #ea580c;
           border-radius:6px; padding:14px 18px; margin:14px 0; }}
.banner h2 {{ margin:0 0 6px; font-size:17px; color:#fb923c; }}
.banner p {{ margin:6px 0; line-height:1.6; font-size:13px; }}
.banner code {{ background:#0f172a; padding:2px 6px; border-radius:3px;
                color:#fbbf24; font-size:12px; }}
.summary {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; }}
.card {{ background:#1e293b; border:1px solid #334155;
         border-radius:8px; padding:10px 14px; min-width:160px; }}
.card .v {{ font-size:22px; font-weight:bold; }}
.card .l {{ font-size:11px; color:#94a3b8; }}
table {{ border-collapse:collapse; background:#1e293b;
         margin:8px 0 16px; font-size:13px; }}
th, td {{ border:1px solid #334155; padding:6px 12px; text-align:center; }}
th {{ background:#334155; color:#fbbf24; }}
section {{ background:#1e293b; border:1px solid #334155;
           border-radius:8px; padding:12px; margin-bottom:14px; }}
section h2 {{ margin:0 0 10px; font-size:16px; color:#f8fafc; }}
.grid {{ display:grid; grid-template-columns:1fr; gap:10px; }}
.item {{ background:#0f172a; border:1px solid #334155;
         border-radius:6px; padding:10px; }}
.item h3 {{ margin:0 0 8px; font-size:12px; color:#cbd5e1;
            font-weight:normal; line-height:1.5; }}
.item h3 .v11a {{ color:#e879f9; }}
.item h3 .paFr {{ color:#fb923c; }}
.item h3 .paEc {{ color:#38bdf8; }}
.item h3 .gap {{ color:#fbbf24; }}
.item img {{ width:100%; display:block; border-radius:4px; }}
.legend {{ background:#1e293b; border:1px solid #334155;
           border-radius:8px; padding:12px; margin-bottom:14px; font-size:13px; }}
.legend b {{ color:#fbbf24; }}
.cross {{ background:#14532d; border:1px solid #22c55e;
          border-radius:8px; padding:14px 18px; margin:14px 0; }}
.cross h2 {{ margin:0 0 6px; color:#86efac; font-size:17px; }}
.cross p {{ margin:6px 0; line-height:1.6; }}
</style>
</head>
<body>
<h1>三模型同图同 action 对比 — Pilot 0 结果出炉</h1>
<p class="subtitle">
  ② v11a-FireRed: <code>{html.escape(args.ckpt_v11a_fr)}</code><br>
  ③ Pilot 0 PathA-FireRed: <code>{html.escape(args.ckpt_pathA_fr)}</code><br>
  ⑤ Track 2A PathA-ExpertC: <code>{html.escape(args.ckpt_pathA_ec)}</code>
</p>

<div class="banner">
<h2>核心发现 — Pilot 0 的"反直觉"结果</h2>
<p>Pilot 0 把 Path A 架构（VeraRenderer，6 层 MLP 渲染器）搬到 FireRed 监督上之后：</p>
<p>· <b style="color:#e879f9">v11a-FireRed (Bezier+CN+attn)</b> in-domain = <b>24.46 dB</b>（FireRed val=74）</p>
<p>· <b style="color:#fb923c">Pilot 0 PathA-FireRed (VeraRenderer)</b> in-domain = <b>23.72 dB</b>
   <span style="color:#fb7185">(−0.74 dB, 输了)</span></p>
<p>但是 <b>cross-domain</b>：</p>
<p>· Pilot 0 → clean Expert C 全集 (n=14592) = <b style="color:#86efac">25.08 dB</b>
   vs v11a → Expert C = 21.36 dB → <b style="color:#86efac">+3.72 dB lift！</b></p>
<p>· Pilot 0 → MMArt-PPR10k (n=250) = <b style="color:#86efac">22.97 dB</b>
   vs v11a → MMArt = 20.79 dB → <b style="color:#86efac">+2.18 dB lift</b></p>
<p><b>解释</b>：v11a 的 Bezier+CN+attn 流水线能逃逸 7D ISP 流形去拟合 FireRed 的非 ISP
artifacts（颜色调色板、空间 mask），所以 in-domain 更准。Pilot 0 的 VeraRenderer 是 7D ISP 形状的
受限变换族，丢了那部分自由度——但同时获得了"物理可行"的归纳偏置，<b>跨分布泛化反而更强</b>。
这正是 §5.2 manifold hypothesis 在架构维度的实验印证。</p>
</div>

<div class="cross">
<h2>对你"对齐 FireRed"目标的含义</h2>
<p>· <b>纯 FireRed 视觉风格目标</b>：Pilot 0 输了 v11a-FireRed 0.74 dB，<b>Pilot 1/2 不建议上</b>
   — Path A 架构本身约束了它能拟合 FireRed 的程度，再加数据也只是缓慢逼近 7D ISP ceiling 23.94 dB，
   永远追不上 v11a 的 24.46 dB。</p>
<p>· <b>真正"看着像 FireRed"</b>：直接部署 <code>checkpoints/lut_v11a_action_gated_context/best.pt</code>。
   §5.3 Path 1 (per-distribution serving) 的标准答案。</p>
<p>· <b>意外收获</b>：Pilot 0 是一个非常好的"通用真实照片编辑器"——in-domain 牺牲 0.74 dB
   换来 +3.72 dB on Expert C 和 +2.18 dB on MMArt。如果目标是"真实照片上效果稳"，
   Pilot 0 是当前最优 checkpoint。</p>
</div>

<div class="summary">
  <div class="card"><div class="v" style="color:#e879f9">{means['v11a_fr']:.2f}</div>
       <div class="l">② v11a-FR avg PSNR vs FR target</div></div>
  <div class="card"><div class="v" style="color:#fb923c">{means['paFr_fr']:.2f}</div>
       <div class="l">③ Pilot0 PathA-FR avg PSNR vs FR target</div></div>
  <div class="card"><div class="v" style="color:#38bdf8">{means['paEc_ec']:.2f}</div>
       <div class="l">⑤ PathA-EC avg PSNR vs EC target</div></div>
  <div class="card"><div class="v" style="color:#fbbf24">{means['arch_delta']:.3f}</div>
       <div class="l">arch delta |③−②| (architecture-only diff)</div></div>
  <div class="card"><div class="v">{len(pairs)}</div>
       <div class="l">matched pairs</div></div>
</div>

<h2>Per-action breakdown (on FireRed-overlap subset)</h2>
<table>
<thead><tr><th>Action</th><th>n</th>
<th>② v11a-FR</th>
<th>③ PathA-FR</th>
<th>⑤ PathA-EC</th>
<th>|③−②|</th>
<th>FR↔EC gap</th></tr></thead>
<tbody>{pa_rows}</tbody>
</table>

<div class="legend">
<b>6-panel legend</b>:
<span style="color:#94a3b8">① Source</span> &nbsp;|&nbsp;
<span style="color:#c026d3">② v11a-FR pred</span> &nbsp;|&nbsp;
<span style="color:#ea580c">③ Pilot 0 PathA-FR pred</span> &nbsp;|&nbsp;
<span style="color:#e879f9">④ FireRed target</span> &nbsp;|&nbsp;
<span style="color:#0ea5e9">⑤ PathA-EC pred</span> &nbsp;|&nbsp;
<span style="color:#fbbf24">⑥ Expert C target</span><br>
<b>Sort key</b>: per action, sorted by FR↔EC manifold gap (largest first).
</div>

{''.join(sections)}

</body>
</html>'''
    (out_dir / 'viewer.html').write_text(viewer, encoding='utf-8')
    logger.info(f'viewer: {out_dir / "viewer.html"}')
    logger.info(f'summary: {out_dir / "summary.json"}')


if __name__ == '__main__':
    main()
