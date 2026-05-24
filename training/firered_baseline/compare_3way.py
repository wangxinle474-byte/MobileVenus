"""三方对比: v1 / v2 / v3 (或更多) per-action primary R + L1 像素 + 总览."""
from __future__ import annotations

import argparse
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
    PARAM_NAMES, ACTIONS, ACTION_TO_IDX, ACTION_PRIMARY_PARAM,
    FireRed7DModel, denormalize_t, build_data,
)


def load_image(path: Path, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def eval_ckpt(ckpt_path: str, val_s: list, device, image_size: int = 256,
              render_size: int = 256):
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

    # 按 action 收集 model_render vs FR L1, primary param R
    by_action = {a: {'L1_om': [], 'L1_mt': [], 'L1_oi': [], 'L1_it': [],
                     'L1_ot': [], 'pred_phys': [], 'inv_phys': []}
                 for a in ACTIONS}

    with torch.no_grad():
        for s in val_s:
            a = s['action']
            try:
                orig_render = load_image(Path(s['orig_path']),
                                         render_size).to(device)
                target_render = load_image(Path(s['target_path']),
                                            render_size).to(device)
            except Exception:
                continue

            orig_pil = Image.open(s['orig_path']).convert('RGB')
            x_in = tf(orig_pil).unsqueeze(0).to(device)
            a_oh = torch.zeros(1, len(ACTIONS), device=device)
            a_oh[0, ACTION_TO_IDX[a]] = 1.0
            pred_norm = model(x_in, a_oh)[0]
            P_pred = {p: float(denormalize_t(p, pred_norm[i].cpu()).item())
                      for i, p in enumerate(PARAM_NAMES)}

            P_inv = s['P_inferred']
            params_pred_t = {p: torch.tensor([P_pred[p]], device=device,
                                              dtype=torch.float32)
                             for p in PARAM_NAMES}
            params_inv_t = {p: torch.tensor([P_inv.get(p, 0.0)], device=device,
                                             dtype=torch.float32)
                            for p in PARAM_NAMES}
            model_render = apply_diff_isp(orig_render, params_pred_t)
            inv_render = apply_diff_isp(orig_render, params_inv_t)

            d = by_action[a]
            d['L1_om'].append(F.l1_loss(orig_render, model_render).item())
            d['L1_oi'].append(F.l1_loss(orig_render, inv_render).item())
            d['L1_ot'].append(F.l1_loss(orig_render, target_render).item())
            d['L1_mt'].append(F.l1_loss(model_render, target_render).item())
            d['L1_it'].append(F.l1_loss(inv_render, target_render).item())
            d['pred_phys'].append([P_pred[p] for p in PARAM_NAMES])
            d['inv_phys'].append([P_inv.get(p, 0.0) for p in PARAM_NAMES])

    # 汇总 per-action primary R + L1(model, FR)
    summary = {}
    overall_L1_mt, overall_L1_om = [], []
    for a in ACTIONS:
        d = by_action[a]
        if not d['L1_mt']:
            continue
        primary = ACTION_PRIMARY_PARAM[a]
        pi = PARAM_NAMES.index(primary)
        pred = np.array([row[pi] for row in d['pred_phys']])
        inv = np.array([row[pi] for row in d['inv_phys']])
        mae = float(np.mean(np.abs(pred - inv)))
        r = (float(np.corrcoef(pred, inv)[0, 1])
             if pred.std() > 1e-6 and inv.std() > 1e-6 else 0.0)
        summary[a] = {
            'primary': primary,
            'mae': mae,
            'r': r,
            'L1_mt_mean': float(np.mean(d['L1_mt'])),
            'L1_om_mean': float(np.mean(d['L1_om'])),
            'n': len(d['L1_mt']),
        }
        overall_L1_mt.extend(d['L1_mt'])
        overall_L1_om.extend(d['L1_om'])
    summary['__overall__'] = {
        'L1_mt_mean': float(np.mean(overall_L1_mt)),
        'L1_om_mean': float(np.mean(overall_L1_om)),
        'n': len(overall_L1_mt),
    }
    return summary, ckpt.get('val_loss', float('nan')), ckpt.get('epoch', -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+',
                    default=['checkpoints/firered_v1/best.pt',
                             'checkpoints/firered_v2/best.pt',
                             'checkpoints/firered_v3/best.pt'])
    ap.add_argument('--names', nargs='+',
                    default=['v1', 'v2', 'v3'])
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_s, val_s = build_data(
        Path(args.jsonl), args.val_ratio,
        ('A excellent', 'B good', 'C acceptable'), args.seed)

    results = {}
    for ckpt, name in zip(args.ckpts, args.names):
        if not Path(ckpt).exists():
            print(f'skip {name}: {ckpt} not found')
            continue
        print(f'\nEvaluating {name}: {ckpt}')
        results[name], val_loss, epoch = eval_ckpt(ckpt, val_s, device)
        print(f'  val_loss={val_loss:.4f}  Ep={epoch}')

    # 表格
    print(f'\n{"="*120}')
    print(f'三方对比表 (val 73 样本, per-action primary param)')
    print(f'{"="*120}')

    header = f'{"action":<12s}'
    for name in args.names:
        if name in results:
            header += f' | {name:>22s}'
    print(header)
    print('-' * 120)

    for a in ACTIONS:
        line = f'{a:<12s}'
        for name in args.names:
            if name not in results:
                continue
            if a not in results[name]:
                line += f' | {"N/A":>22s}'
                continue
            v = results[name][a]
            line += (f' | MAE{v["mae"]:>6.1f} R{v["r"]:>+5.2f} '
                     f'L1m{v["L1_mt_mean"]:.3f}')
        print(line)

    print('-' * 120)
    line = f'{"OVERALL":<12s}'
    for name in args.names:
        if name not in results:
            continue
        ov = results[name].get('__overall__', {})
        if ov:
            line += (f' | L1(o,m)={ov["L1_om_mean"]:.4f} '
                     f'L1(m,FR)={ov["L1_mt_mean"]:.4f}     ')
    print(line)

    # 简要总结
    print(f'\n{"="*120}')
    print('总结: per-action primary R')
    print(f'{"="*120}')
    for a in ACTIONS:
        rs = []
        for name in args.names:
            if name in results and a in results[name]:
                rs.append(f'{name}={results[name][a]["r"]:+.2f}')
        if rs:
            print(f'  {a:<12s} {"  ".join(rs)}')

    print('\n总结: L1(model_render, FireRed_edit)  (越小越好)')
    for name in args.names:
        if name not in results:
            continue
        ov = results[name].get('__overall__', {})
        print(f'  {name:<5s} L1(m,FR) = {ov["L1_mt_mean"]:.4f}')


if __name__ == '__main__':
    main()
