"""诊断: 全 73 val 样本算 panel 间像素 L1 + 参数偏离统计.

输出:
- 平均 L1(orig, model_render):       小 → 模型几乎不改图 (under-prediction)
- 平均 L1(model_render, inv_render): 小 → 模型 ≈ inverse_fit (理想)
- 平均 L1(orig, FireRed_edit):       变化幅度的真值
- 平均 L1(model_render, FireRed_edit): 模型实际效果差距
- 每个 action 的 |P_pred - P_inferred| 在物理空间 / 归一化空间的统计
"""
from __future__ import annotations

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
    PARAM_NORM, denormalize_t, normalize_param, build_data,
)


def load_image(path: Path, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/firered_v1/best.pt')
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--render_size', type=int, default=256)
    args = ap.parse_args()
    ckpt_path = args.ckpt
    jsonl = args.jsonl
    render_size = args.render_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})
    model = FireRed7DModel(image_size=saved_args.get('image_size', 256),
                           dropout=saved_args.get('dropout', 0.3)).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    train_s, val_s = build_data(
        Path(jsonl), val_ratio=saved_args.get('val_ratio', 0.2),
        tier_filter=('A excellent', 'B good', 'C acceptable'),
        seed=saved_args.get('seed', 42))

    model_input_tf = T.Compose([
        T.Resize((saved_args.get('image_size', 256),) * 2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 累计统计
    by_action = {a: {'L1_orig_model': [], 'L1_model_inv': [],
                     'L1_orig_inv': [], 'L1_orig_target': [],
                     'L1_model_target': [],
                     'L1_inv_target': [],
                     'param_l1_phys': [], 'param_l1_norm': [],
                     'wb_pred': [], 'wb_inv': []}
                 for a in ACTIONS}

    with torch.no_grad():
        for s in val_s:
            a = s['action']
            try:
                orig_render = load_image(s['orig_path'], render_size).to(device)
                target_render = load_image(s['target_path'], render_size).to(device)
            except Exception as e:
                print(f'skip {s["source_image"]}: {e}')
                continue

            # 模型预测
            orig_pil = Image.open(s['orig_path']).convert('RGB')
            x_in = model_input_tf(orig_pil).unsqueeze(0).to(device)
            a_oh = torch.zeros(1, len(ACTIONS), device=device)
            a_oh[0, ACTION_TO_IDX[a]] = 1.0
            pred_norm = model(x_in, a_oh)[0]  # (7,)
            P_pred = {p: float(denormalize_t(p, pred_norm[i].cpu()).item())
                      for i, p in enumerate(PARAM_NAMES)}

            P_inv = s['P_inferred']

            # 渲染 model + inv
            params_pred_t = {p: torch.tensor([P_pred[p]], device=device,
                                              dtype=torch.float32)
                             for p in PARAM_NAMES}
            params_inv_t = {p: torch.tensor([P_inv.get(p, 0.0)],
                                             device=device, dtype=torch.float32)
                            for p in PARAM_NAMES}
            model_render = apply_diff_isp(orig_render, params_pred_t)
            inv_render = apply_diff_isp(orig_render, params_inv_t)

            # 像素 L1
            L1_orig_model = F.l1_loss(orig_render, model_render).item()
            L1_model_inv = F.l1_loss(model_render, inv_render).item()
            L1_orig_inv = F.l1_loss(orig_render, inv_render).item()
            L1_orig_target = F.l1_loss(orig_render, target_render).item()
            L1_model_target = F.l1_loss(model_render, target_render).item()
            L1_inv_target = F.l1_loss(inv_render, target_render).item()

            # 参数 L1
            param_l1_phys = float(np.mean([
                abs(P_pred[p] - (P_inv.get(p) or 0.0)) for p in PARAM_NAMES]))
            param_l1_norm = float(np.mean([
                abs(normalize_param(p, P_pred[p])
                    - normalize_param(p, P_inv.get(p) or 0.0))
                for p in PARAM_NAMES]))

            d = by_action[a]
            d['L1_orig_model'].append(L1_orig_model)
            d['L1_model_inv'].append(L1_model_inv)
            d['L1_orig_inv'].append(L1_orig_inv)
            d['L1_orig_target'].append(L1_orig_target)
            d['L1_model_target'].append(L1_model_target)
            d['L1_inv_target'].append(L1_inv_target)
            d['param_l1_phys'].append(param_l1_phys)
            d['param_l1_norm'].append(param_l1_norm)
            d['wb_pred'].append(P_pred['white_balance'])
            d['wb_inv'].append(P_inv.get('white_balance') or 0.0)

    # 汇总
    print(f'\n{"="*100}')
    print(f'诊断: {ckpt_path}  (val 73 张)')
    print(f'{"="*100}')
    print(f'{"action":<11s}{"n":>4s}'
          f'{"L1(orig,model)":>16s}{"L1(model,inv)":>15s}{"L1(orig,inv)":>14s}'
          f'{"L1(orig,FR)":>13s}{"L1(model,FR)":>14s}{"L1(inv,FR)":>12s}'
          f'{"|dP|phys":>10s}{"|dP|norm":>10s}'
          f'{"wb_pred":>10s}{"wb_inv":>10s}')
    overall = {k: [] for k in
               ['L1_orig_model', 'L1_model_inv', 'L1_orig_inv',
                'L1_orig_target', 'L1_model_target', 'L1_inv_target',
                'param_l1_phys', 'param_l1_norm']}
    for a in ACTIONS:
        d = by_action[a]
        if not d['L1_orig_model']:
            continue
        n = len(d['L1_orig_model'])
        for k in overall:
            overall[k].extend(d[k])
        print(f'{a:<11s}{n:>4d}'
              f'{np.mean(d["L1_orig_model"]):>16.4f}'
              f'{np.mean(d["L1_model_inv"]):>15.4f}'
              f'{np.mean(d["L1_orig_inv"]):>14.4f}'
              f'{np.mean(d["L1_orig_target"]):>13.4f}'
              f'{np.mean(d["L1_model_target"]):>14.4f}'
              f'{np.mean(d["L1_inv_target"]):>12.4f}'
              f'{np.mean(d["param_l1_phys"]):>10.2f}'
              f'{np.mean(d["param_l1_norm"]):>10.3f}'
              f'{np.mean(d["wb_pred"]):>10.0f}'
              f'{np.mean(d["wb_inv"]):>10.0f}')
    print('-' * 178)
    n_all = len(overall['L1_orig_model'])
    print(f'{"OVERALL":<11s}{n_all:>4d}'
          f'{np.mean(overall["L1_orig_model"]):>16.4f}'
          f'{np.mean(overall["L1_model_inv"]):>15.4f}'
          f'{np.mean(overall["L1_orig_inv"]):>14.4f}'
          f'{np.mean(overall["L1_orig_target"]):>13.4f}'
          f'{np.mean(overall["L1_model_target"]):>14.4f}'
          f'{np.mean(overall["L1_inv_target"]):>12.4f}'
          f'{np.mean(overall["param_l1_phys"]):>10.2f}'
          f'{np.mean(overall["param_l1_norm"]):>10.3f}')

    print('\n核心结论:')
    L1_om = np.mean(overall['L1_orig_model'])
    L1_oi = np.mean(overall['L1_orig_inv'])
    L1_mt = np.mean(overall['L1_model_target'])
    L1_it = np.mean(overall['L1_inv_target'])
    L1_ot = np.mean(overall['L1_orig_target'])
    print(f'  原图->FireRed_edit 平均改动幅度: L1 = {L1_ot:.4f}')
    print(f'  inv_fit 渲染相比 orig 改动幅度:   L1 = {L1_oi:.4f}  '
          f'({L1_oi/L1_ot*100:.1f}% of FR 改动)')
    print(f'  model 渲染相比 orig 改动幅度:    L1 = {L1_om:.4f}  '
          f'({L1_om/L1_ot*100:.1f}% of FR 改动) '
          f'<- 越小越说明 under-prediction')
    print(f'  inv_render vs FireRed:          L1 = {L1_it:.4f}  '
          f'(7D ISP 上限 — inv_fit 离 FR 多远)')
    print(f'  model_render vs FireRed:        L1 = {L1_mt:.4f}  '
          f'(模型最终效果离 FR 多远)')
    if L1_om < 0.6 * L1_oi:
        print(f'  *** UNDER-PREDICTION CONFIRMED *** model 改动只有 inv_fit '
              f'({L1_om/L1_oi*100:.0f}%)')


if __name__ == '__main__':
    main()
