"""Per-action eval for Path Z (ImageDomainResUNet) checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from models.image_domain_resunet import ImageDomainResUNet
from training.firered_baseline.train_lut import ACTIONS, LUTDataset, build_data, set_actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--split_seed', type=int, default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ca = ckpt['args']
    ckpt_actions = ca.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)
    image_size = ca.get('image_size', 256)
    base_ch = ca.get('base_ch', 48)
    delta_scale = ca.get('delta_scale', 1.0)

    model = ImageDomainResUNet(
        in_ch=3, base_ch=base_ch,
        n_actions=len(ACTIONS), delta_scale=delta_scale,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    split_seed = args.split_seed if args.split_seed is not None \
        else ca.get('split_seed', ca.get('seed', 42))
    _, val_s = build_data(
        Path(args.jsonl), ca.get('val_ratio', 0.2),
        ('A excellent', 'B good', 'C acceptable'), split_seed,
        action_filter=ACTIONS)
    ds = LUTDataset(val_s, image_size, is_train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    per_action = {a: [] for a in ACTIONS}
    with torch.no_grad():
        for batch in loader:
            refined = model(batch['orig'].to(device),
                            batch['action_onehot'].to(device))
            target = batch['target'].to(device)
            mse_per = ((refined - target) ** 2).mean(dim=[1, 2, 3]).clamp(min=1e-10)
            psnr_per = (-10.0 * torch.log10(mse_per)).cpu().tolist()
            for action, psnr in zip(batch['action'], psnr_per):
                per_action[action].append(float(psnr))

    all_vals = [v for lst in per_action.values() for v in lst]
    overall = sum(all_vals) / max(len(all_vals), 1)

    print(f'\ncheckpoint: {args.ckpt}')
    print(f'best_val_psnr={ckpt.get("val_psnr", 0):.2f}dB @ Ep{ckpt.get("epoch", "?")}')
    print(f'params: base_ch={base_ch}, delta_scale={delta_scale}')
    print(f'eval_overall={overall:.2f}dB')

    print('\nPer-action PSNR:')
    for action, values in sorted(per_action.items(),
                                 key=lambda item: sum(item[1]) / max(len(item[1]), 1)):
        if not values:
            continue
        vals = sorted(values)
        mean = sum(vals) / len(vals)
        p10 = vals[max(0, len(vals) // 10)]
        p90 = vals[min(len(vals) - 1, 9 * len(vals) // 10)]
        print(f'  {action:12s} n={len(vals):2d}  mean={mean:.2f}  '
              f'p10={p10:.2f}  p90={p90:.2f}')


if __name__ == '__main__':
    main()
