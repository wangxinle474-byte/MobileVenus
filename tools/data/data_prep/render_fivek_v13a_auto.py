"""全量 FiveK 数据集 v13a 自主决策 (z-score) 渲染.

Pipeline:
  1. 校准: 在 N 张样本上算每个 action 的 L1 delta 均值/std
  2. 对每张图: 跑全部 7 个 action, 选 max z-score 作为模型决策
  3. 保存 refined output + manifest JSONL

Output:
  E:/Data/dataset/fivek_v13a_auto/
    <stem>.png             (256x256, 8-bit RGB)
  E:/Data/dataset/fivek_v13a_auto/manifest.jsonl
    {"name": "...", "action": "wb", "z_score": 1.23, "all_z": {...}}

用法:
  D:\\anaconda\\envs\\Venus\\python.exe tools/data/data_prep/render_fivek_v13a_auto.py
  D:\\anaconda\\envs\\Venus\\python.exe tools/data/data_prep/render_fivek_v13a_auto.py --max 100  # 测试
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from models.firered_residual_refiner import FireRedResidualRefiner  # noqa: E402
from training.firered_baseline.train_lut import (  # noqa: E402
    ACTIONS, ACTION_TO_IDX, NamedCurvesPredictor, set_actions,
)

try:
    import torchvision.transforms as T
except ImportError:
    raise ImportError('需要 torchvision。用 Venus conda: '
                      'D:\\anaconda\\envs\\Venus\\python.exe')

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def load_v11d_base(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    ca = ckpt['args']
    if ca.get('actions') is not None:
        set_actions(ca['actions'])
    model = NamedCurvesPredictor(
        n_colors=ca.get('nc_n_colors', 3),
        n_control_points=ca.get('nc_n_control_points', 7),
        use_attention=ca.get('nc_use_attention', False),
        per_action_curves=ca.get('nc_per_action_curves', False),
        use_7d_anchor=ca.get('nc_use_7d_anchor', True),
        use_context=ca.get('nc_use_context', False),
        action_gated_context=ca.get('nc_action_gated_context', False),
        use_action_context=ca.get('nc_use_action_context', False),
        use_region_basis=ca.get('nc_use_region_basis', False),
        use_region_param_delta=ca.get('nc_use_region_param_delta', False),
        use_learned_cn=ca.get('nc_use_learned_cn', False),
        use_wb_head=ca.get('nc_use_wb_head', False),
        use_nilut_residual=ca.get('nc_use_nilut_residual', False),
        use_vera_renderer=ca.get('nc_use_vera_renderer', False),
        nilut_hidden=ca.get('nilut_hidden', 32),
        nilut_n_layers=ca.get('nilut_n_layers', 3),
        nilut_n_freq=ca.get('nilut_n_freq', 4),
        nilut_gate_init=ca.get('nilut_gate_init', 1.0),
        use_implicit_head=ca.get('use_implicit_head', False),
        implicit_head_base_ch=ca.get('implicit_head_base_ch', 32),
        implicit_head_gate_init=ca.get('implicit_head_gate_init', 0.0),
        image_size=ca.get('image_size', 256),
        n_actions=len(ACTIONS),
        dropout=ca.get('dropout', 0.5),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ca


def load_v13a_refiner(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    ca = ckpt['args']
    refiner = FireRedResidualRefiner(
        base_ch=ca.get('base_ch', 32),
        n_actions=len(ACTIONS),
        delta_scale=ca.get('delta_scale', 0.5),
    ).to(device)
    refiner.load_state_dict(ckpt['model_state_dict'], strict=True)
    refiner.eval()
    for p in refiner.parameters():
        p.requires_grad_(False)
    info = {'val_psnr': ckpt.get('val_psnr', 0),
            'epoch': ckpt.get('epoch', '?')}
    return refiner, info


def preprocess(img: Image.Image, image_size: int, device: torch.device):
    resize = T.Resize((image_size, image_size))
    to_tensor = T.ToTensor()
    norm = T.Normalize(NORM_MEAN, NORM_STD)
    img_r = resize(img)
    orig_t = to_tensor(img_r).unsqueeze(0).to(device)
    enc_t = norm(orig_t.squeeze(0)).unsqueeze(0).to(device)
    return enc_t, orig_t


def run_all_actions(base_model, refiner, enc_input, orig_t, device):
    """Run all 7 actions, return refined images + L1 deltas."""
    results = {}
    with torch.no_grad():
        for action in ACTIONS:
            a_idx = ACTION_TO_IDX[action]
            a_oh = torch.zeros(1, len(ACTIONS), dtype=torch.float32,
                               device=device)
            a_oh[0, a_idx] = 1.0
            base_out, _, _, _ = base_model(enc_input, orig_t, a_oh)
            refined, _ = refiner(orig_t, base_out, a_oh)
            l1_delta = (refined - orig_t).abs().mean().item()
            results[action] = {
                'refined': refined,
                'l1_delta': l1_delta,
            }
    return results


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255
           ).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt',
                    default='checkpoints/lut_v11d_firered_7actions_6537/best.pt')
    ap.add_argument('--refiner_ckpt',
                    default='checkpoints/v13a_firered_refine_7actions/best.pt')
    ap.add_argument('--jpeg_dir', default=r'E:\Data\dataset\fivek_jpeg')
    ap.add_argument('--out_dir', default=r'E:\Data\dataset\fivek_v13a_auto')
    ap.add_argument('--gt_json',
                    default='data/fivek_expert_abcde_params.json',
                    help='used to define the 5000 image stem list')
    ap.add_argument('--expert', default='C')
    ap.add_argument('--n_calib', type=int, default=200)
    ap.add_argument('--max', type=int, default=None,
                    help='limit number of images for testing')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print(f'[load] v11d base: {args.base_ckpt}')
    base_model, base_args = load_v11d_base(Path(args.base_ckpt), device)
    image_size = base_args.get('image_size', 256)
    print(f'[load] v13a refiner: {args.refiner_ckpt}')
    refiner, refiner_info = load_v13a_refiner(
        Path(args.refiner_ckpt), device)
    print(f'[load] refiner val_psnr={refiner_info["val_psnr"]:.2f}dB '
          f'image_size={image_size} actions={list(ACTIONS)}')

    # Get image stems from gt_json (Expert C subset = 5000 images)
    with open(args.gt_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    stems = [s['image_name'].rsplit('.', 1)[0]
             for s in data['samples']
             if s['expert'] == args.expert]
    stems = sorted(set(stems))  # dedup, deterministic order
    print(f'[data] Expert {args.expert}: {len(stems)} unique stems')

    # Filter to images that have JPEG
    jpeg_dir = Path(args.jpeg_dir)
    valid_stems = []
    for stem in stems:
        if (jpeg_dir / f'{stem}.jpg').exists():
            valid_stems.append(stem)
    print(f'[data] JPEG matches: {len(valid_stems)}')

    if args.max is not None:
        valid_stems = valid_stems[:args.max]
        print(f'[data] limited to first {args.max}')

    # ── Calibration ──
    import random
    rng = random.Random(args.seed)
    n_calib = min(args.n_calib, len(valid_stems))
    calib_stems = rng.sample(valid_stems, n_calib)
    print(f'\n[calib] Computing per-action L1 baseline on {n_calib} samples...')
    per_action_deltas = {a: [] for a in ACTIONS}
    t0 = time.time()
    for j, stem in enumerate(calib_stems):
        if (j + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'  calib [{j+1}/{n_calib}]  {elapsed:.1f}s')
        img = Image.open(jpeg_dir / f'{stem}.jpg').convert('RGB')
        enc_input, orig_t = preprocess(img, image_size, device)
        results = run_all_actions(base_model, refiner, enc_input, orig_t, device)
        for action, r in results.items():
            per_action_deltas[action].append(r['l1_delta'])

    action_stats = {}
    for action in ACTIONS:
        arr = np.array(per_action_deltas[action])
        action_stats[action] = {
            'mean': float(arr.mean()),
            'std': float(arr.std() + 1e-6),
        }
    print('[calib] Per-action L1 baseline:')
    for action in ACTIONS:
        s = action_stats[action]
        print(f'  {action:12s}  mean={s["mean"]:.4f}  std={s["std"]:.4f}')

    # Save calib for later reuse
    calib_path = out_dir / '_calibration.json'
    with open(calib_path, 'w', encoding='utf-8') as f:
        json.dump({
            'n_calib': n_calib,
            'seed': args.seed,
            'action_stats': action_stats,
            'base_ckpt': args.base_ckpt,
            'refiner_ckpt': args.refiner_ckpt,
        }, f, indent=2)

    # ── Full render loop ──
    print(f'\n[render] Processing {len(valid_stems)} images...')
    manifest_path = out_dir / 'manifest.jsonl'
    action_counts = {a: 0 for a in ACTIONS}
    t0 = time.time()
    skipped = 0

    with open(manifest_path, 'w', encoding='utf-8') as fout:
        for i, stem in enumerate(valid_stems):
            out_png = out_dir / f'{stem}.png'
            if out_png.exists():
                skipped += 1
                continue

            try:
                img = Image.open(jpeg_dir / f'{stem}.jpg').convert('RGB')
                enc_input, orig_t = preprocess(img, image_size, device)
                results = run_all_actions(base_model, refiner, enc_input,
                                          orig_t, device)

                # Compute z-scores
                all_z = {}
                for action, r in results.items():
                    s = action_stats[action]
                    all_z[action] = (r['l1_delta'] - s['mean']) / s['std']

                # Pick action with max z-score
                model_action = max(all_z.items(), key=lambda kv: kv[1])[0]
                model_z = all_z[model_action]
                action_counts[model_action] += 1

                # Save refined output
                refined = results[model_action]['refined']
                tensor_to_pil(refined).save(out_png, format='PNG',
                                            optimize=True)

                # Manifest entry
                fout.write(json.dumps({
                    'name': stem,
                    'action': model_action,
                    'z_score': float(model_z),
                    'l1_delta': float(results[model_action]['l1_delta']),
                    'all_z': {a: float(z) for a, z in all_z.items()},
                    'all_l1': {a: float(r['l1_delta'])
                               for a, r in results.items()},
                }) + '\n')

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(valid_stems) - i - 1) / rate
                    dist_str = ' '.join(
                        f'{a[:3]}={action_counts[a]}' for a in ACTIONS)
                    print(f'  [{i+1}/{len(valid_stems)}] '
                          f'rate={rate:.1f}/s eta={eta/60:.1f}min  '
                          f'[{dist_str}]')

            except Exception as e:
                print(f'  [ERROR] {stem}: {e}')
                continue

    total_elapsed = time.time() - t0
    print(f'\n[done] processed {len(valid_stems) - skipped} images '
          f'(skipped {skipped} existing) in {total_elapsed/60:.1f} min')
    print(f'Action distribution:')
    for a in ACTIONS:
        pct = 100 * action_counts[a] / max(1, len(valid_stems) - skipped)
        print(f'  {a:12s}  {action_counts[a]:5d}  ({pct:.1f}%)')

    # Summary file
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_images': len(valid_stems),
            'rendered': len(valid_stems) - skipped,
            'action_counts': action_counts,
            'action_stats': action_stats,
            'elapsed_min': total_elapsed / 60,
            'base_ckpt': args.base_ckpt,
            'refiner_ckpt': args.refiner_ckpt,
            'refiner_val_psnr': refiner_info['val_psnr'],
            'image_size': image_size,
        }, f, indent=2)
    print(f'[done] manifest: {manifest_path}')
    print(f'[done] summary: {summary_path}')
    print(f'[done] output: {out_dir}')


if __name__ == '__main__':
    main()
