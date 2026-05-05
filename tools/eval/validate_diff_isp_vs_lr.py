"""
\u91cf\u5316 diff_isp \u4e0e\u771f Lightroom \u7684\u6e32\u67d3\u8bef\u5dee\u3002

\u95ee\u9898\u80cc\u666f:
  diff_isp.py \u662f\u6211\u4eec\u81ea\u5df1\u5199\u7684\u8fd1\u4f3c ISP, \u90e8\u5206 op (sRGB/Rec709/EV) \u57fa\u4e8e\u6807\u51c6,
  \u4f46 shadows/highlights/contrast/brightness/clarity \u7684\u5177\u4f53\u5e38\u6570\u90fd\u662f\u62cd\u8111\u888b\u5b9a\u7684,
  \u4ece\u672a\u4e0e\u771f LR \u5bf9\u8fc7\u3002

\u8be5\u811a\u672c\u5728 FiveK \u4e0a\u505a\u91cf\u5316:
  orig_jpg  -- diff_isp(orig, xmp_params) -->  fake_render
                                                    \u2191 PSNR / SSIM / \u0394E
  expert_jpg  (Lightroom \u6e32\u67d3\u7684\u5b9e\u7269)  \u2500\u2500

\u8f93\u5165\u6570\u636e:
  --orig_dir       fivek_jpeg/            rawpy \u9ed8\u8ba4\u89e3\u9a6c\u8d5b\u514b JPG (5000 \u5f20)
  --expert_dir     fivek_expert_c/        LR \u6e32\u67d3\u7684 Expert JPG (AutoDL 289M)
  --params_json    fivek_expert_params.json  5 \u4e13\u5bb6 XMP \u6570\u503c (46167 \u8bb0\u5f55)
  --expert_filter  'c' \u6216 'mean' \u6216 'default'  (\u9ed8\u8ba4 mean: \u5bf9\u540c\u4e00\u56fe\u7684\u591a\u6761\u8bb0\u5f55\u6c42\u5e73\u5747)

\u8f93\u51fa:
  --out_dir   outputs/validate_diff_isp_vs_lr/
    per_sample.json    \u6bcf\u5f20\u56fe\u7684 PSNR/SSIM/\u0394E + \u4f7f\u7528\u7684\u53c2\u6570
    summary.json       mean / median / std
    visualize/          \u524d N \u4e2a sample \u7684 [orig | fake | expert | diff] \u62fc\u56fe

\u8fd0\u884c:
  # \u5c0f\u5c3a\u5ea6\u8c03\u8bd5 (5 \u5f20)
  python tools/eval/validate_diff_isp_vs_lr.py --n 5 --visualize
  # \u5168\u91cf (1000 \u5f20 - AutoDL \u4e0a)
  python tools/eval/validate_diff_isp_vs_lr.py --n 1000 --expert_dir /root/autodl-tmp/fivek_expert_c
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.diff_isp import apply_diff_isp

# NeuralISP 使用 6 参数 schema (与 training/fivek_8param/config.py 保持一致)
NISP_PARAM_NAMES = ['ev_compensation', 'white_balance', 'contrast',
                    'shadows', 'highlights', 'saturation']
NISP_PARAM_RANGES = {
    'ev_compensation': (-3.0, 3.0),
    'white_balance': (2000.0, 10000.0),
    'contrast': (-100.0, 100.0),
    'shadows': (-100.0, 100.0),
    'highlights': (-100.0, 100.0),
    'saturation': (-100.0, 100.0),
}


def build_nisp_param_tensor(params: dict) -> torch.Tensor:
    """将 XMP physical dict 转为 NeuralISP 期望的 (1, 6) [-1,1] 张量。"""
    out = []
    for k in NISP_PARAM_NAMES:
        v = float(params.get(k, 0.0))
        lo, hi = NISP_PARAM_RANGES[k]
        norm = 2.0 * (v - lo) / (hi - lo) - 1.0
        norm = max(-1.0, min(1.0, norm))
        out.append(norm)
    return torch.tensor([out], dtype=torch.float32)


def load_neural_isp(ckpt_path, device):
    """加载 训练好的 NeuralISP。"""
    from models.neural_isp import NeuralISP
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {'param_dim': 6, 'base_ch': 32, 'n_res_blocks': 4})
    model = NeuralISP(param_dim=cfg.get('param_dim', 6),
                      base_ch=cfg.get('base_ch', 32),
                      n_res_blocks=cfg.get('n_res_blocks', 4))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    n_total, _ = model.count_params()
    print(f'[INFO] NeuralISP loaded: {n_total/1e3:.0f}K params  '
          f'(val_loss={ckpt.get("val_loss", "?")})')
    return model


# ---------- \u6307\u6807 ----------

def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 1.0) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 100.0
    return 10.0 * np.log10(max_val * max_val / mse)


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """skimage \u7070\u5ea6 SSIM (\u7b80\u5316)\u3002\u4f9d\u8d56 scikit-image\u3002"""
    try:
        from skimage.metrics import structural_similarity
        ga = a.mean(axis=-1) if a.ndim == 3 else a
        gb = b.mean(axis=-1) if b.ndim == 3 else b
        return float(structural_similarity(ga, gb, data_range=1.0))
    except ImportError:
        # \u6ca1\u88c5 scikit-image \u7528\u624b\u5de5\u7b80\u7248\u56de\u9000
        return float('nan')


def deltaE_ciede2000(a: np.ndarray, b: np.ndarray) -> float:
    """CIEDE2000 \u8272\u5dee (\u57fa\u4e8e Lab), \u4f9d\u8d56 scikit-image\u3002\u8f93\u5165\u662f [0,1] sRGB."""
    try:
        from skimage.color import rgb2lab, deltaE_ciede2000 as de
        lab1 = rgb2lab(a.astype(np.float32))
        lab2 = rgb2lab(b.astype(np.float32))
        return float(np.mean(de(lab1, lab2)))
    except ImportError:
        return float('nan')


# ---------- \u6570\u636e ----------

def load_image(path: Path, size: int) -> np.ndarray:
    """\u52a0\u8f7d \u7f29\u653e\u5230 (size, size), \u8f93\u51fa [0,1] HxWxC float."""
    pil = Image.open(path).convert('RGB')
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


def build_param_dict(params: dict) -> dict:
    """\u5c06 XMP \u6570\u503c dict \u8f6c\u4e3a apply_diff_isp \u671f\u671b\u7684 torch tensor dict (B=1)."""
    keys = ['ev_compensation', 'white_balance', 'contrast',
            'brightness', 'shadows', 'highlights',
            'saturation', 'vibrance', 'clarity']
    defaults = {
        'ev_compensation': 0.0, 'white_balance': 5500.0, 'contrast': 0.0,
        'brightness': 0.0, 'shadows': 0.0, 'highlights': 0.0,
        'saturation': 0.0, 'vibrance': 0.0, 'clarity': 0.0,
    }
    out = {}
    for k in keys:
        v = float(params.get(k, defaults[k]))
        out[k] = torch.tensor([v], dtype=torch.float32)
    return out


def aggregate_params_per_image(records, expert_filter='mean'):
    """
    \u6839\u636e expert_filter \u5bf9\u540c\u4e00\u56fe\u7684\u591a\u6761\u8bb0\u5f55\u805a\u5408:
      - 'mean': \u6240\u6709\u8bb0\u5f55\u6c42\u5747\u503c
      - 'first': \u9996\u6761\u8bb0\u5f55
      - '<expert_label>': \u53ea\u4fdd\u7559 expert==\u8be5\u6807\u7b7e\u7684 \u518d\u6c42\u5747
    """
    per_img = defaultdict(list)
    for r in records:
        per_img[r['image_name']].append(r)

    param_keys = ['ev_compensation', 'white_balance', 'contrast', 'brightness',
                  'shadows', 'highlights', 'saturation', 'vibrance', 'clarity']
    result = {}
    for img_name, recs in per_img.items():
        if expert_filter != 'mean' and expert_filter != 'first':
            # \u53ea\u4fdd\u7559\u6307\u5b9a expert
            recs = [r for r in recs if r.get('expert') == expert_filter]
            if not recs:
                continue
        if expert_filter == 'first':
            recs = recs[:1]
        agg = {k: float(np.mean([r.get(k, 0) for r in recs])) for k in param_keys}
        result[img_name] = agg
    return result


# ---------- \u53ef\u89c6\u5316 ----------

def save_comparison_grid(out_path, orig, fake, expert, info_lines):
    """\u62fc [orig | fake | expert | diff(fake,expert)x4] \u4fdd\u5b58\u3002"""
    import matplotlib.pyplot as plt
    diff = np.clip(np.abs(fake - expert) * 4, 0, 1)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    titles = ['Original (rawpy)', 'ISP output (ours)', 'Expert (Lightroom GT)',
              '|fake - expert| \u00d7 4']
    for ax, img, t in zip(axs, [orig, fake, expert, diff], titles):
        ax.imshow(img)
        ax.set_title(t)
        ax.axis('off')
    fig.suptitle('\n'.join(info_lines), fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()


# ---------- \u4e3b\u903b\u8f91 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig_dir',
                    default=r'E:\Data\dataset\fivek_jpeg',
                    help='rawpy \u89e3\u9a6c\u7684 \u4e2d\u6027 JPG \u76ee\u5f55')
    ap.add_argument('--expert_dir',
                    default=r'E:\Data\dataset\fivek_expert_c',
                    help='LR \u6e32\u67d3\u7684 Expert JPG \u76ee\u5f55 (\u672c\u5730\u53ef\u80fd\u4e0d\u5b58\u5728 \u2192 AutoDL /root/autodl-tmp/fivek_expert_c)')
    ap.add_argument('--params_json',
                    default='data/fivek_expert_params.json')
    ap.add_argument('--expert_filter', default='mean',
                    choices=['mean', 'first', 'default',
                             '42962A54-F9BA-11DB-B851-000D93313A24',
                             'ED7AD140-FA03-11DB-AB5E-00145166C8C8'],
                    help='\u540c\u4e00\u56fe\u7684\u591a\u6761\u8bb0\u5f55\u800c\u6c47\u805a\u65b9\u5f0f')
    ap.add_argument('--n', type=int, default=100, help='\u6837\u672c\u6570 (0=\u5168\u90e8)')
    ap.add_argument('--image_size', type=int, default=512)
    ap.add_argument('--out_dir', default='outputs/validate_diff_isp_vs_lr')
    ap.add_argument('--visualize', action='store_true', help='\u4fdd\u5b58\u524d 10 \u4e2a\u6837\u672c\u7684\u62fc\u56fe')
    ap.add_argument('--n_vis', type=int, default=10)
    ap.add_argument('--only_images', default='',
                    help='\u9017\u53f7\u5206\u9694\u7684 stem \u5217\u8868 (\u5982 "a4802-DSC_0133,a3794-kme_583"), \u4ec5\u5904\u7406\u8fd9\u4e9b\u56fe\u5e76\u53ef\u89c6\u5316 (\u8986\u76d6 --n)')
    ap.add_argument('--isp_mode', default='diff_isp',
                    choices=['diff_isp', 'neural_isp'],
                    help='使用 手写 diff_isp 还是 训练过的 NeuralISP')
    ap.add_argument('--nisp_ckpt',
                    default='/root/autodl-tmp/checkpoints/neural_isp/best.pt',
                    help='NeuralISP checkpoint (仅 isp_mode=neural_isp 时使用)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / 'visualize'
    vis_dir.mkdir(exist_ok=True)

    # ----- ISP model -----
    nisp_model = None
    if args.isp_mode == 'neural_isp':
        nisp_model = load_neural_isp(args.nisp_ckpt, device)
    print(f'[INFO] isp_mode = {args.isp_mode}')

    orig_dir = Path(args.orig_dir)
    expert_dir = Path(args.expert_dir)
    params_path = PROJECT_ROOT / args.params_json

    # ----- \u52a0\u8f7d params -----
    print(f'[INFO] Loading params from {params_path}')
    data = json.load(open(params_path, encoding='utf-8'))
    records = data['samples']
    print(f'[INFO] {len(records)} records, {len(set(r["image_name"] for r in records))} unique images')

    img2params = aggregate_params_per_image(records, args.expert_filter)
    print(f'[INFO] Aggregated to {len(img2params)} images  (expert_filter={args.expert_filter})')

    # ----- \u5339\u914d \u6709\u6548\u6837\u672c (\u4e24\u4fa7 JPG \u90fd\u5b58\u5728) -----
    valid_samples = []
    missing_orig = missing_exp = 0
    for img_name, params in img2params.items():
        stem = img_name.replace('.dng', '')
        orig_p = orig_dir / f'{stem}.jpg'
        exp_p = expert_dir / f'{stem}.jpg'
        if not orig_p.exists():
            missing_orig += 1
            continue
        if not exp_p.exists():
            missing_exp += 1
            continue
        valid_samples.append((stem, orig_p, exp_p, params))

    print(f'[INFO] {len(valid_samples)} valid pairs '
          f'(missing_orig={missing_orig}, missing_expert={missing_exp})')

    if not valid_samples:
        print('[ERR] No valid pairs found.')
        print(f'      orig_dir exists={orig_dir.exists()}  expert_dir exists={expert_dir.exists()}')
        if not expert_dir.exists():
            print(f'      Hint: Expert JPGs may be on AutoDL. Try:')
            print(f'        python tools/eval/validate_diff_isp_vs_lr.py '
                  f'--expert_dir /root/autodl-tmp/fivek_expert_c')
        sys.exit(1)

    if args.only_images.strip():
        targets = set(x.strip() for x in args.only_images.split(',') if x.strip())
        valid_samples = [s for s in valid_samples if s[0] in targets]
        print(f'[INFO] Filtered to --only_images: {len(valid_samples)} pairs')
    elif args.n > 0 and len(valid_samples) > args.n:
        # \u968f\u673a\u5747\u5300\u91c7\u6837 (\u79cd\u5b50\u56fa\u5b9a\u53ef\u590d\u73b0)
        rng = np.random.default_rng(42)
        idxs = rng.choice(len(valid_samples), args.n, replace=False)
        valid_samples = [valid_samples[i] for i in sorted(idxs)]
        print(f'[INFO] Sub-sampled to {len(valid_samples)} pairs')

    # ----- \u8dd1\u6bcf\u6837\u672c -----
    print(f'[INFO] device={device}  image_size={args.image_size}')
    to_tensor = transforms.ToTensor()
    per_sample = []

    for i, (stem, orig_p, exp_p, params) in enumerate(valid_samples):
        orig_np = load_image(orig_p, args.image_size)
        expert_np = load_image(exp_p, args.image_size)

        orig_t = torch.from_numpy(orig_np.transpose(2, 0, 1)
                                  ).unsqueeze(0).float().to(device)

        with torch.no_grad():
            if args.isp_mode == 'neural_isp':
                nisp_params = build_nisp_param_tensor(params).to(device)
                fake = nisp_model(orig_t, nisp_params)
            else:
                param_t = {k: v.to(device) for k, v in build_param_dict(params).items()}
                fake = apply_diff_isp(orig_t, param_t)
            fake = torch.nan_to_num(fake, nan=0.5).clamp(0, 1)
        fake_np = fake[0].cpu().numpy().transpose(1, 2, 0)

        # \u6307\u6807
        ps = psnr(fake_np, expert_np)
        ss = ssim_gray(fake_np, expert_np)
        de = deltaE_ciede2000(fake_np, expert_np)

        # \u4e3a\u5bf9\u7167 \u4e5f\u7b97 orig vs expert \u76f8\u5f53\u4e8e"\u4e0d\u505a\u4efb\u4f55 edit \u7684\u57fa\u7ebf"
        ps_id = psnr(orig_np, expert_np)
        ss_id = ssim_gray(orig_np, expert_np)
        de_id = deltaE_ciede2000(orig_np, expert_np)

        per_sample.append({
            'image': stem,
            'params_used': params,
            'psnr_fake_vs_expert': ps,
            'ssim_fake_vs_expert': ss,
            'deltaE_fake_vs_expert': de,
            'psnr_orig_vs_expert': ps_id,
            'ssim_orig_vs_expert': ss_id,
            'deltaE_orig_vs_expert': de_id,
        })

        if args.visualize and i < args.n_vis:
            info = [
                f'{stem}   expert_filter={args.expert_filter}',
                f'ev={params["ev_compensation"]:+.2f}  wb={params["white_balance"]:.0f}  '
                f'contrast={params["contrast"]:+.0f}  brightness={params["brightness"]:+.0f}  '
                f'sat={params["saturation"]:+.0f}',
                f'PSNR fake-vs-exp = {ps:.2f} dB (baseline orig-vs-exp = {ps_id:.2f})  '
                f'\u0394E = {de:.2f} (baseline {de_id:.2f})'
            ]
            save_comparison_grid(vis_dir / f'{i:03d}_{stem}.jpg',
                                  orig_np, fake_np, expert_np, info)

        if (i + 1) % 20 == 0 or i == 0:
            print(f'  [{i+1}/{len(valid_samples)}] {stem}  '
                  f'PSNR {ps:.2f}/{ps_id:.2f}  \u0394E {de:.2f}/{de_id:.2f}')

    # ----- \u6c47\u603b -----
    def stats(key):
        vals = [s[key] for s in per_sample if not np.isnan(s[key])]
        if not vals:
            return dict(mean=None, median=None, std=None, min=None, max=None)
        return dict(mean=float(np.mean(vals)), median=float(np.median(vals)),
                    std=float(np.std(vals)),
                    min=float(np.min(vals)), max=float(np.max(vals)))

    summary = {
        'n_samples': len(per_sample),
        'isp_mode': args.isp_mode,
        'expert_filter': args.expert_filter,
        'image_size': args.image_size,
        'psnr_fake_vs_expert': stats('psnr_fake_vs_expert'),
        'ssim_fake_vs_expert': stats('ssim_fake_vs_expert'),
        'deltaE_fake_vs_expert': stats('deltaE_fake_vs_expert'),
        'psnr_orig_vs_expert_BASELINE': stats('psnr_orig_vs_expert'),
        'ssim_orig_vs_expert_BASELINE': stats('ssim_orig_vs_expert'),
        'deltaE_orig_vs_expert_BASELINE': stats('deltaE_orig_vs_expert'),
    }

    json.dump(per_sample, open(out_dir / 'per_sample.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    json.dump(summary, open(out_dir / 'summary.json', 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)

    print('\n' + '='*70)
    print(f'{args.isp_mode}  vs  Lightroom Expert  ({summary["n_samples"]} samples)')
    print('='*70)
    for metric in ['psnr', 'ssim', 'deltaE']:
        s_our = summary[f'{metric}_fake_vs_expert']
        s_base = summary[f'{metric}_orig_vs_expert_BASELINE']
        if s_our['mean'] is None:
            continue
        better = '\u2191 ours wins' if (metric == 'deltaE' and s_our['mean'] < s_base['mean']) \
                 or (metric in ('psnr', 'ssim') and s_our['mean'] > s_base['mean']) else '\u2193 ours loses'
        print(f'  {metric.upper():8s}  '
              f'ours={s_our["mean"]:.3f} \u00b1 {s_our["std"]:.3f}   '
              f'baseline(orig=GT)={s_base["mean"]:.3f}   {better}')

    print(f'\n[DONE] Results at {out_dir}')
    print(f'    per_sample.json   summary.json   visualize/ ({args.n_vis if args.visualize else 0} images)')


if __name__ == '__main__':
    main()
