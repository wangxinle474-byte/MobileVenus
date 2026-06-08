"""对比预览: 模型预测参数渲染 vs Expert C GT 参数渲染。

3-panel: Before (orig JPEG) | Model Prediction | Expert C GT
均通过 apply_diff_isp 渲染, 保证渲染引擎一致。

用法:
  python tools/data/data_prep/preview_model_vs_expert.py --n 12
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch.nn as nn
from models.diff_isp import apply_diff_isp  # noqa: E402
from models.vision_encoder import MobileViTSmall  # noqa: E402

# ── inline from training/expert_c_baseline/train.py (avoid torchvision dep) ──
PARAM_NAMES = [
    'white_balance', 'brightness', 'contrast',
    'shadows', 'highlights', 'saturation', 'clarity',
]
PARAM_NORM = {
    'white_balance': {'center': 6000.0, 'scale': 4000.0,
                      'clip': (2000.0, 10000.0)},
    'brightness':    {'center': 0.0,    'scale': 100.0,
                      'clip': (-150.0, 150.0)},
    'contrast':      {'center': 0.0,    'scale': 100.0,
                      'clip': (-100.0, 100.0)},
    'shadows':       {'center': 0.0,    'scale': 100.0,
                      'clip': (-100.0, 100.0)},
    'highlights':    {'center': 0.0,    'scale': 100.0,
                      'clip': (-100.0, 100.0)},
    'saturation':    {'center': 0.0,    'scale': 100.0,
                      'clip': (-100.0, 100.0)},
    'clarity':       {'center': 0.0,    'scale': 100.0,
                      'clip': (-100.0, 100.0)},
}


class ExpertC7DModel(nn.Module):
    """MobileViTSmall (~2.89M) + Linear(384 -> 7) head."""
    def __init__(self, image_size: int = 224, visual_dim: int = 384):
        super().__init__()
        self.encoder = MobileViTSmall(
            image_size=image_size, output_dim=visual_dim,
            use_se=True, use_fpn=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Dropout(0.1),
            nn.Linear(visual_dim, visual_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(visual_dim // 2, 7),
            nn.Tanh(),
        )

    def forward(self, images):
        feat = self.encoder(images)
        return self.head(feat)

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── helpers ──────────────────────────────────────────────────────────

def denormalize(name: str, norm_val: float) -> float:
    cfg = PARAM_NORM[name]
    return norm_val * cfg['scale'] + cfg['center']


def load_image_tensor(path: str | Path, size: int) -> torch.Tensor:
    """Load image → (1, 3, H, W) float [0, 1]."""
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def make_panel(images: list, labels: list, font_h: int = 22) -> Image.Image:
    h, w = images[0].size[1], images[0].size[0]
    n = len(images)
    pad = 4
    panel = Image.new('RGB', (w * n + pad * (n - 1), h + font_h + 4),
                       (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype('arial.ttf', font_h - 4)
    except Exception:
        font = ImageFont.load_default()
    for i, (im, lab) in enumerate(zip(images, labels)):
        x = i * (w + pad)
        panel.paste(im, (x, font_h + 4))
        draw.text((x + 4, 2), lab, fill=(20, 20, 20), font=font)
    return panel


def params_str(d: dict) -> str:
    parts = []
    for p in PARAM_NAMES:
        v = d.get(p, 0.0)
        if p == 'white_balance':
            parts.append(f'WB={v:.0f}K')
        else:
            parts.append(f'{p[:4]}={v:+.1f}')
    return '  '.join(parts)


# ── main ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/expert_c_v1/best.pt')
    ap.add_argument('--gt_json', default='data/fivek_expert_abcde_params.json')
    ap.add_argument('--jpeg_dir', default=r'E:\Data\dataset\fivek_jpeg')
    ap.add_argument('--expert', default='C')
    ap.add_argument('--out_dir', default='outputs/preview_model_vs_expert')
    ap.add_argument('--n', type=int, default=12, help='number of samples')
    ap.add_argument('--render_size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f'[ERR] checkpoint not found: {ckpt_path}')
        return 1
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})
    image_size = saved_args.get('image_size', 224)

    model = ExpertC7DModel(image_size=image_size).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'[model] loaded {ckpt_path.name}  '
          f'(Ep {ckpt.get("epoch", "?")}, val_loss={ckpt.get("val_loss", "?"):.4f})')
    if 'val_mae' in ckpt:
        mae = ckpt['val_mae']
        print(f'[model] val MAE: '
              f'{", ".join(f"{p[:4]}={mae[i]:.1f}" for i, p in enumerate(PARAM_NAMES))}')

    # 2. Load Expert C GT params
    with open(args.gt_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    expert_records = {s['image_name']: s
                      for s in data['samples']
                      if s['expert'] == args.expert}
    print(f'[data] Expert {args.expert}: {len(expert_records)} GT records')

    # 3. Find available JPEGs with GT
    jpeg_dir = Path(args.jpeg_dir)
    available = []
    for name, rec in expert_records.items():
        stem = name.rsplit('.', 1)[0]
        jpg = jpeg_dir / f'{stem}.jpg'
        if jpg.exists():
            available.append((name, str(jpg), rec))
    print(f'[data] JPEG matches: {len(available)}')

    # Sample
    rng = random.Random(args.seed)
    samples = rng.sample(available, min(args.n, len(available)))

    # 4. Render loop
    previews = []
    for i, (dng_name, jpg_path, gt_rec) in enumerate(samples):
        stem = dng_name.rsplit('.', 1)[0]
        print(f'[{i+1}/{len(samples)}] {stem}')

        # Load image for rendering
        img_render = load_image_tensor(jpg_path, args.render_size).to(device)

        # Model prediction — manual resize + normalize (no torchvision)
        orig_pil = Image.open(jpg_path).convert('RGB')
        resized = orig_pil.resize((image_size, image_size), Image.LANCZOS)
        arr = np.asarray(resized).astype(np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        x_model = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(x_model)[0]  # (7,) in [-1, 1]

        P_pred = {}
        for j, p in enumerate(PARAM_NAMES):
            P_pred[p] = denormalize(p, pred_norm[j].item())

        # Map to diff_isp params dict
        pred_params_t = {p: torch.tensor([P_pred[p]], dtype=torch.float32)
                         for p in PARAM_NAMES}
        with torch.no_grad():
            pred_render = apply_diff_isp(img_render, pred_params_t).clamp(0, 1)

        # Expert C GT params → diff_isp render
        gt_p = gt_rec['params']
        P_gt = {}
        for p in PARAM_NAMES:
            P_gt[p] = float(gt_p.get(p, 0) or 0)

        gt_params_t = {p: torch.tensor([P_gt[p]], dtype=torch.float32)
                       for p in PARAM_NAMES}
        with torch.no_grad():
            gt_render = apply_diff_isp(img_render, gt_params_t).clamp(0, 1)

        # 3-panel
        orig_pil_r = tensor_to_pil(img_render)
        pred_pil = tensor_to_pil(pred_render)
        gt_pil = tensor_to_pil(gt_render)

        panel = make_panel(
            [orig_pil_r, pred_pil, gt_pil],
            [f'Before: {stem}',
             f'Model Pred',
             f'Expert {args.expert} GT'],
        )
        panel_path = out_dir / f'{i+1:02d}_{stem}.jpg'
        panel.save(panel_path, quality=92)

        # Print params comparison
        print(f'  Pred: {params_str(P_pred)}')
        print(f'  GT:   {params_str(P_gt)}')

        previews.append({
            'stem': stem,
            'panel': panel_path.name,
            'P_pred': P_pred,
            'P_gt': P_gt,
        })

    # 6. Contact sheet
    if previews:
        panels = [Image.open(out_dir / p['panel']) for p in previews]
        pw, ph = panels[0].size
        cols = min(3, len(panels))
        rows_count = (len(panels) + cols - 1) // cols
        contact = Image.new('RGB', (pw * cols + 8 * (cols - 1),
                                     ph * rows_count + 8 * (rows_count - 1)),
                            (240, 240, 240))
        for idx, p in enumerate(panels):
            r, c = divmod(idx, cols)
            contact.paste(p, (c * (pw + 8), r * (ph + 8)))
        contact_path = out_dir / '_contact.jpg'
        contact.save(contact_path, quality=88)
        print(f'\n[done] contact sheet: {contact_path}')

    # 7. Summary JSON
    summary = {
        'checkpoint': str(ckpt_path),
        'expert': args.expert,
        'n_samples': len(previews),
        'previews': previews,
    }
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f'[done] summary: {summary_path}')
    print(f'[done] output dir: {out_dir.resolve()}')
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
