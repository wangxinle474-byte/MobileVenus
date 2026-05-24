"""v13 single-image editing CLI.

Loads v11-d base + v13 residual refiner and edits one image with a chosen
action (or all 7 actions). Mimics the exact preprocessing used at training time
so results are faithful to the trained distribution.

Usage:
  # Single action
  python tools/v13_edit_image.py --image path/to/in.jpg --action contrast \\
      --out path/to/out.jpg

  # All 7 actions, saves grid + individual files
  python tools/v13_edit_image.py --image path/to/in.jpg --all_actions \\
      --out_dir path/to/out_dir/

  # Override checkpoints (defaults below match v13a/v11d 7-action ckpts)
  python tools/v13_edit_image.py --image in.jpg --action wb --out out.jpg \\
      --ckpt checkpoints/v13a_firered_refine_7actions/best.pt \\
      --base_ckpt checkpoints/lut_v11d_firered_7actions_6537/best.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.firered_residual_refiner import FireRedResidualRefiner  # noqa: E402
from training.firered_baseline.train_lut import (  # noqa: E402
    ACTIONS, set_actions,
)
from training.firered_baseline.train_v13_firered_refine import (  # noqa: E402
    load_v11d_base,
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('v13_edit')


ACTION_PROMPT = {
    'contrast':   'Increase contrast',
    'saturation': 'Enhance saturation',
    'shadows':    'Lift shadows',
    'highlights': 'Recover highlights',
    'wb':         'Apply warmer white balance',
    'brightness': 'Increase brightness',
    'clarity':    'Enhance clarity and sharpness',
}

DEFAULT_CKPT = 'checkpoints/v13a_firered_refine_7actions/best.pt'
DEFAULT_BASE = 'checkpoints/lut_v11d_firered_7actions_6537/best.pt'


def _resolve_base_ckpt(v13_args: dict, cli_base_ckpt: str | None) -> Path:
    if cli_base_ckpt:
        return Path(cli_base_ckpt)
    saved = v13_args.get('base_ckpt')
    if saved and Path(saved).exists():
        return Path(saved)
    if Path(DEFAULT_BASE).exists():
        return Path(DEFAULT_BASE)
    raise RuntimeError('Could not locate v11-d base ckpt. Pass --base_ckpt.')


def _preprocess(image_path: Path, size: int, device: torch.device):
    """Load image and produce (enc_input, orig) tensors in batch dim 1."""
    img = Image.open(image_path).convert('RGB')
    to_tensor = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    norm = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
    orig = to_tensor(img).unsqueeze(0).to(device)         # (1, 3, H, W) [0,1]
    enc_input = norm(orig.squeeze(0)).unsqueeze(0)        # (1, 3, H, W) norm
    return enc_input, orig, img.size  # (W_orig, H_orig)


def _tensor_to_pil(x: torch.Tensor) -> Image.Image:
    """(1, 3, H, W) [0, 1] → PIL."""
    arr = (x.clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy() * 255)
    return Image.fromarray(arr.astype('uint8'))


def _action_onehot(action: str, device: torch.device) -> torch.Tensor:
    if action not in ACTIONS:
        raise ValueError(f'unknown action: {action}. valid: {ACTIONS}')
    idx = ACTIONS.index(action)
    oh = torch.zeros(1, len(ACTIONS), device=device)
    oh[0, idx] = 1.0
    return oh


@torch.no_grad()
def edit_one(base_model, refiner, enc_input, orig, action: str,
             device: torch.device) -> tuple[Image.Image, Image.Image]:
    """Returns (base_pil, refined_pil)."""
    oh = _action_onehot(action, device)
    base_out, _, _, _ = base_model(enc_input, orig, oh)
    refined, _ = refiner(orig, base_out, oh)
    return _tensor_to_pil(base_out), _tensor_to_pil(refined)


def _make_grid(images: list[Image.Image], labels: list[str],
               cols: int = 4, gap: int = 6,
               bg: tuple = (15, 23, 42)) -> Image.Image:
    """Lay images out in a grid with caption strips above each."""
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('arial.ttf', 14)
    except OSError:
        font = ImageFont.load_default()

    W, H = images[0].size
    cap_h = 22
    cell_w = W
    cell_h = H + cap_h
    rows = (len(images) + cols - 1) // cols
    grid_w = cols * cell_w + (cols + 1) * gap
    grid_h = rows * cell_h + (rows + 1) * gap
    canvas = Image.new('RGB', (grid_w, grid_h), bg)
    draw = ImageDraw.Draw(canvas)

    for i, (img, label) in enumerate(zip(images, labels)):
        r, c = divmod(i, cols)
        x = gap + c * (cell_w + gap)
        y = gap + r * (cell_h + gap)
        draw.rectangle([x, y, x + cell_w, y + cap_h - 2], fill=(30, 41, 59))
        draw.text((x + 6, y + 4), label, fill=(226, 232, 240), font=font)
        canvas.paste(img, (x, y + cap_h))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True, help='input image path')
    ap.add_argument('--action', default=None,
                    choices=list(ACTION_PROMPT.keys()),
                    help='which action to apply (skip if --all_actions)')
    ap.add_argument('--all_actions', action='store_true',
                    help='run all 7 actions and save a grid + each individual')
    ap.add_argument('--out', default=None,
                    help='output path (single action mode)')
    ap.add_argument('--out_dir', default=None,
                    help='output directory (all_actions mode)')
    ap.add_argument('--ckpt', default=DEFAULT_CKPT,
                    help=f'v13 refiner ckpt (default: {DEFAULT_CKPT})')
    ap.add_argument('--base_ckpt', default=None,
                    help='v11-d base ckpt (default: read from v13 args)')
    ap.add_argument('--image_size', type=int, default=None,
                    help='override; default reads from v13 args (256)')
    ap.add_argument('--save_comparison', action='store_true',
                    help='also save orig|base|refined triplet')
    ap.add_argument('--device', default='cuda',
                    help='cuda or cpu')
    args = ap.parse_args()

    if args.action is None and not args.all_actions:
        ap.error('must specify --action or --all_actions')
    if args.all_actions and not args.out_dir:
        ap.error('--all_actions requires --out_dir')
    if args.action and not args.out and not args.save_comparison:
        ap.error('--action requires --out (or --save_comparison + --out_dir)')

    device = torch.device(args.device if torch.cuda.is_available()
                          else 'cpu')
    logger.info(f'device: {device}')

    # ── Load v13 refiner ckpt ──
    v13_ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    v13_args = v13_ckpt['args']

    # CRITICAL: set global ACTIONS to match training before loading anything
    ckpt_actions = v13_args.get('actions')
    if ckpt_actions is not None:
        set_actions(ckpt_actions)
    logger.info(f'actions: {ACTIONS}')

    base_ckpt_path = _resolve_base_ckpt(v13_args, args.base_ckpt)
    base_model, base_args = load_v11d_base(base_ckpt_path, device)

    image_size = args.image_size or v13_args.get(
        'image_size', base_args.get('image_size', 256))
    logger.info(f'image_size: {image_size}')

    refiner = FireRedResidualRefiner(
        base_ch=v13_args.get('base_ch', 32),
        n_actions=len(ACTIONS),
        delta_scale=v13_args.get('delta_scale', 0.5),
    ).to(device)
    refiner.load_state_dict(v13_ckpt['model_state_dict'], strict=True)
    refiner.eval()
    logger.info(f'v13 ckpt: {args.ckpt} '
                f'val_psnr={v13_ckpt.get("val_psnr", 0):.2f}dB '
                f'@ Ep{v13_ckpt.get("epoch", "?")}')

    # ── Preprocess input image ──
    img_path = Path(args.image)
    if not img_path.exists():
        ap.error(f'input image not found: {img_path}')
    enc_input, orig, orig_size = _preprocess(img_path, image_size, device)
    logger.info(f'loaded {img_path} ({orig_size[0]}x{orig_size[1]} → '
                f'{image_size}x{image_size} for inference)')

    orig_pil = _tensor_to_pil(orig)

    # ── Run inference ──
    if args.all_actions:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        grid_images = [orig_pil]
        grid_labels = ['orig']
        for action in ACTIONS:
            base_pil, refined_pil = edit_one(
                base_model, refiner, enc_input, orig, action, device)
            stem = img_path.stem
            refined_pil.save(out_dir / f'{stem}_{action}_v13.jpg', quality=92)
            base_pil.save(out_dir / f'{stem}_{action}_base.jpg', quality=92)
            grid_images.append(refined_pil)
            grid_labels.append(f'{action} (v13)')
            logger.info(f'  [{action}] saved')
        # Pad to 8 cells if 7 actions + 1 orig
        grid = _make_grid(grid_images, grid_labels, cols=4)
        grid_path = out_dir / f'{img_path.stem}_all_actions_grid.jpg'
        grid.save(grid_path, quality=92)
        logger.info(f'grid: {grid_path}')
    else:
        base_pil, refined_pil = edit_one(
            base_model, refiner, enc_input, orig, args.action, device)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            refined_pil.save(out_path, quality=92)
            logger.info(f'saved refined: {out_path}')
        if args.save_comparison:
            cmp = _make_grid(
                [orig_pil, base_pil, refined_pil],
                ['orig', 'v11-d base', f'v13 refined ({args.action})'],
                cols=3)
            cmp_path = (Path(args.out).with_suffix('.compare.jpg')
                        if args.out else Path(args.out_dir) /
                        f'{img_path.stem}_{args.action}_compare.jpg')
            cmp_path.parent.mkdir(parents=True, exist_ok=True)
            cmp.save(cmp_path, quality=92)
            logger.info(f'saved comparison: {cmp_path}')


if __name__ == '__main__':
    main()
