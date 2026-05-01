"""可视化 AesExpert delta >= threshold 的样本: 原图 vs 增强图 并排对比."""
import json, sys, os
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.fivek_8param.config import PARAM_NAMES
from models.diff_isp import apply_diff_isp


def render_augmented(orig_pil, aug_params, size=512):
    """用 diff_isp 渲染增强图."""
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    tensor = tf(orig_pil).unsqueeze(0)
    params_torch = {
        p: torch.tensor([float(aug_params.get(p, 0))], dtype=torch.float32)
        for p in PARAM_NAMES
    }
    with torch.no_grad():
        rendered = apply_diff_isp(tensor, params_torch).clamp(0, 1)
    arr = (rendered[0].numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_label(img, text, position='top'):
    """在图片上加文字标签."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    y = 5 if position == 'top' else img.height - 25
    # 黑底白字
    draw.rectangle([0, y-2, img.width, y+24], fill='black')
    draw.text((10, y), text, fill='white', font=font)
    return img


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--delta_min', type=float, default=4.0)
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--out_dir', default='outputs/data/delta_vis')
    ap.add_argument('--max_samples', type=int, default=20)
    args = ap.parse_args()

    # Load data
    orig_aes = json.load(open(ROOT / 'outputs/data/fivek_orig_aesexpert.json', encoding='utf-8'))
    orig_map = {}
    for s in orig_aes.get('samples', []):
        if s.get('aesexpert_overall') is not None:
            orig_map[s['image']] = s

    aug_data = json.load(open(ROOT / 'outputs/data/aug_aesexpert.json', encoding='utf-8'))

    results = []
    for s in aug_data.get('samples', []):
        aug_score = s.get('aesexpert_overall')
        if aug_score is None:
            continue
        img = s.get('image', '')
        if img not in orig_map:
            continue
        orig_score = orig_map[img]['aesexpert_overall']
        delta = aug_score - orig_score
        if delta >= args.delta_min:
            results.append({
                'image': img,
                'style': s.get('style', '?'),
                'orig_score': orig_score,
                'aug_score': aug_score,
                'delta': delta,
                'orig_path': orig_map[img].get('image_path', ''),
                'aug_params': s.get('augmented_params', {}),
            })

    results.sort(key=lambda x: (-x['delta'], -x['aug_score']))
    results = results[:args.max_samples]
    print(f'Delta >= {args.delta_min}: {len(results)} samples (showing {len(results)})')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate comparison images
    pairs = []
    for i, r in enumerate(results):
        orig_path = Path(r['orig_path'])
        if not orig_path.exists():
            print(f'  SKIP {orig_path}')
            continue

        orig_pil = Image.open(orig_path).convert('RGB')
        orig_resized = orig_pil.copy()
        orig_resized.thumbnail((args.size, args.size))

        aug_pil = render_augmented(orig_pil, r['aug_params'], args.size)

        # Add labels
        orig_labeled = orig_resized.resize((args.size, args.size))
        add_label(orig_labeled, f"Original  AesExpert={r['orig_score']:.1f}")
        add_label(aug_pil, f"Augmented AesExpert={r['aug_score']:.1f}  delta={r['delta']:+.1f}  style={r['style']}")

        # Side by side
        combo = Image.new('RGB', (args.size * 2 + 4, args.size), (40, 40, 40))
        combo.paste(orig_labeled, (0, 0))
        combo.paste(aug_pil, (args.size + 4, 0))

        safe_style = r['style'].replace('?', 'unknown').replace('/', '_').replace('\\', '_')
        fname = f"{i+1:02d}_{r['image']}_{safe_style}_d{r['delta']:+.0f}.jpg"
        combo.save(out_dir / fname, quality=90)
        pairs.append((fname, r))
        print(f'  [{i+1}] {fname}')

    # Generate HTML gallery
    html = ['<html><head><meta charset="utf-8">',
            '<title>AesExpert Delta >= %.1f</title>' % args.delta_min,
            '<style>body{background:#222;color:#eee;font-family:monospace;padding:20px}',
            'img{max-width:100%;margin:8px 0}',
            '.card{background:#333;border-radius:8px;padding:12px;margin:16px 0}',
            'h2{color:#7df}</style></head><body>',
            '<h1>AesExpert Delta >= %.1f (%d samples)</h1>' % (args.delta_min, len(pairs))]

    for fname, r in pairs:
        html.append('<div class="card">')
        html.append(f'<h2>{r["image"]} | {r["style"]} | orig={r["orig_score"]:.1f} → aug={r["aug_score"]:.1f} (delta={r["delta"]:+.1f})</h2>')
        p = r['aug_params']
        params_str = ', '.join(f'{k}={p.get(k,0):.2f}' for k in PARAM_NAMES if k in p)
        html.append(f'<p>Params: {params_str}</p>')
        html.append(f'<img src="{fname}">')
        html.append('</div>')

    html.append('</body></html>')
    html_path = out_dir / 'gallery.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html))
    print(f'\nHTML gallery: {html_path}')


if __name__ == '__main__':
    main()
