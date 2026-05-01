import os
import sys
import io
import json
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


DEGRADATION_TYPES = [
    'heavy_blur',
    'pixelate',
    'jpeg_artifacts',
    'noise_blur',
    'posterize_color_cast',
    'extreme_dark',
    'extreme_bright',
    'mixed_destroy',
]

SYNTHETIC_SCORE_BY_TYPE = {
    'heavy_blur': 1.7,
    'pixelate': 1.3,
    'jpeg_artifacts': 1.6,
    'noise_blur': 1.1,
    'posterize_color_cast': 0.8,
    'extreme_dark': 1.0,
    'extreme_bright': 1.1,
    'mixed_destroy': 0.5,
}


def list_fivek_images(jpeg_dir: str, max_images: int, seed: int) -> List[Path]:
    rng = np.random.RandomState(seed)
    paths = sorted(Path(jpeg_dir).glob('*.jpg'))
    rng.shuffle(paths)
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality, optimize=False)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def add_gaussian_noise(img: Image.Image, sigma: float, rng: np.random.RandomState) -> Image.Image:
    arr = np.asarray(img).astype(np.float32) / 255.0
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def color_cast(img: Image.Image, rgb_gain: Tuple[float, float, float]) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    gain = np.array(rgb_gain, dtype=np.float32).reshape(1, 1, 3)
    arr = np.clip(arr * gain, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def degrade_image(img: Image.Image, kind: str, rng: np.random.RandomState) -> Image.Image:
    img = img.convert('RGB')
    w, h = img.size

    if kind == 'heavy_blur':
        out = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(10, 22))))
        out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.25, 0.55)))
        return out

    if kind == 'pixelate':
        small_side = int(rng.choice([12, 16, 20, 24, 32]))
        scale = small_side / min(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        out = img.resize((nw, nh), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.NEAREST)
        out = jpeg_recompress(out, int(rng.uniform(5, 18)))
        return out

    if kind == 'jpeg_artifacts':
        out = img
        for _ in range(int(rng.randint(2, 5))):
            out = jpeg_recompress(out, int(rng.uniform(2, 10)))
        out = ImageEnhance.Sharpness(out).enhance(float(rng.uniform(0.0, 0.4)))
        return out

    if kind == 'noise_blur':
        out = add_gaussian_noise(img, float(rng.uniform(0.18, 0.38)), rng)
        out = out.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(2, 7))))
        out = jpeg_recompress(out, int(rng.uniform(5, 18)))
        return out

    if kind == 'posterize_color_cast':
        out = ImageOps.posterize(img, bits=int(rng.choice([2, 3])))
        cast = rng.choice(['red', 'green', 'blue', 'yellow'])
        gains = {
            'red': (2.2, 0.35, 0.35),
            'green': (0.35, 2.2, 0.35),
            'blue': (0.35, 0.45, 2.3),
            'yellow': (2.1, 1.7, 0.25),
        }[cast]
        out = color_cast(out, gains)
        out = ImageEnhance.Color(out).enhance(float(rng.uniform(2.5, 4.0)))
        return out

    if kind == 'extreme_dark':
        out = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.02, 0.12)))
        out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.2, 0.6)))
        out = add_gaussian_noise(out, float(rng.uniform(0.04, 0.12)), rng)
        return out

    if kind == 'extreme_bright':
        out = ImageEnhance.Brightness(img).enhance(float(rng.uniform(4.5, 8.0)))
        out = ImageEnhance.Contrast(out).enhance(float(rng.uniform(0.15, 0.45)))
        out = ImageEnhance.Color(out).enhance(float(rng.uniform(0.0, 0.4)))
        return out

    if kind == 'mixed_destroy':
        out = img
        chain = rng.choice(DEGRADATION_TYPES[:-1], size=int(rng.randint(3, 5)), replace=False)
        for k in chain:
            out = degrade_image(out, k, rng)
        return out

    raise ValueError(kind)


@torch.no_grad()
def score_pil(img: Image.Image, metric, device: torch.device, image_size: int) -> float:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    return float(metric(tensor).mean().item())


def load_musiq(device: torch.device):
    import pyiqa
    metric = pyiqa.create_metric('musiq-ava', device=device)
    for p in metric.parameters():
        p.requires_grad = False
    return metric


def sample_synthetic_score(kind: str, rng: np.random.RandomState) -> float:
    base = SYNTHETIC_SCORE_BY_TYPE.get(kind, 1.5)
    score = base + float(rng.normal(0.0, 0.15))
    return float(np.clip(score, 0.2, 1.95))


def generate_low_score_images(args):
    rng = np.random.RandomState(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    musiq = None if args.no_musiq else load_musiq(device)

    image_paths = list_fivek_images(args.jpeg_dir, args.max_images, args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    attempts = 0
    t0 = time.time()

    for img_idx, img_path in enumerate(image_paths):
        src = Image.open(img_path).convert('RGB')
        stem = img_path.stem
        source_score = score_pil(src, musiq, device, args.score_size) if args.score_source and musiq is not None else None

        kept_for_image = 0
        for variant_idx in range(args.variants_per):
            accepted = False
            best_record = None
            max_retries = 1 if args.no_musiq else args.retries
            for retry in range(max_retries):
                attempts += 1
                kind = str(rng.choice(DEGRADATION_TYPES))
                degraded = degrade_image(src, kind, rng)
                musiq_score = None if musiq is None else score_pil(degraded, musiq, device, args.score_size)
                synthetic_score = sample_synthetic_score(kind, rng)
                score = synthetic_score if args.no_musiq else musiq_score

                if best_record is None or score < best_record['score']:
                    best_record = {
                        'image': degraded,
                        'score': score,
                        'musiq_score': musiq_score,
                        'synthetic_score': synthetic_score,
                        'kind': kind,
                        'retry': retry,
                    }

                if score < args.target_score:
                    accepted = True
                    break

            record = best_record
            if accepted or args.keep_best:
                out_name = f'{stem}_low_{kept_for_image:02d}_{record["kind"]}_score{record["score"]:.2f}.jpg'
                out_path = out_dir / out_name
                record['image'].save(out_path, quality=90)
                samples.append({
                    'image': stem,
                    'source_path': str(img_path),
                    'low_image_path': str(out_path),
                    'quality_label': 'low',
                    'label_score': float(record['synthetic_score']),
                    'target_score': args.target_score,
                    'musiq_score': None if record['musiq_score'] is None else float(record['musiq_score']),
                    'source_musiq': source_score,
                    'degradation_type': record['kind'],
                    'accepted_below_target': bool(record['score'] < args.target_score),
                    'retry': int(record['retry']),
                })
                kept_for_image += 1

        if (img_idx + 1) % 20 == 0:
            accepted_count = sum(1 for s in samples if s['accepted_below_target'])
            logger.info(
                f'[{img_idx+1}/{len(image_paths)}] samples={len(samples)} '
                f'below{args.target_score}={accepted_count} attempts={attempts} '
                f'elapsed={time.time()-t0:.1f}s'
            )

    scores = np.array([s['label_score'] for s in samples], dtype=np.float32) if samples else np.array([])
    musiq_scores = np.array([s['musiq_score'] for s in samples if s['musiq_score'] is not None], dtype=np.float32)
    accepted_count = sum(1 for s in samples if s['accepted_below_target'])

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'method': 'low_score_pixel_degradation',
            'target_score': args.target_score,
            'count': len(samples),
            'accepted_below_target': accepted_count,
            'score_definition': 'label_score is a synthetic low-quality training label in [0, 2); musiq_score is optional reference.',
            'degradation_types': DEGRADATION_TYPES,
            'samples': samples,
        }, f, ensure_ascii=False, indent=2)

    if len(scores) > 0:
        logger.info(f'保存: {out_json}')
        logger.info(
            f'label_score: mean={scores.mean():.2f} std={scores.std():.2f} '
            f'min={scores.min():.2f} max={scores.max():.2f} '
            f'below{args.target_score}={accepted_count}/{len(samples)} '
            f'({accepted_count/max(len(samples),1)*100:.1f}%)'
        )
        if len(musiq_scores) > 0:
            logger.info(
                f'MUSIQ reference: mean={musiq_scores.mean():.2f} std={musiq_scores.std():.2f} '
                f'min={musiq_scores.min():.2f} max={musiq_scores.max():.2f}'
            )
    else:
        logger.warning('没有生成样本')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpeg_dir', default=r'E:\dataset\fivek_jpeg')
    parser.add_argument('--output_dir', default='outputs/data/low_score_images')
    parser.add_argument('--output_json', default='outputs/data/low_score_images.json')
    parser.add_argument('--max_images', type=int, default=100)
    parser.add_argument('--variants_per', type=int, default=2)
    parser.add_argument('--retries', type=int, default=8)
    parser.add_argument('--target_score', type=float, default=2.0)
    parser.add_argument('--score_size', type=int, default=512)
    parser.add_argument('--score_source', action='store_true')
    parser.add_argument('--keep_best', action='store_true')
    parser.add_argument('--no_musiq', action='store_true')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()

    generate_low_score_images(args)
