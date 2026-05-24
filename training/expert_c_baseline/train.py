"""C-GT \u76d1\u7763\u56de\u5f52 baseline \u8bad\u7ec3: MobileViTSmall + Linear 384\u21927.

\u8bad\u7ec3\u76ee\u6807: \u4ece (orig_jpg) \u76f4\u63a5\u9884\u6d4b Expert C \u7684 7D ISP \u53c2\u6570 (\u5f52\u4e00\u5316 [-1, 1]).
\u4e0d\u9700\u8981 Expert C \u6e32\u67d3 JPEG (\u53ea\u7528 GT \u53c2\u6570\u76d1\u7763).

\u8f93\u51fa: checkpoints/expert_c_baseline/{best.pt, last.pt, history.json}
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.vision_encoder import MobileViTSmall  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================
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


def normalize(name: str, value: float) -> float:
    cfg = PARAM_NORM[name]
    v = max(cfg['clip'][0], min(cfg['clip'][1], value))
    return (v - cfg['center']) / cfg['scale']


def denormalize(name: str, norm_value: float) -> float:
    cfg = PARAM_NORM[name]
    return norm_value * cfg['scale'] + cfg['center']


def denormalize_t(name: str, t: torch.Tensor) -> torch.Tensor:
    cfg = PARAM_NORM[name]
    return t * cfg['scale'] + cfg['center']


# ============================================================
# Dataset
# ============================================================
class ExpertCDataset(Dataset):
    """\u8fd4\u56de (orig_jpg \u5f52\u4e00\u5316\u540e Tensor, 7D \u5f52\u4e00\u5316\u53c2\u6570)."""

    def __init__(self, samples: list, jpeg_dir: Path, image_size: int = 224,
                 is_train: bool = True):
        self.samples = samples
        self.jpeg_dir = Path(jpeg_dir)
        self.image_size = image_size
        self.is_train = is_train
        self.resize = T.Resize((image_size, image_size))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.hflip = T.RandomHorizontalFlip(1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        stem = s['image_name'].rsplit('.', 1)[0]
        path = self.jpeg_dir / f'{stem}.jpg'
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            logger.warning(f'load fail {path}: {e}')
            img = Image.new('RGB', (self.image_size, self.image_size))
        img = self.resize(img)
        if self.is_train and random.random() < 0.5:
            img = self.hflip(img)
        t = self.normalize(self.to_tensor(img))

        params_norm = torch.tensor(
            [normalize(p, s['params'].get(p) or 0.0)
             for p in PARAM_NAMES], dtype=torch.float32)
        return {
            'image': t,
            'params_norm': params_norm,
            'image_name': s['image_name'],
        }


# ============================================================
# Model
# ============================================================
class ExpertC7DModel(nn.Module):
    """MobileViTSmall (~2.89M) + Linear(384 -> 7) head."""

    def __init__(self, image_size: int = 224, visual_dim: int = 384):
        super().__init__()
        self.encoder = MobileViTSmall(
            image_size=image_size,
            output_dim=visual_dim,
            use_se=True,
            use_fpn=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Dropout(0.1),
            nn.Linear(visual_dim, visual_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(visual_dim // 2, 7),
            nn.Tanh(),  # [-1, 1]
        )

    def forward(self, images):
        feat = self.encoder(images)        # (B, 384)
        return self.head(feat)             # (B, 7)


# ============================================================
# Loss / Metrics
# ============================================================
def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                      weights: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2 * weights.view(1, -1)).mean()


def per_param_metrics(pred_norm: torch.Tensor, target_norm: torch.Tensor):
    """\u8fd4\u56de\u7269\u7406\u7a7a\u95f4 per-param MAE / Pearson R."""
    pred_phys = torch.stack(
        [denormalize_t(p, pred_norm[:, i]) for i, p in enumerate(PARAM_NAMES)],
        dim=1)
    tgt_phys = torch.stack(
        [denormalize_t(p, target_norm[:, i]) for i, p in enumerate(PARAM_NAMES)],
        dim=1)
    mae = (pred_phys - tgt_phys).abs().mean(dim=0).cpu().numpy()

    # Pearson R
    p_np = pred_phys.cpu().numpy()
    t_np = tgt_phys.cpu().numpy()
    pearson_r = []
    for i in range(7):
        p = p_np[:, i]
        t = t_np[:, i]
        if p.std() < 1e-6 or t.std() < 1e-6:
            pearson_r.append(0.0)
        else:
            pearson_r.append(float(np.corrcoef(p, t)[0, 1]))
    return mae, np.array(pearson_r)


# ============================================================
# Data prep
# ============================================================
def build_data(gt_json: Path, jpeg_dir: Path, expert: str = 'C',
               val_ratio: float = 0.1, seed: int = 42):
    with open(gt_json, encoding='utf-8') as f:
        gt = json.load(f)
    samples = [s for s in gt['samples'] if s['expert'] == expert]
    logger.info(f'Expert {expert}: {len(samples)} GT records')

    # \u8fc7\u6ee4: \u539f\u56fe\u5fc5\u987b\u5b58\u5728 + WB \u6709\u503c
    kept = []
    for s in samples:
        stem = s['image_name'].rsplit('.', 1)[0]
        p = jpeg_dir / f'{stem}.jpg'
        if not p.exists():
            continue
        if s['params'].get('white_balance') is None:
            # \u7f3a\u5931 WB \u7684\u4e0d\u8981 (\u5176\u4ed6\u53c2\u6570 None \u4e5f\u8bdd, \u8fd9\u91cc\u53ea\u68c0 wb)
            continue
        kept.append(s)
    logger.info(f'After filter (jpeg+wb): {len(kept)}')

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_val = int(len(kept) * val_ratio)
    val = kept[:n_val]
    train = kept[n_val:]
    logger.info(f'train={len(train)}, val={len(val)}')
    return train, val


# ============================================================
# Trainer
# ============================================================
def train_one_epoch(model, loader, opt, loss_w, device, log_every=50):
    model.train()
    tot = 0.0
    n = 0
    for i, batch in enumerate(loader):
        img = batch['image'].to(device, non_blocking=True)
        tgt = batch['params_norm'].to(device, non_blocking=True)
        pred = model(img)
        loss = weighted_mse_loss(pred, tgt, loss_w)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item()
        n += 1
        if (i + 1) % log_every == 0:
            logger.info(f'  [{i+1}/{len(loader)}] loss={tot/n:.4f}')
    return tot / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_w, device):
    model.eval()
    losses = 0.0
    all_pred = []
    all_tgt = []
    n = 0
    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        tgt = batch['params_norm'].to(device, non_blocking=True)
        pred = model(img)
        loss = weighted_mse_loss(pred, tgt, loss_w)
        losses += loss.item()
        all_pred.append(pred.cpu())
        all_tgt.append(tgt.cpu())
        n += 1
    losses /= max(n, 1)
    pred_all = torch.cat(all_pred, dim=0)
    tgt_all = torch.cat(all_tgt, dim=0)
    mae_phys, r = per_param_metrics(pred_all, tgt_all)
    return losses, mae_phys, r


def mean_baseline_metrics(train_samples, val_samples):
    """\u8ba1\u7b97 \u201c\u9884\u6d4b\u8bad\u7ec3\u96c6\u5747\u503c\u201d \u7684 baseline metrics (sanity check upper bound)."""
    means = {}
    for p in PARAM_NAMES:
        vs = [s['params'].get(p) or 0.0 for s in train_samples]
        means[p] = float(np.mean(vs))
    val_mae = []
    for p in PARAM_NAMES:
        vs = [s['params'].get(p) or 0.0 for s in val_samples]
        val_mae.append(float(np.mean([abs(v - means[p]) for v in vs])))
    return means, np.array(val_mae)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt_json', default='data/fivek_expert_abcde_params.json')
    ap.add_argument('--jpeg_dir', default=r'E:\Data\dataset\fivek_jpeg')
    ap.add_argument('--expert', default='C', choices=list('ABCDE'))
    ap.add_argument('--out_dir', default='checkpoints/expert_c_baseline')
    ap.add_argument('--image_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--clarity_weight', type=float, default=0.1,
                    help='clarity \u51e0\u4e4e\u5168 0, \u52a8\u636e\u9650\u4ec5\u4e0e MAE \u5360\u6bd4\u4f4e')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device={device}, lr={args.lr}, batch={args.batch_size}, '
                f'epochs={args.epochs}')

    # \u6570\u636e
    train_s, val_s = build_data(
        Path(args.gt_json), Path(args.jpeg_dir), args.expert,
        args.val_ratio, args.seed)
    train_ds = ExpertCDataset(train_s, args.jpeg_dir, args.image_size, True)
    val_ds = ExpertCDataset(val_s, args.jpeg_dir, args.image_size, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # \u9884\u6d4b\u5747\u503c baseline
    mean_params, mean_mae = mean_baseline_metrics(train_s, val_s)
    logger.info('\n=== "predict mean" baseline (val MAE physical) ===')
    for i, p in enumerate(PARAM_NAMES):
        logger.info(f'  {p:<15s} mean_pred={mean_params[p]:+8.2f}  '
                    f'val_mae={mean_mae[i]:8.2f}')

    # \u6a21\u578b
    model = ExpertC7DModel(image_size=args.image_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'\nmodel: {n_params/1e6:.2f}M params')

    opt = optim.AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Per-param loss weights (clarity 低权重, 其它等权)
    loss_w = torch.ones(7, device=device)
    loss_w[6] = args.clarity_weight  # clarity
    logger.info(f'loss weights: {loss_w.cpu().tolist()}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, loss_w, device,
                                  log_every=args.log_every)
        val_loss, val_mae, val_r = evaluate(model, val_loader, loss_w, device)
        sched.step()
        elapsed = time.time() - t0

        log_line = (f'Ep {epoch:3d}/{args.epochs}  '
                    f'train={tr_loss:.4f}  val={val_loss:.4f}  '
                    f'elapsed={elapsed:.1f}s  lr={sched.get_last_lr()[0]:.2e}')
        logger.info(log_line)

        per_param_str = '  '.join(
            f'{p[:4]}=MAE{val_mae[i]:.1f},R{val_r[i]:.2f}'
            for i, p in enumerate(PARAM_NAMES))
        logger.info(f'  per-param: {per_param_str}')

        history.append({
            'epoch': epoch,
            'train_loss': tr_loss,
            'val_loss': val_loss,
            'val_mae': val_mae.tolist(),
            'val_pearson_r': val_r.tolist(),
            'lr': sched.get_last_lr()[0],
            'elapsed': elapsed,
        })
        json.dump(history, open(out_dir / 'history.json', 'w'), indent=2)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae.tolist(),
                'val_pearson_r': val_r.tolist(),
                'args': vars(args),
            }, out_dir / 'best.pt')
            logger.info(f'  * best val={val_loss:.4f}  saved')

    # final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_mae': val_mae.tolist(),
        'val_pearson_r': val_r.tolist(),
        'args': vars(args),
    }, out_dir / 'last.pt')

    logger.info(f'\n{"="*78}\n[DONE] best_val_loss={best_val:.4f}')
    logger.info(f'  checkpoint: {out_dir/"best.pt"}')
    logger.info(f'  history:    {out_dir/"history.json"}\n{"="*78}')


if __name__ == '__main__':
    main()
