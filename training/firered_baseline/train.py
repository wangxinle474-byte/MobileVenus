"""FireRed 伪标签条件回归训练 baseline.

训练目标: (orig_jpg, action ∈ {contrast,saturation,shadows,highlights,wb})
        → P_inferred 7D ISP 参数 (FireRed 风格).

数据源: outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl (N=499)
过滤:   tier ∈ {A excellent, B good, C acceptable} (372 张); 丢 D fail (127 张)
监督:   tier-aware loss weight (A=1.2, B=1.0, C=0.5)
模型:   MobileViTSmall(384) + action one-hot 5D + Linear(389→7) + Tanh

vs `expert_c_baseline`: 数据 12× 少 (372 vs 4498) → 强 dropout, color jitter, 早停.
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
ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}

# 与 expert_c_baseline 一致 (Lightroom 物理空间)
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

TIER_WEIGHTS = {
    'A excellent': 1.2,
    'B good':      1.0,
    'C acceptable': 0.5,
    # 'D fail' 不进训练
}

# 每个 action 主导的参数 (用于 action-aware loss boost)
ACTION_PRIMARY_PARAM = {
    'contrast':   'contrast',
    'saturation': 'saturation',
    'shadows':    'shadows',
    'highlights': 'highlights',
    'wb':         'white_balance',
}


def normalize_param(name: str, value: float) -> float:
    cfg = PARAM_NORM[name]
    v = max(cfg['clip'][0], min(cfg['clip'][1], value))
    return (v - cfg['center']) / cfg['scale']


def denormalize_t(name: str, t: torch.Tensor) -> torch.Tensor:
    cfg = PARAM_NORM[name]
    return t * cfg['scale'] + cfg['center']


# ============================================================
# Dataset
# ============================================================
class FireRedPseudoDataset(Dataset):
    """返回 (image_tensor, action_onehot, params_norm, tier_weight)."""

    def __init__(self, samples: list, image_size: int = 256,
                 is_train: bool = True):
        self.samples = samples
        self.image_size = image_size
        self.is_train = is_train

        self.resize = T.Resize((image_size, image_size))
        self.to_tensor = T.ToTensor()
        self.norm = T.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.color_jitter = T.ColorJitter(
            brightness=0.05, contrast=0.05, saturation=0.05, hue=0.0)
        self.hflip = T.RandomHorizontalFlip(1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s['orig_path']).convert('RGB')
        except Exception as e:
            logger.warning(f'load fail {s["orig_path"]}: {e}')
            img = Image.new('RGB', (self.image_size, self.image_size))
        img = self.resize(img)
        if self.is_train:
            if random.random() < 0.5:
                img = self.hflip(img)
            img = self.color_jitter(img)
        t = self.norm(self.to_tensor(img))

        action_idx = ACTION_TO_IDX[s['action']]
        action_oh = torch.zeros(len(ACTIONS), dtype=torch.float32)
        action_oh[action_idx] = 1.0

        P = s['P_inferred']
        params_norm = torch.tensor(
            [normalize_param(p, P.get(p) or 0.0) for p in PARAM_NAMES],
            dtype=torch.float32)

        tier_w = TIER_WEIGHTS.get(s['quality_tier'], 0.0)

        return {
            'image': t,
            'action_onehot': action_oh,
            'action_idx': action_idx,
            'params_norm': params_norm,
            'tier_weight': torch.tensor(tier_w, dtype=torch.float32),
            'source_image': s['source_image'],
            'action': s['action'],
        }


# ============================================================
# Model
# ============================================================
class FireRed7DModel(nn.Module):
    """MobileViTSmall(384) + action one-hot 5D → Linear(389→64→7) Tanh."""

    def __init__(self, image_size: int = 256, visual_dim: int = 384,
                 n_actions: int = 5, dropout: float = 0.3):
        super().__init__()
        self.encoder = MobileViTSmall(
            image_size=image_size,
            output_dim=visual_dim,
            use_se=True,
            use_fpn=True,
        )
        self.action_emb = nn.Linear(n_actions, 32)
        fused_dim = visual_dim + 32
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 7),
            nn.Tanh(),
        )

    def forward(self, images, action_onehot):
        feat = self.encoder(images)            # (B, 384)
        a = self.action_emb(action_onehot)     # (B, 32)
        fused = torch.cat([feat, a], dim=1)    # (B, 416)
        return self.head(fused)                # (B, 7) ∈ [-1, 1]


# ============================================================
# Loss / Metrics
# ============================================================
def weighted_mse_loss(pred, target, param_w, sample_w):
    """sample_w: (B,) tier weight; param_w: (7,)."""
    err2 = (pred - target) ** 2 * param_w.view(1, -1)  # (B, 7)
    err2 = err2.mean(dim=1) * sample_w                  # (B,)
    if sample_w.sum() < 1e-6:
        return err2.mean()
    return err2.sum() / (sample_w.sum() + 1e-6)


def per_param_metrics(pred_norm, target_norm):
    pred_phys = torch.stack(
        [denormalize_t(p, pred_norm[:, i]) for i, p in enumerate(PARAM_NAMES)],
        dim=1)
    tgt_phys = torch.stack(
        [denormalize_t(p, target_norm[:, i]) for i, p in enumerate(PARAM_NAMES)],
        dim=1)
    mae = (pred_phys - tgt_phys).abs().mean(dim=0).cpu().numpy()
    p_np = pred_phys.cpu().numpy()
    t_np = tgt_phys.cpu().numpy()
    pearson_r = []
    for i in range(7):
        a, b = p_np[:, i], t_np[:, i]
        if a.std() < 1e-6 or b.std() < 1e-6:
            pearson_r.append(0.0)
        else:
            pearson_r.append(float(np.corrcoef(a, b)[0, 1]))
    return mae, np.array(pearson_r)


def per_action_param_metrics(pred_norm, target_norm, action_idxs):
    """对每个 action, 算其 primary 参数的 MAE/R."""
    pred_phys = torch.stack(
        [denormalize_t(p, pred_norm[:, i]) for i, p in enumerate(PARAM_NAMES)],
        dim=1).cpu().numpy()
    tgt_phys = torch.stack(
        [denormalize_t(p, target_norm[:, i]) for i, p in enumerate(PARAM_NAMES)],
        dim=1).cpu().numpy()
    action_idxs = action_idxs.cpu().numpy()
    out = {}
    for ai, action in enumerate(ACTIONS):
        mask = action_idxs == ai
        if mask.sum() < 2:
            continue
        primary = ACTION_PRIMARY_PARAM[action]
        pi = PARAM_NAMES.index(primary)
        a = pred_phys[mask, pi]
        b = tgt_phys[mask, pi]
        mae = float(np.mean(np.abs(a - b)))
        r = (float(np.corrcoef(a, b)[0, 1])
             if a.std() > 1e-6 and b.std() > 1e-6 else 0.0)
        out[action] = {'primary': primary, 'mae': mae, 'r': r,
                       'n': int(mask.sum())}
    return out


# ============================================================
# Data prep
# ============================================================
def build_data(jsonl_path: Path, val_ratio: float = 0.2,
               tier_filter=('A excellent', 'B good', 'C acceptable'),
               seed: int = 42):
    samples = []
    n_total = 0
    n_orig_missing = 0
    tier_counts = {}
    for line in open(jsonl_path, encoding='utf-8'):
        r = json.loads(line)
        n_total += 1
        tier = r.get('quality_tier', 'D fail')
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if tier not in tier_filter:
            continue
        if not Path(r['orig_path']).exists():
            n_orig_missing += 1
            continue
        samples.append(r)
    logger.info(f'jsonl total: {n_total}; tier dist: {tier_counts}')
    logger.info(f'after filter ({list(tier_filter)}): {len(samples)} '
                f'(orig missing: {n_orig_missing})')

    # 按 (action, tier) 分层抽样保证 val 多样性
    rng = random.Random(seed)
    by_action_tier = {}
    for s in samples:
        k = (s['action'], s['quality_tier'])
        by_action_tier.setdefault(k, []).append(s)
    val, train = [], []
    for k, lst in by_action_tier.items():
        rng.shuffle(lst)
        n_val = max(1, int(len(lst) * val_ratio))
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    logger.info(f'train={len(train)}, val={len(val)} '
                f'(action-tier stratified)')
    return train, val


# ============================================================
# Trainer
# ============================================================
def train_one_epoch(model, loader, opt, param_w, device, log_every=50):
    model.train()
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        img = batch['image'].to(device, non_blocking=True)
        a_oh = batch['action_onehot'].to(device, non_blocking=True)
        tgt = batch['params_norm'].to(device, non_blocking=True)
        sw = batch['tier_weight'].to(device, non_blocking=True)
        pred = model(img, a_oh)
        loss = weighted_mse_loss(pred, tgt, param_w, sw)
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
def evaluate(model, loader, param_w, device):
    model.eval()
    losses, n = 0.0, 0
    all_pred, all_tgt, all_act = [], [], []
    for batch in loader:
        img = batch['image'].to(device, non_blocking=True)
        a_oh = batch['action_onehot'].to(device, non_blocking=True)
        tgt = batch['params_norm'].to(device, non_blocking=True)
        sw = batch['tier_weight'].to(device, non_blocking=True)
        pred = model(img, a_oh)
        loss = weighted_mse_loss(pred, tgt, param_w, sw)
        losses += loss.item()
        n += 1
        all_pred.append(pred.cpu())
        all_tgt.append(tgt.cpu())
        all_act.append(batch['action_idx'])
    losses /= max(n, 1)
    pred_all = torch.cat(all_pred, dim=0)
    tgt_all = torch.cat(all_tgt, dim=0)
    act_all = torch.cat(all_act, dim=0)
    mae, r = per_param_metrics(pred_all, tgt_all)
    per_act = per_action_param_metrics(pred_all, tgt_all, act_all)
    return losses, mae, r, per_act


def mean_baseline(train_samples, val_samples):
    """对每个 action, 用训练集均值作为预测 → val per-action primary MAE."""
    by_action_train = {}
    for s in train_samples:
        by_action_train.setdefault(s['action'], []).append(s)
    means = {}  # {action: {param: mean}}
    for a, lst in by_action_train.items():
        means[a] = {p: float(np.mean(
            [si['P_inferred'].get(p) or 0.0 for si in lst])) for p in PARAM_NAMES}
    out = {}
    for a in ACTIONS:
        primary = ACTION_PRIMARY_PARAM[a]
        val_a = [s for s in val_samples if s['action'] == a]
        if not val_a or a not in means:
            continue
        m = means[a][primary]
        mae = float(np.mean([abs((s['P_inferred'].get(primary) or 0.0) - m)
                             for s in val_a]))
        out[a] = {'primary': primary, 'mean_pred': m, 'mae': mae,
                  'n': len(val_a)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl',
                    default='outputs/inverse_fit_pilot/fivek_500_master/'
                            'pseudo_labels.jsonl')
    ap.add_argument('--out_dir', default='checkpoints/firered_v1')
    ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_every', type=int, default=20)
    ap.add_argument('--clarity_weight', type=float, default=0.1)
    ap.add_argument('--patience', type=int, default=8,
                    help='early stop patience (轮数)')
    ap.add_argument('--include_d', action='store_true',
                    help='也加入 D fail (默认 false)')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'device={device}, lr={args.lr}, batch={args.batch_size}, '
                f'epochs={args.epochs}, dropout={args.dropout}')

    tier_filter = ['A excellent', 'B good', 'C acceptable']
    if args.include_d:
        tier_filter.append('D fail')
    train_s, val_s = build_data(
        Path(args.jsonl), args.val_ratio, tuple(tier_filter), args.seed)

    # action 分布
    from collections import Counter
    logger.info(f'train action dist: {dict(Counter([s["action"] for s in train_s]))}')
    logger.info(f'val   action dist: {dict(Counter([s["action"] for s in val_s]))}')

    train_ds = FireRedPseudoDataset(train_s, args.image_size, is_train=True)
    val_ds = FireRedPseudoDataset(val_s, args.image_size, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # baseline (per-action, primary param)
    base = mean_baseline(train_s, val_s)
    logger.info('\n=== "predict per-action mean" baseline (val MAE) ===')
    for a in ACTIONS:
        if a in base:
            b = base[a]
            logger.info(f'  {a:<11s} primary={b["primary"]:<14s} '
                        f'mean_pred={b["mean_pred"]:+8.2f}  '
                        f'mae={b["mae"]:8.2f}  n={b["n"]}')

    # 模型
    model = FireRed7DModel(image_size=args.image_size,
                           dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'\nmodel: {n_params/1e6:.2f}M params')

    opt = optim.AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    param_w = torch.ones(7, device=device)
    param_w[6] = args.clarity_weight  # clarity
    logger.info(f'param weights: {param_w.cpu().tolist()}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float('inf')
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, param_w, device,
                                  log_every=args.log_every)
        val_loss, val_mae, val_r, val_per_act = evaluate(
            model, val_loader, param_w, device)
        sched.step()
        elapsed = time.time() - t0

        logger.info(f'Ep {epoch:3d}/{args.epochs}  '
                    f'train={tr_loss:.4f}  val={val_loss:.4f}  '
                    f'elapsed={elapsed:.1f}s  lr={sched.get_last_lr()[0]:.2e}')

        per_param_str = '  '.join(
            f'{p[:4]}=MAE{val_mae[i]:.1f},R{val_r[i]:+.2f}'
            for i, p in enumerate(PARAM_NAMES))
        logger.info(f'  per-param: {per_param_str}')

        per_act_str = '  '.join(
            f'{a[:4]}({v["primary"][:4]})=MAE{v["mae"]:.1f},R{v["r"]:+.2f}'
            for a, v in val_per_act.items())
        logger.info(f'  per-action(primary): {per_act_str}')

        history.append({
            'epoch': epoch,
            'train_loss': tr_loss,
            'val_loss': val_loss,
            'val_mae': val_mae.tolist(),
            'val_pearson_r': val_r.tolist(),
            'val_per_action': val_per_act,
            'lr': sched.get_last_lr()[0],
            'elapsed': elapsed,
        })
        json.dump(history, open(out_dir / 'history.json', 'w'),
                  indent=2, default=float)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae.tolist(),
                'val_pearson_r': val_r.tolist(),
                'val_per_action': val_per_act,
                'baseline_per_action': base,
                'args': vars(args),
            }, out_dir / 'best.pt')
            logger.info(f'  * best val={val_loss:.4f}  saved')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f'  early stop: no improve for {no_improve} epochs')
                break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_mae': val_mae.tolist(),
        'val_pearson_r': val_r.tolist(),
        'val_per_action': val_per_act,
        'args': vars(args),
    }, out_dir / 'last.pt')

    logger.info(f'\n{"="*78}\n[DONE] best_val_loss={best_val:.4f} @ Ep{best_epoch}')
    logger.info(f'  checkpoint: {out_dir/"best.pt"}')
    logger.info(f'  history:    {out_dir/"history.json"}\n{"="*78}')


if __name__ == '__main__':
    main()
