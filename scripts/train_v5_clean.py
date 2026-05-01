#!/usr/bin/env python3
import sys, time, logging
sys.path.insert(0, '/root/autodl-tmp/MobileVenus')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
from training.fivek_8param.dataset_expert import build_expert_datasets
from models.diff_isp import apply_diff_isp, ssim_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
    logging.FileHandler('/root/autodl-tmp/train_v5_clean.log'),
    logging.StreamHandler()])
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = Path('/root/autodl-tmp/checkpoints/distill_v5/stage_b')
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS, BS, LR, IMG_W, PARAM_W = 50, 16, 1e-4, 0.5, 0.1

def denorm(pn):
    r = {}
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        r[name] = (lo + (pn[:, i].float() + 1.0) / 2.0 * (hi - lo)).to(device)
    return r

train_ds, val_ds = build_expert_datasets(
    data_file='/root/autodl-tmp/data/fivek_expert_params.json',
    consensus_file='/root/autodl-tmp/data/fivek_expert_consensus.json',
    orig_jpeg_dir='/root/autodl-tmp/fivek_jpeg',
    expert_dirs={'c': '/root/autodl-tmp/fivek_expert_c'},
    image_size=224, val_ratio=0.1)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BS, shuffle=False, num_workers=4, pin_memory=True)
logger.info(f'Train={len(train_ds)} Val={len(val_ds)}')

sa = SemanticDistillModel()
st = torch.load('/root/autodl-tmp/checkpoints/distill_v5/stage_a/best.pt', map_location='cpu', weights_only=False)
sa.load_state_dict(st['model_state_dict'])
model = DistillParamModel(sa).to(device)
model.freeze_backbone()

optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
best_val = float('inf')

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    if epoch == 11:
        model.unfreeze_backbone()
        optimizer = optim.AdamW([
            {'params': model.decoder.parameters(),            'lr': LR * 0.5},
            {'params': model.vision_encoder.parameters(),     'lr': LR * 0.05},
            {'params': model.semantic_projector.parameters(), 'lr': LR * 0.05},
        ], weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)
        logger.info('★ backbone unfrozen')

    model.train()
    sl, si, sp, n = 0., 0., 0., 0
    for batch in train_loader:
        img    = batch['image'].to(device)
        raw    = batch['raw_image'].to(device).float()
        expert = batch['expert_image'].to(device).float()
        params_gt = batch['params'].to(device).float()
        weights   = batch['weights'].to(device).float()

        out = model(img)
        pn  = out['params_norm'].float()
        phys = denorm(pn)
        with torch.no_grad():
            pred_img = apply_diff_isp(raw, {k: v.detach() for k, v in phys.items()})
        i_loss  = F.l1_loss(pred_img, expert) + ssim_loss(pred_img, expert)  # for monitoring
        mse     = ((pn - params_gt)**2 * weights).sum() / (weights.sum() + 1e-8)
        loss    = mse  # gradient only through param mse, ISP detached

        if not torch.isfinite(loss):
            logger.warning(f'skip batch (loss={loss.item()})')
            continue

        optimizer.zero_grad()
        loss.backward()

        # Replace NaN/Inf gradients with 0 instead of skipping
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sl += loss.item(); si += i_loss.item(); sp += mse.item(); n += 1

    model.eval()
    vs, vn = 0., 0
    with torch.no_grad():
        for batch in val_loader:
            raw    = batch['raw_image'].to(device).float()
            expert = batch['expert_image'].to(device).float()
            img    = batch['image'].to(device)
            pn     = model(img)['params_norm'].float()
            pi     = apply_diff_isp(raw, denorm(pn))
            v      = F.l1_loss(pi, expert) + ssim_loss(pi, expert)
            if torch.isfinite(v): vs += v.item(); vn += 1
    vl = vs / max(vn, 1)
    scheduler.step()

    logger.info(f'Ep {epoch}/{EPOCHS} loss={sl/max(n,1):.4f} img={si/max(n,1):.4f} p={sp/max(n,1):.4f} val={vl:.4f} n={n} t={time.time()-t0:.0f}s')
    ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': vl}
    torch.save(ckpt, OUT_DIR / 'last.pt')
    if vl < best_val:
        best_val = vl
        torch.save(ckpt, OUT_DIR / 'best.pt')
        logger.info(f'  ★ best={best_val:.4f}')

logger.info(f'Done. best_val={best_val:.4f}')
