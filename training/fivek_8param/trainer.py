"""
训练循环: 8参数 + 一致性加权 + 可选语义蒸馏
"""
import os
import json
import math
import time
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import TrainConfig, PARAM_NAMES, PARAM_RANGES
from .dataset import build_datasets, denormalize_param
from .model import FiveK8ParamModel
from .loss import CombinedLoss

logger = logging.getLogger(__name__)


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """计算各参数的 MAE (原始范围)"""
    mae = {}
    with torch.no_grad():
        for i, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_RANGES[name]
            pred_raw = (pred[:, i] + 1.0) / 2.0 * (hi - lo) + lo
            tgt_raw = (target[:, i] + 1.0) / 2.0 * (hi - lo) + lo
            mae[name] = (pred_raw - tgt_raw).abs().mean().item()
    return mae


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, epoch, log_every=50,
):
    model.train()
    total_metrics = {}
    n = 0
    for i, batch in enumerate(loader):
        images = batch['image'].to(device)
        params = batch['params'].to(device)
        weights = batch['weights'].to(device)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            outputs = model(images, return_embedding=True)
            loss, metrics = criterion(
                outputs['params_norm'], params, weights,
                student_emb=outputs.get('semantic_emb'),
            )

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n += 1

        if (i + 1) % log_every == 0:
            avg = total_metrics['total'] / n
            logger.info(f"  [{i+1}/{len(loader)}] loss={avg:.4f}")

    return {k: v / n for k, v in total_metrics.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_metrics = {}
    all_mae = {}
    n = 0
    for batch in loader:
        images = batch['image'].to(device)
        params = batch['params'].to(device)
        weights = batch['weights'].to(device)

        outputs = model(images, return_embedding=True)
        _, metrics = criterion(outputs['params_norm'], params, weights)

        mae = compute_mae(outputs['params_norm'], params)
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        for k, v in mae.items():
            all_mae[k] = all_mae.get(k, 0) + v
        n += 1

    avg_metrics = {k: v / n for k, v in total_metrics.items()}
    avg_mae = {k: v / n for k, v in all_mae.items()}
    return avg_metrics, avg_mae


def train(cfg: TrainConfig):
    """主训练入口"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()],
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"设备: {device}")

    # 数据
    train_ds, val_ds = build_datasets(
        cfg.data_file, cfg.consensus_file, cfg.jpeg_dir,
        cfg.image_size, cfg.val_ratio, cfg.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # 模型
    model = FiveK8ParamModel(
        image_size=cfg.image_size,
        visual_dim=cfg.visual_dim,
        semantic_dim=cfg.semantic_dim,
        decoder_hidden=cfg.decoder_hidden,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params/1e6:.2f}M")

    # 损失
    criterion = CombinedLoss(
        use_consensus=cfg.consensus_weight,
        distill_weight=cfg.distill_weight,
    )

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr,
    )
    scaler = torch.amp.GradScaler('cuda') if cfg.fp16 and device.type == 'cuda' else None

    # 输出
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'config.json', 'w') as f:
        json.dump(vars(cfg), f, indent=2)

    best_val = float('inf')
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        logger.info(f"Epoch {epoch}/{cfg.epochs} (lr={optimizer.param_groups[0]['lr']:.2e})")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg.log_every,
        )
        val_metrics, val_mae = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        mae_str = ' | '.join(f"{k}={v:.2f}" for k, v in val_mae.items())
        logger.info(
            f"Epoch {epoch} | train={train_metrics['total']:.4f} | "
            f"val={val_metrics['total']:.4f} | {mae_str} | {elapsed:.1f}s"
        )

        # 保存历史
        record = {'epoch': epoch, 'train': train_metrics, 'val': val_metrics, 'mae': val_mae,
                   'lr': optimizer.param_groups[0]['lr']}
        history.append(record)

        # 保存最优
        if val_metrics['total'] < best_val:
            best_val = val_metrics['total']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_metrics': val_metrics, 'val_mae': val_mae},
                       out_dir / 'best.pt')
            logger.info(f"  ★ 新最优 val_loss={best_val:.4f}")

    # 保存最终
    torch.save({'epoch': cfg.epochs, 'model_state_dict': model.state_dict()},
               out_dir / 'final.pt')

    with open(out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"\n训练完成! 最优 val_loss={best_val:.4f}")
    logger.info(f"权重保存到: {out_dir}")
