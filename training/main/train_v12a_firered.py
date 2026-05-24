from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from models.diff_isp import apply_diff_isp, ssim_loss
from models.refinement_net_v4 import RefinementNetV4
from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
from training.semantic_distill.config import DistillConfig
from training.semantic_distill.model import DistillParamModel, SemanticDistillModel
from models.legacy_v8_param_model import LegacyDistillParamModel


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ACTIONS = ['contrast', 'saturation', 'shadows', 'highlights', 'wb', 'brightness', 'clarity']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}


class FireRedV12ADataset(Dataset):
    def __init__(self, samples, repo_root: Path, orig_dir: Path | None,
                 is_train: bool = True, val_action_dropout: bool = False):
        self.samples = samples
        self.repo_root = repo_root
        self.orig_dir = orig_dir
        self.is_train = is_train
        self.val_action_dropout = val_action_dropout
        self.resize_224 = T.Resize((224, 224))
        self.resize_512 = T.Resize((512, 512))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.hflip = T.RandomHorizontalFlip(1.0)

    def __len__(self):
        return len(self.samples)

    def _resolve_path(self, path_value: str, fallback_dir: Path | None = None) -> Path:
        p = Path(path_value)
        if p.exists():
            return p
        if not p.is_absolute():
            q = self.repo_root / p
            if q.exists():
                return q
        if fallback_dir is not None:
            q = fallback_dir / p.name
            if q.exists():
                return q
        return p

    def __getitem__(self, idx):
        s = self.samples[idx]
        orig_path = self._resolve_path(s['orig_path'], self.orig_dir)
        target_path = self._resolve_path(s['target_path'])
        try:
            pil_orig = Image.open(orig_path).convert('RGB')
            pil_target = Image.open(target_path).convert('RGB')
        except Exception:
            image_224 = torch.zeros(3, 224, 224)
            raw_512 = torch.zeros(3, 512, 512)
            target_512 = torch.zeros(3, 512, 512)
            action_oh = torch.zeros(len(ACTIONS), dtype=torch.float32)
            return {
                'image_224': image_224,
                'raw_512': raw_512,
                'target_512': target_512,
                'action_onehot': action_oh,
            }

        do_flip = self.is_train and random.random() < 0.5

        orig_224 = self.resize_224(pil_orig)
        orig_512 = self.resize_512(pil_orig)
        target_512 = self.resize_512(pil_target)
        if do_flip:
            orig_224 = self.hflip(orig_224)
            orig_512 = self.hflip(orig_512)
            target_512 = self.hflip(target_512)

        action = s.get('action') or s.get('tone_target')
        action_oh = torch.zeros(len(ACTIONS), dtype=torch.float32)
        if action in ACTION_TO_IDX:
            action_oh[ACTION_TO_IDX[action]] = 1.0
        if self.val_action_dropout and not self.is_train:
            action_oh.zero_()

        return {
            'image_224': self.normalize(self.to_tensor(orig_224)),
            'raw_512': self.to_tensor(orig_512),
            'target_512': self.to_tensor(target_512),
            'action_onehot': action_oh,
        }


class ActionConditionedRefinementNetV4(nn.Module):
    def __init__(self, base_ch=64, n_actions=7):
        super().__init__()
        self.refine = RefinementNetV4(base_ch=base_ch)
        self.action_affine = nn.Sequential(
            nn.Linear(n_actions, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )
        nn.init.zeros_(self.action_affine[-1].weight)
        nn.init.zeros_(self.action_affine[-1].bias)

    def forward(self, x, action_onehot):
        affine = self.action_affine(action_onehot).view(-1, 6, 1, 1)
        gamma, beta = affine[:, :3], affine[:, 3:]
        x_cond = (x * (1.0 + 0.25 * torch.tanh(gamma))
                  + 0.25 * torch.tanh(beta)).clamp(0, 1)
        return self.refine(x_cond)

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def make_denorm_fn():
    lo_list = [PARAM_RANGES[p][0] for p in PARAM_NAMES]
    hi_list = [PARAM_RANGES[p][1] for p in PARAM_NAMES]

    def denorm(params_norm, device):
        lo = torch.tensor(lo_list, device=device, dtype=params_norm.dtype)
        hi = torch.tensor(hi_list, device=device, dtype=params_norm.dtype)
        params_phys = lo.unsqueeze(0) + (params_norm + 1.0) / 2.0 * (hi - lo).unsqueeze(0)
        return {name: params_phys[:, i] for i, name in enumerate(PARAM_NAMES)}

    return denorm


def psnr_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred.float(), target.float(), reduction='none')
    mse = mse.flatten(1).mean(dim=1).clamp_min(eps)
    return (10.0 * torch.log10(1.0 / mse)).mean()


def load_jsonl(path: Path):
    samples = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def split_by_source(samples, val_ratio: float, seed: int):
    sources = sorted({s.get('source_image') or Path(s['orig_path']).name for s in samples})
    rng = random.Random(seed)
    rng.shuffle(sources)
    n_val = max(1, int(len(sources) * val_ratio))
    val_sources = set(sources[:n_val])
    train = [s for s in samples if (s.get('source_image') or Path(s['orig_path']).name) not in val_sources]
    val = [s for s in samples if (s.get('source_image') or Path(s['orig_path']).name) in val_sources]
    return train, val


def load_param_model(asset_root: Path, param_version: str, device: torch.device):
    cfg = DistillConfig()
    pm_path = asset_root / 'checkpoints' / f'distill_{param_version}' / 'stage_b' / 'best.pt'
    try:
        stage_a = SemanticDistillModel(
            image_size=cfg.image_size,
            visual_dim=cfg.visual_dim,
            semantic_dim=cfg.semantic_dim,
            text_dim=cfg.text_dim,
        )
        stage_a_path = asset_root / 'checkpoints' / 'distill_v6' / 'stage_a' / 'best.pt'
        stage_a_state = torch.load(str(stage_a_path), map_location='cpu', weights_only=False)
        stage_a.load_state_dict(stage_a_state['model_state_dict'])
        param_model = DistillParamModel(stage_a, decoder_hidden=cfg.decoder_hidden)
        pm_state = torch.load(str(pm_path), map_location='cpu', weights_only=False)
        param_model.load_state_dict(pm_state['model_state_dict'])
    except RuntimeError:
        logger.warning('Param model standard load failed, trying legacy v8 loader')
        pm_state = torch.load(str(pm_path), map_location='cpu', weights_only=False)
        param_model = LegacyDistillParamModel(
            image_size=cfg.image_size,
            visual_dim=cfg.visual_dim,
            semantic_dim=cfg.semantic_dim,
            decoder_hidden=cfg.decoder_hidden,
        )
        param_model.load_state_dict(pm_state['model_state_dict'])
    param_model.to(device).eval()
    for p in param_model.parameters():
        p.requires_grad = False
    return param_model


def load_refine_init(model: ActionConditionedRefinementNetV4, path: Path):
    if not path.exists():
        return False
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.refine.load_state_dict(state, strict=False)
    logger.info(f'Loaded refine init: {path} missing={len(missing)} unexpected={len(unexpected)}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', default='outputs/firered_v12a_existing/pseudo_labels.jsonl')
    ap.add_argument('--asset_root', default='')
    ap.add_argument('--orig_dir', default='')
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--accum_steps', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--base_ch', type=int, default=64)
    ap.add_argument('--param_version', default='v8')
    ap.add_argument('--phase1_epochs', type=int, default=5)
    ap.add_argument('--l1_weight', type=float, default=1.0)
    ap.add_argument('--ssim_weight', type=float, default=0.5)
    ap.add_argument('--phase2_l1_weight', type=float, default=0.1)
    ap.add_argument('--musiq_weight', type=float, default=0.5)
    ap.add_argument('--clipiqa_weight', type=float, default=0.2)
    ap.add_argument('--clipiqa_target', type=float, default=0.7)
    ap.add_argument('--resume', default='')
    ap.add_argument('--init_refine_ckpt', default='')
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    default_asset = Path('/root/autodl-tmp') if (Path('/root/autodl-tmp') / 'checkpoints').exists() else PROJECT_ROOT
    asset_root = Path(args.asset_root) if args.asset_root else default_asset
    out_dir = Path(args.out_dir) if args.out_dir else asset_root / 'checkpoints' / 'refinement_v12a_firered'
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_dir = Path(args.orig_dir) if args.orig_dir else None

    logger.info(f'Device: {device}')
    logger.info(f'asset_root={asset_root}')
    logger.info(f'out_dir={out_dir}')

    samples = load_jsonl(PROJECT_ROOT / args.jsonl if not Path(args.jsonl).is_absolute() else Path(args.jsonl))
    train_s, val_s = split_by_source(samples, args.val_ratio, args.seed)
    logger.info(f'Data: records={len(samples)} train={len(train_s)} val={len(val_s)} unique_split_by_source')

    train_ds = FireRedV12ADataset(train_s, PROJECT_ROOT, orig_dir, is_train=True)
    val_ds = FireRedV12ADataset(val_s, PROJECT_ROOT, orig_dir, is_train=False)
    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(
        val_ds, args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    denorm = make_denorm_fn()
    param_model = load_param_model(asset_root, args.param_version, device)
    logger.info(f'Param model {args.param_version} loaded')

    model = ActionConditionedRefinementNetV4(base_ch=args.base_ch, n_actions=len(ACTIONS)).to(device)
    total_p, train_p = model.count_params()
    logger.info(f'ActionConditioned RefineNetV4: total={total_p/1e6:.2f}M trainable={train_p/1e6:.2f}M')

    if args.init_refine_ckpt:
        load_refine_init(model, Path(args.init_refine_ckpt))
    else:
        default_init = asset_root / 'checkpoints' / 'refinement_v4' / 'best.pt'
        if default_init.exists():
            load_refine_init(model, default_init)

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        logger.info(f'Resumed from {args.resume} epoch={start_epoch - 1}')

    musiq_model = None
    clipiqa_model = None
    try:
        import pyiqa
        musiq_model = pyiqa.create_metric('musiq-ava', device=device)
        for p in musiq_model.parameters():
            p.requires_grad = False
        logger.info('MUSIQ-AVA loaded')
        if args.clipiqa_weight > 0:
            clipiqa_model = pyiqa.create_metric('clipiqa+', device=device)
            for p in clipiqa_model.parameters():
                p.requires_grad = False
            logger.info('CLIPIQA+ loaded')
    except Exception as e:
        logger.warning(f'pyiqa metrics disabled: {e}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - start_epoch + 1), eta_min=args.lr * 0.01)

    best_l1 = float('inf')
    best_musiq = 0.0
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        use_aesthetic = epoch > args.phase1_epochs and musiq_model is not None
        if use_aesthetic:
            l1_w = args.phase2_l1_weight
            ssim_w = 0.0
            musiq_w = args.musiq_weight
            clip_w = args.clipiqa_weight
            phase = 'P2-MUSIQ'
        else:
            l1_w = args.l1_weight
            ssim_w = args.ssim_weight
            musiq_w = 0.0
            clip_w = 0.0
            phase = 'P1-L1'

        model.train()
        optimizer.zero_grad()
        sums = {'total': 0.0, 'l1': 0.0, 'ssim': 0.0, 'psnr': 0.0, 'musiq': 0.0, 'clip': 0.0}
        n = 0

        for i, batch in enumerate(train_loader):
            img_224 = batch['image_224'].to(device, non_blocking=True)
            raw_512 = batch['raw_512'].to(device, non_blocking=True)
            target_512 = batch['target_512'].to(device, non_blocking=True)
            action_oh = batch['action_onehot'].to(device, non_blocking=True)

            with torch.no_grad():
                out = param_model(img_224)
                pred_phys = denorm(out['norm_params'], device)
                rendered = apply_diff_isp(raw_512.float(), pred_phys).clamp(0, 1)
                rendered = torch.nan_to_num(rendered, nan=0.5)

            enhanced = model(rendered, action_oh)
            l1_loss = F.l1_loss(enhanced, target_512.float())
            ssim_v = ssim_loss(enhanced, target_512.float())
            psnr_v = psnr_metric(enhanced.detach(), target_512.float())
            musiq_loss = torch.tensor(0.0, device=device)
            clip_loss = torch.tensor(0.0, device=device)
            if use_aesthetic:
                musiq_loss = -musiq_model(enhanced).mean()
                if clipiqa_model is not None:
                    clip_score = clipiqa_model(enhanced)
                    clip_loss = F.relu(args.clipiqa_target - clip_score.mean())

            total = l1_w * l1_loss + ssim_w * ssim_v + musiq_w * musiq_loss + clip_w * clip_loss
            (total / args.accum_steps).backward()

            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(train_loader):
                grad_ok = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grad_ok = False
                        p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                if grad_ok:
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                optimizer.zero_grad()

            sums['total'] += total.item()
            sums['l1'] += l1_loss.item()
            sums['ssim'] += ssim_v.item()
            sums['psnr'] += psnr_v.item()
            sums['musiq'] += musiq_loss.item()
            sums['clip'] += clip_loss.item()
            n += 1

            if (i + 1) % args.log_every == 0:
                logger.info(
                    f'  [{i+1}/{len(train_loader)}] total={sums["total"]/n:.4f} '
                    f'l1={sums["l1"]/n:.4f} ssim={sums["ssim"]/n:.4f} '
                    f'psnr={sums["psnr"]/n:.2f} '
                    f'musiq_l={sums["musiq"]/n:.4f} clip_l={sums["clip"]/n:.4f}')

        model.eval()
        v_l1 = v_psnr = v_musiq = v_clip = 0.0
        v_n = v_aes_n = 0
        with torch.no_grad():
            for batch in val_loader:
                img_224 = batch['image_224'].to(device, non_blocking=True)
                raw_512 = batch['raw_512'].to(device, non_blocking=True)
                target_512 = batch['target_512'].to(device, non_blocking=True)
                action_oh = batch['action_onehot'].to(device, non_blocking=True)
                out = param_model(img_224)
                pred_phys = denorm(out['norm_params'], device)
                rendered = apply_diff_isp(raw_512.float(), pred_phys).clamp(0, 1)
                rendered = torch.nan_to_num(rendered, nan=0.5)
                enhanced = model(rendered, action_oh)
                v_l1 += F.l1_loss(enhanced, target_512.float()).item()
                v_psnr += psnr_metric(enhanced, target_512.float()).item()
                if musiq_model is not None:
                    v_musiq += musiq_model(enhanced).mean().item()
                    v_aes_n += 1
                if clipiqa_model is not None:
                    v_clip += clipiqa_model(enhanced).mean().item()
                v_n += 1

        scheduler.step()
        val_l1 = v_l1 / max(v_n, 1)
        val_psnr = v_psnr / max(v_n, 1)
        val_musiq = v_musiq / max(v_aes_n, 1)
        val_clip = v_clip / max(v_n, 1)
        elapsed = time.time() - t0
        logger.info(
            f'Ep {epoch}/{args.epochs} [{phase}] train={sums["total"]/max(n,1):.4f} '
            f'l1={sums["l1"]/max(n,1):.4f} psnr={sums["psnr"]/max(n,1):.2f} '
            f'val_l1={val_l1:.4f} val_PSNR={val_psnr:.2f} '
            f'val_MUSIQ={val_musiq:.2f} val_CLIP={val_clip:.3f} {elapsed:.1f}s')

        history.append({
            'epoch': epoch,
            'phase': phase,
            'train': {k: v / max(n, 1) for k, v in sums.items()},
            'val_l1': val_l1,
            'val_psnr': val_psnr,
            'val_musiq': val_musiq,
            'val_clipiqa': val_clip,
            'time_sec': elapsed,
        })

        save_it = (val_musiq > best_musiq) if use_aesthetic else (val_l1 < best_l1)
        if save_it:
            best_l1 = min(best_l1, val_l1)
            best_musiq = max(best_musiq, val_musiq)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_l1': val_l1,
                'val_psnr': val_psnr,
                'val_musiq': val_musiq,
                'val_clipiqa': val_clip,
                'config': vars(args),
                'actions': ACTIONS,
            }, out_dir / 'best.pt')
            logger.info(f'  * best saved: l1={val_l1:.4f} PSNR={val_psnr:.2f} MUSIQ={val_musiq:.2f} CLIP={val_clip:.3f}')

        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, out_dir / f'ep{epoch}.pt')
        with (out_dir / 'history.json').open('w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict()}, out_dir / 'final.pt')
    logger.info(f'DONE best_l1={best_l1:.4f} best_MUSIQ={best_musiq:.2f} out={out_dir}')


if __name__ == '__main__':
    main()
