"""
Distill v12 — 大模型 + 512 训练 + MUSIQ 主导 + CLIPIQA 辅助

核心升级 (vs v11):
  1. RefinementNetV2: U-Net 编解码, ~1M 参数 (v11=130K)
  2. 512×512 训练 (v11=224, MUSIQ 在 512 信号更强)
  3. Phase 2: MUSIQ 主导 (weight=0.5) + CLIPIQA 辅助 (weight=0.2)
  4. L1 权重大幅降低 (不再追求匹配 Expert C, 而是超越)

Pipeline:
  原图(224) → param_model(冻结v8) → diff_isp(512渲染) → RefinementNetV2(训练) → 高质量输出

AutoDL:
  python training/main/train_v12_refine_hd.py
  python training/main/train_v12_refine_hd.py --epochs 80 --musiq_weight 1.0
"""
import sys, json, time, argparse, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path('/root/autodl-tmp')
sys.path.insert(0, str(ROOT / 'IntelligenceCamera'))


def _make_denorm_fn():
    from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
    import torch
    lo_list = [PARAM_RANGES[p][0] for p in PARAM_NAMES]
    hi_list = [PARAM_RANGES[p][1] for p in PARAM_NAMES]
    def denorm(params_norm, device):
        lo = torch.tensor(lo_list, device=device, dtype=params_norm.dtype)
        hi = torch.tensor(hi_list, device=device, dtype=params_norm.dtype)
        params_phys = lo.unsqueeze(0) + (params_norm + 1.0) / 2.0 * (hi - lo).unsqueeze(0)
        return {name: params_phys[:, i] for i, name in enumerate(PARAM_NAMES)}
    return PARAM_NAMES, PARAM_RANGES, denorm


class FiveKHiResDataset:
    """
    双分辨率数据集:
      - image_224: (3, 224, 224) ImageNet normalized — param_model 输入
      - raw_512:   (3, 512, 512) [0,1] — diff_isp 渲染用
      - expert_512:(3, 512, 512) [0,1] — Expert C GT (Phase 1 L1 目标)
      - params/weights: 同之前
    """
    def __init__(self, samples, consensus_scores, is_train=True, val_expert='c'):
        import torch
        from torchvision import transforms as T
        from training.fivek_8param.config import PARAM_NAMES, normalize_param, PARAM_RANGES
        from PIL import Image
        import random as _random

        self.samples = samples
        self.consensus = consensus_scores
        self.is_train = is_train
        self.val_expert = val_expert
        self.PARAM_NAMES = PARAM_NAMES
        self.PARAM_RANGES = PARAM_RANGES
        self.normalize_param = normalize_param
        self._random = _random
        self._Image = Image
        self._torch = torch

        self.resize_224 = T.Resize((224, 224))
        self.resize_512 = T.Resize((512, 512))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.hflip = T.RandomHorizontalFlip(1.0)

        logger.info(f'{"Train" if is_train else "Val"} HiRes dataset: {len(samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        expert_paths = s.get('expert_paths', {})

        if self.is_train and expert_paths:
            chosen = self._random.choice(list(expert_paths.keys()))
        else:
            chosen = self.val_expert
        expert_path = expert_paths.get(chosen) or s.get('expert_path', '')

        try:
            pil_orig = self._Image.open(s['orig_path']).convert('RGB')
            pil_expert = self._Image.open(expert_path).convert('RGB')
        except Exception as e:
            dummy_224 = self._torch.zeros(3, 224, 224)
            dummy_512 = self._torch.zeros(3, 512, 512)
            zeros = self._torch.zeros(len(self.PARAM_NAMES))
            return {
                'image_224': dummy_224, 'raw_512': dummy_512,
                'expert_512': dummy_512, 'params': zeros,
                'weights': self._torch.ones(len(self.PARAM_NAMES)) * 0.5,
            }

        # Random flip (same for both)
        do_flip = self.is_train and self._random.random() < 0.5

        # 224 for param model
        orig_224 = self.resize_224(pil_orig)
        if do_flip:
            orig_224 = self.hflip(orig_224)
        image_224 = self.normalize(self.to_tensor(orig_224))

        # 512 for rendering + expert GT
        orig_512 = self.resize_512(pil_orig)
        expert_512 = self.resize_512(pil_expert)
        if do_flip:
            orig_512 = self.hflip(orig_512)
            expert_512 = self.hflip(expert_512)
        raw_512 = self.to_tensor(orig_512)
        expert_512_t = self.to_tensor(expert_512)

        # Params
        phys_params = {p: float(s['mean_params'][p]) for p in self.PARAM_NAMES}
        params = self._torch.tensor(
            [self.normalize_param(p, phys_params[p]) for p in self.PARAM_NAMES],
            dtype=self._torch.float32,
        )

        dng_name = s['dng_name']
        if dng_name in self.consensus:
            per_param = self.consensus[dng_name]['per_param']
            weights = self._torch.tensor(
                [per_param.get(p, 0.5) for p in self.PARAM_NAMES],
                dtype=self._torch.float32,
            )
        else:
            weights = self._torch.ones(len(self.PARAM_NAMES), dtype=self._torch.float32) * 0.5

        return {
            'image_224': image_224,       # (3, 224, 224) ImageNet norm
            'raw_512': raw_512,           # (3, 512, 512) [0,1]
            'expert_512': expert_512_t,   # (3, 512, 512) [0,1]
            'params': params,
            'weights': weights,
        }


def build_hires_datasets(params_file, consensus_file, orig_dir, expert_dirs,
                         val_ratio=0.1, seed=42, val_expert='c'):
    """构建 HiRes 数据集"""
    import random
    with open(params_file) as f:
        all_params = json.load(f)
    with open(consensus_file) as f:
        consensus = json.load(f)

    from training.fivek_8param.config import PARAM_NAMES

    samples = []
    for dng_name, info in all_params.items():
        orig_path = Path(orig_dir) / f'{dng_name}.jpg'
        if not orig_path.exists():
            continue

        expert_paths = {}
        for ex, ex_dir in expert_dirs.items():
            ep = Path(ex_dir) / f'{dng_name}.jpg'
            if ep.exists():
                expert_paths[ex] = str(ep)

        if val_expert not in expert_paths:
            continue

        mean_params = {}
        for p in PARAM_NAMES:
            vals = [v for v in info.get('experts', {}).values() if p in v]
            if vals:
                mean_params[p] = sum(v[p] for v in vals) / len(vals)
            else:
                mean_params[p] = info.get('mean', {}).get(p, 0.0)

        samples.append({
            'orig_path': str(orig_path),
            'expert_paths': expert_paths,
            'mean_params': mean_params,
            'dng_name': dng_name,
        })

    random.seed(seed)
    random.shuffle(samples)
    n_val = int(len(samples) * val_ratio)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    train_ds = FiveKHiResDataset(train_samples, consensus, is_train=True, val_expert=val_expert)
    val_ds = FiveKHiResDataset(val_samples, consensus, is_train=False, val_expert=val_expert)
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8,
                        help='512×512 + 大模型, batch 需要小一些')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--l1_weight', type=float, default=1.0,
                        help='Phase1 L1 权重')
    parser.add_argument('--ssim_weight', type=float, default=0.5)
    parser.add_argument('--musiq_weight', type=float, default=0.5,
                        help='Phase2 MUSIQ 权重 (远大于 v11 的 0.05)')
    parser.add_argument('--clipiqa_weight', type=float, default=0.2,
                        help='Phase2 CLIPIQA+ 辅助权重')
    parser.add_argument('--phase1_epochs', type=int, default=10,
                        help='Phase 1 warmup (仅 L1+SSIM)')
    parser.add_argument('--phase2_l1_weight', type=float, default=0.1,
                        help='Phase2 L1 权重 (大幅降低)')
    parser.add_argument('--musiq_target', type=float, default=5.5,
                        help='MUSIQ hinge loss 目标 (从8.5降到5.5, 让梯度有区分度)')
    parser.add_argument('--clipiqa_target', type=float, default=0.7,
                        help='CLIPIQA hinge loss 目标')
    parser.add_argument('--accum_steps', type=int, default=8,
                        help='Gradient accumulation steps (effective batch = batch_size * accum)')
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--param_version', type=str, default='v8')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的 checkpoint 路径')
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
    from training.semantic_distill.config import DistillConfig
    from models.diff_isp import apply_diff_isp, ssim_loss
    from models.refinement_net_v4 import RefinementNetV4

    PARAM_NAMES, _, denorm = _make_denorm_fn()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    logger.info(f'Config: base_ch={args.base_ch} musiq_w={args.musiq_weight} '
                f'musiq_target={args.musiq_target} phase2_l1={args.phase2_l1_weight} '
                f'accum={args.accum_steps} effective_batch={args.batch_size * args.accum_steps}')

    # ── 数据 (复用 build_expert_datasets + HiRes wrapper) ──
    from training.fivek_8param.dataset_expert import build_expert_datasets

    class HiResWrapper:
        """Wraps base 224 dataset to also provide 512 images"""
        def __init__(self, base_dataset):
            self.base = base_dataset
            from torchvision import transforms as T
            self.resize_512 = T.Resize((512, 512))
            self.to_tensor = T.ToTensor()
        def __len__(self): return len(self.base)
        def __getitem__(self, idx):
            item = self.base[idx]
            import random as _random
            from PIL import Image as _Image
            s = self.base.samples[idx]
            expert_paths = s.get('expert_paths', {})
            if self.base.is_train and expert_paths:
                chosen = _random.choice(list(expert_paths.keys()))
            else:
                chosen = self.base.val_expert
            expert_path = expert_paths.get(chosen) or ''
            try:
                pil_orig = _Image.open(s['orig_path']).convert('RGB')
                pil_expert = _Image.open(expert_path).convert('RGB')
                raw_512 = self.to_tensor(self.resize_512(pil_orig))
                expert_512 = self.to_tensor(self.resize_512(pil_expert))
            except:
                raw_512 = torch.zeros(3, 512, 512)
                expert_512 = torch.zeros(3, 512, 512)
            return {
                'image_224': item['image'],
                'raw_512': raw_512, 'expert_512': expert_512,
                'params': item['params'], 'weights': item['weights'],
            }

    expert_dirs = {}
    for ex in ['a', 'b', 'c', 'd', 'e']:
        p = ROOT / f'fivek_expert_{ex}'
        if p.exists():
            expert_dirs[ex] = str(p)

    train_ds_base, val_ds_base = build_expert_datasets(
        str(ROOT / 'data' / 'fivek_expert_params.json'),
        str(ROOT / 'data' / 'fivek_expert_consensus.json'),
        orig_jpeg_dir=str(ROOT / 'fivek_jpeg'), expert_dirs=expert_dirs,
        image_size=224, val_ratio=args.val_ratio, seed=args.seed, val_expert='c',
    )
    train_ds = HiResWrapper(train_ds_base)
    val_ds = HiResWrapper(val_ds_base)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    logger.info(f'Data: train={len(train_ds)}, val={len(val_ds)}')

    # ── 冻结参数模型 ──
    cfg = DistillConfig()
    stage_a = SemanticDistillModel(
        image_size=cfg.image_size, visual_dim=cfg.visual_dim,
        semantic_dim=cfg.semantic_dim, text_dim=cfg.text_dim,
    )
    sa_state = torch.load(str(ROOT / 'checkpoints' / 'distill_v6' / 'stage_a' / 'best.pt'),
                          map_location='cpu', weights_only=False)
    stage_a.load_state_dict(sa_state['model_state_dict'])
    param_model = DistillParamModel(stage_a, decoder_hidden=cfg.decoder_hidden)
    pm_ckpt = ROOT / 'checkpoints' / f'distill_{args.param_version}' / 'stage_b' / 'best.pt'
    if pm_ckpt.exists():
        pm_state = torch.load(str(pm_ckpt), map_location='cpu', weights_only=False)
        param_model.load_state_dict(pm_state['model_state_dict'])
        logger.info(f'Param model {args.param_version} loaded')
    else:
        logger.error(f'Not found: {pm_ckpt}'); return

    param_model.to(device).eval()
    for p in param_model.parameters():
        p.requires_grad = False

    # ── RefinementNet V4 ──
    refine = RefinementNetV4(base_ch=args.base_ch).to(device)
    total_p, train_p = refine.count_params()
    logger.info(f'RefinementNetV4: {total_p/1e6:.2f}M params ({train_p/1e6:.2f}M trainable)')

    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        refine.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        logger.info(f'Resumed from {args.resume} (epoch {start_epoch-1})')

    # ── MUSIQ + CLIPIQA ──
    musiq_model = None
    clipiqa_model = None
    try:
        import pyiqa
    except Exception as e:
        logger.warning(f'pyiqa import failed: {e}')
    else:
        try:
            musiq_model = pyiqa.create_metric('musiq-ava', device=device)
            for p in musiq_model.parameters():
                p.requires_grad = False
            logger.info('MUSIQ-AVA loaded')
        except Exception as e:
            logger.warning(f'MUSIQ load failed: {e}')

        if args.clipiqa_weight > 0:
            try:
                clipiqa_model = pyiqa.create_metric('clipiqa+', device=device)
                for p in clipiqa_model.parameters():
                    p.requires_grad = False
                logger.info('CLIPIQA+ loaded')
            except Exception as e:
                logger.warning(f'CLIPIQA+ load failed: {e}')

    # ── 优化器 ──
    optimizer = optim.AdamW(refine.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - start_epoch + 1, eta_min=1e-6)

    out_dir = ROOT / 'checkpoints' / 'refinement_v4'
    out_dir.mkdir(parents=True, exist_ok=True)

    best_musiq = 0.0
    best_l1 = float('inf')
    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        use_aesthetic = epoch > args.phase1_epochs and musiq_model is not None

        # Phase 2: 切换权重
        if use_aesthetic:
            l1_w = args.phase2_l1_weight       # 0.1 (大幅降低)
            ssim_w = 0.0                       # Phase2 不用 SSIM
            musiq_w = args.musiq_weight         # 0.5
            clipiqa_w = args.clipiqa_weight     # 0.2
            phase = 'P2-MUSIQ'
        else:
            l1_w = args.l1_weight
            ssim_w = args.ssim_weight
            musiq_w = 0.0
            clipiqa_w = 0.0
            phase = 'P1-L1'

        # ── Train ──
        refine.train()
        s_l1 = s_ssim = s_musiq = s_clip = s_total = s_aes = 0.0
        n = n_aes = 0

        for i, batch in enumerate(train_loader):
            img_224 = batch['image_224'].to(device)
            raw_512 = batch['raw_512'].to(device)
            expert_512 = batch['expert_512'].to(device)

            # 冻结管线: param(224) → diff_isp(512)
            with torch.no_grad():
                out = param_model(img_224)
                pred_phys = denorm(out['norm_params'], device)
                rendered = apply_diff_isp(raw_512.float(), pred_phys).clamp(0, 1)
                rendered = torch.nan_to_num(rendered, nan=0.5)

            # RefinementNet (有梯度)
            enhanced = refine(rendered)

            # L1 + SSIM vs Expert C
            l1_loss = F.l1_loss(enhanced, expert_512.float())
            ssim_v = ssim_loss(enhanced, expert_512.float())

            # MUSIQ + CLIPIQA (Phase 2)
            musiq_loss = torch.tensor(0.0, device=device)
            clip_loss = torch.tensor(0.0, device=device)

            if use_aesthetic:
                if musiq_model is not None:
                    # 直接最大化 MUSIQ 分数
                    musiq_loss = -musiq_model(enhanced).mean()
                if clipiqa_model is not None:
                    clip_score = clipiqa_model(enhanced)
                    # CLIPIQA+ 范围 0-1, 目标可配置
                    clip_loss = F.relu(args.clipiqa_target - clip_score.mean())

            total = (l1_w * l1_loss + ssim_w * ssim_v
                     + musiq_w * musiq_loss + clipiqa_w * clip_loss)

            accum = args.accum_steps
            loss_scaled = total / accum
            loss_scaled.backward()

            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                # NaN check
                grad_ok = True
                for p in refine.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        grad_ok = False; break
                if grad_ok:
                    nn.utils.clip_grad_norm_(refine.parameters(), 5.0)
                    optimizer.step()
                else:
                    logger.warning(f'NaN grad at batch {i}')
                optimizer.zero_grad()

            s_l1 += l1_loss.item()
            s_ssim += ssim_v.item()
            s_musiq += musiq_loss.item()
            s_clip += clip_loss.item()
            s_total += total.item()
            n += 1

            # MUSIQ 监控
            if (i + 1) % 3 == 0 and musiq_model is not None:
                with torch.no_grad():
                    aes = musiq_model(enhanced.detach()).mean().item()
                    s_aes += aes; n_aes += 1

            if (i + 1) % args.log_every == 0 and n > 0:
                aes_str = f' MUSIQ={s_aes/max(n_aes,1):.2f}' if n_aes > 0 else ''
                clip_str = f' clip_l={s_clip/n:.4f}' if use_aesthetic else ''
                logger.info(
                    f'  [{i+1}/{len(train_loader)}] '
                    f'total={s_total/n:.4f} l1={s_l1/n:.4f} ssim={s_ssim/n:.4f}'
                    f' musiq_l={s_musiq/n:.4f}{clip_str}{aes_str}'
                )

        # ── Val ──
        refine.eval()
        v_l1 = v_aes = v_clip = 0.0
        v_n = v_n_aes = 0

        with torch.no_grad():
            for batch in val_loader:
                img_224 = batch['image_224'].to(device)
                raw_512 = batch['raw_512'].to(device)
                expert_512 = batch['expert_512'].to(device)

                out = param_model(img_224)
                pred_phys = denorm(out['norm_params'], device)
                rendered = apply_diff_isp(raw_512.float(), pred_phys).clamp(0, 1)
                rendered = torch.nan_to_num(rendered, nan=0.5)

                enhanced = refine(rendered)
                v_l1 += F.l1_loss(enhanced, expert_512.float()).item()

                if musiq_model is not None:
                    v_aes += musiq_model(enhanced).mean().item()
                    v_n_aes += 1
                if clipiqa_model is not None:
                    v_clip += clipiqa_model(enhanced).mean().item()

                v_n += 1

        val_l1 = v_l1 / max(v_n, 1)
        val_aes = v_aes / max(v_n_aes, 1)
        val_clip = v_clip / max(v_n, 1)
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            f'Ep {epoch}/{args.epochs} [{phase}] | '
            f'train={s_total/max(n,1):.4f} l1={s_l1/max(n,1):.4f} | '
            f'val_l1={val_l1:.4f} val_MUSIQ={val_aes:.2f} val_CLIP={val_clip:.3f} | '
            f'{elapsed:.1f}s'
        )

        history.append({
            'epoch': epoch, 'phase': phase,
            'val_l1': val_l1, 'val_musiq': val_aes, 'val_clipiqa': val_clip,
        })

        # 保存逻辑
        save_it = False
        if use_aesthetic:
            # Phase 2: MUSIQ 优先
            if val_aes > best_musiq:
                best_musiq = val_aes; save_it = True
        else:
            # Phase 1: L1
            if val_l1 < best_l1:
                best_l1 = val_l1; save_it = True

        if save_it:
            torch.save({
                'epoch': epoch,
                'model_state_dict': refine.state_dict(),
                'val_l1': val_l1, 'val_musiq': val_aes, 'val_clipiqa': val_clip,
                'config': {'base_ch': args.base_ch, 'param_version': args.param_version},
            }, out_dir / 'best.pt')
            logger.info(f'  ★ best: l1={val_l1:.4f} MUSIQ={val_aes:.2f} CLIP={val_clip:.3f}')

        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': refine.state_dict()},
                       out_dir / f'ep{epoch}.pt')

    torch.save({'epoch': args.epochs, 'model_state_dict': refine.state_dict()},
               out_dir / 'final.pt')
    json.dump(history, open(out_dir / 'history.json', 'w'), indent=2)
    logger.info(f'v12 done! best_MUSIQ={best_musiq:.2f} best_l1={best_l1:.4f}')


if __name__ == '__main__':
    main()
