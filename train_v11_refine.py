"""v11 — RefinementNet + MUSIQ-guided training. Phase1: L1+SSIM, Phase2: +MUSIQ loss"""
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--ssim_weight', type=float, default=0.5)
    parser.add_argument('--musiq_weight', type=float, default=0.05)
    parser.add_argument('--phase1_epochs', type=int, default=20)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--n_blocks', type=int, default=6)
    parser.add_argument('--param_version', type=str, default='v8')
    args = parser.parse_args()

    import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
    from training.fivek_8param.dataset_expert import build_expert_datasets
    from training.semantic_distill.config import DistillConfig
    from models.diff_isp import apply_diff_isp, ssim_loss
    from models.refinement_net import RefinementNet

    PARAM_NAMES, _, denorm = _make_denorm_fn()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    expert_dirs = {}
    for ex in ['a','b','c','d','e']:
        p = ROOT / f'fivek_expert_{ex}'
        if p.exists(): expert_dirs[ex] = str(p)

    train_ds, val_ds = build_expert_datasets(
        str(ROOT/'data'/'fivek_expert_params.json'), str(ROOT/'data'/'fivek_expert_consensus.json'),
        orig_jpeg_dir=str(ROOT/'fivek_jpeg'), expert_dirs=expert_dirs,
        image_size=args.image_size, val_ratio=args.val_ratio, seed=args.seed, val_expert='c',
    )
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f'Data: train={len(train_ds)}, val={len(val_ds)}')

    # 冻结的参数模型
    cfg = DistillConfig()
    stage_a = SemanticDistillModel(image_size=cfg.image_size, visual_dim=cfg.visual_dim, semantic_dim=cfg.semantic_dim, text_dim=cfg.text_dim)
    sa_state = torch.load(str(ROOT/'checkpoints'/'distill_v6'/'stage_a'/'best.pt'), map_location='cpu', weights_only=False)
    stage_a.load_state_dict(sa_state['model_state_dict'])
    param_model = DistillParamModel(stage_a, decoder_hidden=cfg.decoder_hidden)
    pm_ckpt = ROOT/'checkpoints'/f'distill_{args.param_version}'/'stage_b'/'best.pt'
    if pm_ckpt.exists():
        pm_state = torch.load(str(pm_ckpt), map_location='cpu', weights_only=False)
        param_model.load_state_dict(pm_state['model_state_dict'])
        logger.info(f'Param model {args.param_version} loaded')
    else:
        logger.error(f'Not found: {pm_ckpt}'); return
    param_model.to(device).eval()
    for p in param_model.parameters(): p.requires_grad = False

    # RefinementNet
    refine = RefinementNet(base_ch=args.base_ch, n_blocks=args.n_blocks).to(device)
    total_p, _ = refine.count_params()
    logger.info(f'RefinementNet: {total_p/1e3:.1f}K params')

    # MUSIQ
    musiq_model = None
    try:
        import pyiqa
        musiq_model = pyiqa.create_metric('musiq-ava', device=device)
        for p in musiq_model.parameters(): p.requires_grad = False
        logger.info('MUSIQ loaded')
    except Exception as e:
        logger.warning(f'MUSIQ failed: {e}')

    optimizer = optim.AdamW(refine.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    out_dir = ROOT / 'checkpoints' / 'refinement'
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf'); best_aes = 0.0; history = []

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        use_musiq = epoch > args.phase1_epochs and musiq_model is not None
        phase = 'P2-MUSIQ' if use_musiq else 'P1-L1'

        refine.train()
        s_l1=s_ssim=s_musiq=s_total=s_aes=0.0; n=n_aes=0

        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            raw_image = batch['raw_image'].to(device)
            expert_image = batch['expert_image'].to(device)

            with torch.no_grad():
                out = param_model(images)
                pred_phys = denorm(out['params_norm'], device)
                rendered = apply_diff_isp(raw_image.float(), pred_phys).clamp(0,1)
                rendered = torch.nan_to_num(rendered, nan=0.5)

            enhanced = refine(rendered)
            l1_loss = F.l1_loss(enhanced, expert_image.float())
            ssim_v = ssim_loss(enhanced, expert_image.float())

            musiq_loss = torch.tensor(0.0, device=device)
            if use_musiq:
                musiq_score = musiq_model(enhanced)
                musiq_loss = F.relu(8.5 - musiq_score.mean())

            total = args.l1_weight*l1_loss + args.ssim_weight*ssim_v + args.musiq_weight*musiq_loss

            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(refine.parameters(), 1.0)
            optimizer.step()

            s_l1 += l1_loss.item(); s_ssim += ssim_v.item(); s_musiq += musiq_loss.item()
            s_total += total.item(); n += 1

            if (i+1) % 5 == 0 and musiq_model is not None:
                with torch.no_grad():
                    aes = musiq_model(enhanced.detach()).mean().item()
                    s_aes += aes; n_aes += 1

            if (i+1) % args.log_every == 0 and n > 0:
                aes_str = f' MUSIQ={s_aes/max(n_aes,1):.2f}' if n_aes > 0 else ''
                logger.info(f'  [{i+1}/{len(train_loader)}] total={s_total/n:.4f} l1={s_l1/n:.4f} ssim={s_ssim/n:.4f}{aes_str}')

        refine.eval()
        v_l1=v_aes=0.0; v_n=v_n_aes=0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                raw_image = batch['raw_image'].to(device)
                expert_image = batch['expert_image'].to(device)
                out = param_model(images)
                pred_phys = denorm(out['params_norm'], device)
                rendered = apply_diff_isp(raw_image.float(), pred_phys).clamp(0,1)
                rendered = torch.nan_to_num(rendered, nan=0.5)
                enhanced = refine(rendered)
                v_l1 += F.l1_loss(enhanced, expert_image.float()).item()
                if musiq_model is not None:
                    v_aes += musiq_model(enhanced).mean().item(); v_n_aes += 1
                v_n += 1

        val_l1 = v_l1/max(v_n,1); val_aes = v_aes/max(v_n_aes,1)
        scheduler.step()
        elapsed = time.time() - t0
        logger.info(f'Ep {epoch}/{args.epochs} [{phase}] | train={s_total/max(n,1):.4f} l1={s_l1/max(n,1):.4f} | val_l1={val_l1:.4f} val_MUSIQ={val_aes:.2f} | {elapsed:.1f}s')
        history.append({'epoch':epoch,'phase':phase,'val_l1':val_l1,'val_musiq':val_aes})

        save_it = False
        if use_musiq and val_aes > best_aes:
            best_aes = val_aes; save_it = True
        elif not use_musiq and val_l1 < best_val:
            best_val = val_l1; save_it = True
        if save_it:
            torch.save({'epoch':epoch,'model_state_dict':refine.state_dict(),'val_l1':val_l1,'val_musiq':val_aes,
                         'config':{'base_ch':args.base_ch,'n_blocks':args.n_blocks,'param_version':args.param_version}}, out_dir/'best.pt')
            logger.info(f'  ★ best: l1={val_l1:.4f} MUSIQ={val_aes:.2f}')

        if epoch % 10 == 0:
            torch.save({'epoch':epoch,'model_state_dict':refine.state_dict()}, out_dir/f'ep{epoch}.pt')

    torch.save({'epoch':args.epochs,'model_state_dict':refine.state_dict()}, out_dir/'final.pt')
    json.dump(history, open(out_dir/'history.json','w'), indent=2)
    logger.info(f'v11 done! best_l1={best_val:.4f} best_MUSIQ={best_aes:.2f}')

if __name__ == '__main__':
    main()
