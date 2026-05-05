"""训练轻量神经 ISP: (raw_image, params) → expert_image"""
import sys, json, time, argparse, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path('/root/autodl-tmp')
CODE_DIR = ROOT / 'IntelligenceCamera'
sys.path.insert(0, str(CODE_DIR))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--base_ch', type=int, default=32)
    parser.add_argument('--n_res_blocks', type=int, default=4)
    parser.add_argument('--l1_weight', type=float, default=1.0)
    parser.add_argument('--ssim_weight', type=float, default=1.0)
    parser.add_argument('--aesthetic_every', type=int, default=10)
    args = parser.parse_args()

    import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from training.fivek_8param.dataset_expert import build_expert_datasets
    from training.fivek_8param.config import PARAM_NAMES
    from models.neural_isp import NeuralISP, neural_isp_loss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    expert_dirs = {}
    for ex in ['a','b','c','d','e']:
        p = ROOT / f'fivek_expert_{ex}'
        if p.exists(): expert_dirs[ex] = str(p)
    logger.info(f'Expert: {list(expert_dirs.keys())}')

    FIVEK_PARAMS = ROOT / 'data' / 'fivek_expert_params.json'
    FIVEK_CONSENS = ROOT / 'data' / 'fivek_expert_consensus.json'
    train_ds, val_ds = build_expert_datasets(
        str(FIVEK_PARAMS), str(FIVEK_CONSENS),
        orig_jpeg_dir=str(ROOT/'fivek_jpeg'), expert_dirs=expert_dirs,
        image_size=args.image_size, val_ratio=args.val_ratio,
        seed=args.seed, val_expert='c',
    )
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f'Data: train={len(train_ds)}, val={len(val_ds)}')

    model = NeuralISP(param_dim=len(PARAM_NAMES), base_ch=args.base_ch, n_res_blocks=args.n_res_blocks).to(device)
    total_p, train_p = model.count_params()
    logger.info(f'NeuralISP: {total_p/1e3:.1f}K params')

    aesthetic_eval = None
    try:
        from training.aesthetic_loss import AestheticLoss
        aesthetic_eval = AestheticLoss(metric_name='musiq-ava', target_score=5.0, input_size=224, device=str(device))
        if not aesthetic_eval.available: aesthetic_eval = None
    except: pass

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    out_dir = ROOT / 'checkpoints' / 'neural_isp'
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf'); best_aes = 0.0; history = []

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        s_total=s_l1=s_ssim=s_aes=0.0; n=n_aes=0

        for i, batch in enumerate(train_loader):
            raw_image = batch['raw_image'].to(device)
            expert_image = batch['expert_image'].to(device)
            params = batch['params'].to(device)

            enhanced = model(raw_image, params)
            losses = neural_isp_loss(enhanced, expert_image, l1_weight=args.l1_weight, ssim_weight=args.ssim_weight)

            optimizer.zero_grad()
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            s_total += losses['total'].item(); s_l1 += losses['l1'].item(); s_ssim += losses['ssim'].item(); n += 1

            if (i+1) % args.aesthetic_every == 0 and aesthetic_eval is not None:
                with torch.no_grad():
                    _, aes_score = aesthetic_eval(enhanced)
                    s_aes += aes_score.mean().item(); n_aes += 1

            if (i+1) % args.log_every == 0 and n > 0:
                aes_str = f' aes={s_aes/max(n_aes,1):.2f}' if n_aes > 0 else ''
                logger.info(f'  [{i+1}/{len(train_loader)}] loss={s_total/n:.4f} l1={s_l1/n:.4f} ssim={s_ssim/n:.4f}{aes_str}')

        model.eval()
        v_total=v_l1=v_aes=0.0; v_n=v_n_aes=0
        with torch.no_grad():
            for batch in val_loader:
                raw_image = batch['raw_image'].to(device)
                expert_image = batch['expert_image'].to(device)
                params = batch['params'].to(device)
                enhanced = model(raw_image, params)
                losses = neural_isp_loss(enhanced, expert_image, l1_weight=args.l1_weight, ssim_weight=args.ssim_weight)
                v_total += losses['total'].item(); v_l1 += losses['l1'].item(); v_n += 1
                if aesthetic_eval is not None:
                    _, aes_score = aesthetic_eval(enhanced)
                    v_aes += aes_score.mean().item(); v_n_aes += 1

        val_loss = v_total/max(v_n,1); val_l1 = v_l1/max(v_n,1); val_aes = v_aes/max(v_n_aes,1)
        scheduler.step()
        elapsed = time.time() - t0
        logger.info(f'Ep {epoch}/{args.epochs} | train={s_total/max(n,1):.4f} l1={s_l1/max(n,1):.4f} | val={val_loss:.4f} l1={val_l1:.4f} aes={val_aes:.2f} | {elapsed:.1f}s')

        history.append({'epoch':epoch, 'train_loss':s_total/max(n,1), 'val_loss':val_loss, 'val_l1':val_l1, 'val_aes':val_aes})

        if val_loss < best_val:
            best_val = val_loss
            if val_aes > best_aes: best_aes = val_aes
            torch.save({'epoch':epoch, 'model_state_dict':model.state_dict(), 'val_loss':val_loss, 'val_l1':val_l1, 'val_aes':val_aes,
                         'config':{'param_dim':len(PARAM_NAMES),'base_ch':args.base_ch,'n_res_blocks':args.n_res_blocks}}, out_dir/'best.pt')
            logger.info(f'  * best: val={val_loss:.4f} l1={val_l1:.4f} aes={val_aes:.2f}')

        if epoch % 10 == 0:
            torch.save({'epoch':epoch,'model_state_dict':model.state_dict()}, out_dir/f'ep{epoch}.pt')

    torch.save({'epoch':args.epochs,'model_state_dict':model.state_dict()}, out_dir/'final.pt')
    json.dump(history, open(out_dir/'history.json','w'), indent=2)
    logger.info(f'NeuralISP done! best_val={best_val:.4f} best_aes={best_aes:.2f}')

if __name__ == '__main__':
    main()
