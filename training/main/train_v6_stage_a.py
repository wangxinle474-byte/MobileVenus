"""
Distill v6 — Stage A: 语义对齐预训练
数据: 5K COCO (Venus美学文本) + 4500 FiveK (同域图片)
目标: 比 v4 Stage A 更强的语义对齐，同时兼顾域内分布

AutoDL 用法:
  python training/main/train_v6_stage_a.py \
    --coco_embed /root/autodl-tmp/data/coco5k_text_embeddings.npz \
    --coco_img_root /root/AADB_images \
    --fivek_embed /root/autodl-tmp/data/fivek_text_embeddings.npz \
    --fivek_img_root /root/autodl-tmp/fivek_jpeg \
    --output_dir /root/autodl-tmp/checkpoints/distill_v6 \
    --epochs 30 --batch_size 32 --lr 5e-4

若无 FiveK 文本 embedding，只用 COCO:
  python training/main/train_v6_stage_a.py \
    --coco_embed /root/autodl-tmp/data/coco5k_text_embeddings.npz \
    --coco_img_root /root/AADB_images \
    --output_dir /root/autodl-tmp/checkpoints/distill_v6 \
    --epochs 30
"""
import sys, json, argparse, logging, time
from pathlib import Path

_this = Path(__file__).parent
sys.path.insert(0, str(_this))
# AutoDL: MobileVenus 项目在同级 MobileVenus 子目录
for _candidate in [_this / 'MobileVenus', _this.parent / 'MobileVenus']:
    if (_candidate / 'training').exists():
        sys.path.insert(0, str(_candidate))
        break

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────── Dataset ───────────────────────────

class VenusAlignDataset(Dataset):
    """单来源图文对齐数据集"""

    def __init__(self, embed_file: str, image_root: str,
                 image_size: int = 224, augment: bool = True):
        data = np.load(embed_file, allow_pickle=True)
        self.image_names = list(data['image_names'])
        self.embeddings  = data['embeddings'].astype(np.float32)
        self.image_root  = Path(image_root)
        self.text_dim    = int(data['text_dim'])

        valid = []
        for i, name in enumerate(self.image_names):
            if self._find(name) is not None:
                valid.append(i)
        self.valid_indices = valid
        logger.info(f'  [{Path(embed_file).stem}] {len(valid)}/{len(self.image_names)} images found')

        if augment:
            self.tf = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def _find(self, name: str):
        for ext in ['.jpg', '.jpeg', '.png', '']:
            p = self.image_root / (name + ext)
            if p.exists():
                return p
        return None

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real = self.valid_indices[idx]
        img  = Image.open(str(self._find(self.image_names[real]))).convert('RGB')
        return {'image': self.tf(img),
                'text_emb': torch.from_numpy(self.embeddings[real])}


# ─────────────────────────── Loss ───────────────────────────

class SemanticAlignLoss(torch.nn.Module):
    def __init__(self, uniformity_weight=0.1):
        super().__init__()
        self.uw = uniformity_weight

    def forward(self, s_emb, t_emb):
        import torch.nn.functional as F
        s = F.normalize(s_emb, dim=-1)
        t = F.normalize(t_emb, dim=-1)
        cos  = (s * t).sum(-1)
        loss = (1.0 - cos).mean()

        if self.uw > 0 and s.size(0) > 1:
            sq = torch.cdist(s, s, p=2).pow(2)
            mask = torch.triu(torch.ones_like(sq), diagonal=1).bool()
            unif = torch.logsumexp(-2 * sq[mask], 0) - torch.log(mask.sum().float().clamp(1))
            loss = loss + self.uw * unif

        return loss, {'total': loss.item(), 'cos_sim_mean': cos.mean().item()}


# ─────────────────────────── Train ───────────────────────────

def run_epoch(model, loader, crit, opt, scaler, device, is_train):
    from training.semantic_distill.model import SemanticDistillModel
    total_loss, total_cos, n = 0, 0, 0
    for batch in loader:
        imgs    = batch['image'].to(device)
        t_emb   = batch['text_emb'].to(device)
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            out      = model(imgs, t_emb)
            loss, m  = crit(out['student_emb'], out['teacher_emb'])
        if is_train:
            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
        total_loss += m['total']
        total_cos  += m['cos_sim_mean']
        n += 1
    return total_loss / n, total_cos / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_embed',   default='', help='COCO 文本 embedding (可选，留空跳过)')
    parser.add_argument('--coco_img_root',default='/root/AADB_images')
    parser.add_argument('--fivek_embed',  default='', help='FiveK 文本 embedding (可选)')
    parser.add_argument('--fivek_img_root',default='/root/autodl-tmp/fivek_jpeg')
    parser.add_argument('--output_dir',   default='checkpoints/distill_v6')
    parser.add_argument('--epochs',       type=int,   default=30)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=5e-4)
    parser.add_argument('--image_size',   type=int,   default=224)
    parser.add_argument('--val_ratio',    type=float, default=0.1)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--fp16',         action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # ── 数据 ──
    logger.info('构建数据集...')
    datasets = []
    text_dim = 384  # MiniLM default

    if args.coco_embed and Path(args.coco_embed).exists():
        coco_ds = VenusAlignDataset(args.coco_embed, args.coco_img_root,
                                     args.image_size, augment=True)
        if len(coco_ds) > 0:
            datasets.append(coco_ds)
            text_dim = coco_ds.text_dim
    elif args.coco_embed:
        logger.warning(f'COCO embed 文件不存在，跳过: {args.coco_embed}')

    if args.fivek_embed and Path(args.fivek_embed).exists():
        fivek_ds = VenusAlignDataset(args.fivek_embed, args.fivek_img_root,
                                      args.image_size, augment=True)
        if len(fivek_ds) > 0:
            datasets.append(fivek_ds)
            text_dim = fivek_ds.text_dim
    elif args.fivek_embed:
        logger.warning(f'FiveK embed 文件不存在，跳过: {args.fivek_embed}')

    assert datasets, 'No valid images found!'
    full_ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]

    n_val   = int(len(full_ds) * args.val_ratio)
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)
    logger.info(f'Train={n_train}, Val={n_val}, Total={len(full_ds)}')

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── 模型 ──
    from training.semantic_distill.model import SemanticDistillModel
    model = SemanticDistillModel(image_size=args.image_size, text_dim=text_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'模型参数: {n_params/1e6:.2f}M')

    crit    = SemanticAlignLoss(uniformity_weight=0.1)
    opt     = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler  = torch.amp.GradScaler('cuda') if args.fp16 and device.type == 'cuda' else None

    out_dir = Path(args.output_dir) / 'stage_a'
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    history  = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_cos = run_epoch(model, train_loader, crit, opt, scaler, device, True)
        model.eval()
        with torch.no_grad():
            vl_loss, vl_cos = run_epoch(model, val_loader, crit, None, None, device, False)
        sched.step()
        elapsed = time.time() - t0

        logger.info(f'Ep {epoch}/{args.epochs} | '
                    f'train={tr_loss:.4f} cos={tr_cos:.3f} | '
                    f'val={vl_loss:.4f} cos={vl_cos:.3f} | {elapsed:.1f}s')

        history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_cos': tr_cos,
                        'val_loss': vl_loss, 'val_cos': vl_cos})

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': best_val}, out_dir / 'best.pt')
            logger.info(f'  ★ best={best_val:.4f}')

    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict()},
               out_dir / 'final.pt')
    json.dump(history, open(out_dir / 'history.json', 'w'), indent=2)
    logger.info(f'Stage A 完成! best_val={best_val:.4f}')


if __name__ == '__main__':
    main()
