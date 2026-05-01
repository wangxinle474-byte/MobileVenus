"""
Stage C: 文本条件化 ISP 参数预测 — 训练脚本

基于 Stage B v8 模型, 加入文本指令条件化:
  Image + "提高曝光，增加饱和度" → ISP 6参数

训练数据: generate_instruction_data.py 生成的 instruction_data.json
前置条件: Stage B v8 checkpoint (distill_v8/stage_b/best.pt)

用法:
  # 1. 生成指令数据
  python tools/data/generate_instruction_data.py \
    --fivek_params data/fivek_expert_params.json \
    --ppr10k_params data/ppr10k_params.json \
    --output data/instruction_data.json

  # 2. 训练 Stage C
  python train_stage_c.py
  python train_stage_c.py --fusion cross_attn  # 交叉注意力 (消融)
"""
import sys
import json
import time
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# AutoDL 路径
ROOT = Path('/root/autodl-tmp')
CODE_DIR = ROOT / 'IntelligenceCamera'
sys.path.insert(0, str(CODE_DIR))

# 本地路径 fallback
LOCAL_ROOT = Path(__file__).parent
if not ROOT.exists():
    ROOT = LOCAL_ROOT
    CODE_DIR = LOCAL_ROOT
    sys.path.insert(0, str(CODE_DIR))


def main():
    parser = argparse.ArgumentParser(description='Stage C: 文本条件化训练')
    parser.add_argument('--stage_b_ckpt', default='',
                        help='Stage B checkpoint (默认自动查找)')
    parser.add_argument('--instruction_data', default='',
                        help='指令数据 JSON')
    parser.add_argument('--fusion', default='film', choices=['film', 'cross_attn'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--text_dropout', type=float, default=0.2)
    parser.add_argument('--consistency_weight', type=float, default=0.3)
    parser.add_argument('--base_align_weight', type=float, default=0.1)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_every', type=int, default=50)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from training.semantic_distill.model import SemanticDistillModel, DistillParamModel
    from training.semantic_distill.config import DistillConfig
    from training.text_condition.model import TextConditionedModel, build_char_vocab
    from training.text_condition.dataset import build_instruction_datasets
    from training.fivek_8param.config import PARAM_NAMES, PARAM_RANGES
    from training.fivek_8param.loss import DistillParamLoss

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}  Fusion: {args.fusion}')

    # ── 路径 ──
    data_dir = ROOT / 'data'
    ckpt_dir = ROOT / 'checkpoints'

    # Stage B checkpoint
    if args.stage_b_ckpt:
        stage_b_path = Path(args.stage_b_ckpt)
    else:
        # 优先 v8, 次选 v7/v6
        for ver in ['v8', 'v7', 'v6']:
            p = ckpt_dir / f'distill_{ver}' / 'stage_b' / 'best.pt'
            if p.exists():
                stage_b_path = p
                break
        else:
            raise FileNotFoundError('找不到 Stage B checkpoint')
    logger.info(f'Stage B: {stage_b_path}')

    # Stage A checkpoint (构建 Stage B 模型需要)
    stage_a_path = ckpt_dir / 'distill_v6' / 'stage_a' / 'best.pt'
    if not stage_a_path.exists():
        raise FileNotFoundError(f'找不到 Stage A: {stage_a_path}')

    # 指令数据
    if args.instruction_data:
        instr_path = Path(args.instruction_data)
    else:
        instr_path = data_dir / 'instruction_data.json'
    if not instr_path.exists():
        raise FileNotFoundError(
            f'指令数据不存在: {instr_path}\n'
            f'请先运行: python tools/data/generate_instruction_data.py')

    # ── 加载 Stage B 模型 ──
    cfg = DistillConfig()
    stage_a = SemanticDistillModel(
        image_size=cfg.image_size, visual_dim=cfg.visual_dim,
        semantic_dim=cfg.semantic_dim, text_dim=cfg.text_dim,
    )
    state_a = torch.load(str(stage_a_path), map_location='cpu', weights_only=False)
    stage_a.load_state_dict(state_a['model_state_dict'])

    stage_b = DistillParamModel(stage_a, decoder_hidden=cfg.decoder_hidden)
    state_b = torch.load(str(stage_b_path), map_location='cpu', weights_only=False)
    stage_b.load_state_dict(state_b['model_state_dict'])
    logger.info('Stage B 模型加载完成')

    # ── 构建 Stage C 模型 ──
    model = TextConditionedModel(
        stage_b_model=stage_b,
        vocab_size=5000,
        max_text_len=64,
        text_embed_dim=128,
        text_hidden_dim=256,
        text_num_layers=2,
        text_num_heads=4,
        text_output_dim=cfg.semantic_dim,
        fusion_type=args.fusion,
        decoder_hidden=cfg.decoder_hidden,
    ).to(device)

    counts = model.param_count()
    logger.info(f'Stage C 参数: total={counts["total"]/1e6:.2f}M  '
                f'trainable={counts["trainable"]/1e6:.2f}M  '
                f'(text_enc={counts["text_encoder"]/1e6:.2f}M  '
                f'fusion={counts["fusion"]/1e3:.1f}K  '
                f'decoder={counts["cond_decoder"]/1e6:.2f}M)')

    # ── 数据集 ──
    image_dirs = {}
    fivek_dir = ROOT / 'fivek_jpeg'
    if fivek_dir.exists():
        image_dirs['fivek'] = str(fivek_dir)
    ppr10k_dir = ROOT / 'PPR10K' / 'train_val_images_tif_360p'
    if ppr10k_dir.exists():
        image_dirs['ppr10k'] = str(ppr10k_dir)
    logger.info(f'图片目录: {list(image_dirs.keys())}')

    train_ds, val_ds, vocab = build_instruction_datasets(
        str(instr_path), image_dirs,
        image_size=args.image_size,
        text_dropout=args.text_dropout,
        seed=args.seed,
    )

    # 保存词表
    out_dir = Path(ROOT / 'checkpoints' / 'distill_v8' / 'stage_c')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # ── 优化器 ──
    param_crit = DistillParamLoss(align_weight=0.0, use_consensus=True)

    optimizer = optim.AdamW(model.trainable_params(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val = float('inf')
    history = []

    # ── 训练循环 ──
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        s_total = s_param = s_consist = s_base = 0.0
        n = 0

        for i, batch in enumerate(train_loader):
            images     = batch['image'].to(device)
            text_ids   = batch['text_ids'].to(device)
            params_gt  = batch['params'].to(device)
            base_gt    = batch['base_params'].to(device)
            has_text   = batch['has_text'].to(device)

            # 有文本的样本: 用文本条件化
            out_text = model(images, text_ids, return_base=True)
            pred_text = out_text['params_norm']
            B = pred_text.shape[0]
            w = torch.ones(B, pred_text.shape[1], device=device)
            p_loss_text, _ = param_crit(pred_text, params_gt, w)

            # 无文本的样本: 应与 Stage B 基准一致
            out_no_text = model(images, text_ids=None, return_base=True)
            pred_no_text = out_no_text['params_norm']

            # 一致性损失: 无文本时应接近 Stage B 基准
            base_pred = out_text['base_params_norm'].detach()
            consistency_loss = F.mse_loss(pred_no_text, base_pred)

            # 基准对齐: 无文本时也应接近 GT (保持 Stage B 能力)
            p_loss_base, _ = param_crit(pred_no_text, params_gt, w)

            total_loss = (p_loss_text +
                          args.consistency_weight * consistency_loss +
                          args.base_align_weight * p_loss_base)

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)
            optimizer.step()

            s_total   += total_loss.item()
            s_param   += p_loss_text.item()
            s_consist += consistency_loss.item()
            s_base    += p_loss_base.item()
            n += 1

            if (i + 1) % args.log_every == 0:
                logger.info(
                    f'  [{i+1}/{len(train_loader)}] '
                    f'loss={s_total/n:.4f} p_text={s_param/n:.4f} '
                    f'consist={s_consist/n:.4f} base={s_base/n:.4f}'
                )

        # ── 验证 ──
        model.eval()
        v_text, v_no_text, v_n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                images    = batch['image'].to(device)
                text_ids  = batch['text_ids'].to(device)
                params_gt = batch['params'].to(device)

                # 有文本
                out = model(images, text_ids)
                B = out['params_norm'].shape[0]
                w = torch.ones(B, out['params_norm'].shape[1], device=device)
                p_loss, _ = param_crit(out['params_norm'], params_gt, w)
                v_text += p_loss.item()

                # 无文本
                out_nt = model(images, text_ids=None)
                p_loss_nt, _ = param_crit(out_nt['params_norm'], params_gt, w)
                v_no_text += p_loss_nt.item()

                v_n += 1

        val_with_text = v_text / max(v_n, 1)
        val_no_text = v_no_text / max(v_n, 1)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f'Ep {epoch}/{args.epochs} | '
            f'train={s_total/n:.4f} | '
            f'val_text={val_with_text:.4f} val_no_text={val_no_text:.4f} | '
            f'{elapsed:.1f}s'
        )

        history.append({
            'epoch': epoch,
            'train_loss': s_total / n,
            'train_param': s_param / n,
            'train_consistency': s_consist / n,
            'val_with_text': val_with_text,
            'val_no_text': val_no_text,
        })

        if val_with_text < best_val:
            best_val = val_with_text
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_text': val_with_text,
                'val_no_text': val_no_text,
                'fusion_type': args.fusion,
                'vocab_size': len(vocab),
            }, out_dir / 'best.pt')
            logger.info(f'  * best={best_val:.4f}')

    # ── 保存 ──
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
    }, out_dir / 'final.pt')
    json.dump(history, open(out_dir / 'history.json', 'w'), indent=2)

    logger.info(f'\n{"="*60}')
    logger.info(f'Stage C done!')
    logger.info(f'  best val (with text): {best_val:.4f}')
    logger.info(f'  val (no text):        {val_no_text:.4f}')
    logger.info(f'  fusion: {args.fusion}')
    logger.info(f'  trainable params: {counts["trainable"]/1e6:.2f}M')
    logger.info(f'  checkpoint: {out_dir / "best.pt"}')
    logger.info(f'{"="*60}')


if __name__ == '__main__':
    main()
