"""
AutoDL 端 Distill v4 训练脚本
解决域偏移问题: Stage A 使用 FiveK 原始图片 (与 Stage B 同域)

前置条件 (AutoDL):
  /root/autodl-tmp/fivek_stage_a.json   — Venus 对 FiveK 的分析 (由 gen_fivek_stage_a.py 生成)
  /root/autodl-tmp/fivek_jpeg/          — FiveK 原始图片 (5121 张)
  /root/autodl-tmp/data/
    fivek_expert_params.json            — FiveK 专家参数 (需上传)
    fivek_expert_consensus.json         — FiveK 共识参数 (需上传)
  /root/autodl-tmp/MobileVenus/         — 训练代码 (需上传)

用法:
  cd /root/autodl-tmp
  python train_distill_v4_autodl.py
  python train_distill_v4_autodl.py --skip_embed   # 跳过嵌入步骤 (已有 .npz)
  python train_distill_v4_autodl.py --skip_stage_a  # 仅跑 Stage B
"""
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── AutoDL 路径配置 ──────────────────────────────────────────────
ROOT          = Path('/root/autodl-tmp')
FIVEK_JSON    = ROOT / 'fivek_stage_a.json'
FIVEK_JPEG    = ROOT / 'fivek_jpeg'
DATA_DIR      = ROOT / 'data'
EMBED_NPZ     = ROOT / 'fivek_text_embeddings.npz'
CODE_DIR      = ROOT / 'MobileVenus'
OUTPUT_DIR    = ROOT / 'checkpoints' / 'distill_v4'

FIVEK_PARAMS  = DATA_DIR / 'fivek_expert_params.json'
FIVEK_CONSENS = DATA_DIR / 'fivek_expert_consensus.json'


def check_prerequisites():
    """检查前置文件"""
    errors = []
    for p, desc in [
        (FIVEK_JSON,    'fivek_stage_a.json'),
        (FIVEK_JPEG,    'fivek_jpeg 目录'),
        (FIVEK_PARAMS,  'fivek_expert_params.json'),
        (FIVEK_CONSENS, 'fivek_expert_consensus.json'),
        (CODE_DIR,      'MobileVenus 代码目录'),
    ]:
        if not p.exists():
            errors.append(f'  ❌ 缺少: {p}  ({desc})')
        else:
            logger.info(f'  ✅ {p}')

    if errors:
        logger.error('缺少前置文件:')
        for e in errors:
            logger.error(e)
        sys.exit(1)

    # 检查 JSON 条数
    with open(FIVEK_JSON, encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f'fivek_stage_a.json: {len(data)} 条')
    if len(data) < 500:
        logger.warning(f'数据量较少 ({len(data)} 条)，建议 ≥1000 条')


def step1_embed_texts():
    """Step 1: 从 fivek_stage_a.json 提取文本并编码为 MiniLM embeddings"""
    logger.info('=' * 60)
    logger.info('Step 1: 文本编码 (MiniLM → .npz)')

    if EMBED_NPZ.exists():
        import numpy as np
        d = np.load(str(EMBED_NPZ))
        logger.info(f'  已有 {EMBED_NPZ} ({len(d["image_names"])} 条)，跳过')
        return

    sys.path.insert(0, str(CODE_DIR))
    from training.semantic_distill.embed_texts import extract_long_answers, deduplicate_by_image, encode_texts
    import numpy as np

    logger.info('  提取长文本回答...')
    pairs = extract_long_answers(str(FIVEK_JSON), min_len=50)
    logger.info(f'  找到 {len(pairs)} 条')
    merged = deduplicate_by_image(pairs)
    logger.info(f'  合并后 {len(merged)} 张图片')

    image_names = [m[0] for m in merged]
    texts = [m[1][:1000] for m in merged]  # MiniLM max 256 tokens

    logger.info('  编码文本 (all-MiniLM-L6-v2)...')
    embeddings = encode_texts(texts, batch_size=128)
    logger.info(f'  embedding shape: {embeddings.shape}')

    np.savez_compressed(
        str(EMBED_NPZ),
        image_names=np.array(image_names),
        embeddings=embeddings,
        text_dim=embeddings.shape[1],
    )
    logger.info(f'  保存: {EMBED_NPZ} ({EMBED_NPZ.stat().st_size/1024:.0f} KB)')


def step2_train_stage_a():
    """Step 2: Stage A — 视觉→语义对齐 (FiveK 图片 + FiveK 文本嵌入)"""
    logger.info('=' * 60)
    logger.info('Step 2: Stage A 训练 (视觉-语义对齐, FiveK 域)')

    sys.path.insert(0, str(CODE_DIR))
    from training.semantic_distill.config import DistillConfig
    from training.semantic_distill.trainer import train_stage_a

    cfg = DistillConfig(
        text_embed_file   = str(EMBED_NPZ),
        venus_image_root  = str(FIVEK_JPEG),
        fivek_data_file   = str(FIVEK_PARAMS),
        fivek_consensus_file = str(FIVEK_CONSENS),
        fivek_jpeg_dir    = str(FIVEK_JPEG),
        output_dir        = str(OUTPUT_DIR),
        stage_a_epochs    = 25,
        stage_a_lr        = 5e-4,
        stage_a_batch_size = 32,
        num_workers       = 4,
        fp16              = True,
    )

    logger.info(f'  配置: epochs={cfg.stage_a_epochs}, lr={cfg.stage_a_lr}, batch={cfg.stage_a_batch_size}')
    logger.info(f'  数据: {EMBED_NPZ} + {FIVEK_JPEG}')
    logger.info(f'  输出: {OUTPUT_DIR}/stage_a/')

    t0 = time.time()
    train_stage_a(cfg)
    logger.info(f'  Stage A 完成, 耗时 {(time.time()-t0)/60:.1f} min')
    return cfg


def step3_train_stage_b(cfg):
    """Step 3: Stage B — 参数解码器微调 (FiveK 专家标注)"""
    logger.info('=' * 60)
    logger.info('Step 3: Stage B 训练 (FiveK 参数预测)')

    sys.path.insert(0, str(CODE_DIR))
    from training.semantic_distill.trainer import train_stage_b

    cfg.stage_b_epochs    = 35
    cfg.stage_b_lr        = 1e-4
    cfg.stage_b_batch_size = 16

    logger.info(f'  配置: epochs={cfg.stage_b_epochs}, lr={cfg.stage_b_lr}')
    logger.info(f'  数据: {FIVEK_PARAMS}')
    logger.info(f'  输出: {OUTPUT_DIR}/stage_b/')

    t0 = time.time()
    train_stage_b(cfg)
    logger.info(f'  Stage B 完成, 耗时 {(time.time()-t0)/60:.1f} min')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_embed',   action='store_true', help='跳过文本嵌入 (已有 .npz)')
    parser.add_argument('--skip_stage_a', action='store_true', help='跳过 Stage A (仅跑 Stage B)')
    args = parser.parse_args()

    logger.info('=' * 60)
    logger.info('  Distill v4 训练 — FiveK 域 Stage A (消除域偏移)')
    logger.info('=' * 60)

    logger.info('检查前置文件...')
    check_prerequisites()

    if not args.skip_embed:
        step1_embed_texts()
    else:
        logger.info('[跳过] Step 1: 文本嵌入')

    if not args.skip_stage_a:
        cfg = step2_train_stage_a()
    else:
        logger.info('[跳过] Step 2: Stage A')
        sys.path.insert(0, str(CODE_DIR))
        from training.semantic_distill.config import DistillConfig
        cfg = DistillConfig(
            text_embed_file      = str(EMBED_NPZ),
            venus_image_root     = str(FIVEK_JPEG),
            fivek_data_file      = str(FIVEK_PARAMS),
            fivek_consensus_file = str(FIVEK_CONSENS),
            fivek_jpeg_dir       = str(FIVEK_JPEG),
            output_dir           = str(OUTPUT_DIR),
        )

    step3_train_stage_b(cfg)

    logger.info('=' * 60)
    logger.info('Distill v4 训练完成!')
    logger.info(f'模型权重: {OUTPUT_DIR}/stage_a/best.pt')
    logger.info(f'          {OUTPUT_DIR}/stage_b/best.pt')
    logger.info('下一步: 下载 checkpoints/distill_v4/ 到本地, 运行 eval_psnr_ssim.py')


if __name__ == '__main__':
    main()
