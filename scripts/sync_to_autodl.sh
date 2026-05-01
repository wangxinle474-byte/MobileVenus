#!/bin/bash
# ================================================================
# AutoDL 同步脚本 — 只上传训练相关代码
#
# 用法:
#   bash scripts/sync_to_autodl.sh
#   bash scripts/sync_to_autodl.sh <autodl_ip:port>
#
# 默认目标: /root/autodl-tmp/IntelligenceCamera/
# ================================================================

# AutoDL SSH 地址 (从 AutoDL 控制台获取, 格式: root@connect.xxx.seetacloud.com -p xxxxx)
AUTODL_HOST="${1:-root@connect.westb.seetacloud.com}"
AUTODL_PORT="${2:-12345}"
REMOTE_DIR="/root/autodl-tmp/IntelligenceCamera"

# 项目根目录
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== 同步训练代码到 AutoDL ==="
echo "  本地: $LOCAL_DIR"
echo "  远程: $AUTODL_HOST:$REMOTE_DIR (port $AUTODL_PORT)"
echo ""

# 使用 rsync 只同步训练所需文件
rsync -avz --progress \
    -e "ssh -p $AUTODL_PORT" \
    --include='*.py' \
    --include='*.yaml' \
    --include='*.txt' \
    --include='*.md' \
    \
    --include='models/***' \
    --include='training/***' \
    --include='scripts/***' \
    --include='tools/eval/***' \
    --include='tools/data/***' \
    --include='tools/train/***' \
    --include='tools/__init__.py' \
    \
    --include='train_v12_refine_hd.py' \
    --include='train_v6_stage_a.py' \
    --include='requirements.txt' \
    \
    --exclude='models/mobile_venus.py' \
    --exclude='models/parameter_predictor.py' \
    --exclude='models/language_model.py' \
    --exclude='models/suggestion_generator.py' \
    --exclude='models/text_encoder.py' \
    --exclude='APP/***' \
    --exclude='docs/***' \
    --exclude='images/***' \
    --exclude='examples/***' \
    --exclude='inference/***' \
    --exclude='evaluate/***' \
    --exclude='tools/demo/***' \
    --exclude='tools/plot/***' \
    --exclude='data/***' \
    --exclude='.git/***' \
    --exclude='__pycache__/***' \
    --exclude='*.pyc' \
    --exclude='.gitignore' \
    \
    "$LOCAL_DIR/" "$AUTODL_HOST:$REMOTE_DIR/"

echo ""
echo "=== 同步完成 ==="
echo "在 AutoDL 上运行训练:"
echo "  cd $REMOTE_DIR"
echo "  python scripts/train_v6_stage_a.py"
echo "  python train_v12_refine_hd.py"
