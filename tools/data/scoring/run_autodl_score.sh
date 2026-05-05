#!/bin/bash
# AutoDL 端运行 IP2P 对评分的启动脚本
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /root/autodl-tmp

# 环境检查
echo "=== env check ==="
python -c "import llava, torch; print('llava:', llava.__file__); print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'devs:', torch.cuda.device_count())"
echo "=== gpu ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || echo "nvidia-smi unavailable"
echo "=== start scoring ==="

LIMIT="${1:--1}"   # 第一个参数为 limit, 默认 -1 (全量)
SKIP_DESC="${2:-}" # 第二个参数: --skip_desc 或空

LOG_FILE="/root/autodl-tmp/outputs/score_pilot_100.log"
OUT_JSON="/root/autodl-tmp/outputs/pair_scores.json"

nohup python /root/autodl-tmp/autodl_score_ip2p_pairs.py \
    --pairs_dir /root/autodl-tmp/outputs \
    --output "$OUT_JSON" \
    --limit "$LIMIT" \
    $SKIP_DESC \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo "Out: $OUT_JSON"
disown
sleep 2
ps -p $PID > /dev/null && echo "[RUNNING]" || echo "[FAILED]"
