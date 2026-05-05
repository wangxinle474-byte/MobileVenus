#!/bin/bash
# 在 AutoDL 上下载 AesExpert (qyuan/AesMMIT_LLaVA_v1.5_7b_240325)
# 用于 tools/data/autodl_score_fivek.py 的美学打分.
# 国内机房 hf-mirror 直连快, 一般 5-10 分钟完事.
set -e
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_ENABLE_HF_TRANSFER

TARGET_DIR=/root/autodl-tmp/models/AesExpert
mkdir -p "$TARGET_DIR"

echo "=== Disk space ==="
df -h /root/autodl-tmp | tail -1

echo ""
echo "=== Download AesExpert (~13.2 GB, qyuan/AesMMIT_LLaVA_v1.5_7b_240325) ==="
echo "    NOTE: 历史上有人用过 huang-lin/AesExpert, 已 401 不存在."
echo "    实际官方权重在 qyuan/AesMMIT_LLaVA_v1.5_7b_240325 (commit 89697323)"

huggingface-cli download qyuan/AesMMIT_LLaVA_v1.5_7b_240325 \
  --local-dir "$TARGET_DIR" \
  --max-workers 8 \
  2>&1 | tail -30

echo ""
echo "=== Verify ==="
du -sh "$TARGET_DIR"
ls -lh "$TARGET_DIR" | head -20

echo ""
echo "=== Quick run example ==="
echo "  python tools/data/autodl_score_fivek.py \\"
echo "      --jpeg_dir /root/autodl-tmp/fivek_jpeg \\"
echo "      --output /root/autodl-tmp/outputs/fivek_aesexpert_scores.json"
