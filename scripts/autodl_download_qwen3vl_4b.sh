#!/bin/bash
# \u4e0b\u8f7d Qwen3-VL-4B-Instruct \u5230 AutoDL, \u7528\u4e8e LoRA SFT
set -e
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_ENABLE_HF_TRANSFER

TARGET_DIR=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct
mkdir -p "$TARGET_DIR"

echo "=== Disk space ==="
df -h /root/autodl-tmp | tail -1

echo ""
echo "=== Download Qwen3-VL-4B-Instruct (~8 GB) ==="
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$TARGET_DIR" \
  --max-workers 8 \
  2>&1 | tail -30

echo ""
echo "=== Verify ==="
du -sh "$TARGET_DIR"
ls -lh "$TARGET_DIR" | head -20
