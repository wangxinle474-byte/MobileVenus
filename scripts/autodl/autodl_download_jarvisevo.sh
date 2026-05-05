#!/bin/bash
# AutoDL \u4e0a\u901a\u8fc7 HF \u955c\u50cf\u4e0b\u8f7d JarvisEvo 8B \u6743\u91cd
# \u7528\u4e8e\u4e3a 800 \u4e2a ArtEdit-Bench \u6837\u672c\u751f\u6210 CoT pseudo \u6807\u7b7e
set -e
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
unset HF_HUB_ENABLE_HF_TRANSFER

TARGET_DIR=/root/autodl-tmp/checkpoints/pretrained/JarvisEvo
mkdir -p "$TARGET_DIR"

echo "=== Disk space check ==="
df -h /root/autodl-tmp | tail -1
echo ""

echo "=== Download JarvisEvo/JarvisEvo (approximately 16-17 GB) ==="
echo "Target: $TARGET_DIR"
echo "Mirror: $HF_ENDPOINT"
echo ""

huggingface-cli download JarvisEvo/JarvisEvo \
  --local-dir "$TARGET_DIR" \
  --max-workers 8 \
  2>&1 | tail -50

echo ""
echo "=== Post-download verification ==="
du -sh "$TARGET_DIR"
ls -lh "$TARGET_DIR" | head -30
