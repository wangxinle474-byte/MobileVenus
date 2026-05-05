#!/bin/bash
# AutoDL \u4e0a\u4e0b\u8f7d JarvisEvo \u7684 ArtEdit-Bench \u6570\u636e\u96c6 (\u901a\u8fc7 HF \u955c\u50cf)
set -e
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
# hf_transfer rust accelerator \u5728\u955c\u50cf\u4e0a\u6301\u7eed\u9519 (XET CDN \u4e0d\u8986\u76d6), \u6539\u7528 requests
unset HF_HUB_ENABLE_HF_TRANSFER

# 1. \u786e\u8ba4 mirror \u80fd\u901a
echo "=== Test HF mirror connectivity ==="
curl -s --connect-timeout 5 -I "$HF_ENDPOINT" | head -3

# 2. \u786e\u8ba4 hf_transfer \u5df2\u88c5
pip install -q hf_transfer huggingface_hub 2>&1 | tail -2

# 3. \u4e0b\u8f7d ArtEdit-Bench \u6570\u636e\u96c6 (805 MB)
mkdir -p /root/autodl-tmp/datasets/ArtEdit-Bench
echo "=== Downloading ArtEdit-Bench dataset (~805 MB) ==="
huggingface-cli download JarvisEvo/ArtEdit-Bench \
  --repo-type dataset \
  --local-dir /root/autodl-tmp/datasets/ArtEdit-Bench \
  --max-workers 8

echo ""
echo "=== Verify download ==="
du -sh /root/autodl-tmp/datasets/ArtEdit-Bench
ls -lh /root/autodl-tmp/datasets/ArtEdit-Bench/ | head -20
