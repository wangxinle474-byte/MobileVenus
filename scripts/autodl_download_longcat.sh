#!/bin/bash
# AutoDL 后台下载 LongCat-Image-Edit-Turbo 从 ModelScope.
# 用法: bash scripts/autodl_download_longcat.sh
set -e
cd /root/autodl-tmp/IntelligenceCamera
mkdir -p logs
nohup /root/miniconda3/bin/python tools/data/autodl_download_longcat.py \
    > logs/longcat_download.log 2>&1 &
echo "PID=$!"
sleep 2
echo "--- log head ---"
head -5 logs/longcat_download.log 2>/dev/null || echo "(log not flushed yet)"
