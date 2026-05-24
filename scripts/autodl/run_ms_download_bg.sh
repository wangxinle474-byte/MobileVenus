#!/bin/bash
# 后台启动 ModelScope FireRed 下载 + python -u 强制 unbuffered (实时日志)
PY=/root/miniconda3/bin/python
SCRIPT=/root/autodl-tmp/IntelligenceCamera/scripts/autodl/download_firered_modelscope.py
LOG=/root/autodl-tmp/firered_hf_pilot/ms_download.log

mkdir -p /root/autodl-tmp/firered_hf_pilot/

# 若有旧实例先杀
pkill -9 -f download_firered_modelscope.py 2>/dev/null
sleep 1

cd /root/autodl-tmp
nohup $PY -u $SCRIPT > $LOG 2>&1 &
PID=$!
echo "[PID] $PID"
echo "[LOG] $LOG"
sleep 4
echo
echo "=== first 4s of log ==="
tail -n 50 $LOG 2>/dev/null
