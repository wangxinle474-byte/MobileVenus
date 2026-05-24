#!/bin/bash
PY=/root/miniconda3/bin/python
SCRIPT=/root/autodl-tmp/IntelligenceCamera/scripts/autodl/download_firered_ms_retry.py
LOG=/root/autodl-tmp/firered_hf_pilot/ms_retry.log

mkdir -p /root/autodl-tmp/firered_hf_pilot/

pkill -9 -f download_firered 2>/dev/null
sleep 1

cd /root/autodl-tmp
nohup $PY -u $SCRIPT > $LOG 2>&1 &
PID=$!
echo "[PID] $PID"
echo "[LOG] $LOG"
sleep 4
echo
echo "=== first 4s ==="
tail -n 40 $LOG 2>/dev/null
