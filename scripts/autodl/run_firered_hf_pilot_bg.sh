#!/bin/bash
# 后台启动 FireRed HF pilot (idx=600 单张) 用 nohup 抗 SSH 断线
# 日志写 /root/autodl-tmp/firered_hf_pilot/pilot.log

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=1     # 用 hf_transfer 加速下载 (如已装)

PY=/root/miniconda3/bin/python
SCRIPT=/root/autodl-tmp/IntelligenceCamera/tools/data/editor_models/run_firered_hf_local.py
LOG=/root/autodl-tmp/firered_hf_pilot/pilot.log

mkdir -p /root/autodl-tmp/firered_hf_pilot/out

# 若已有同名进程，先杀
PIDS_OLD=$(pgrep -f run_firered_hf_local.py)
if [ -n "$PIDS_OLD" ]; then
    echo "[KILL] killing old pid(s): $PIDS_OLD"
    kill -9 $PIDS_OLD 2>/dev/null
    sleep 1
fi

cd /root/autodl-tmp
nohup $PY $SCRIPT \
    --captions /root/autodl-tmp/firered_hf_pilot/0600_pilot.json \
    --input_dir /root/autodl-tmp/firered_hf_pilot/origs \
    --out_dir /root/autodl-tmp/firered_hf_pilot/out \
    --cpu_offload \
    --seed 42 \
    --steps 8 \
    --cfg 1.0 \
    > $LOG 2>&1 &

PID=$!
echo "[PID] $PID"
echo "[LOG] $LOG"
echo "[ENV] HF_HOME=$HF_HOME"
echo "[ENV] HF_ENDPOINT=$HF_ENDPOINT"
echo
echo "Started in background. Use:"
echo "  tail -n 100 -F $LOG"
echo "  ls -la /root/autodl-tmp/firered_hf_pilot/out/"
