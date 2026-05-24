#!/bin/bash
# 后台启动 FireRed HF 1.0 inference (idx=600 单张 pilot)
# 用 ModelScope 下载到本地的模型路径 + 明确指定 v1.0 LoRA

PY=/root/miniconda3/bin/python
SCRIPT=/root/autodl-tmp/IntelligenceCamera/tools/data/editor_models/run_firered_hf_local.py
LOG=/root/autodl-tmp/firered_hf_pilot/inference.log

BASE_DIR=/root/autodl-tmp/ms_models/FireRedTeam/FireRed-Image-Edit-1___0
LORA_DIR=/root/autodl-tmp/ms_models/FireRedTeam/FireRed-Image-Edit-1___0-Lightning
LORA_NAME=FireRed-Image-Edit-1.0-Lightning-8steps-v1.0.safetensors

# 健康检查
if [ ! -d "$BASE_DIR" ]; then echo "[ERROR] base dir missing: $BASE_DIR"; exit 1; fi
if [ ! -f "$LORA_DIR/$LORA_NAME" ]; then echo "[ERROR] LoRA file missing: $LORA_DIR/$LORA_NAME"; exit 1; fi

mkdir -p /root/autodl-tmp/firered_hf_pilot/out

# Kill 旧实例
pkill -9 -f run_firered_hf_local.py 2>/dev/null
sleep 1

cd /root/autodl-tmp
nohup $PY -u $SCRIPT \
    --captions /root/autodl-tmp/firered_hf_pilot/0600_pilot.json \
    --input_dir /root/autodl-tmp/firered_hf_pilot/origs \
    --out_dir /root/autodl-tmp/firered_hf_pilot/out \
    --base_model "$BASE_DIR" \
    --lora_repo "$LORA_DIR" \
    --lora_weight_name "$LORA_NAME" \
    --sequential_offload \
    --vae_slice_tile \
    --seed 42 \
    --steps 8 \
    --cfg 1.0 \
    > $LOG 2>&1 &

PID=$!
echo "[PID] $PID"
echo "[LOG] $LOG"
echo "[BASE] $BASE_DIR"
echo "[LORA] $LORA_DIR/$LORA_NAME"
sleep 4
echo
echo "=== first 4s log ==="
tail -n 30 $LOG 2>/dev/null
