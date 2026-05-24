#!/bin/bash
# steps=4 速度测试 (vs steps=8 baseline 81.1s/张)
# 跑两张: idx=600 (steps=8 跑过), idx=497 (新)
# 风险: Lightning LoRA 训练目标 8 steps, 4 steps 可能 underfit

PY=/root/miniconda3/bin/python
SCRIPT=/root/autodl-tmp/IntelligenceCamera/tools/data/editor_models/run_firered_hf_local.py
LOG=/root/autodl-tmp/firered_hf_pilot/inference_steps4.log

BASE_DIR=/root/autodl-tmp/ms_models/FireRedTeam/FireRed-Image-Edit-1___0
LORA_DIR=/root/autodl-tmp/ms_models/FireRedTeam/FireRed-Image-Edit-1___0-Lightning
LORA_NAME=FireRed-Image-Edit-1.0-Lightning-8steps-v1.0.safetensors

# 健康检查
if [ ! -d "$BASE_DIR" ]; then echo "[ERROR] base dir missing: $BASE_DIR"; exit 1; fi
if [ ! -f "$LORA_DIR/$LORA_NAME" ]; then echo "[ERROR] LoRA file missing"; exit 1; fi
if [ ! -f /root/autodl-tmp/firered_hf_pilot/origs/0600.png ]; then echo "[ERROR] 0600.png missing"; exit 1; fi
if [ ! -f /root/autodl-tmp/firered_hf_pilot/origs/0497.png ]; then echo "[ERROR] 0497.png missing"; exit 1; fi
if [ ! -f /root/autodl-tmp/firered_hf_pilot/steps4_pilot.json ]; then echo "[ERROR] caption json missing"; exit 1; fi

mkdir -p /root/autodl-tmp/firered_hf_pilot/out_steps4

pkill -9 -f run_firered_hf_local.py 2>/dev/null
sleep 1

cd /root/autodl-tmp
nohup $PY -u $SCRIPT \
    --captions /root/autodl-tmp/firered_hf_pilot/steps4_pilot.json \
    --input_dir /root/autodl-tmp/firered_hf_pilot/origs \
    --out_dir /root/autodl-tmp/firered_hf_pilot/out_steps4 \
    --base_model "$BASE_DIR" \
    --lora_repo "$LORA_DIR" \
    --lora_weight_name "$LORA_NAME" \
    --sequential_offload \
    --vae_slice_tile \
    --seed 42 \
    --steps 4 \
    --cfg 1.0 \
    > $LOG 2>&1 &

PID=$!
echo "[PID] $PID"
echo "[LOG] $LOG"
echo "[STEPS] 4 (vs baseline 8)"
echo "[INPUTS] idx=600 + idx=497"
sleep 4
echo
echo "=== first 4s log ==="
tail -n 30 $LOG 2>/dev/null
