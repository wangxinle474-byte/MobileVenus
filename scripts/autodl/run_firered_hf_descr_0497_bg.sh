#!/bin/bash
# description-style 0497 复现测试 — HF 1.0 本地是否仍然 'ignored orig'
# 用 steps=8 严格匹配 1.1 API 默认设置, seed=42 + cfg=1.0

PY=/root/miniconda3/bin/python
SCRIPT=/root/autodl-tmp/IntelligenceCamera/tools/data/editor_models/run_firered_hf_local.py
LOG=/root/autodl-tmp/firered_hf_pilot/inference_descr_0497.log

BASE_DIR=/root/autodl-tmp/ms_models/FireRedTeam/FireRed-Image-Edit-1___0
LORA_DIR=/root/autodl-tmp/ms_models/FireRedTeam/FireRed-Image-Edit-1___0-Lightning
LORA_NAME=FireRed-Image-Edit-1.0-Lightning-8steps-v1.0.safetensors

if [ ! -f /root/autodl-tmp/firered_hf_pilot/origs/0497.png ]; then echo "[ERROR] 0497.png missing"; exit 1; fi
if [ ! -f /root/autodl-tmp/firered_hf_pilot/0497_descr_pilot.json ]; then echo "[ERROR] caption json missing"; exit 1; fi

mkdir -p /root/autodl-tmp/firered_hf_pilot/out_descr_0497

pkill -9 -f run_firered_hf_local.py 2>/dev/null
sleep 1

cd /root/autodl-tmp
nohup $PY -u $SCRIPT \
    --captions /root/autodl-tmp/firered_hf_pilot/0497_descr_pilot.json \
    --input_dir /root/autodl-tmp/firered_hf_pilot/origs \
    --out_dir /root/autodl-tmp/firered_hf_pilot/out_descr_0497 \
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
echo "[CAPTION] photo of a porcelain artwork, higher contrast, balanced exposure, neutral white balance"
echo "[STEPS] 8 (strict match with 1.1 API)"
sleep 4
echo
echo "=== first 4s log ==="
tail -n 30 $LOG 2>/dev/null
