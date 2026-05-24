#!/bin/bash
# Path Y: batch FireRed inference for 1500 samples (5 actions × 300 images)
# Run on AutoDL L20 48GB. Estimated ~5h total.
#
# Prerequisites:
#   1. Source images synced to /root/autodl-tmp/fivek_jpeg/
#   2. Per-action caption JSONs in /root/autodl-tmp/pathY_captions/
#   3. FireRed model at /root/autodl-tmp/ms_models/
#
# Usage:
#   bash /root/autodl-tmp/IntelligenceCamera/scripts/autodl/run_pathY_firered_batch.sh

set -euo pipefail

PY=/root/miniconda3/bin/python
SCRIPT=/root/autodl-tmp/IntelligenceCamera/tools/data/editor_models/run_firered_hf_local.py
CAPTIONS_DIR=/root/autodl-tmp/pathY_captions
INPUT_DIR=/root/autodl-tmp/fivek_jpeg
OUT_BASE=/root/autodl-tmp/pathY_outputs
LOG_DIR=/root/autodl-tmp/pathY_logs

BASE_DIR=/root/autodl-tmp/ms_models/FireRedTeam/FireRed-Image-Edit-1___0
LORA_DIR=/root/autodl-tmp/ms_models/FireRedTeam/FireRed-Image-Edit-1___0-Lightning
LORA_NAME=FireRed-Image-Edit-1.0-Lightning-8steps-v1.0.safetensors

ACTIONS="wb highlights saturation shadows contrast"

# Sanity checks
[ ! -d "$BASE_DIR" ] && echo "[ERROR] base model missing: $BASE_DIR" && exit 1
[ ! -f "$LORA_DIR/$LORA_NAME" ] && echo "[ERROR] LoRA missing: $LORA_DIR/$LORA_NAME" && exit 1
[ ! -d "$INPUT_DIR" ] && echo "[ERROR] input_dir missing: $INPUT_DIR" && exit 1
[ ! -d "$CAPTIONS_DIR" ] && echo "[ERROR] captions_dir missing: $CAPTIONS_DIR" && exit 1

mkdir -p "$OUT_BASE" "$LOG_DIR"

echo "=== Path Y FireRed batch inference ==="
echo "  actions: $ACTIONS"
echo "  input:   $INPUT_DIR"
echo "  output:  $OUT_BASE/<action>/"
echo "  logs:    $LOG_DIR/"
echo ""

TOTAL_START=$(date +%s)

for ACTION in $ACTIONS; do
    CAPTION_FILE="$CAPTIONS_DIR/pathY_${ACTION}.json"
    OUT_DIR="$OUT_BASE/$ACTION"
    LOG="$LOG_DIR/${ACTION}.log"

    if [ ! -f "$CAPTION_FILE" ]; then
        echo "[SKIP] $ACTION: caption file not found: $CAPTION_FILE"
        continue
    fi

    # Count existing outputs to support resume
    mkdir -p "$OUT_DIR"
    N_DONE=$(find "$OUT_DIR" -maxdepth 1 -name '*.png' -type f 2>/dev/null | wc -l)
    N_TOTAL=$($PY -c "import json; d=json.load(open('$CAPTION_FILE')); print(len(d['samples']))")
    echo "[START] $ACTION: $N_DONE/$N_TOTAL already done"

    if [ "$N_DONE" -ge "$N_TOTAL" ]; then
        echo "  -> all done, skipping"
        continue
    fi

    T0=$(date +%s)
    $PY -u $SCRIPT \
        --captions "$CAPTION_FILE" \
        --input_dir "$INPUT_DIR" \
        --out_dir "$OUT_DIR" \
        --base_model "$BASE_DIR" \
        --lora_repo "$LORA_DIR" \
        --lora_weight_name "$LORA_NAME" \
        --sequential_offload \
        --vae_slice_tile \
        --seed 42 \
        --steps 8 \
        --cfg 1.0 \
        2>&1 | tee "$LOG"
    T1=$(date +%s)
    echo "[DONE] $ACTION: $((T1-T0))s"
    echo ""
done

TOTAL_END=$(date +%s)
echo "=== ALL DONE: $((TOTAL_END-TOTAL_START))s total ==="
echo "Results in: $OUT_BASE/"
