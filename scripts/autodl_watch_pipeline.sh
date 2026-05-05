#!/bin/bash
# \u5b9e\u65f6\u76d1\u63a7 IntelligenceCamera LoRA PoC \u6d41\u6c34\u7ebf
# \u7528\u6cd5: watch -n 3 bash /root/autodl-tmp/IntelligenceCamera/scripts/autodl_watch_pipeline.sh

OUTPUTS=/root/autodl-tmp/IntelligenceCamera/outputs

echo "=============================================================================="
echo "  IntelligenceCamera Pipeline Monitor   $(date '+%H:%M:%S')"
echo "=============================================================================="
echo ""

# GPU
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null | \
  awk -F, '{printf "  %s | used: %s / free: %s | util: %s\n", $1, $2, $3, $4}'
echo ""

# Disk
echo "--- Disk ---"
df -h /root/autodl-tmp | tail -1 | awk '{printf "  /autodl-tmp: %s used / %s free (%s)\n", $3, $4, $5}'
echo ""

# Processes
echo "--- Running processes ---"
VLLM_PID=$(pgrep -f 'vllm serve' | head -1)
LABEL_PID=$(pgrep -f 'generate_cot_pseudo' | head -1)
TRAIN_PID=$(pgrep -f 'llamafactory' | head -1)

if [[ -n "$VLLM_PID" ]]; then
  echo "  [RUN] vLLM serve PID=$VLLM_PID"
else
  echo "  [   ] vLLM serve (not started)"
fi

if [[ -n "$LABEL_PID" ]]; then
  echo "  [RUN] pseudo labeling PID=$LABEL_PID"
else
  echo "  [   ] pseudo labeling (not started)"
fi

if [[ -n "$TRAIN_PID" ]]; then
  echo "  [RUN] LoRA training PID=$TRAIN_PID"
else
  echo "  [   ] LoRA training (not started)"
fi
echo ""

# vLLM health
echo "--- vLLM health (port 8086) ---"
HEALTH=$(curl -s -m 2 http://localhost:8086/health 2>/dev/null)
if [[ -n "$HEALTH" ]] || curl -s -m 2 http://localhost:8086/v1/models 2>/dev/null | grep -q jarvisevo; then
  echo "  [OK] vLLM accepting requests"
else
  echo "  [--] vLLM not responding"
fi
echo ""

# Pseudo label progress
echo "--- Pseudo labels ---"
LABEL_FILE=/root/autodl-tmp/datasets/ArtEdit-Bench/pseudo_labels.jsonl
if [[ -f "$LABEL_FILE" ]]; then
  N_TOTAL=$(wc -l < "$LABEL_FILE")
  N_OK=$(grep -c '"tool_call":' "$LABEL_FILE")
  echo "  labels written: $N_TOTAL | with tool_call: $N_OK"
else
  echo "  (no labels file yet)"
fi
echo ""

# Log tails
echo "--- Last 5 lines of each log ---"
for logname in vllm_serve labels_run lora_sft_train; do
  LOG_FILE="$OUTPUTS/${logname}.log"
  if [[ -f "$LOG_FILE" ]]; then
    echo ""
    echo "  [$logname]"
    tail -5 "$LOG_FILE" | sed 's/^/    /'
  fi
done
