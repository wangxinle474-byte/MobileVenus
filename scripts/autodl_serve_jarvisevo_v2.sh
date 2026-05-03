#!/bin/bash
# \u542f JarvisEvo-8B vLLM 0.11.0 serve (\u5355\u5361 RTX 5090 32GB, tp=1)
set -e
export PATH=/root/miniconda3/bin:$PATH

MODEL_DIR=/root/autodl-tmp/checkpoints/pretrained/JarvisEvo
PORT=8086

echo "=== GPU status ==="
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader
echo ""

echo "=== Start vLLM serve ==="
echo "  Model: $MODEL_DIR"
echo "  Port : $PORT"
echo "  API  : jarvisevo (key=0)"
echo ""

# vllm 0.11.0 \u7684\u591a\u6a21\u6001\u9650\u5236\u7528 JSON: --limit-mm-per-prompt '{"image":2}'
VLLM_WORKER_MULTIPROC_METHOD=spawn \
  vllm serve "$MODEL_DIR" \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port $PORT \
    --api-key 0 \
    --served-model-name jarvisevo \
    --max-model-len 16384 \
    --limit-mm-per-prompt '{"image":2}' \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --trust-remote-code
