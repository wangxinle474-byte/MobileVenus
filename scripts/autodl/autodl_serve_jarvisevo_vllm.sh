#!/bin/bash
# \u542f JarvisEvo-8B vLLM \u670d\u52a1 (\u5355\u5361 RTX 5090 32GB, tp=1)
# \u7528\u4e8e\u5bf9 800 ArtEdit \u6837\u672c\u751f\u6210 pseudo \u6807\u7b7e
set -e
export PATH=/root/miniconda3/bin:$PATH

MODEL_DIR=/root/autodl-tmp/checkpoints/pretrained/JarvisEvo

if [[ ! -d "$MODEL_DIR" ]] || [[ -z $(ls "$MODEL_DIR"/*.safetensors 2>/dev/null) ]]; then
  echo "[ERR] JarvisEvo \u6743\u91cd\u672a\u5728 $MODEL_DIR, \u7b49\u4e0b\u8f7d\u5b8c\u518d\u8dd1"
  exit 1
fi

# \u4f9d\u8d56: vllm
python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 || {
  echo '[INFO] Installing vllm...'
  pip install -q vllm==0.11.0
}

echo "=== GPU status ==="
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader

echo ""
echo "=== Start vLLM (single GPU, tp=1, max_model_len=16384) ==="
# \u964d\u4f4e max_model_len \u4ece 20480 \u2192 16384 \u4ee5\u8282\u7701 KV cache \u663e\u5b58
# gpu_memory_utilization=0.85 \u7559\u70b9\u7a7a\u95f4\u7ed9 activations
VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve "$MODEL_DIR" \
  --tensor-parallel-size 1 \
  --port 8086 \
  --api-key 0 \
  --served-model-name jarvisevo \
  --max-model-len 16384 \
  --limit-mm-per-prompt.image 2 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --trust-remote-code
