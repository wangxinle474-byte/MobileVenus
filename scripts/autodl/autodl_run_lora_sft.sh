#!/bin/bash
# \u5728 AutoDL \u5355\u5361 RTX 5090 \u4e0a \u8dd1 LoRA SFT \u8bad\u7ec3 Qwen3-VL-4B
# \u524d\u63d0:
#   1. Qwen3-VL-4B \u5df2\u4e0b (/root/autodl-tmp/models/Qwen3-VL-4B-Instruct)
#   2. pseudo labels \u5df2\u751f\u6210 (/root/autodl-tmp/datasets/ArtEdit-Bench/pseudo_labels.jsonl)
#   3. LLaMA-Factory \u5b89\u88c5\u597d (\u6216 src/sft_rft pip install -e .)
set -e
export PATH=/root/miniconda3/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

LF_ROOT=/root/autodl-tmp/JarvisEvo_repo/src/sft_rft
DATA_DIR=/root/autodl-tmp/datasets/ArtEdit-Bench/sharegpt
CONFIG=/root/autodl-tmp/IntelligenceCamera/configs/lora_qwen3vl_4b_sft.yaml

# Step 1: \u8fc1\u79fb\u6570\u636e\u96c6\u6ce8\u518c\u5230 LLaMA-Factory data/dataset_info.json
echo "=== Step 1: Register ArtEdit_LoRA datasets in LLaMA-Factory ==="
LF_DATA_INFO=$LF_ROOT/data/dataset_info.json

if [[ ! -f "$LF_DATA_INFO" ]]; then
  echo "[ERR] LLaMA-Factory data/dataset_info.json \u4e0d\u5b58\u5728, \u786e\u4fdd $LF_ROOT \u5df2\u5b89\u88c5"
  exit 1
fi

# \u5408\u5e76\u6211\u4eec\u7684 snippet \u5230 LF \u7684 dataset_info
python - <<PY
import json
lf_path = "$LF_DATA_INFO"
snippet_path = "$DATA_DIR/dataset_info_snippet.json"
with open(lf_path, encoding='utf-8') as f:
    lf = json.load(f)
with open(snippet_path, encoding='utf-8') as f:
    snip = json.load(f)
lf.update(snip)
with open(lf_path, 'w', encoding='utf-8') as f:
    json.dump(lf, f, indent=2, ensure_ascii=False)
print(f"[OK] Registered {list(snip.keys())} into {lf_path}")
PY

# Step 2: \u8dd1 LoRA SFT
echo ""
echo "=== Step 2: LoRA SFT training ==="
cd "$LF_ROOT"
FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG" 2>&1 | tee /root/autodl-tmp/IntelligenceCamera/outputs/lora_sft_train.log
