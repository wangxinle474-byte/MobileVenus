#!/bin/bash
# AutoDL \u4e0a\u8dd1 validate_diff_isp_vs_lr.py \u7684\u5305\u88c5\u811a\u672c\u3002
# \u5904\u7406 conda \u6fc0\u6d3b + \u4f9d\u8d56\u68c0\u67e5 + \u7edd\u5bf9\u8def\u5f84\u3002
set -e

# \u663e\u5f0f\u52a0 conda \u5230 PATH (\u975e\u4ea4\u4e92\u5f0f ssh \u4e0b .bashrc \u4e0d\u52a0\u8f7d)
export PATH=/root/miniconda3/bin:$PATH

cd /root/autodl-tmp/IntelligenceCamera

# \u4f9d\u8d56\u68c0\u67e5
python -c "import skimage" 2>/dev/null || pip install -q scikit-image

echo "=== Environment ==="
python -c "import torch, skimage, PIL; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}  skimage={skimage.__version__}')"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | head -1

echo "=== Running validation ==="
python tools/eval/validate_diff_isp_vs_lr.py \
  --orig_dir /root/autodl-tmp/fivek_jpeg \
  --expert_dir /root/autodl-tmp/fivek_expert_c \
  --params_json /root/autodl-tmp/data/fivek_expert_params.json \
  --expert_filter mean \
  --n 100 \
  --image_size 512 \
  --visualize \
  --n_vis 10 \
  --out_dir outputs/validate_diff_isp_vs_lr

echo "=== Summary ==="
cat outputs/validate_diff_isp_vs_lr/summary.json
