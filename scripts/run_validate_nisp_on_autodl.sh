#!/bin/bash
# AutoDL: \u8dd1 NeuralISP \u9a8c\u8bc1\u5bf9\u6bd4 diff_isp \u57fa\u7ebf
set -e
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/IntelligenceCamera

echo "=== Running NeuralISP validation ==="
python tools/eval/validate_diff_isp_vs_lr.py \
  --orig_dir /root/autodl-tmp/fivek_jpeg \
  --expert_dir /root/autodl-tmp/fivek_expert_c \
  --params_json /root/autodl-tmp/data/fivek_expert_params.json \
  --expert_filter mean \
  --n 100 \
  --image_size 512 \
  --visualize \
  --n_vis 10 \
  --isp_mode neural_isp \
  --nisp_ckpt /root/autodl-tmp/checkpoints/neural_isp/best.pt \
  --out_dir outputs/validate_neural_isp

echo "=== Summary (NeuralISP) ==="
cat outputs/validate_neural_isp/summary.json
