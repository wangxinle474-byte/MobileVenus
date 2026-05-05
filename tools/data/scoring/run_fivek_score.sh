#!/bin/bash
# Run AesExpert scoring on FiveK JPEG dataset on AutoDL.
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp
export PYTHONPATH=/root/autodl-tmp/LLaVA-main:$PYTHONPATH

echo '=== env check ==='
python -c 'import llava, torch; print("llava:", llava.__file__); print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())'

echo '=== gpu ==='
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

echo '=== start scoring fivek ==='
LOG=/root/autodl-tmp/outputs/score_fivek.log
OUT=/root/autodl-tmp/outputs/fivek_aesexpert_scores.json

nohup python autodl_score_fivek.py \
    --jpeg_dir /root/autodl-tmp/fivek_jpeg \
    --output $OUT > $LOG 2>&1 &
echo PID: $!
echo Log: $LOG
echo Out: $OUT
sleep 3
head -20 $LOG 2>/dev/null || echo '(log not yet created)'
