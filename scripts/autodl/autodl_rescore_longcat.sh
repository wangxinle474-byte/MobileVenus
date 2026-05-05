#!/bin/bash
# 仅重跑评分 (LongCat 推理已完成, 不重跑).
set -e
cd /root/autodl-tmp/IntelligenceCamera
mkdir -p logs

echo "============================================="
echo "[SCORE A] sceneA"
echo "============================================="
/root/miniconda3/bin/python -u tools/eval/score_longcat_edits.py \
    --input_dir outputs/longcat_compare_sceneA \
    --captions data/compare_5_captions.json \
    --out outputs/longcat_score_sceneA.json \
    --group_label sceneA 2>&1 | tee logs/longcat_score_sceneA.log

echo ""
echo "============================================="
echo "[SCORE B] editB"
echo "============================================="
/root/miniconda3/bin/python -u tools/eval/score_longcat_edits.py \
    --input_dir outputs/longcat_compare_editB \
    --captions data/compare_5_captions_edit.json \
    --out outputs/longcat_score_editB.json \
    --group_label editB 2>&1 | tee logs/longcat_score_editB.log

echo ""
echo "[DONE] scores: outputs/longcat_score_sceneA.json + outputs/longcat_score_editB.json"
