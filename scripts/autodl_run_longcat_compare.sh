#!/bin/bash
# AutoDL 跑 LongCat 两组对照 (场景描述 vs 编辑指令) 共 10 张推理.
# 前置: LongCat 已下载到 /root/autodl-tmp/cache/modelscope.
set -e
cd /root/autodl-tmp/IntelligenceCamera
mkdir -p logs outputs/longcat_compare_sceneA outputs/longcat_compare_editB

# RTX 5090 32GB: 不用 offload, 全 GPU 跑, 单张 Turbo 4-NFE ~5-10 秒
COMMON_ARGS="--skip_download --offload none --steps 4"

echo "============================================="
echo "[A] 场景描述风格 → outputs/longcat_compare_sceneA/"
echo "============================================="
/root/miniconda3/bin/python -u tools/data/local_run_longcat_turbo.py \
    --captions data/compare_5_captions.json \
    --out_dir outputs/longcat_compare_sceneA \
    $COMMON_ARGS 2>&1 | tee logs/longcat_sceneA.log

echo ""
echo "============================================="
echo "[B] 编辑指令风格 → outputs/longcat_compare_editB/"
echo "============================================="
/root/miniconda3/bin/python -u tools/data/local_run_longcat_turbo.py \
    --captions data/compare_5_captions_edit.json \
    --out_dir outputs/longcat_compare_editB \
    $COMMON_ARGS 2>&1 | tee logs/longcat_editB.log

echo ""
echo "============================================="
echo "[SCORE A] Qwen3-VL-4B \u5bf9 sceneA \u6253\u5206 (5 \u5bf9)"
echo "============================================="
/root/miniconda3/bin/python -u tools/eval/score_longcat_edits.py \
    --input_dir outputs/longcat_compare_sceneA \
    --captions data/compare_5_captions.json \
    --out outputs/longcat_score_sceneA.json \
    --group_label sceneA 2>&1 | tee logs/longcat_score_sceneA.log

echo ""
echo "============================================="
echo "[SCORE B] Qwen3-VL-4B \u5bf9 editB \u6253\u5206 (5 \u5bf9)"
echo "============================================="
/root/miniconda3/bin/python -u tools/eval/score_longcat_edits.py \
    --input_dir outputs/longcat_compare_editB \
    --captions data/compare_5_captions_edit.json \
    --out outputs/longcat_score_editB.json \
    --group_label editB 2>&1 | tee logs/longcat_score_editB.log

echo ""
echo "[ALL DONE] images: outputs/longcat_compare_sceneA/ + outputs/longcat_compare_editB/"
echo "          scores: outputs/longcat_score_sceneA.json + outputs/longcat_score_editB.json"
ls -la outputs/longcat_compare_sceneA/ outputs/longcat_compare_editB/ outputs/longcat_score_*.json 2>/dev/null
