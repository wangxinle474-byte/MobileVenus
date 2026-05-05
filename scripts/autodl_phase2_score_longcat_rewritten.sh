#!/bin/bash
# Phase 2 (AutoDL, 新布局): 评分第 5 组 = LongCat 用重写 prompt 跑出的编辑结果
#
# 运行前提:
#   - 已跑过 phase 1, data/compare_5_captions_edit_rewritten.json 存在
#   - 本地用重写 prompt 跑 LongCat 生成 outputs/compare_5/longcat_editB_rewritten/<idx>.png
#   - 上传: scp -r outputs/compare_5/longcat_editB_rewritten root@<host>:.../outputs/compare_5/
set -e
cd /root/autodl-tmp/IntelligenceCamera
mkdir -p logs outputs/compare_5/scores

PY=/root/miniconda3/bin/python

INDIR=outputs/compare_5/longcat_editB_rewritten
ORIG=outputs/compare_5/originals
CAPS=data/compare_5_captions_edit_rewritten.json
OUT=outputs/compare_5/scores/longcat_editB_rewritten_10pt.json

if [ ! -d "$INDIR" ]; then
    echo "[ERR] 未找到 $INDIR, 请先本地跑 LongCat 并上传"
    exit 1
fi
if [ ! -f "$CAPS" ]; then
    echo "[ERR] 未找到 $CAPS, 请先跑 phase 1"
    exit 1
fi

echo "============================================="
echo "[Phase 2] 评分 LongCat-editB-rewritten (第 5 组)"
echo "============================================="
$PY -u tools/eval/score_longcat_edits.py \
    --input_dir "$INDIR" \
    --originals_dir "$ORIG" \
    --captions "$CAPS" \
    --out "$OUT" \
    --group_label longcat_editB_rewritten 2>&1 | tee logs/phase2_score_longcat_rewritten.log

echo ""
echo "============================================="
echo "[Phase 2 DONE]"
echo "  scores: $OUT"
echo "============================================="
echo ""
echo "[NEXT] 本地跑全量展示:"
echo "  scp root@<host>:/root/autodl-tmp/IntelligenceCamera/$OUT outputs/compare_5/scores/"
echo "  python tools/eval/present_3way_10pt.py"
