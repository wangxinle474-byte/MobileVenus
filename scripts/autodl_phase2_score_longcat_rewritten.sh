#!/bin/bash
# Phase 2 (AutoDL): \u8bc4\u5206\u7b2c 5 \u7ec4 = LongCat \u7528\u91cd\u5199 prompt \u8dd1\u51fa\u7684\u7f16\u8f91\u7ed3\u679c
#
# \u8fd0\u884c\u524d\u63d0:
#   - \u5df2\u8dd1\u8fc7 phase 1 \u62ff\u5230 data/compare_5_captions_edit_rewritten.json
#   - \u672c\u5730\u7528\u91cd\u5199 prompt \u8dd1 LongCat \u751f\u6210 outputs/longcat_compare_editB_rewritten/
#   - \u8be5\u76ee\u5f55\u542b 0xxx_orig.png + 0xxx_longcat.png \u5404 5 \u5f20
#   - \u4e0a\u4f20\u5230 AutoDL: scp -r outputs/longcat_compare_editB_rewritten ...
set -e
cd /root/autodl-tmp/IntelligenceCamera
mkdir -p logs

PY=/root/miniconda3/bin/python

INDIR=outputs/longcat_compare_editB_rewritten
CAPS=data/compare_5_captions_edit_rewritten.json
OUT=outputs/longcat_score_editB_rewritten_10pt.json

if [ ! -d "$INDIR" ]; then
    echo "[ERR] \u672a\u627e\u5230 $INDIR, \u8bf7\u5148\u672c\u5730\u8dd1 LongCat \u5e76\u4e0a\u4f20"
    exit 1
fi
if [ ! -f "$CAPS" ]; then
    echo "[ERR] \u672a\u627e\u5230 $CAPS, \u8bf7\u5148\u8dd1 phase 1"
    exit 1
fi

echo "============================================="
echo "[Phase 2] \u8bc4\u5206 LongCat-editB-rewritten (\u7b2c 5 \u7ec4)"
echo "============================================="
$PY -u tools/eval/score_longcat_edits.py \
    --input_dir "$INDIR" \
    --captions "$CAPS" \
    --out "$OUT" \
    --group_label editB_longcat_rewritten \
    --edit_suffix longcat 2>&1 | tee logs/phase2_score_longcat_rewritten.log

echo ""
echo "============================================="
echo "[Phase 2 DONE]"
echo "  scores: $OUT"
echo "============================================="
echo ""
echo "[NEXT] \u672c\u5730\u8dd1\u5168\u91cf\u5c55\u793a:"
echo "  scp root@<host>:/root/autodl-tmp/IntelligenceCamera/$OUT outputs/"
echo "  python tools/eval/present_3way_10pt.py"
