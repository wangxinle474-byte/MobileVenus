#!/bin/bash
# 4 组 1-10 分评分 (新布局):
#   LongCat-sceneA / LongCat-editB / FireRed-editB-norewrite / FireRed-editB-rewrite
# 读: outputs/compare_5/originals/<idx>.png + outputs/compare_5/<group>/<idx>.png
# 写: outputs/compare_5/scores/<group>_10pt.json
set -e
cd /root/autodl-tmp/IntelligenceCamera
mkdir -p logs outputs/compare_5/scores

PY=/root/miniconda3/bin/python
ORIG=outputs/compare_5/originals

run_group() {
    local name=$1
    local indir=$2
    local captions=$3
    local out=$4
    echo ""
    echo "============================================="
    echo "[$name]"
    echo "============================================="
    $PY -u tools/eval/score_longcat_edits.py \
        --input_dir "$indir" \
        --originals_dir "$ORIG" \
        --captions "$captions" \
        --out "$out" \
        --group_label "$name" 2>&1 | tee "logs/score10_$name.log"
}

run_group "longcat_sceneA" \
    outputs/compare_5/longcat_sceneA \
    data/compare_5_captions.json \
    outputs/compare_5/scores/longcat_sceneA_10pt.json

run_group "longcat_editB" \
    outputs/compare_5/longcat_editB \
    data/compare_5_captions_edit.json \
    outputs/compare_5/scores/longcat_editB_10pt.json

run_group "firered_editB" \
    outputs/compare_5/firered_editB \
    data/compare_5_captions_edit.json \
    outputs/compare_5/scores/firered_editB_10pt.json

run_group "firered_editB_rewrite" \
    outputs/compare_5/firered_editB_rewrite \
    data/compare_5_captions_edit.json \
    outputs/compare_5/scores/firered_editB_rewrite_10pt.json

echo ""
echo "[ALL DONE]"
ls -la outputs/compare_5/scores/*.json 2>/dev/null
