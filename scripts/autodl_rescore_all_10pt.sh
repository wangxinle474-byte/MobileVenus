#!/bin/bash
# 四组 1-10 分评分: LongCat-sceneA / LongCat-editB / FireRed-editB / FireRed-editB-rewrite
set -e
cd /root/autodl-tmp/IntelligenceCamera
mkdir -p logs

PY=/root/miniconda3/bin/python

run_group() {
    local name=$1
    local indir=$2
    local captions=$3
    local out=$4
    local suffix=$5
    echo ""
    echo "============================================="
    echo "[$name]  suffix=$suffix"
    echo "============================================="
    $PY -u tools/eval/score_longcat_edits.py \
        --input_dir "$indir" \
        --captions "$captions" \
        --out "$out" \
        --group_label "$name" \
        --edit_suffix "$suffix" 2>&1 | tee "logs/score10_$name.log"
}

run_group "sceneA_longcat" \
    outputs/longcat_compare_sceneA \
    data/compare_5_captions.json \
    outputs/longcat_score_sceneA_10pt.json \
    longcat

run_group "editB_longcat" \
    outputs/longcat_compare_editB \
    data/compare_5_captions_edit.json \
    outputs/longcat_score_editB_10pt.json \
    longcat

run_group "editB_firered" \
    outputs/firered_compare_editB \
    data/compare_5_captions_edit.json \
    outputs/firered_score_editB_10pt.json \
    firered

run_group "editB_firered_rewrite" \
    outputs/firered_compare_editB_rewrite \
    data/compare_5_captions_edit.json \
    outputs/firered_score_editB_rewrite_10pt.json \
    firered

echo ""
echo "[ALL DONE]"
ls -la outputs/*_10pt.json 2>/dev/null
