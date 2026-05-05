#!/bin/bash
# Phase 1 (AutoDL \u4e0a\u4e00\u952e\u8dd1):
#   1. Qwen3-VL \u5c06 editB \u6307\u4ee4\u91cd\u5199\u4e3a\u5b8c\u6574\u573a\u666f prompt
#      \u8f93\u51fa: data/compare_5_captions_edit_rewritten.json
#   2. \u8bc4\u5206\u73b0\u6709 4 \u7ec4 (LongCat-sceneA / LongCat-editB /
#      FireRed-editB / FireRed-editB-rewrite)
#
# \u8fd0\u884c\u5b8c\u540e:
#   - \u5c06 data/compare_5_captions_edit_rewritten.json scp \u56de\u672c\u5730
#   - \u672c\u5730 LongCat \u8dd1\u91cd\u5199\u540e\u7684 prompt \u2192 outputs/longcat_compare_editB_rewritten/
#   - scp \u4e0a\u4f20 \u2192 \u8c03\u7528 phase 2 \u8bc4\u5206\u7b2c 5 \u7ec4
set -e
cd /root/autodl-tmp/IntelligenceCamera
mkdir -p logs

PY=/root/miniconda3/bin/python

echo ""
echo "============================================="
echo "[Phase 1.1] Qwen3-VL rewrite editB \u2192 scene prompt"
echo "============================================="
$PY -u tools/data/autodl_rewrite_edit_to_scene.py \
    --captions_in data/compare_5_captions_edit.json \
    --out data/compare_5_captions_edit_rewritten.json \
    --originals_dir outputs/compare_5/originals \
    2>&1 | tee logs/phase1_rewrite.log

echo ""
echo "============================================="
echo "[Phase 1.2] \u8bc4\u5206\u73b0\u6709 4 \u7ec4 (1-10 \u5206)"
echo "============================================="
bash scripts/autodl_rescore_all_10pt.sh

echo ""
echo "============================================="
echo "[Phase 1 DONE]"
echo "  rewritten captions: data/compare_5_captions_edit_rewritten.json"
echo "  scores: outputs/compare_5/scores/*.json (4 \u4e2a)"
echo "============================================="
echo ""
echo "[NEXT] \u5728\u672c\u5730\u8dd1:"
echo "  scp -P <port> root@<host>:/root/autodl-tmp/IntelligenceCamera/data/compare_5_captions_edit_rewritten.json data/"
echo "  python tools/data/local_run_longcat_turbo.py \\"
echo "    --captions data/compare_5_captions_edit_rewritten.json \\"
echo "    --out_dir outputs/compare_5/longcat_editB_rewritten \\"
echo "    --skip_download --use_4bit --steps 4 --offload sequential"
echo "  scp -r outputs/compare_5/longcat_editB_rewritten root@<host>:/root/autodl-tmp/IntelligenceCamera/outputs/compare_5/"
echo "  ssh root@<host> 'bash scripts/autodl_phase2_score_longcat_rewritten.sh'"
