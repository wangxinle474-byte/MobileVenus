# Path X v2 retraining: same arch as v1 (frozen v11a backbone + implicit residual head),
# but trained on v2 master jsonl which includes Path Y +1500 FireRed samples.
#
# Prerequisites:
#   1. Path Y outputs complete (5 actions × 300 PNGs)
#   2. Ingest pipeline complete:
#        python tools/data/data_prep/ingest_pathY_results.py
#      → produces outputs/inverse_fit_pilot/pathY_2000_master/pseudo_labels.jsonl
#   3. v11a backbone checkpoint: checkpoints/lut_v11a_action_gated_context/best.pt
#
# Usage:
#   .\scripts\local\train_pathX_v2.ps1
#
# Expected runtime: ~3 h on RTX 4080
# Target: WB PSNR 22.50 → ≥25 dB if sample-limited, ~22.50 if structurally non-ISP

$ErrorActionPreference = "Stop"

# Derive project root from script location to avoid non-ASCII path issues
$PROJECT = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $PROJECT

$JSONL_V2 = "outputs/inverse_fit_pilot/pathY_2000_master/pseudo_labels.jsonl"
$BACKBONE = "checkpoints/lut_v11a_action_gated_context/best.pt"
$OUT_DIR  = "checkpoints/lut_v11a_pathX_v2_implicit_head"

# Sanity checks
if (-not (Test-Path $JSONL_V2)) {
    Write-Error "v2 jsonl not found: $JSONL_V2"
    Write-Host "  Run: python tools/data/data_prep/ingest_pathY_results.py"
    exit 1
}
if (-not (Test-Path $BACKBONE)) {
    Write-Error "backbone ckpt not found: $BACKBONE"
    exit 1
}

# Show v2 jsonl record count + per-action breakdown
Write-Host "=== v2 jsonl stats ===" -ForegroundColor Cyan
python -c @"
import json
from collections import Counter
n = 0; ac = Counter()
with open(r'$JSONL_V2', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        n += 1
        ac[r.get('action', '?')] += 1
print(f'total: {n}')
for a, c in sorted(ac.items()):
    print(f'  {a:12s}: {c}')
"@

Write-Host "`n=== Path X v2 training ===" -ForegroundColor Cyan
Write-Host "  jsonl:    $JSONL_V2"
Write-Host "  backbone: $BACKBONE"
Write-Host "  out_dir:  $OUT_DIR"
Write-Host "  expected: ~3h on RTX 4080`n"

# Train (same hyperparams as v1 Path X, only --jsonl differs)
python training/firered_baseline/train_lut.py `
    --jsonl $JSONL_V2 `
    --out_dir $OUT_DIR `
    --image_size 256 `
    --batch_size 4 `
    --epochs 40 `
    --lr 3e-4 `
    --lut_lr 1e-3 `
    --weight_decay 1e-3 `
    --dropout 0.5 `
    --val_ratio 0.2 `
    --seed 42 `
    --patience 10 `
    --l1_weight 1.0 `
    --ssim_weight 0.5 `
    --smooth_weight 1e-4 `
    --mono_weight 1e-2 `
    --coarse_weight 0.5 `
    --param_weight 0.05 `
    --named_curves `
    --nc_n_colors 3 `
    --nc_n_control_points 7 `
    --nc_use_7d_anchor `
    --nc_use_context `
    --nc_action_gated_context `
    --use_implicit_head `
    --implicit_head_base_ch 32 `
    --implicit_head_gate_init 1.0 `
    --implicit_head_lr 5e-4 `
    --backbone_ckpt $BACKBONE `
    --freeze_backbone

Write-Host "`n=== Training complete ===" -ForegroundColor Green
Write-Host "  ckpt: $OUT_DIR\best.pt"
Write-Host "`nNext: per-action eval"
Write-Host "  python tools/eval_v11_named_curves.py \\"
Write-Host "      --ckpt $OUT_DIR/best.pt \\"
Write-Host "      --jsonl $JSONL_V2 \\"
Write-Host "      --out_dir outputs/eval_pathX_v2 \\"
Write-Host "      --split_seed 42"
