# Path Y end-to-end: ingest → train Path X v2 → eval per-action.
#
# Usage (run AFTER Path Y data generation completes):
#   .\scripts\local\pathY_full_pipeline.ps1 [-WbOnly]
#
# WbOnly mode: only ingests wb action (early validation of WB hypothesis,
#              ~3h total: 0min ingest WB-only + ~3h train + ~5min eval).
# Full mode:   ingests all 5 actions (~15-30 min ingest + ~3h train + ~5min eval).

param(
    [switch]$WbOnly = $false,
    [switch]$SkipIngest = $false,
    [switch]$SkipTrain = $false,
    [switch]$SkipEval = $false
)

$ErrorActionPreference = "Stop"
# Derive project root from script location to avoid non-ASCII path issues
$PROJECT = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $PROJECT

$OUT_NAME = if ($WbOnly) { "pathY_wb_only_master" } else { "pathY_2000_master" }
$JSONL_V2 = "outputs/inverse_fit_pilot/$OUT_NAME/pseudo_labels.jsonl"
$CKPT_DIR = if ($WbOnly) { "checkpoints/lut_v11a_pathX_v2_wb_only" } else { "checkpoints/lut_v11a_pathX_v2_implicit_head" }
$EVAL_DIR = if ($WbOnly) { "outputs/eval_pathX_v2_wb_only" } else { "outputs/eval_pathX_v2" }

Write-Host "=== Path Y Full Pipeline ===" -ForegroundColor Cyan
Write-Host "  Mode      : $(if ($WbOnly) {'WB-only'} else {'Full (5 actions)'})"
Write-Host "  v2 jsonl  : $JSONL_V2"
Write-Host "  ckpt dir  : $CKPT_DIR"
Write-Host "  eval dir  : $EVAL_DIR"
Write-Host ""

# === Step 1: Ingest ===
if (-not $SkipIngest) {
    Write-Host "[1/3] Ingest" -ForegroundColor Yellow
    $t0 = Get-Date

    if ($WbOnly) {
        # Custom ingest: only wb. Reuses ingest_pathY_results.py - it auto-skips
        # actions whose PNG dir is missing, so if only wb has PNGs it'll handle that.
        # But we want a unique output name to avoid mixing with full v2.
        python tools/data/data_prep/ingest_pathY_results.py `
            --pngs_root outputs/teacher_edits/pathY_outputs `
            --out_name $OUT_NAME
    } else {
        python tools/data/data_prep/ingest_pathY_results.py `
            --pngs_root outputs/teacher_edits/pathY_outputs `
            --out_name $OUT_NAME
    }
    $dt = (Get-Date) - $t0
    Write-Host "  Done in $($dt.TotalMinutes.ToString('F1')) min" -ForegroundColor Green
} else {
    Write-Host "[1/3] Ingest SKIPPED" -ForegroundColor Gray
}

if (-not (Test-Path $JSONL_V2)) {
    Write-Error "v2 jsonl not produced: $JSONL_V2"
    exit 1
}

# Show stats
Write-Host "`n  v2 jsonl stats:"
python -c @"
import json
from collections import Counter
n = 0; ac = Counter(); tc = Counter()
with open(r'$JSONL_V2', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        n += 1
        ac[r.get('action', '?')] += 1
        tc[r.get('quality_tier', '?')] += 1
print(f'    total: {n}')
print(f'    actions:')
for a, c in sorted(ac.items()):
    print(f'      {a:12s}: {c}')
print(f'    tiers:')
for t, c in sorted(tc.items()):
    print(f'      {t:15s}: {c}')
"@

# === Step 2: Train Path X v2 ===
if (-not $SkipTrain) {
    Write-Host "`n[2/3] Train Path X v2" -ForegroundColor Yellow
    $t0 = Get-Date

    python training/firered_baseline/train_lut.py `
        --jsonl $JSONL_V2 `
        --out_dir $CKPT_DIR `
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
        --backbone_ckpt "checkpoints/lut_v11a_action_gated_context/best.pt" `
        --freeze_backbone

    $dt = (Get-Date) - $t0
    Write-Host "  Done in $($dt.TotalMinutes.ToString('F1')) min" -ForegroundColor Green
} else {
    Write-Host "`n[2/3] Train SKIPPED" -ForegroundColor Gray
}

# === Step 3: Per-action eval ===
if (-not $SkipEval) {
    Write-Host "`n[3/3] Per-action eval" -ForegroundColor Yellow
    $t0 = Get-Date

    python tools/eval_v11_named_curves.py `
        --ckpt "$CKPT_DIR/best.pt" `
        --jsonl $JSONL_V2 `
        --out_dir $EVAL_DIR `
        --split_seed 42

    $dt = (Get-Date) - $t0
    Write-Host "  Done in $($dt.TotalMinutes.ToString('F1')) min" -ForegroundColor Green

    # Quick comparison vs Path X v1
    Write-Host "`n=== WB hypothesis verdict ===" -ForegroundColor Cyan
    Write-Host "  Path X v1 (372 train):"
    Write-Host "    overall: 25.28 dB, wb: 22.50 dB, highlights: 25.99 dB"
    Write-Host "  Path X v2 (this run):"
    if (Test-Path "$EVAL_DIR/per_action_psnr.json") {
        python -c @"
import json
d = json.load(open(r'$EVAL_DIR/per_action_psnr.json', encoding='utf-8'))
print(f'    overall: {d.get(\"overall\", \"?\"):.2f} dB' if isinstance(d.get('overall'), (int, float)) else '    overall: ' + str(d.get('overall', '?')))
for a in ['wb', 'highlights', 'saturation', 'shadows', 'contrast']:
    v = d.get(a, '?')
    print(f'    {a:12s}: {v:.2f} dB' if isinstance(v, (int, float)) else f'    {a:12s}: {v}')
"@
    } else {
        Write-Host "    (eval output not found - check $EVAL_DIR)"
    }
}

Write-Host "`n=== Pipeline complete ===" -ForegroundColor Green
