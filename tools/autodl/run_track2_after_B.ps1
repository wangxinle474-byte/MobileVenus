# tools/run_track2_after_B.ps1
# Chain that runs the remaining Track 2 work after Path B has finished:
#   Step 1: eval Path B on three sets (~5 min)
#   Step 2: train Path A from scratch (~12-15 h)
#   Step 3: eval Path A on three sets (~5 min)
#   Step 4: print pointers for the four-cell decision in docs/track2_results.md
#
# Prerequisite: Path B training process has already exited (either trip-wire
# or natural 80-epoch completion). The script reads checkpoints/lut_track2B_*/
# best.pt and refuses to run if it's missing.
#
# Usage from project root:
#   pwsh tools\run_track2_after_B.ps1
#
# Skip steps:
#   pwsh tools\run_track2_after_B.ps1 -SkipEvalB         # skip eval B
#   pwsh tools\run_track2_after_B.ps1 -SkipTrainA        # only eval B
#   pwsh tools\run_track2_after_B.ps1 -SkipEvalA         # train A but don't eval
#
# Custom Path A out_dir (useful for capacity sweep on B if four-cell needs more):
#   pwsh tools\run_track2_after_B.ps1 -AOutDir checkpoints\lut_track2A_alt
#
# All output is tee'd to a log file alongside each step.

[CmdletBinding()]
param(
    [string]$BCkpt = "checkpoints\lut_track2B_nilut_clean_seed42\best.pt",
    [string]$AOutDir = "checkpoints\lut_track2A_vera_clean_seed42",
    [string]$EvalBDir = "outputs\eval_track2\B",
    [string]$EvalADir = "outputs\eval_track2\A",
    [switch]$SkipEvalB,
    [switch]$SkipTrainA,
    [switch]$SkipEvalA
)

$ErrorActionPreference = 'Continue'
# PowerShell 7.4+ converts native-command stderr+nonzero-exit into terminating
# errors when $ErrorActionPreference is 'Stop'.  Python's logging module writes
# INFO/DEBUG to stderr by default, which would abort this chain prematurely.
# Explicit $LASTEXITCODE checks below provide the real error gating.
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$startTime = Get-Date

function Write-Step {
    param([string]$msg)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host ""
}

Write-Host "Track 2 chain start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "  B ckpt:   $BCkpt"
Write-Host "  A outdir: $AOutDir"
Write-Host "  Eval B:   $EvalBDir"
Write-Host "  Eval A:   $EvalADir"

# ---- Sanity: B checkpoint must exist ----
if (-not (Test-Path $BCkpt)) {
    Write-Host ""
    Write-Host "[ERROR] Path B checkpoint not found: $BCkpt" -ForegroundColor Red
    Write-Host "        Has Path B finished?"
    Write-Host "        Check:   Get-ChildItem checkpoints\lut_track2B_nilut_clean_seed42\"
    exit 1
}

$ACkpt = Join-Path $AOutDir 'best.pt'

# ============================================================
# STEP 1: eval Path B on three sets
# ============================================================
if (-not $SkipEvalB) {
    Write-Step "STEP 1/3: eval Path B on three sets (clean ExpC / FireRed74 / MMArt250)"
    if (-not (Test-Path $EvalBDir)) { New-Item -ItemType Directory -Path $EvalBDir -Force | Out-Null }
    $evalBLog = Join-Path $EvalBDir 'eval_log.txt'

    python tools\eval_track2.py `
        --ckpt $BCkpt `
        --out_dir $EvalBDir `
        2>&1 | Tee-Object -FilePath $evalBLog

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] eval B failed with exit $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Path B eval: $EvalBDir\track2_eval.json" -ForegroundColor Green
} else {
    Write-Host "[skip] Step 1 (eval B)" -ForegroundColor Yellow
}

# ============================================================
# STEP 2: train Path A from scratch
# ============================================================
if (-not $SkipTrainA) {
    Write-Step "STEP 2/3: train Path A (VeraRenderer, no Bezier; ETA ~12-15 h)"
    if (-not (Test-Path $AOutDir)) { New-Item -ItemType Directory -Path $AOutDir -Force | Out-Null }
    $aLog = "$AOutDir\..\lut_track2A_log.txt"

    # ⚠ Path A intentionally OMITS --nc_use_7d_anchor / --nc_use_context /
    # --nc_action_gated_context.  Forward branch in train_lut.py:1131-1136
    # short-circuits to VeraRenderer when nc_use_vera_renderer is true; the
    # parametric anchor flags become dead weight that only confuse
    # param_head training.
    python -u training\firered_baseline\train_lut.py `
        --jsonl outputs\fivek_expert_c_master\pseudo_labels.jsonl `
        --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
        --nc_use_vera_renderer `
        --vera_latent_dim 64 --vera_hidden 128 --vera_n_layers 6 `
        --vera_gate_init 1.0 `
        --param_weight 0.05 --dropout 0.5 `
        --epochs 80 --seed 42 `
        --out_dir $AOutDir `
        *>&1 | Tee-Object -FilePath $aLog

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Path A training failed with exit $LASTEXITCODE" -ForegroundColor Red
        Write-Host "        Log: $aLog"
        exit 1
    }
    if (-not (Test-Path $ACkpt)) {
        Write-Host "[ERROR] Path A finished but $ACkpt does not exist" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Path A training: $ACkpt" -ForegroundColor Green
} else {
    Write-Host "[skip] Step 2 (train A)" -ForegroundColor Yellow
}

# ============================================================
# STEP 3: eval Path A on three sets
# ============================================================
if (-not $SkipEvalA) {
    if (-not (Test-Path $ACkpt)) {
        Write-Host "[skip] Step 3 (eval A) -- $ACkpt does not exist" -ForegroundColor Yellow
    } else {
        Write-Step "STEP 3/3: eval Path A on three sets"
        if (-not (Test-Path $EvalADir)) { New-Item -ItemType Directory -Path $EvalADir -Force | Out-Null }
        $evalALog = Join-Path $EvalADir 'eval_log.txt'

        python tools\eval_track2.py `
            --ckpt $ACkpt `
            --out_dir $EvalADir `
            2>&1 | Tee-Object -FilePath $evalALog

        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] eval A failed with exit $LASTEXITCODE" -ForegroundColor Red
            exit 1
        }
        Write-Host "[OK] Path A eval: $EvalADir\track2_eval.json" -ForegroundColor Green
    }
} else {
    Write-Host "[skip] Step 3 (eval A)" -ForegroundColor Yellow
}

# ============================================================
# Summary
# ============================================================
$elapsed = (Get-Date) - $startTime
Write-Step "Chain complete  --  elapsed: $([int]$elapsed.TotalHours)h $([int]($elapsed.TotalMinutes % 60))m"

Write-Host "Next steps (manual):" -ForegroundColor Green
Write-Host "  1. Open docs\track2_results.md and fill the TBD cells:"
Write-Host "       - §3 Path B table from $EvalBDir\track2_eval.json"
Write-Host "       - §4 Path A table from $EvalADir\track2_eval.json"
Write-Host "  2. Pick the matching row in §5 four-cell decision table"
Write-Host "  3. Decide: standalone Track 2 paper or merge into Track 1 §4.9"
Write-Host ""
Write-Host "Quick view of overall numbers:" -ForegroundColor Green
Write-Host '  python -c "import json; d=json.load(open(r''' + $EvalBDir + '\track2_eval.json'')); print(''B'', {k: v[''subsets''].get(''leak_free'', v[''subsets''].get(''full'', {})).get(''overall'') for k,v in d[''eval_sets''].items()})"'
Write-Host '  python -c "import json; d=json.load(open(r''' + $EvalADir + '\track2_eval.json'')); print(''A'', {k: v[''subsets''].get(''leak_free'', v[''subsets''].get(''full'', {})).get(''overall'') for k,v in d[''eval_sets''].items()})"'
