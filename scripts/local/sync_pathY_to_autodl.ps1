# Path Y: sync source images + caption JSONs to AutoDL
# Run from local Windows machine.
#
# Prerequisites:
#   - AutoDL instance running (L20 48GB)
#   - SSH configured (see scripts/local/tail_autodl_log.ps1 for host/port)
#
# Usage:
#   .\scripts\local\sync_pathY_to_autodl.ps1

$ErrorActionPreference = "Stop"

# === CONFIG (update these to match your AutoDL instance) ===
$AUTODL_HOST = "root@connect.bjb2.seetacloud.com"
$AUTODL_PORT = 35345  # UPDATE if instance changed
$REMOTE_BASE = "/root/autodl-tmp"
# ===========================================================

$PROJECT = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$FIVEK_LOCAL = "E:\Data\dataset\fivek_jpeg"
$SOURCE_LIST = "$PROJECT\data\teacher_edits_fivek_pathY_300_sources.txt"
$CAPTIONS_DIR = "$PROJECT\data\pathY_captions"

Write-Host "=== Path Y: sync to AutoDL ===" -ForegroundColor Cyan
Write-Host "  host: $AUTODL_HOST`:$AUTODL_PORT"
Write-Host "  remote: $REMOTE_BASE"

# 1. Create remote directories
Write-Host "`n[1/3] Creating remote directories..." -ForegroundColor Yellow
ssh -p $AUTODL_PORT $AUTODL_HOST "mkdir -p $REMOTE_BASE/pathY_captions $REMOTE_BASE/pathY_outputs $REMOTE_BASE/pathY_logs"

# 2. Upload per-action caption JSONs
Write-Host "`n[2/3] Uploading caption JSONs..." -ForegroundColor Yellow
scp -P $AUTODL_PORT "$CAPTIONS_DIR\pathY_*.json" "${AUTODL_HOST}:$REMOTE_BASE/pathY_captions/"
Write-Host "  uploaded 5 per-action caption files"

# 3. Upload source images (only the 300 new ones)
Write-Host "`n[3/3] Uploading 300 source images..." -ForegroundColor Yellow
$sources = Get-Content $SOURCE_LIST
$tempDir = New-Item -ItemType Directory -Path "$env:TEMP\pathY_sources" -Force

# Check which are already on remote
Write-Host "  checking remote for existing files..."
$remoteList = ssh -p $AUTODL_PORT $AUTODL_HOST "ls $REMOTE_BASE/fivek_jpeg/ 2>/dev/null" 2>$null
$remoteSet = @{}
if ($remoteList) {
    $remoteList -split "`n" | ForEach-Object { $remoteSet[$_.Trim()] = $true }
}

$toUpload = @()
foreach ($src in $sources) {
    $src = $src.Trim()
    if (-not $remoteSet.ContainsKey($src)) {
        $toUpload += $src
    }
}
Write-Host "  $($toUpload.Count) new images to upload ($($sources.Count - $toUpload.Count) already on remote)"

if ($toUpload.Count -gt 0) {
    # Copy to temp dir for batch scp
    foreach ($src in $toUpload) {
        Copy-Item "$FIVEK_LOCAL\$src" "$tempDir\" -ErrorAction SilentlyContinue
    }
    scp -P $AUTODL_PORT -r "$tempDir\*" "${AUTODL_HOST}:$REMOTE_BASE/fivek_jpeg/"
    Write-Host "  uploaded $($toUpload.Count) images"
    Remove-Item $tempDir -Recurse -Force
}

# 4. Sync the inference script
Write-Host "`n[bonus] Syncing inference script..." -ForegroundColor Yellow
scp -P $AUTODL_PORT "$PROJECT\tools\data\editor_models\run_firered_hf_local.py" `
    "${AUTODL_HOST}:$REMOTE_BASE/IntelligenceCamera/tools/data/editor_models/"
scp -P $AUTODL_PORT "$PROJECT\scripts\autodl\run_pathY_firered_batch.sh" `
    "${AUTODL_HOST}:$REMOTE_BASE/IntelligenceCamera/scripts/autodl/"

Write-Host "`n=== Sync complete ===" -ForegroundColor Green
Write-Host "Next: SSH to AutoDL and run:"
Write-Host "  bash $REMOTE_BASE/IntelligenceCamera/scripts/autodl/run_pathY_firered_batch.sh"
Write-Host "  (estimated ~5h on L20 48GB)"
