# Watchdog: poll wb count every 60s, launch wb_cooler when warmer hits 300.
#
# Run in a NEW PowerShell window (alongside the existing wb-warmer launcher window).
#
# Usage:
#   Start-Process powershell -ArgumentList "-NoExit", "-Command", "& {cd 'E:\智能相机\Venus_CVPR2026-main\IntelligenceCamera'; .\scripts\local\launch_wb_cooler_after_warmer.ps1}"

$ErrorActionPreference = "Stop"
# Derive project root from script location to avoid hardcoding paths with
# non-ASCII chars (which break under Windows PowerShell ANSI decoding).
$PROJECT = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $PROJECT

$WB_DIR = "outputs\teacher_edits\pathY_outputs\wb"
$TARGET = 300

Write-Host "=== wb_cooler watchdog ===" -ForegroundColor Cyan
Write-Host "Polling $WB_DIR every 60s for $TARGET PNGs"
Write-Host "(Ctrl+C to abort watchdog without affecting warmer run)`n"

while ($true) {
    if (Test-Path $WB_DIR) {
        $n = (Get-ChildItem $WB_DIR -Filter *.png -ErrorAction SilentlyContinue | Measure-Object).Count
    } else {
        $n = 0
    }
    $ts = (Get-Date).ToString("HH:mm:ss")
    Write-Host "[$ts] wb warmer: $n/$TARGET"

    if ($n -ge $TARGET) {
        Write-Host "`n[OK] wb warmer complete. Launching wb_cooler..." -ForegroundColor Green
        # Launch wb_cooler in this window's foreground (visible progress)
        python scripts/local/run_pathY_api_local.py --actions wb_cooler
        Write-Host "`n=== wb_cooler complete ===" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 60
}
