# AesExpert download progress watcher
# Usage: .\scripts\watch_download.ps1

$target = "E:\AesExpert_HF"
$totalExpected = 14.1e9
$prevSize = 0
$prevTime = Get-Date

Clear-Host
Write-Host "========== AesExpert Download Monitor ==========" -ForegroundColor Cyan
Write-Host "Target: $target" -ForegroundColor Gray
Write-Host "Expected: $([math]::Round($totalExpected/1GB,1)) GB" -ForegroundColor Gray
Write-Host "Press Ctrl+C to exit`n" -ForegroundColor Gray

while ($true) {
    try {
        $files = Get-ChildItem $target -Recurse -File -ErrorAction SilentlyContinue
        $currentSize = ($files | Measure-Object -Property Length -Sum).Sum
        $now = Get-Date

        $dt = ($now - $prevTime).TotalSeconds
        $speed = if ($dt -gt 0) { ($currentSize - $prevSize) / $dt } else { 0 }

        $pct = [math]::Round($currentSize / $totalExpected * 100, 1)
        $doneGB = [math]::Round($currentSize / 1GB, 2)
        $speedMBs = [math]::Round($speed / 1MB, 2)

        $remaining = $totalExpected - $currentSize
        $etaSec = if ($speed -gt 0) { [math]::Round($remaining / $speed) } else { 0 }
        $etaH = [math]::Floor($etaSec / 3600)
        $etaM = [math]::Floor(($etaSec % 3600) / 60)
        $etaStr = if ($etaSec -gt 0) { "${etaH}h${etaM}m" } else { "--" }

        $barLen = 40
        $filled = [math]::Floor($pct / 100 * $barLen)
        $bar = "[" + ("=" * $filled) + (" " * ($barLen - $filled)) + "]"

        $timestamp = $now.ToString("HH:mm:ss")
        $line = "$timestamp  $bar  $pct%  ${doneGB}GB  |  $speedMBs MB/s  |  ETA: $etaStr          "
        Write-Host "`r$line" -NoNewline

        $prevSize = $currentSize
        $prevTime = $now

        if ($pct -ge 100) {
            Write-Host "`n`n[DONE] Download complete!" -ForegroundColor Green
            break
        }
    } catch {
        Write-Host "`nError: $_" -ForegroundColor Red
    }

    Start-Sleep -Seconds 2
}
