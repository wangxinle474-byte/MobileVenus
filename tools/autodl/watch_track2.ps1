# tools/watch_track2.ps1
# Real-time follower for the Track 2 chain (eval B -> train A -> eval A).
# Polls every $IntervalSec seconds, auto-switches to whichever log is
# currently being written, and prints only new bytes since last read.
#
# Usage from project root (in a separate PowerShell window):
#   pwsh tools\watch_track2.ps1
#   pwsh tools\watch_track2.ps1 -IntervalSec 1     # faster polling
#
# Ctrl+C to exit.

param(
    [int]$IntervalSec = 2
)

$logs = @(
    @{ Name = 'eval B';  Path = 'outputs\eval_track2\B\eval_log.txt' },
    @{ Name = 'train A'; Path = 'checkpoints\lut_track2A_log.txt'    },
    @{ Name = 'eval A';  Path = 'outputs\eval_track2\A\eval_log.txt' }
)

Write-Host "Track 2 chain follower" -ForegroundColor Cyan
Write-Host "  poll interval: ${IntervalSec}s, Ctrl+C to exit" -ForegroundColor DarkCyan
Write-Host "  tracked logs:" -ForegroundColor DarkCyan
$logs | ForEach-Object { Write-Host ("    [{0,-7}] {1}" -f $_.Name, $_.Path) -ForegroundColor DarkCyan }
Write-Host ("-" * 72)

$lastLog  = ''
$lastSize = 0L

while ($true) {
    # Filter to logs that exist and pick the most-recently-modified
    $existing = @()
    foreach ($l in $logs) {
        if (Test-Path $l.Path) {
            $info = Get-Item $l.Path
            $existing += [PSCustomObject]@{
                Name         = $l.Name
                Path         = $l.Path
                LastWriteTime= $info.LastWriteTime
                Length       = $info.Length
            }
        }
    }

    if (-not $existing) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] no logs yet, waiting..." -ForegroundColor Yellow
        Start-Sleep -Seconds $IntervalSec
        continue
    }

    $active = $existing | Sort-Object LastWriteTime -Descending | Select-Object -First 1

    # Banner when switching to a new log
    if ($active.Path -ne $lastLog) {
        Write-Host ""
        Write-Host ("=" * 72) -ForegroundColor Green
        Write-Host ("[{0}] >>> following: {1,-7} ({2})" -f (Get-Date -Format 'HH:mm:ss'), $active.Name, $active.Path) -ForegroundColor Green
        Write-Host ("=" * 72) -ForegroundColor Green
        Write-Host ""
        $lastLog  = $active.Path
        # On switch, dump the last ~30 lines for context, then continue from current end
        $tail = Get-Content $active.Path -Tail 30 -ErrorAction SilentlyContinue
        if ($tail) { $tail | ForEach-Object { Write-Host $_ } }
        $lastSize = $active.Length
        Start-Sleep -Seconds $IntervalSec
        continue
    }

    # Print only the new bytes since last read (efficient for high-rate logs)
    if ($active.Length -gt $lastSize) {
        try {
            $fs = [System.IO.File]::Open($active.Path, 'Open', 'Read', 'ReadWrite')
            $fs.Seek($lastSize, 'Begin') | Out-Null
            $sr = New-Object System.IO.StreamReader($fs)
            $chunk = $sr.ReadToEnd()
            $sr.Close()
            $fs.Close()
            if ($chunk) {
                # Trim trailing newline to avoid double-spacing
                Write-Host -NoNewline $chunk
            }
            $lastSize = $active.Length
        } catch {
            Write-Host "[warn] read failed: $_" -ForegroundColor Yellow
        }
    }

    Start-Sleep -Seconds $IntervalSec
}
