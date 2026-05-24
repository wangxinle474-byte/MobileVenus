# Path Y 本地进度监控 - 每 30 秒刷新
#
# Usage:
#   .\scripts\local\watch_pathY_local.ps1
#   Ctrl+C 退出

$ErrorActionPreference = "Stop"
$OUT_BASE = "outputs\teacher_edits\pathY_outputs"
$ACTIONS = @("wb", "highlights", "saturation", "shadows", "contrast")
$ACTION_N = 300
$TOTAL = $ACTIONS.Count * $ACTION_N

$prev_total = $null
$prev_time = $null

while ($true) {
    Clear-Host
    $now = Get-Date
    Write-Host "=== Path Y 本地进度 @ $($now.ToString('HH:mm:ss')) ===" -ForegroundColor Cyan
    Write-Host ""

    # 1. 进程检查
    $python_procs = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -match "run_firered|run_pathY" -or
        (Get-CimInstance Win32_Process -Filter "ProcessId = $($_.Id)" -ErrorAction SilentlyContinue).CommandLine -match "run_firered|run_pathY"
    }
    if ($python_procs) {
        $n_proc = ($python_procs | Measure-Object).Count
        Write-Host "● ALIVE  ($n_proc python process(es))" -ForegroundColor Green
    } else {
        # Fallback: check by command line via WMI
        $wmi = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue | Where-Object {
            $_.CommandLine -match "run_firered|run_pathY"
        }
        if ($wmi) {
            $n_proc = ($wmi | Measure-Object).Count
            Write-Host "● ALIVE  ($n_proc python process(es))" -ForegroundColor Green
        } else {
            Write-Host "○ STOPPED  (no run_firered/run_pathY python process found)" -ForegroundColor Red
        }
    }
    Write-Host ""

    # 2. 各 action 完成数
    Write-Host "Per-action:" -ForegroundColor Yellow
    $total_done = 0
    foreach ($action in $ACTIONS) {
        $dir = Join-Path $OUT_BASE $action
        if (Test-Path $dir) {
            $n = (Get-ChildItem $dir -Filter *.png -ErrorAction SilentlyContinue | Measure-Object).Count
        } else {
            $n = 0
        }
        $total_done += $n
        $pct = [int]($n * 100 / $ACTION_N)
        $bar_len = 20
        $filled = [int]($pct * $bar_len / 100)
        $empty = $bar_len - $filled
        $bar = ('#' * $filled) + ('.' * $empty)

        $mark = " "
        $color = "White"
        if ($n -ge $ACTION_N) {
            $mark = "✓"
            $color = "Cyan"
        } elseif ($n -gt 0 -and $n -lt $ACTION_N) {
            $mark = "*"
            $color = "Green"
        }
        Write-Host ("  {0} {1,-12} [{2}] {3,3}/{4}  ({5,3}%)" -f $mark, $action, $bar, $n, $ACTION_N, $pct) -ForegroundColor $color
    }

    # 3. 总进度
    $tpct = [int]($total_done * 100 / $TOTAL)
    $tbar_len = 40
    $tfilled = [int]($tpct * $tbar_len / 100)
    $tempty = $tbar_len - $tfilled
    $tbar = ('#' * $tfilled) + ('.' * $tempty)
    Write-Host ""
    Write-Host ("Total: [{0}] {1}/{2}  ({3}%)" -f $tbar, $total_done, $TOTAL, $tpct) -ForegroundColor Yellow

    # 4. 速率 + ETA
    if ($prev_total -ne $null -and $prev_time -ne $null) {
        $dt = ($now - $prev_time).TotalMinutes
        $dn = $total_done - $prev_total
        if ($dt -gt 0 -and $dn -gt 0) {
            $rate = $dn / $dt  # samples per min
            $remaining = $TOTAL - $total_done
            $eta_min = $remaining / $rate
            $eta_str = if ($eta_min -lt 60) {
                "$([int]$eta_min) min"
            } else {
                "$([math]::Round($eta_min / 60, 1)) h"
            }
            Write-Host ("  rate: {0:F2} smp/min,  ETA: {1}" -f $rate, $eta_str) -ForegroundColor Gray
        }
    }

    # 5. 最近 log 行 (前一个 action 完成或当前 action 启动)
    $log_dir = Join-Path $OUT_BASE "_logs"
    if (Test-Path $log_dir) {
        $latest_log = Get-ChildItem $log_dir -Filter "*.log" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest_log) {
            Write-Host ""
            Write-Host "Latest log ($($latest_log.Name)):" -ForegroundColor Gray
            $tail = Get-Content $latest_log.FullName -Tail 3 -ErrorAction SilentlyContinue
            if ($tail) {
                foreach ($line in $tail) {
                    if ($line.Length -gt 110) { $line = $line.Substring(0, 110) }
                    Write-Host "  $line" -ForegroundColor DarkGray
                }
            } else {
                Write-Host "  (log empty - subprocess stdout buffering, PNG count is source of truth)" -ForegroundColor DarkGray
            }
        }
    }

    $prev_total = $total_done
    $prev_time = $now

    Write-Host ""
    Write-Host "Refreshes every 30s. Ctrl+C to stop." -ForegroundColor DarkGray
    Start-Sleep -Seconds 30
}
