# 本地续传 AesExpert v2: 不依赖 hf_hub, 用 requests 走代理直接下载
# 用法:
#   .\scripts\local_resume_aesexpert_v2.ps1            # 前台 (能看进度)
#   .\scripts\local_resume_aesexpert_v2.ps1 -Background # 后台独立进程

param(
    [switch]$Background
)

$pyHelper = "$PSScriptRoot\..\tools\data\local_resume_aesexpert_v2.py"
$logFile = "E:\AesExpert_HF\.cache\download_resume_v2.log"

if (-not (Test-Path "E:\AesExpert_HF\.cache")) {
    New-Item -ItemType Directory -Path "E:\AesExpert_HF\.cache" -Force | Out-Null
}

if ($Background) {
    Write-Host "[INFO] 后台续传 v2" -ForegroundColor Cyan
    Write-Host "[INFO] 日志: $logFile" -ForegroundColor Cyan
    $proc = Start-Process -FilePath "python" `
        -ArgumentList @($pyHelper) `
        -WindowStyle Hidden `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError "$logFile.err" `
        -PassThru
    Write-Host "[INFO] PID: $($proc.Id)" -ForegroundColor Yellow
    Write-Host "[INFO] 监控: Get-Content '$logFile' -Wait -Tail 30" -ForegroundColor Yellow
    Write-Host "[INFO] 终止: Stop-Process -Id $($proc.Id)" -ForegroundColor Yellow
} else {
    Write-Host "[INFO] 前台续传 v2 (Ctrl+C 中止, 已下载部分会保留)" -ForegroundColor Cyan
    python $pyHelper
}
