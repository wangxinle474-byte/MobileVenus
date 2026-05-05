# 本地续传 AesExpert (huang-lin/AesExpert) 到 E:\AesExpert_HF
# 用 hf-mirror.com 镜像加速国内访问
# 用法:
#   .\scripts\local_resume_aesexpert.ps1            # 前台跑 (能看进度,Ctrl+C 中止)
#   .\scripts\local_resume_aesexpert.ps1 -Background # 后台独立进程跑 (关 Cascade 也不影响)

param(
    [switch]$Background
)

$target = "E:\AesExpert_HF"
$logFile = "E:\AesExpert_HF\.cache\download_resume.log"
$pyHelper = "$PSScriptRoot\..\tools\data\local_resume_aesexpert.py"

if (-not (Test-Path "E:\AesExpert_HF\.cache")) {
    New-Item -ItemType Directory -Path "E:\AesExpert_HF\.cache" -Force | Out-Null
}

if ($Background) {
    Write-Host "[INFO] 后台独立进程启动续传" -ForegroundColor Cyan
    Write-Host "[INFO] 日志: $logFile" -ForegroundColor Cyan
    $proc = Start-Process -FilePath "python" `
        -ArgumentList @($pyHelper) `
        -WindowStyle Hidden `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError "$logFile.err" `
        -PassThru
    Write-Host "[INFO] PID: $($proc.Id)  -- 关 Cascade/PowerShell 也不影响" -ForegroundColor Yellow
    Write-Host "[INFO] 监控: Get-Content '$logFile' -Wait -Tail 20" -ForegroundColor Yellow
    Write-Host "[INFO] 终止: Stop-Process -Id $($proc.Id)" -ForegroundColor Yellow
} else {
    Write-Host "[INFO] 前台续传 AesExpert (Ctrl+C 可中止, 已下载部分会保留)" -ForegroundColor Cyan
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    $env:HF_HUB_ENABLE_HF_TRANSFER = "0"
    python $pyHelper
}
