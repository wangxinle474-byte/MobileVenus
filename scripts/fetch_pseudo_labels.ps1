# ================================================================
# 从 AutoDL 拉 pseudo_labels.jsonl 到本地, 供 inspect_pseudo_labels.py 检验.
#
# 用法:
#   .\scripts\fetch_pseudo_labels.ps1
#   .\scripts\fetch_pseudo_labels.ps1 -Host "root@connect.westb.seetacloud.com" -Port 12345
#
# 前提: AutoDL 已开机 + 10+ 秒 (SSH 就绪)
# ================================================================

param(
    [string]$RemoteHost = "root@connect.westb.seetacloud.com",
    [int]$Port = 12345,
    [string]$Remote = "/root/autodl-tmp/datasets/ArtEdit-Bench/pseudo_labels.jsonl",
    [string]$LocalDir = "data\pseudo_labels"
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$LocalFull = Join-Path $ProjectRoot (Join-Path $LocalDir "pseudo_labels.jsonl")
$LocalDirFull = Split-Path $LocalFull -Parent
if (-not (Test-Path $LocalDirFull)) {
    New-Item -ItemType Directory -Path $LocalDirFull -Force | Out-Null
}

Write-Host "=== Fetch pseudo_labels.jsonl from AutoDL ===" -ForegroundColor Cyan
Write-Host "  remote: ${RemoteHost}:${Remote} (port $Port)"
Write-Host "  local:  $LocalFull"
Write-Host ""

# 1) quick ssh sanity check
Write-Host "[1/3] ssh connectivity check ..." -ForegroundColor Yellow
$t0 = Get-Date
$remoteTest = ssh -p $Port -o ConnectTimeout=10 -o StrictHostKeyChecking=no $RemoteHost "test -f $Remote && stat -c '%s' $Remote" 2>&1
$dt = (Get-Date) - $t0
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] ssh 不通 ($([math]::Round($dt.TotalSeconds,1))s)" -ForegroundColor Red
    Write-Host "  错误: $remoteTest"
    Write-Host "  提示: AutoDL 可能未开机, 或 SSH 端口/host 变了"
    exit 1
}
$remoteSize = [int64]$remoteTest
$remoteSizeMB = [math]::Round($remoteSize / 1MB, 2)
Write-Host "[OK] ssh up ($([math]::Round($dt.TotalSeconds,1))s), remote file = $remoteSizeMB MB" -ForegroundColor Green

# 2) scp
Write-Host "`n[2/3] scp download ..." -ForegroundColor Yellow
$t0 = Get-Date
scp -P $Port -o ConnectTimeout=10 -o StrictHostKeyChecking=no "${RemoteHost}:${Remote}" $LocalFull
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] scp 失败" -ForegroundColor Red
    exit 2
}
$dt = (Get-Date) - $t0
$localSize = (Get-Item $LocalFull).Length
$localSizeMB = [math]::Round($localSize / 1MB, 2)
Write-Host "[OK] downloaded $localSizeMB MB in $([math]::Round($dt.TotalSeconds,1))s" -ForegroundColor Green
if ($localSize -ne $remoteSize) {
    Write-Host "[WARN] size mismatch: local=$localSize remote=$remoteSize" -ForegroundColor Yellow
}

# 3) 提示下一步
Write-Host "`n[3/3] Next step:" -ForegroundColor Yellow
Write-Host "  python tools\data\inspect_pseudo_labels.py --jsonl $LocalFull" -ForegroundColor Cyan
Write-Host ""
Write-Host "  (记得跑完关掉 AutoDL 省钱)" -ForegroundColor DarkGray
