# Upload CSGO + Qwen-Image-CN comparison files to AutoDL
#
# Usage:
#   .\scripts\upload_compare_5.ps1
#   .\scripts\upload_compare_5.ps1 -RemoteHost "root@connect.xxx.com" -Port 12345

param(
    [string]$RemoteHost = "root@connect.westb.seetacloud.com",
    [int]$Port = 12345,
    [string]$RemoteDir = "/root/autodl-tmp/IntelligenceCamera"
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "=== Upload CSGO/Qwen comparison files ===" -ForegroundColor Green
Write-Host ("  local : " + $ProjectRoot)
Write-Host ("  remote: " + $RemoteHost + ":" + $RemoteDir + " (port " + $Port + ")")
Write-Host ""

# 1. SSH connectivity test
Write-Host "[1/4] Test SSH..." -ForegroundColor Yellow
$sshTest = ssh -p $Port -o ConnectTimeout=10 -o BatchMode=yes $RemoteHost "echo OK; nvidia-smi -L 2>&1 | head -2" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "X SSH failed. Check:" -ForegroundColor Red
    Write-Host "  1. AutoDL instance running" -ForegroundColor Red
    Write-Host "  2. SSH port correct (-Port arg)" -ForegroundColor Red
    Write-Host "  3. SSH key configured" -ForegroundColor Red
    Write-Host ("Error: " + $sshTest) -ForegroundColor DarkRed
    exit 1
}
Write-Host "OK:" -ForegroundColor Green
Write-Host ($sshTest -join "`n") -ForegroundColor DarkGray

# 2. Create remote dirs
Write-Host "`n[2/4] Create remote dirs..." -ForegroundColor Yellow
$mkdirCmd = "mkdir -p " + $RemoteDir + "/tools/data " + $RemoteDir + "/data " + $RemoteDir + "/outputs/ip2p_pilot_100 /root/autodl-tmp/logs"
ssh -p $Port $RemoteHost $mkdirCmd

# 3. Upload files
Write-Host "`n[3/4] Upload files..." -ForegroundColor Yellow

$Files = @(
    "tools/data/autodl_run_csgo.py",
    "tools/data/autodl_run_qwen_cn.py",
    "data/compare_5_captions.json",
    "outputs/ip2p_pilot_100/0071_orig.png",
    "outputs/ip2p_pilot_100/0110_orig.png",
    "outputs/ip2p_pilot_100/0194_orig.png",
    "outputs/ip2p_pilot_100/0448_orig.png",
    "outputs/ip2p_pilot_100/0808_orig.png"
)

$uploaded = 0
foreach ($rel in $Files) {
    $relWin = $rel.Replace("/", "\")
    $localFile = Join-Path $ProjectRoot $relWin
    if (-not (Test-Path $localFile)) {
        Write-Host ("  X missing: " + $rel) -ForegroundColor Red
        continue
    }
    $remoteFile = $RemoteDir + "/" + $rel
    scp -P $Port $localFile ($RemoteHost + ":" + $remoteFile) 2>$null
    if ($LASTEXITCODE -eq 0) {
        $size = (Get-Item $localFile).Length
        $sizeKB = [math]::Round($size / 1024, 1)
        Write-Host ("  + " + $rel + " (" + $sizeKB + " KB)") -ForegroundColor DarkGray
        $uploaded++
    } else {
        Write-Host ("  X " + $rel + " (scp fail)") -ForegroundColor Red
    }
}

Write-Host ("`n[4/4] Done: " + $uploaded + " / " + $Files.Count + " uploaded") -ForegroundColor Green

Write-Host "`n=== AutoDL run commands ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "ssh -p $Port $RemoteHost"
Write-Host "cd $RemoteDir"
Write-Host ""
Write-Host "# Install deps (if not already):"
Write-Host "pip install -U diffusers transformers accelerate safetensors huggingface_hub einops"
Write-Host ""
Write-Host "# 1. CSGO inference (~14GB VRAM, ~10 min for 5 images):"
Write-Host "python tools/data/autodl_run_csgo.py 2>&1 | tee /root/autodl-tmp/logs/csgo_run.log"
Write-Host ""
Write-Host "# 2. Qwen-Image-CN-Inpainting (~40GB model, offload mode, ~30 min):"
Write-Host "python tools/data/autodl_run_qwen_cn.py --mode offload 2>&1 | tee /root/autodl-tmp/logs/qwen_run.log"
Write-Host ""
Write-Host "# 3. Check results:"
Write-Host "ls -lh outputs/csgo_compare/ outputs/qwen_cn_compare/"
Write-Host ""
Write-Host "=== Download results (run locally) ===" -ForegroundColor Cyan
Write-Host "scp -P $Port -r `"${RemoteHost}:${RemoteDir}/outputs/csgo_compare`" outputs/"
Write-Host "scp -P $Port -r `"${RemoteHost}:${RemoteDir}/outputs/qwen_cn_compare`" outputs/"
