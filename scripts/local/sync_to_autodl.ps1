# ================================================================
# AutoDL 同步脚本 (Windows PowerShell) — 只上传训练相关代码
#
# 用法:
#   .\scripts\sync_to_autodl.ps1
#   .\scripts\sync_to_autodl.ps1 -Host "root@connect.westb.seetacloud.com" -Port 12345
#
# 前提: 安装 scp (Windows 10+ 自带 OpenSSH)
# ================================================================

param(
    [string]$RemoteHost = "root@connect.westb.seetacloud.com",
    [int]$Port = 12345,
    [string]$RemoteDir = "/root/autodl-tmp/IntelligenceCamera"
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "=== 同步训练代码到 AutoDL ===" -ForegroundColor Green
Write-Host "  本地: $ProjectRoot"
Write-Host "  远程: ${RemoteHost}:${RemoteDir} (port $Port)"
Write-Host ""

# 训练所需的文件列表
$TrainFiles = @(
    # 模型核心 (排除 DEPRECATED 文件)
    "models/__init__.py",
    "models/vision_encoder.py",
    "models/semantic_bridge.py",
    "models/diff_isp.py",
    "models/refinement_net_v4.py",
    "models/isp_pipeline.py",
    "models/aesthetic_scorer.py",

    # 训练框架
    "training/__init__.py",
    "training/aesthetic_loss.py",
    "training/dataset.py",
    "training/distillation.py",
    "training/train_config.yaml",
    "training/train_aadb_aesthetic.py",

    # semantic_distill
    "training/semantic_distill/__init__.py",
    "training/semantic_distill/__main__.py",
    "training/semantic_distill/config.py",
    "training/semantic_distill/embed_texts.py",
    "training/semantic_distill/loss.py",
    "training/semantic_distill/model.py",
    "training/semantic_distill/text_dataset.py",
    "training/semantic_distill/trainer.py",

    # text_condition
    "training/text_condition/__init__.py",
    "training/text_condition/config.py",
    "training/text_condition/model.py",

    # fivek_8param
    "training/fivek_8param/__init__.py",
    "training/fivek_8param/__main__.py",
    "training/fivek_8param/config.py",
    "training/fivek_8param/dataset.py",
    "training/fivek_8param/dataset_expert.py",
    "training/fivek_8param/dataset_ppr10k.py",
    "training/fivek_8param/loss.py",
    "training/fivek_8param/model.py",
    "training/fivek_8param/trainer.py",

    # 训练脚本 (legacy_training 归档 + 根目录活动)
    "scripts/legacy_training/train_v6_stage_a.py",
    "scripts/legacy_training/train_v6_stage_b.py",
    "scripts/legacy_training/train_v7_stage_b.py",
    "scripts/legacy_training/train_stage_c.py",
    "scripts/legacy_training/train_v13_multiscale.py",
    "train_v12_refine_hd.py",
    "train_v6_stage_a.py",

    # 工具 (训练/数据/评估)
    "tools/eval/*.py",
    "tools/data/*.py",
    "tools/train/*.py",

    # 配置
    "requirements.txt"
)

# 创建远程目录结构
Write-Host "创建远程目录..." -ForegroundColor Yellow
$Dirs = @(
    "models", "training", "training/semantic_distill", "training/text_condition",
    "training/fivek_8param", "scripts", "scripts/legacy_training", "tools", "tools/eval", "tools/data", "tools/train"
)
$DirCmd = ($Dirs | ForEach-Object { "mkdir -p $RemoteDir/$_" }) -join "; "
ssh -p $Port $RemoteHost "$DirCmd"

# 逐个上传
Write-Host "上传训练文件..." -ForegroundColor Yellow
$uploaded = 0
foreach ($pattern in $TrainFiles) {
    $localFiles = Get-ChildItem -Path "$ProjectRoot\$($pattern -replace '/', '\')" -ErrorAction SilentlyContinue
    foreach ($f in $localFiles) {
        $relativePath = $f.FullName.Substring($ProjectRoot.Length + 1) -replace '\\', '/'
        $remoteFile = "$RemoteDir/$relativePath"
        scp -P $Port $f.FullName "${RemoteHost}:${remoteFile}" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ $relativePath" -ForegroundColor DarkGray
            $uploaded++
        } else {
            Write-Host "  ✗ $relativePath" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "=== 同步完成: $uploaded 个文件 ===" -ForegroundColor Green
Write-Host ""
Write-Host "在 AutoDL 上运行:" -ForegroundColor Cyan
Write-Host "  cd $RemoteDir"
Write-Host "  python scripts/legacy_training/train_v6_stage_a.py"
Write-Host "  python train_v12_refine_hd.py"
