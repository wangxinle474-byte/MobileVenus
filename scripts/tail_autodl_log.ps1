# 实时查看 AutoDL 上的远程日志文件
# 用法:
#   .\scripts\tail_autodl_log.ps1                            # 默认看 install.log
#   .\scripts\tail_autodl_log.ps1 -Log lora_sft_train.log    # 看训练日志
#   .\scripts\tail_autodl_log.ps1 -Log install.log -Lines 50 # 看最后 50 行
#
# Ctrl+C 退出
param(
    [string]$Log = 'install.log',
    [int]$Lines = 100,
    [string]$Host = 'connect.bjb2.seetacloud.com',
    [int]$Port = 35345,
    [string]$User = 'root'
)

$RemoteLogDir = '/root/autodl-tmp/IntelligenceCamera/logs'
$RemotePath = "$RemoteLogDir/$Log"

Write-Host "=== Tail AutoDL log: $RemotePath ===" -ForegroundColor Cyan
Write-Host "    Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

# 用 ssh + tail -F 实时追踪 (-F 文件可以暂时不存在)
ssh -p $Port "$User@$Host" "tail -n $Lines -F $RemotePath"
