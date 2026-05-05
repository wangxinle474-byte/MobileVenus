# Resume LongCat-Image-Edit-Turbo download via hf-mirror (more reliable from CN)
# Uses huggingface-cli CLI which auto-resumes from existing .incomplete files

$ErrorActionPreference = 'Stop'

$env:HF_ENDPOINT = 'https://hf-mirror.com'
$env:HF_HOME = 'E:\cache\huggingface'
$env:HUGGINGFACE_HUB_CACHE = 'E:\cache\huggingface\hub'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '60'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
# DO NOT set HF_HUB_ENABLE_HF_TRANSFER (rust downloader sometimes dies silently)
Remove-Item Env:HF_HUB_ENABLE_HF_TRANSFER -ErrorAction SilentlyContinue
# Clear any stale proxy settings
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue

Write-Host '=== LongCat resume via hf-mirror ==='
Write-Host "HF_ENDPOINT = $env:HF_ENDPOINT"
Write-Host "HF_HOME     = $env:HF_HOME"
Write-Host ''

# Note: use hub cache layout (not local_dir) so it continues from existing cache
huggingface-cli download meituan-longcat/LongCat-Image-Edit-Turbo `
    --cache-dir 'E:\cache\huggingface\hub' `
    --max-workers 4
