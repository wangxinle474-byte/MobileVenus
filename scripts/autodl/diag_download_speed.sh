#!/bin/bash
# 诊断 HF/ModelScope 下载速度 + hf_transfer 是否装
set +e

echo "=== hf_transfer 是否装 ==="
/root/miniconda3/bin/python -c "import hf_transfer; print('OK', hf_transfer.__version__)" 2>&1 | head -3

echo
echo "=== modelscope 是否装 ==="
/root/miniconda3/bin/python -c "import modelscope; print('OK', modelscope.__version__)" 2>&1 | head -3

echo
echo "=== 镜像速度测试 (5s timeout each, 测 config.json 这种小文件 + safetensors 前 5MB) ==="
echo "[hf-mirror.com] config.json:"
curl -o /dev/null -s -w "  code=%{http_code} speed=%{speed_download}B/s size=%{size_download}B\n" \
    "https://hf-mirror.com/FireRedTeam/FireRed-Image-Edit-1.0/resolve/main/config.json" --max-time 5

echo "[huggingface.co] config.json:"
curl -o /dev/null -s -w "  code=%{http_code} speed=%{speed_download}B/s size=%{size_download}B\n" \
    "https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.0/resolve/main/config.json" --max-time 5

echo "[modelscope.cn] config.json:"
curl -o /dev/null -s -w "  code=%{http_code} speed=%{speed_download}B/s size=%{size_download}B\n" \
    "https://modelscope.cn/api/v1/models/FireRedTeam/FireRed-Image-Edit-1.0/repo?Revision=master&FilePath=config.json" --max-time 5

echo
echo "[hf-mirror.com] 大文件前 5MB 速度测试 (10s):"
curl -o /tmp/spdtest_hfmirror.bin -s -w "  code=%{http_code} speed=%{speed_download}B/s downloaded=%{size_download}B\n" \
    -r 0-5242880 \
    "https://hf-mirror.com/FireRedTeam/FireRed-Image-Edit-1.0/resolve/main/model.safetensors.index.json" --max-time 10
rm -f /tmp/spdtest_hfmirror.bin

echo
echo "=== 当前下载进度 ==="
du -sh /root/autodl-tmp/hf_cache 2>/dev/null
echo
ls -la /root/autodl-tmp/hf_cache/hub/ 2>/dev/null
echo
echo "[hf_cache 内大文件 > 10MB]"
find /root/autodl-tmp/hf_cache -type f -size +10M 2>/dev/null | head -20 | xargs -I{} du -h {} 2>/dev/null

echo
echo "=== PID 2995 (HF pilot 进程) 状态 ==="
ps -p 2995 -o pid,etime,pcpu,pmem,rss 2>/dev/null || echo "[DEAD]"
echo
echo "[DONE]"
