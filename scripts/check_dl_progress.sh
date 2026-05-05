#!/bin/bash
# \u5feb\u901f\u68c0\u67e5\u4e24\u4e2a\u4e0b\u8f7d\u7684\u5b9e\u65f6\u8fdb\u5ea6
JE_CACHE=/root/autodl-tmp/checkpoints/pretrained/JarvisEvo/.cache/huggingface/download
QW_CACHE=/root/autodl-tmp/models/Qwen3-VL-4B-Instruct/.cache/huggingface/download

T0=$(du -sm "$JE_CACHE" 2>/dev/null | cut -f1)
Q0=$(du -sm "$QW_CACHE" 2>/dev/null | cut -f1)
sleep 20
T1=$(du -sm "$JE_CACHE" 2>/dev/null | cut -f1)
Q1=$(du -sm "$QW_CACHE" 2>/dev/null | cut -f1)

echo "JarvisEvo cache: ${T0} MB -> ${T1} MB  (delta: $((T1-T0)) MB in 20s)"
echo "Qwen3VL cache:   ${Q0} MB -> ${Q1} MB  (delta: $((Q1-Q0)) MB in 20s)"
echo ""
echo "JarvisEvo shards: $(ls /root/autodl-tmp/checkpoints/pretrained/JarvisEvo/*.safetensors 2>/dev/null | wc -l)/4"
echo "Qwen3VL shards:   $(ls /root/autodl-tmp/models/Qwen3-VL-4B-Instruct/*.safetensors 2>/dev/null | wc -l)/2"
echo ""
echo "DL processes still running: $(ps aux | grep huggingface-cli | grep -v grep | wc -l)"
echo ""
df -h /root/autodl-tmp | tail -1
