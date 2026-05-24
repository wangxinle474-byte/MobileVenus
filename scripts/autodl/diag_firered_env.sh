#!/bin/bash
# AutoDL FireRed HF pilot 环境诊断脚本
# 检查 GPU 型号 / 磁盘空间 / 已缓存模型 / Python+torch+diffusers 版本
set +e   # 不让单条失败中断整个脚本

echo "=== 1. nvidia-smi ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader 2>&1
echo

echo "=== 2. CUDA toolkit (nvcc) ==="
nvcc --version 2>&1 | tail -2 || echo "(nvcc not installed, ok if torch self-contained)"
echo

echo "=== 3. disk /root/autodl-tmp ==="
df -h /root/autodl-tmp 2>&1 | head -3
echo

echo "=== 4. top dirs in /root/autodl-tmp ==="
du -sh /root/autodl-tmp/*/ 2>/dev/null | sort -h | tail -15
echo

echo "=== 5. HF cache hub entries ==="
ls -1 /root/autodl-tmp/hf_cache/hub/ 2>/dev/null | head -30
echo "(... total $(ls -1 /root/autodl-tmp/hf_cache/hub/ 2>/dev/null | wc -l) entries)"
echo

echo "=== 6. Search HF cache for FireRed / Qwen-Image-Edit ==="
ls -1 /root/autodl-tmp/hf_cache/hub/ 2>/dev/null | grep -iE 'firered|qwen.*image.*edit' || echo "(no FireRed/QwenImageEdit cached)"
echo

echo "=== 7. Search ModelScope cache too ==="
ls -1 ~/.cache/modelscope/hub/ 2>/dev/null | head -10 || echo "(no modelscope cache)"
echo

echo "=== 8. python+torch+cuda ==="
python -c "
import sys, torch
print('python:', sys.version.split()[0])
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
    print('vram total:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
" 2>&1
echo

echo "=== 9. diffusers/transformers/accelerate ==="
python -c "
mods = ['diffusers', 'transformers', 'accelerate', 'safetensors', 'huggingface_hub', 'PIL']
for m in mods:
    try:
        mod = __import__(m)
        v = getattr(mod, '__version__', '?')
        print(f'  {m:<20s} {v}')
    except ImportError:
        print(f'  {m:<20s} (NOT INSTALLED)')
" 2>&1
echo

echo "=== 10. free memory ==="
free -h | head -3
echo

echo "=== 11. env vars ==="
env | grep -iE 'hf_|huggingface|modelscope|cuda_visible' | head -10
echo

echo "=== DONE ==="
