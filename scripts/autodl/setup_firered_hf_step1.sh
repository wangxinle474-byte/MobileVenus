#!/bin/bash
# Step 1: 删 Venus-Q-Stage1 副本 + 探测 Python + 检查依赖
set +e

echo "=== 1.1 删 Venus-Q-Stage1 副本 ==="
ls -ld /root/autodl-tmp/models/Venus-Q-Stage1/ 2>/dev/null
echo "[deleting /root/autodl-tmp/models/Venus-Q-Stage1/ ...]"
rm -rf /root/autodl-tmp/models/Venus-Q-Stage1/
if [ -d /root/autodl-tmp/models/Venus-Q-Stage1/ ]; then
    echo "[ERROR] still exists"
else
    echo "[OK] removed (root copy at /root/autodl-tmp/Venus-Q-Stage1/ still here)"
fi
ls -ld /root/autodl-tmp/Venus-Q-Stage1/ 2>/dev/null | head -1
echo

echo "=== 1.2 磁盘状态 (after delete) ==="
df -h /root/autodl-tmp | head -3
echo

echo "=== 1.3 寻找 Python ==="
echo "PATH: $PATH"
echo "which python: $(which python 2>/dev/null || echo NA)"
echo "which python3: $(which python3 2>/dev/null || echo NA)"
echo "which conda: $(which conda 2>/dev/null || echo NA)"
echo "ls /root/miniconda3/bin/python*:"
ls -la /root/miniconda3/bin/python* 2>/dev/null | head -5
echo "ls /root/miniconda3/envs/:"
ls -la /root/miniconda3/envs/ 2>/dev/null | head -10
echo

echo "=== 1.4 conda envs ==="
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda env list 2>&1
else
    echo "(no conda.sh)"
fi
echo

echo "=== 1.5 Python 库版本 ==="
PY="/root/miniconda3/bin/python"
if [ ! -x "$PY" ]; then
    PY=$(which python 2>/dev/null || which python3 || echo "")
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "[ERROR] no python found"
    exit 1
fi
echo "Using PY=$PY"
$PY --version
$PY << 'PYEOF'
import sys
print('python:', sys.version.split()[0])
print('sys.executable:', sys.executable)
print()
mods = ['torch', 'diffusers', 'transformers', 'accelerate', 'safetensors', 'huggingface_hub', 'PIL', 'gradio_client', 'sentencepiece', 'bitsandbytes', 'optimum']
for m in mods:
    try:
        mod = __import__(m)
        v = getattr(mod, '__version__', '?')
        print(f'  {m:<22s} {v}')
    except ImportError as e:
        print(f'  {m:<22s} (NOT INSTALLED)')

print()
try:
    import torch
    print('cuda:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('device:', torch.cuda.get_device_name(0))
        print('vram total:', round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2), 'GB')
        print('vram free :', round(torch.cuda.mem_get_info(0)[0]/1024**3, 2), 'GB')
        cc = torch.cuda.get_device_capability(0)
        print('compute capability:', cc, '(sm_%d%d)' % cc)
except Exception as e:
    print('[ERROR] torch:', e)
PYEOF

echo
echo "=== 1.6 diffusers 是否含 QwenImageEditPlusPipeline ==="
$PY -c "
try:
    from diffusers import QwenImageEditPlusPipeline
    print('[OK] QwenImageEditPlusPipeline import OK')
except ImportError as e:
    print('[MISS]', e)
" 2>&1

echo
echo "=== DONE step 1 ==="
