#!/bin/bash
# \u5728 AutoDL \u5b89\u88c5 vllm \u7528\u4e8e JarvisEvo-8B \u63a8\u7406
# vllm 0.11.x \u517c\u5bb9 transformers 4.57.x, \u4e0d\u5f71\u54cd LLaMA-Factory
set -e
export PATH=/root/miniconda3/bin:$PATH

echo "=== pre-check ==="
python --version
pip show torch 2>/dev/null | grep -E 'Name|Version' | head -2 || echo 'torch missing'
pip show transformers 2>/dev/null | grep -E 'Name|Version' | head -2

echo ""
echo "=== install vllm 0.11.0 (compatible with transformers 4.57) ==="
pip install -q vllm==0.11.0 2>&1 | tail -10

echo ""
echo "=== verify ==="
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import transformers; print('transformers', transformers.__version__)"
