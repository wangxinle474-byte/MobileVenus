#!/bin/bash
# \u5728 AutoDL \u5b89\u88c5 JarvisEvo \u7684 LLaMA-Factory fork
set -e
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com

REPO_DIR=/root/autodl-tmp/JarvisEvo_repo

# Step 1: clone the JarvisEvo repo (走 ghproxy 镜像, AutoDL 直连 GitHub 经常超时)
GH_PROXY="https://ghfast.top/https://github.com"
GH_DIRECT="https://github.com"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "=== Cloning JarvisEvo repo (via ghfast.top mirror) ==="
  for url in "$GH_PROXY/LYL1015/JarvisEvo.git" "$GH_DIRECT/LYL1015/JarvisEvo.git"; do
    echo "[TRY] $url"
    if timeout 180 git clone --depth 1 "$url" "$REPO_DIR"; then
      echo "[OK] cloned via $url"
      break
    else
      echo "[FAIL] $url"
      rm -rf "$REPO_DIR"
    fi
  done
  [[ -d "$REPO_DIR" ]] || { echo "[ERR] all clone attempts failed"; exit 1; }
else
  echo "[INFO] $REPO_DIR already exists, skipping clone"
fi

# Step 2: install LLaMA-Factory (JarvisEvo's fork in src/sft_rft)
cd "$REPO_DIR/src/sft_rft"
echo ""
echo "=== Install LLaMA-Factory (JarvisEvo fork) ==="
pip install -q -e ".[torch,metrics]" --no-build-isolation 2>&1 | tail -5

# \u6279\u91cf\u4f9d\u8d56 (\u964d\u4e00\u4e9b\u4e0d\u5fc5\u8981\u7684 train\u673a\u80fd, \u4f18\u5148\u6eda\u8fc7)
echo ""
echo "=== Verify llamafactory-cli ==="
which llamafactory-cli
llamafactory-cli version 2>&1 | head -3

# Step 3: make sure peft, bitsandbytes are installed (LoRA)
pip install -q peft==0.17.1 bitsandbytes 2>&1 | tail -3

echo ""
echo "=== All set. LLaMA-Factory CLI at: $(which llamafactory-cli) ==="
