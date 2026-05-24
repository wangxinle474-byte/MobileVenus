#!/bin/bash
# Kill HF pilot 进程 + 清 hf_cache 下 FireRed 残留 (.incomplete 文件)
set +e

echo "=== Before kill ==="
ps -ef | grep -E 'run_firered_hf_local|2995' | grep -v grep

echo
echo "=== Kill PID 2995 ==="
kill -9 2995 2>&1
sleep 2
ps -p 2995 -o pid,cmd 2>/dev/null && echo "[STILL ALIVE]" || echo "[DEAD - good]"

# Also kill any other instance just in case
pkill -9 -f run_firered_hf_local.py 2>&1

echo
echo "=== Clean hf_cache FireRed residue ==="
SIZE_BEFORE=$(du -sh /root/autodl-tmp/hf_cache 2>/dev/null | cut -f1)
echo "[BEFORE] hf_cache = $SIZE_BEFORE"

# 删除 .incomplete + 整个 FireRed cache (节省 ~870MB)
rm -rf /root/autodl-tmp/hf_cache/hub/models--FireRedTeam--FireRed-Image-Edit-1.0
rm -rf /root/autodl-tmp/hf_cache/hub/.locks

SIZE_AFTER=$(du -sh /root/autodl-tmp/hf_cache 2>/dev/null | cut -f1)
echo "[AFTER]  hf_cache = $SIZE_AFTER"

echo
echo "=== Disk avail ==="
df -h /root/autodl-tmp | head -3

echo
echo "=== Done ==="
