#!/bin/bash
# 后台启动 LongCat compare 全流程, PID 写入 logs/compare.pid.
# 主日志 logs/compare_full.log (用 tail -f 实时看).
set -e
cd /root/autodl-tmp/IntelligenceCamera
mkdir -p logs

# disown + setsid 双保险, SSH 断开也不影响
setsid nohup bash scripts/autodl_run_longcat_compare.sh \
    > logs/compare_full.log 2>&1 < /dev/null &
PID=$!
echo $PID > logs/compare.pid
disown $PID 2>/dev/null || true

sleep 1
echo "PID=$PID  (saved to logs/compare.pid)"
echo "---"
echo "实时监控 (在 AutoDL 任意 ssh / JupyterLab 终端执行):"
echo "  tail -f /root/autodl-tmp/IntelligenceCamera/logs/compare_full.log"
echo ""
echo "查进程状态:"
echo "  ps -p \$(cat /root/autodl-tmp/IntelligenceCamera/logs/compare.pid)"
echo ""
echo "查 GPU:"
echo "  watch -n 1 nvidia-smi"
