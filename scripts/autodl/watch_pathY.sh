#!/bin/bash
# Path Y 进度监控（在 AutoDL 上跑）
#
# 用法 1: 单次状态
#   bash /root/autodl-tmp/IntelligenceCamera/scripts/autodl/watch_pathY.sh
#
# 用法 2: 每 30s 自动刷新（推荐）
#   watch -n 30 -c bash /root/autodl-tmp/IntelligenceCamera/scripts/autodl/watch_pathY.sh
#
# 用法 3: 实时 tail 主日志（最详细，看每步进度）
#   tail -f /root/autodl-tmp/pathY_logs/_master.log

OUT_BASE=/root/autodl-tmp/pathY_outputs
LOG_DIR=/root/autodl-tmp/pathY_logs
MASTER_LOG=$LOG_DIR/_master.log
ACTIONS="wb highlights saturation shadows contrast"
ACTION_N=300
TOTAL=$((ACTION_N * 5))

# 颜色（terminal 支持时）
if [ -t 1 ]; then
    BOLD=$(tput bold 2>/dev/null || echo "")
    RST=$(tput sgr0 2>/dev/null || echo "")
    GRN=$(tput setaf 2 2>/dev/null || echo "")
    YLW=$(tput setaf 3 2>/dev/null || echo "")
    CYN=$(tput setaf 6 2>/dev/null || echo "")
    RED=$(tput setaf 1 2>/dev/null || echo "")
else
    BOLD=""; RST=""; GRN=""; YLW=""; CYN=""; RED=""
fi

echo "${BOLD}=== Path Y 进度 @ $(date '+%H:%M:%S') ===${RST}"

# 1. 进程状态
PIDS=$(pgrep -f run_firered_hf_local 2>/dev/null)
if [ -n "$PIDS" ]; then
    ETIME=$(ps -p $(echo $PIDS | awk '{print $1}') -o etime= 2>/dev/null | xargs)
    echo "${GRN}● ALIVE${RST}   pid=$PIDS  elapsed=$ETIME"
else
    echo "${RED}○ STOPPED${RST}  (no run_firered_hf_local process)"
fi

# 2. GPU
GPU_INFO=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
echo "  GPU: $GPU_INFO  (used MiB, free MiB, util %)"

# 3. 磁盘
DISK=$(df -h /root/autodl-tmp 2>/dev/null | tail -1 | awk '{printf "%s used / %s total (%s)", $3, $2, $5}')
echo "  Disk: $DISK"
echo

# 4. 各 action 完成数
echo "${BOLD}Per-action:${RST}"
TOTAL_DONE=0
CURRENT_ACTION=""
# 找出当前正在跑的 action（最后一个 [START] 行）
if [ -f "$MASTER_LOG" ]; then
    CURRENT_ACTION=$(grep -E '^\[START\]' "$MASTER_LOG" 2>/dev/null | tail -1 | sed -E 's/^\[START\] (\w+):.*/\1/')
fi

for ACTION in $ACTIONS; do
    N=$(find "$OUT_BASE/$ACTION" -maxdepth 1 -name '*.png' -type f 2>/dev/null | wc -l)
    TOTAL_DONE=$((TOTAL_DONE + N))
    PCT=$((N * 100 / ACTION_N))
    BAR_LEN=20
    FILLED=$((PCT * BAR_LEN / 100))
    EMPTY=$((BAR_LEN - FILLED))
    BAR=$(printf '%*s' "$FILLED" '' | tr ' ' '#')$(printf '%*s' "$EMPTY" '' | tr ' ' '.')

    if [ "$ACTION" = "$CURRENT_ACTION" ]; then
        MARK="${GRN}*${RST}"
        COLOR="$GRN"
    elif [ "$N" -ge "$ACTION_N" ]; then
        MARK="${CYN}✓${RST}"
        COLOR="$CYN"
    else
        MARK=" "
        COLOR=""
    fi
    printf "  %s %-11s [%s] %3d/%d  (%3d%%)\n" "$MARK" "$COLOR$ACTION$RST" "$BAR" "$N" "$ACTION_N" "$PCT"
done

# 5. 总进度
TPCT=$((TOTAL_DONE * 100 / TOTAL))
TBAR_LEN=40
TFILLED=$((TPCT * TBAR_LEN / 100))
TEMPTY=$((TBAR_LEN - TFILLED))
TBAR=$(printf '%*s' "$TFILLED" '' | tr ' ' '#')$(printf '%*s' "$TEMPTY" '' | tr ' ' '.')
echo
echo "${BOLD}Total:${RST}      [$TBAR] $TOTAL_DONE/$TOTAL  (${TPCT}%)"

# 6. 当前 sample / step (从主日志末尾解析)
if [ -f "$MASTER_LOG" ]; then
    echo
    echo "${BOLD}Current:${RST}"
    LAST_SAMPLE=$(grep -E '^\[[0-9]+/[0-9]+\] idx=' "$MASTER_LOG" 2>/dev/null | tail -1)
    LAST_STEP=$(tail -3 "$MASTER_LOG" 2>/dev/null | grep -oE '[0-9]+%\|[^|]+\| [0-9]+/[0-9]+ \[[^]]+\]' | tail -1)
    LAST_SAVED=$(grep -E '\-> saved' "$MASTER_LOG" 2>/dev/null | tail -1)

    [ -n "$LAST_SAMPLE" ] && echo "  ${YLW}$(echo "$LAST_SAMPLE" | head -c 110)${RST}"
    [ -n "$LAST_STEP" ]   && echo "  step: $LAST_STEP"
    [ -n "$LAST_SAVED" ]  && echo "  last: $(echo "$LAST_SAVED" | head -c 90)"
fi

# 7. 错误统计
if [ -f "$MASTER_LOG" ]; then
    N_FAIL=$(grep -c 'FAIL (' "$MASTER_LOG" 2>/dev/null)
    N_OK=$(grep -c '\-> saved' "$MASTER_LOG" 2>/dev/null)
    if [ "$N_FAIL" -gt 0 ]; then
        echo
        echo "${RED}⚠ FAILED: $N_FAIL${RST} | OK: $N_OK"
    fi
fi
