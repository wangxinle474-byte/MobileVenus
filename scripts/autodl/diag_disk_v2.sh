#!/bin/bash
# 深度磁盘诊断：找出 cache/ 和 models/ 内可释放空间
set +e

echo "=== A. /root/autodl-tmp/cache/ 二级目录 ==="
du -sh /root/autodl-tmp/cache/*/ 2>/dev/null | sort -h | tail -20
echo "(total: $(du -sh /root/autodl-tmp/cache 2>/dev/null | cut -f1))"
echo

echo "=== B. /root/autodl-tmp/models/ 二级目录 ==="
du -sh /root/autodl-tmp/models/*/ 2>/dev/null | sort -h | tail -20
echo "(total: $(du -sh /root/autodl-tmp/models 2>/dev/null | cut -f1))"
echo

echo "=== C. /root/autodl-tmp/checkpoints/ 二级目录 ==="
du -sh /root/autodl-tmp/checkpoints/*/ 2>/dev/null | sort -h | tail -20
echo "(total: $(du -sh /root/autodl-tmp/checkpoints 2>/dev/null | cut -f1))"
echo

echo "=== D. /root/autodl-tmp/Venus-Q-Stage1/ 内容 ==="
ls -lh /root/autodl-tmp/Venus-Q-Stage1/ 2>/dev/null | head -20
echo

echo "=== E. /root/autodl-tmp/datasets/ 二级 ==="
du -sh /root/autodl-tmp/datasets/*/ 2>/dev/null | sort -h | tail -20
echo

echo "=== F. /root/autodl-tmp/outputs/ 二级 ==="
du -sh /root/autodl-tmp/outputs/*/ 2>/dev/null | sort -h | tail -20
echo

echo "=== G. 顶级 > 100MB 文件 ==="
find /root/autodl-tmp -maxdepth 3 -type f -size +100M 2>/dev/null | head -30 | xargs -I{} du -h {} 2>/dev/null | sort -h | tail -25
echo

echo "=== H. /root/.cache/ 子目录 (系统盘缓存) ==="
du -sh /root/.cache/*/ 2>/dev/null | sort -h | tail -10
echo "(/root/.cache total: $(du -sh /root/.cache 2>/dev/null | cut -f1))"
echo

echo "=== I. 系统盘 / 状态 ==="
df -h / 2>&1 | head -3
echo

echo "=== J. 其他可写盘 ==="
mount | grep -v 'tmpfs\|proc\|sys\|cgroup\|fuse' | head -20
echo

echo "=== K. autodl-fs (公共文件存储, 不计费) ==="
ls -la /root/autodl-fs/ 2>/dev/null | head -5 || echo "(no autodl-fs)"
df -h /root/autodl-fs 2>/dev/null | head -3
echo

echo "=== L. pip cache + apt cache ==="
du -sh /root/.cache/pip 2>/dev/null
du -sh /var/cache/apt 2>/dev/null
echo

echo "=== DONE ==="
