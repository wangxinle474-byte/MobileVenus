#!/bin/bash
# Step 1.5: 探查 hardlink + PPR10K 冗余 + 可写分区
set +e

echo "=== A. Venus-Q-Stage1 文件 inode 数 (hardlink 数 > 1 说明共享) ==="
ls -li /root/autodl-tmp/Venus-Q-Stage1/*.safetensors 2>/dev/null | head -3
echo "...每个文件第 3 列 (link count) 如果 > 1 就是 hardlink"
echo

echo "=== B. PPR10K 内部 (zip vs 解压目录) ==="
ls -la /root/autodl-tmp/PPR10K/ 2>/dev/null
echo
ls -la /root/autodl-tmp/PPR10K/train_val_images_tif_360p/ 2>/dev/null | head -20
echo
echo "[PPR10K zip + dir size]"
du -sh /root/autodl-tmp/PPR10K/train_val_images_tif_360p/*.zip 2>/dev/null
du -sh /root/autodl-tmp/PPR10K/train_val_images_tif_360p/*/ 2>/dev/null | head -10
echo

echo "=== C. /autodl-pub 是否可写 (公共只读盘) ==="
df -h /autodl-pub 2>/dev/null | head -3
echo "(只读测试)"; touch /autodl-pub/test_write 2>&1 | head -2
echo

echo "=== D. /tmp 状态 ==="
df -h /tmp 2>&1 | head -3
echo

echo "=== E. /root/ 系统盘下其他大目录 ==="
du -sh /root/.cache/* 2>/dev/null | sort -h | tail -10
du -sh /root/* 2>/dev/null | grep -v autodl-tmp | sort -h | tail -10
echo

echo "=== F. 当前 /root/autodl-tmp 真实可用空间 ==="
df -h /root/autodl-tmp 2>&1 | head -3
echo

echo "=== G. AesExpert 子文件 ==="
ls -lh /root/autodl-tmp/models/AesExpert/ 2>/dev/null | head -10
echo

echo "=== H. 可大幅释放空间的候选 ==="
echo "[Top 20 largest files in /root/autodl-tmp]"
find /root/autodl-tmp -maxdepth 4 -type f -size +500M 2>/dev/null | xargs -I{} du -h {} 2>/dev/null | sort -h | tail -20
echo

echo "=== DONE ==="
