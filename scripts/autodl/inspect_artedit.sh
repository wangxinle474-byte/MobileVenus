#!/bin/bash
# \u68c0\u67e5 ArtEdit-Bench \u6570\u636e\u7ed3\u6784
cd /root/autodl-tmp/datasets/ArtEdit-Bench

echo "=== Counts ==="
echo "CN samples: $(ls ArtEdit-Bench-Lr/CN | wc -l)"
echo "EN samples: $(ls ArtEdit-Bench-Lr/EN | wc -l)"
echo "Eval samples: $(ls ArtEdit-Bench-Eval | wc -l)"

echo ""
echo "=== \u5e26 <box> \u7684 CN \u6837\u672c (\u5c40\u90e8\u8c03\u6574) ==="
grep -l "<box>" ArtEdit-Bench-Lr/CN/*/user_want.txt | head -3 | while read f; do
  id=$(basename $(dirname $f))
  echo "[$id] $(cat $f)"
done

echo ""
echo "=== \u4e0d\u5e26 <box> \u7684 CN \u6837\u672c (\u5168\u5c40\u8c03\u6574) ==="
for f in ArtEdit-Bench-Lr/CN/*/user_want.txt; do
  if ! grep -q "<box>" "$f"; then
    id=$(basename $(dirname $f))
    echo "[$id] $(cat $f)"
    count=$((${count:-0}+1))
    [[ $count -ge 3 ]] && break
  fi
done

echo ""
echo "=== \u524d 3 \u4e2a EN \u6837\u672c ==="
for id in $(ls ArtEdit-Bench-Lr/EN | head -3); do
  echo "[$id] $(cat ArtEdit-Bench-Lr/EN/$id/user_want.txt | head -c 200)"
done

echo ""
echo "=== \u6587\u4ef6\u5927\u5c0f (1 \u4e2a\u6837\u672c) ==="
ls -lh ArtEdit-Bench-Lr/CN/1/

echo ""
echo "=== \u603b\u76d8\u5927\u5c0f ==="
du -sh ArtEdit-Bench-Lr ArtEdit-Bench-Eval
