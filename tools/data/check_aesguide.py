import json

d = json.load(open(r'E:/Data/dataset/Venus_data/Benchmark_AesGuide/json/Benchmark_AesGuide.json', 'r', encoding='utf-8'))
keys = list(d.keys())
print(f'Total: {len(keys)} images')

# Length distribution
lens = [len(v) for v in d.values()]
print(f'Length: avg={sum(lens)/len(lens):.0f}  min={min(lens)}  max={max(lens)}')
print(f'  <300: {sum(1 for l in lens if l<300)}')
print(f'  >=400: {sum(1 for l in lens if l>=400)}')
print(f'  >=500: {sum(1 for l in lens if l>=500)}')

# Check pattern markers
markers = ['However', 'suggestion', 'could benefit', 'consider', 'enhance', 'improve', 'recommend', 'may benefit', 'might be enhanced']
for marker in markers:
    count = sum(1 for v in d.values() if marker.lower() in v.lower())
    print(f'  "{marker}": {count}/{len(d)} ({count/len(d)*100:.1f}%)')

print()
print('=== 5 Sample texts (full) ===')
for i, k in enumerate(keys[:5]):
    v = d[k]
    print(f'\n[{i}] {k} ({len(v)} chars)')
    print(f'  {v}')
