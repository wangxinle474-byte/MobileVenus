import json
import re

d = json.load(open(r'data/venus_pseudo_labels.json', 'r', encoding='utf-8'))
labels = d['labels']
print(f'Total: {len(labels)} images')
print()

# Print 5 full samples to understand structure
for i in [0, 100, 500, 1000, 3000]:
    if i < len(labels):
        item = labels[i]
        print(f'[{i}] {item["image"]} (overall={item["overall"]})')
        print(f'  FULL TEXT:')
        print(f'  {item["raw_response"]}')
        print()
        print('-' * 80)

# Analyze patterns
print()
print('=== Pattern Analysis ===')
markers = ['However', 'suggestion', 'could benefit', 'might', 'consider', 'enhance', 'improve', 'recommend']
for marker in markers:
    count = sum(1 for item in labels if marker.lower() in item['raw_response'].lower())
    print(f'  "{marker}": {count}/{len(labels)} ({count/len(labels)*100:.1f}%)')

# Check average length
lens = [len(item['raw_response']) for item in labels]
print(f'\nAvg text length: {sum(lens)/len(lens):.0f} chars  min={min(lens)} max={max(lens)}')
