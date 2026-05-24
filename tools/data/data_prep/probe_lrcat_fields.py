"""列出 lrcat snapshot.text 中出现的所有 develop settings 字段名 + 验证 Import N → expert ABCDE."""
import re
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict

LRCAT = Path(r'E:\Data\dataset\fivek_dataset\raw_photos\fivek.lrcat')
con = sqlite3.connect(f'file:{LRCAT}?mode=ro', uri=True)
cur = con.cursor()

# 1) 找一张同时有 Import / Import 2 / Import 3 的 image, 对比参数
print('='*80, '\n[1] 找一张完整 5 expert snapshot 的 image\n', '='*80, sep='')
cur.execute('''
  SELECT image, COUNT(DISTINCT name) AS nn,
         GROUP_CONCAT(DISTINCT name) AS names
  FROM Adobe_libraryImageDevelopSnapshot
  WHERE name LIKE 'Import%' AND name NOT LIKE '%Zeroed%' AND name NOT LIKE '%/%'
  GROUP BY image
  HAVING nn >= 5
  LIMIT 5
''')
rows = cur.fetchall()
for r in rows:
    print(f'  image={r[0]}  n={r[1]}  names={r[2]}')

if rows:
    img_id = rows[0][0]
    print(f'\n[1b] 选 image_id={img_id} 看 5 个 snapshot 各自的关键参数:')
    cur.execute('''
      SELECT name, text FROM Adobe_libraryImageDevelopSnapshot
      WHERE image=? AND name LIKE 'Import%' AND name NOT LIKE '%Zeroed%' AND name NOT LIKE '%/%'
      ORDER BY name
    ''', (img_id,))
    for name, text in cur.fetchall():
        # 解析关键 key = val
        if not text:
            continue
        ks = ['Brightness', 'Contrast', 'Exposure', 'Shadows', 'Highlights',
              'Recovery', 'FillLight', 'Blacks', 'Saturation', 'Vibrance',
              'Clarity', 'Temperature', 'Tint', 'WhiteBalance', 'ToneCurveName']
        d = {}
        for k in ks:
            m = re.search(rf'\b{k}\s*=\s*([^,\n]+?)(?:,|\s*$|\n)', text)
            if m:
                d[k] = m.group(1).strip()
        print(f'  [{name}]')
        for k, v in d.items():
            print(f'    {k} = {v}')
        print()

# 2) 列出 develop settings text 中所有 key
print('='*80, '\n[2] develop settings text 全部出现的 key (前 200 random snapshot)\n', '='*80, sep='')
cur.execute('''SELECT text FROM Adobe_libraryImageDevelopSnapshot
               WHERE text IS NOT NULL AND name LIKE 'Import%'
               LIMIT 200''')
all_keys = Counter()
for (text,) in cur.fetchall():
    if not text:
        continue
    # 简单 regex 抓 "KeyName = value"
    keys = re.findall(r'^\s*(\w+)\s*=\s*', text, flags=re.MULTILINE)
    for k in keys:
        all_keys[k] += 1

print(f'共 {len(all_keys)} 个 distinct keys (出现频次, 前 60):')
for k, v in all_keys.most_common(60):
    print(f'  {v:4d}  {k}')

# 3) Import N 的 count(unique image)
print('\n' + '='*80, '\n[3] Import N 各自覆盖 image 数 (验证 ABCDE 映射)\n', '='*80, sep='')
cur.execute('''SELECT name, COUNT(DISTINCT image) FROM Adobe_libraryImageDevelopSnapshot
               WHERE name LIKE 'Import%' AND name NOT LIKE '%/%' AND name NOT LIKE '%Zeroed%'
               GROUP BY name ORDER BY name''')
for r in cur.fetchall():
    print(f'  {r[0]:<30s}  unique_images={r[1]}')

con.close()
