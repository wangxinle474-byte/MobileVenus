"""深入探查 lrcat: copyName / history step name / develop settings 真实 GT 位置."""
import re
import sqlite3
from pathlib import Path
from collections import Counter

LRCAT = Path(r'E:\Data\dataset\fivek_dataset\raw_photos\fivek.lrcat')
con = sqlite3.connect(f'file:{LRCAT}?mode=ro', uri=True)
cur = con.cursor()

# 1) Adobe_images.copyName 分布 (virtual copy 区分 expert?)
print('='*80, '\n[1] Adobe_images.copyName 分布 (是否区分 ABCDE?)\n', '='*80, sep='')
cur.execute('''SELECT copyName, COUNT(*) FROM Adobe_images
               GROUP BY copyName ORDER BY 2 DESC LIMIT 20''')
for r in cur.fetchall():
    print(f'  copyName="{r[0]}"   count={r[1]}')

# 2) Adobe_images 总 count + masterImage 分布 (有 masterImage 的就是 virtual copy)
print('\n' + '='*80, '\n[2] Adobe_images masterImage 是否非空 (virtual copy 标记)\n', '='*80, sep='')
cur.execute('SELECT COUNT(*) FROM Adobe_images WHERE masterImage IS NOT NULL')
n_virtual = cur.fetchone()[0]
cur.execute('SELECT COUNT(*) FROM Adobe_images WHERE masterImage IS NULL')
n_master = cur.fetchone()[0]
print(f'  master (masterImage IS NULL): {n_master}')
print(f'  virtual copy (masterImage NOT NULL): {n_virtual}')

# 3) Adobe_libraryImageDevelopHistoryStep.name 分布
print('\n' + '='*80, '\n[3] history step.name 分布 (是否含 "expert" / "retouch")\n', '='*80, sep='')
cur.execute('''SELECT name, COUNT(*) FROM Adobe_libraryImageDevelopHistoryStep
               WHERE name IS NOT NULL
               GROUP BY name ORDER BY 2 DESC LIMIT 40''')
for r in cur.fetchall():
    print(f'  n={r[1]:6}  name="{r[0]}"')

# 4) Adobe_imageDevelopSettings: 有多少 text 是非空?
print('\n' + '='*80, '\n[4] Adobe_imageDevelopSettings text 非空 ratio\n', '='*80, sep='')
cur.execute('SELECT COUNT(*) FROM Adobe_imageDevelopSettings WHERE text IS NOT NULL AND length(text) > 50')
n_with_text = cur.fetchone()[0]
print(f'  text 非空 (len>50): {n_with_text}/96458')

# 看一个非空的 text 样本
cur.execute('''SELECT image, text FROM Adobe_imageDevelopSettings
               WHERE text IS NOT NULL AND length(text) > 200 LIMIT 1''')
r = cur.fetchone()
if r:
    print(f'  sample image_id={r[0]}, text (first 2000 chars):')
    print(r[1][:2000])

# 5) 也许 expert 信息在 Adobe_imageProperties (703 行)?
print('\n' + '='*80, '\n[5] Adobe_imageProperties 看一眼\n', '='*80, sep='')
cur.execute('PRAGMA table_info(Adobe_imageProperties)')
cols = [c[1] for c in cur.fetchall()]
print(f'  cols: {cols}')
cur.execute(f'SELECT * FROM Adobe_imageProperties LIMIT 3')
for r in cur.fetchall():
    print(' ', r)

# 6) AgLibraryCollection (23 行) — 23 个 collection, 可能是 5 expert + 别的分类
print('\n' + '='*80, '\n[6] AgLibraryCollection 名字 (23 行)\n', '='*80, sep='')
cur.execute('SELECT id_local, name, imageCount FROM AgLibraryCollection')
for r in cur.fetchall():
    print(f'  id={r[0]:5}  name="{r[1]}"  imageCount={r[2]}')

# 7) virtualCopies image 的 imageDevelopSettings 看其 text
print('\n' + '='*80, '\n[7] virtual copy 的 develop settings 看其 text (取 image_id 最大的几个)\n', '='*80, sep='')
cur.execute('''SELECT i.id_local, i.copyName, i.masterImage, ds.text
               FROM Adobe_images i
               LEFT JOIN Adobe_imageDevelopSettings ds ON ds.image = i.id_local
               WHERE i.masterImage IS NOT NULL AND ds.text IS NOT NULL
               LIMIT 3''')
for r in cur.fetchall():
    print(f'  img={r[0]} copy={r[1]} master={r[2]}')
    print(f'  text (first 1500):')
    print((r[3] or '')[:1500])
    print()

con.close()
