"""sample lrcat develop settings + snapshot 命名 + image 路径关联."""
import sqlite3
from pathlib import Path

LRCAT = Path(r'E:\Data\dataset\fivek_dataset\raw_photos\fivek.lrcat')
con = sqlite3.connect(f'file:{LRCAT}?mode=ro', uri=True)
cur = con.cursor()

# 1. snapshot 命名 (是否含 expert A/B/C/D/E)
print('='*80, '\nAdobe_libraryImageDevelopSnapshot.name samples (前20条)\n', '='*80, sep='')
cur.execute('SELECT id_local, name, image FROM Adobe_libraryImageDevelopSnapshot LIMIT 20')
for r in cur.fetchall():
    print(f'  id={r[0]:7}  image={r[2]:>8}  name="{r[1]}"')

# 名字 distinct
print('\nname distinct (前30):')
cur.execute('SELECT DISTINCT name, COUNT(*) FROM Adobe_libraryImageDevelopSnapshot '
            'GROUP BY name ORDER BY 2 DESC LIMIT 30')
for r in cur.fetchall():
    print(f'  n={r[1]:6}  name="{r[0]}"')

# 2. Adobe_imageDevelopSettings.text 样本 (text 字段是关键 develop settings)
print('\n' + '='*80, '\nAdobe_imageDevelopSettings 1 sample\n', '='*80, sep='')
cur.execute('SELECT id_local, image, whiteBalance, text FROM Adobe_imageDevelopSettings LIMIT 1')
r = cur.fetchone()
print(f'id_local={r[0]}, image={r[1]}, whiteBalance={r[2]}')
print('text (first 2500 chars):')
print(r[3][:2500] if r[3] else '(empty)')

# 3. Adobe_libraryImageDevelopSnapshot.text 样本 (snapshot 的 develop settings)
print('\n' + '='*80, '\nAdobe_libraryImageDevelopSnapshot 1 sample\n', '='*80, sep='')
cur.execute('SELECT id_local, image, name, text FROM Adobe_libraryImageDevelopSnapshot LIMIT 1')
r = cur.fetchone()
print(f'id_local={r[0]}, image={r[1]}, name="{r[2]}"')
print('text (first 2500 chars):')
print(r[3][:2500] if r[3] else '(empty)')

# 4. Adobe_images 表样本 — 映射 image id 到 file 名
print('\n' + '='*80, '\nAdobe_images sample\n', '='*80, sep='')
cur.execute('PRAGMA table_info(Adobe_images)')
for c in cur.fetchall():
    print(f'  col: {c[1]} ({c[2]})')

# 5. 看 AgLibraryFile 是不是含 dng 文件名
print('\n' + '='*80, '\nAgLibraryFile (5000 rows) sample\n', '='*80, sep='')
cur.execute('PRAGMA table_info(AgLibraryFile)')
cols = [c[1] for c in cur.fetchall()]
print(f'cols: {cols}')
cur.execute(f'SELECT {",".join(cols[:6])} FROM AgLibraryFile LIMIT 3')
for r in cur.fetchall():
    print(' ', r)

con.close()
