"""验证 lrcat join 链: collection 'C' -> CollectionImage -> Adobe_images -> masterImage -> rootFile -> AgLibraryFile."""
import sqlite3
from pathlib import Path

LRCAT = Path(r'E:\Data\dataset\fivek_dataset\raw_photos\fivek.lrcat')
con = sqlite3.connect(f'file:{LRCAT}?mode=ro', uri=True)
cur = con.cursor()

# 1) collection C 的 5 张 sample images: 链路完整 join
print('='*80, '\n[1] collection C 中 5 个 sample 的完整 join 链\n', '='*80, sep='')
cur.execute('''
  SELECT ci.image AS vcopy_id,
         i.copyName,
         i.masterImage AS master_id,
         m.rootFile AS master_root,
         f.baseName,
         f.extension,
         ds.text
  FROM AgLibraryCollectionImage ci
  JOIN AgLibraryCollection c ON c.id_local = ci.collection
  JOIN Adobe_images i ON i.id_local = ci.image
  JOIN Adobe_images m ON m.id_local = i.masterImage
  JOIN AgLibraryFile f ON f.id_local = m.rootFile
  LEFT JOIN Adobe_imageDevelopSettings ds ON ds.image = ci.image
  WHERE c.name = 'C'
  LIMIT 3
''')
rows = cur.fetchall()
for r in rows:
    vcopy_id, copyName, master_id, master_root, baseName, extension, text = r
    print(f'  vcopy_id={vcopy_id} copy="{copyName}" master={master_id} '
          f'rootFile={master_root}  file={baseName}.{extension}')
    print(f'  develop text (first 800):')
    print((text or '(empty)')[:800])
    print()

# 2) collection C 的总数, 看 join 后丢失多少
print('='*80, '\n[2] collection C 总数 vs 完整 join 数\n', '='*80, sep='')
cur.execute('SELECT COUNT(*) FROM AgLibraryCollectionImage WHERE collection = 930899')
n_col = cur.fetchone()[0]
print(f'  collection C images: {n_col}')

cur.execute('''
  SELECT COUNT(*) FROM AgLibraryCollectionImage ci
  JOIN Adobe_images i ON i.id_local = ci.image
  JOIN Adobe_images m ON m.id_local = i.masterImage
  JOIN AgLibraryFile f ON f.id_local = m.rootFile
  WHERE ci.collection = 930899
''')
n_join = cur.fetchone()[0]
print(f'  join with master + rootFile + file: {n_join}')

cur.execute('''
  SELECT COUNT(*) FROM AgLibraryCollectionImage ci
  JOIN Adobe_images i ON i.id_local = ci.image
  JOIN Adobe_images m ON m.id_local = i.masterImage
  JOIN AgLibraryFile f ON f.id_local = m.rootFile
  JOIN Adobe_imageDevelopSettings ds ON ds.image = ci.image
  WHERE ci.collection = 930899
  AND ds.text IS NOT NULL AND length(ds.text) > 50
''')
n_with_text = cur.fetchone()[0]
print(f'  + with non-empty develop text: {n_with_text}')

# 3) 5 个 expert 各自 collection 的 id_local + count
print('\n' + '='*80, '\n[3] 5 expert collection 的 id_local + 含 text 的 record 数\n', '='*80, sep='')
for ename in 'ABCDE':
    cur.execute("SELECT id_local FROM AgLibraryCollection WHERE name=?", (ename,))
    r = cur.fetchone()
    if not r:
        continue
    cid = r[0]
    cur.execute('''
      SELECT COUNT(*) FROM AgLibraryCollectionImage ci
      JOIN Adobe_images i ON i.id_local = ci.image
      JOIN Adobe_imageDevelopSettings ds ON ds.image = ci.image
      WHERE ci.collection = ?
      AND ds.text IS NOT NULL AND length(ds.text) > 50
    ''', (cid,))
    n_text = cur.fetchone()[0]
    print(f'  Expert {ename}: collection_id={cid}, records_with_text={n_text}')

con.close()
