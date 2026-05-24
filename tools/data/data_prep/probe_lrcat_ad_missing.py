"""\u8c03\u67e5 Expert A=2800 / D=2200 \u4e3a\u4f55\u4e0d\u662f 5000."""
import sqlite3
from pathlib import Path

LRCAT = Path(r'E:\Data\dataset\fivek_dataset\raw_photos\fivek.lrcat')
con = sqlite3.connect(f'file:{LRCAT}?mode=ro', uri=True)
cur = con.cursor()

# Expert A collection id = 918542
for ex, cid in [('A', 918542), ('B', 923976), ('C', 930899),
                ('D', 936482), ('E', 954927)]:
    print(f'\n=== Expert {ex} (collection_id={cid}) ===')
    # \u603b virtual copy \u6570
    cur.execute('SELECT COUNT(*) FROM AgLibraryCollectionImage WHERE collection=?', (cid,))
    n_total = cur.fetchone()[0]
    print(f'  total vcopies in collection: {n_total}')

    # \u6709 develop settings (\u4efb\u610f text \u72b6\u6001)
    cur.execute('''SELECT COUNT(*) FROM AgLibraryCollectionImage ci
                   JOIN Adobe_imageDevelopSettings ds ON ds.image = ci.image
                   WHERE ci.collection=?''', (cid,))
    n_any_text = cur.fetchone()[0]
    print(f'  with any develop settings row: {n_any_text}')

    # text non-empty
    cur.execute('''SELECT COUNT(*) FROM AgLibraryCollectionImage ci
                   JOIN Adobe_imageDevelopSettings ds ON ds.image = ci.image
                   WHERE ci.collection=?
                   AND ds.text IS NOT NULL AND length(ds.text) > 50''', (cid,))
    n_text_50 = cur.fetchone()[0]
    print(f'  + text length > 50: {n_text_50}')

    # text length > 200 (\u542b\u591a\u4e2a key)
    cur.execute('''SELECT COUNT(*) FROM AgLibraryCollectionImage ci
                   JOIN Adobe_imageDevelopSettings ds ON ds.image = ci.image
                   WHERE ci.collection=?
                   AND ds.text IS NOT NULL AND length(ds.text) > 200''', (cid,))
    n_text_200 = cur.fetchone()[0]
    print(f'  + text length > 200: {n_text_200}')

    # \u52a0\u5165 file join (\u770b\u662f\u5426\u6709\u5931\u8d25)
    cur.execute('''
      SELECT COUNT(*) FROM AgLibraryCollectionImage ci
      JOIN Adobe_images i ON i.id_local = ci.image
      JOIN Adobe_images m ON m.id_local = i.masterImage
      JOIN AgLibraryFile f ON f.id_local = m.rootFile
      JOIN Adobe_imageDevelopSettings ds ON ds.image = ci.image
      WHERE ci.collection=?
      AND ds.text IS NOT NULL AND length(ds.text) > 50
    ''', (cid,))
    n_full_join = cur.fetchone()[0]
    print(f'  full join (with file): {n_full_join}')

    # \u770b 5 \u4e2a\u51fa\u4e0d\u5728\u7ed3\u679c\u91cc\u7684 \u793a\u4f8b: text \u4e3a\u7a7a\u6216\u7f3a\u5931
    cur.execute('''
      SELECT ci.image, ds.text IS NULL AS text_null, length(ds.text) AS tlen
      FROM AgLibraryCollectionImage ci
      LEFT JOIN Adobe_imageDevelopSettings ds ON ds.image = ci.image
      WHERE ci.collection=?
      AND (ds.text IS NULL OR length(ds.text) < 50)
      LIMIT 5
    ''', (cid,))
    sample = cur.fetchall()
    if sample:
        print(f'  sample of missing: {sample}')

    # \u770b text \u662f\u4ec0\u4e48 (\u53ef\u80fd\u662f "Import" \u9884\u8bbe\u7684\u7b80\u7565 setting)
    cur.execute('''
      SELECT ds.text FROM AgLibraryCollectionImage ci
      JOIN Adobe_imageDevelopSettings ds ON ds.image = ci.image
      WHERE ci.collection=?
      AND ds.text IS NOT NULL AND length(ds.text) BETWEEN 10 AND 100
      LIMIT 1
    ''', (cid,))
    s = cur.fetchone()
    if s:
        print(f'  short-text example (10-100 chars): "{s[0][:100]}"')

con.close()
