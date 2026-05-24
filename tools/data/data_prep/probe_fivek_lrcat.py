"""探查 fivek.lrcat (Lightroom catalog) 表结构与候选 develop settings 字段."""
import sqlite3
from pathlib import Path

LRCAT = Path(r'E:\Data\dataset\fivek_dataset\raw_photos\fivek.lrcat')

con = sqlite3.connect(f'file:{LRCAT}?mode=ro', uri=True)
cur = con.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in cur.fetchall()]
print(f'== {len(tables)} tables ==')

# 看候选 develop 相关的表
keywords = ['develop', 'image', 'process', 'history', 'snapshot',
            'edit', 'master', 'collection', 'rgb', 'tone', 'white']
print('\n== candidate tables (matching dev/image/etc.) ==')
candidates = []
for t in tables:
    tl = t.lower()
    if any(k in tl for k in keywords):
        candidates.append(t)
        cur.execute(f'SELECT COUNT(*) FROM "{t}"')
        n = cur.fetchone()[0]
        print(f'  {t:60s}  rows={n}')

# 看 Develop / settings 的列
print('\n== column info for candidate tables ==')
for t in candidates[:15]:
    cur.execute(f'PRAGMA table_info("{t}")')
    cols = cur.fetchall()
    if 5 <= len(cols) <= 30:
        col_names = [c[1] for c in cols]
        print(f'  {t}:')
        print(f'    cols ({len(col_names)}): {col_names}')

# 看 Adobe_imageDevelopSettings 或 类似命名的表
print('\n== full table list with rowcounts ==')
for t in tables:
    try:
        cur.execute(f'SELECT COUNT(*) FROM "{t}"')
        n = cur.fetchone()[0]
        print(f'  {t:70s}  {n}')
    except Exception as e:
        print(f'  {t:70s}  ERR: {e}')

con.close()
