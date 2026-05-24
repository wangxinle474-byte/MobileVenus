"""\u4ece fivek.lrcat (Lightroom catalog SQLite) \u63d0\u53d6 5 expert (A/B/C/D/E) \u7684\u5f00\u53d1\u8bbe\u7f6e\u771f GT.

Pipeline:
    AgLibraryCollection(name=A/B/C/D/E)
      -> AgLibraryCollectionImage (5000 vcopy / expert)
      -> Adobe_images (virtual copy)  -> masterImage
      -> Adobe_images (master)        -> rootFile
      -> AgLibraryFile (baseName + ext)
    Adobe_imageDevelopSettings.text (Lua-\u98ce\u683c\u8868) -> regex \u62bd 7D \u53c2\u6570

Output: data/fivek_expert_abcde_params.json
  meta {n_images, n_records, version, ...}
  samples [ {image_name, expert, develop_raw, params_7d, params_extra}, ... ]
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime


# ----------------------------- 字段映射规则 ---------------------------------
# Lightroom develop settings 字段 -> 我们的 7D
# LR Process Version (PV) 影响字段语义:
#   PV2003/2010 (LR3-): Brightness/Contrast/FillLight/HighlightRecovery
#   PV2012 (LR4+):       Exposure/Contrast/Highlights/Shadows/Whites/Blacks/Clarity
# 提取时保留原始字段不归一化, 下游决定如何使用.

LUA_KEYS_OF_INTEREST = [
    # 色温 / 白平衡
    'WhiteBalance', 'Temperature', 'Tint',
    'CustomTemperature', 'CustomTint',
    # 曝光 / 亮度
    'Exposure', 'Brightness',
    # 对比度
    'Contrast',
    # 阴影/高光 (PV2010 命名)
    'Shadows', 'HighlightRecovery', 'FillLight',
    # 阴影/高光 (PV2012 命名)
    'Highlights', 'Whites', 'Blacks',
    # 饱和度 / 鲜艳度
    'Saturation', 'Vibrance',
    # 清晰度
    'Clarity',
    # parametric curve
    'ParametricDarks', 'ParametricShadows',
    'ParametricLights', 'ParametricHighlights',
    'ParametricShadowSplit', 'ParametricMidtoneSplit', 'ParametricHighlightSplit',
    # process / version
    'ProcessVersion', 'Version', 'ToneCurveName',
]


def parse_lua_settings(text: str) -> dict:
    """\u4ece Lua-\u98ce\u683c\u7684 develop text \u62bd \u51fa\u611f\u5174\u8da3\u7684 key.

    text \u6837\u5b50: 's = { Brightness = 47, Contrast = 13, ... WhiteBalance = "Custom" }'
    """
    if not text:
        return {}
    out = {}
    for k in LUA_KEYS_OF_INTEREST:
        # \u533a\u5206 \u6570\u503c / \u5b57\u7b26\u4e32 / \u8868
        m = re.search(rf'\b{k}\s*=\s*("[^"]*"|-?\d+(?:\.\d+)?|\{{[^}}]*\}})', text)
        if not m:
            continue
        raw = m.group(1)
        if raw.startswith('"'):
            out[k] = raw.strip('"')
        elif raw.startswith('{'):
            out[k] = raw  # \u4fdd\u7559\u5b8c\u6574 lua table (ToneCurve)
        else:
            try:
                v = float(raw)
                out[k] = int(v) if v.is_integer() else v
            except ValueError:
                out[k] = raw
    return out


def map_to_7d(raw: dict) -> dict:
    """\u628a\u539f\u59cb LR \u53c2\u6570 \u6620\u5c04\u5230 7D (\u4fdd\u6301\u5355\u4f4d/\u8303\u56f4 \u8ddf LR \u4e00\u81f4, \u4e0d\u5f3a\u884c\u5f52\u4e00\u5316).

    Returns:
        {
          white_balance: Temperature in Kelvin (or None if WhiteBalance != Custom),
          brightness:     Brightness or Exposure*stop_scale (PV2012 fallback),
          contrast:       Contrast,
          shadows:        FillLight (PV2010) or Shadows (PV2012),
          highlights:     -HighlightRecovery (PV2010) or Highlights (PV2012),
          saturation:     Saturation,
          clarity:        Clarity (0 if missing),
        }
    """
    pv = raw.get('ProcessVersion') or raw.get('Version') or ''
    is_pv2012 = (str(pv).startswith('6') or 'Camera' in str(raw.get('Version', ''))
                 or 'Highlights' in raw)

    # White balance: \u53d6 Temperature (\u5b9e\u9645\u8272\u6e29 K), \u82e5\u662f "As Shot" \u53ef\u80fd\u4e3a None
    temp = raw.get('Temperature') or raw.get('CustomTemperature')

    if is_pv2012:
        brightness = raw.get('Exposure', 0)  # PV2012 \u7528 Exposure \u53d6\u4ee3 Brightness
        shadows = raw.get('Shadows', 0)
        # PV2012: Highlights = -100~+100 (\u6b63\u503c=\u63d0\u4eae, \u8d1f\u503c=\u538b\u6697)
        highlights = raw.get('Highlights', 0)
    else:
        # PV2010: Brightness \u539f\u672c\u5b9a\u4e49\u662f \u4e2d\u95f4\u8c03 (-150~+150)
        brightness = raw.get('Brightness', 0)
        # PV2010: FillLight = 0~100 (\u63d0\u9ad8\u9634\u5f71), Shadows = 0~100 (\u9ed1\u8272 clipping)
        # \u91c7\u7528 FillLight \u4f5c\u4e3a\u201cshadows\u63d0\u4eae\u201d\u7684\u4e3b\u8981\u6307\u6807; \u539fShadows \u662f\u9ed1\u8272 clipping
        shadows = raw.get('FillLight', 0)
        # PV2010: HighlightRecovery = 0~100 (\u538b\u4f4e\u9ad8\u5149); \u8f6c\u4e3a\u8d1f\u503c\u4fdd\u6301\u4e0e PV2012 \u4e00\u81f4
        highlights = -raw.get('HighlightRecovery', 0)

    return {
        'white_balance': float(temp) if temp is not None else None,
        'brightness': float(brightness),
        'contrast': float(raw.get('Contrast', 0)),
        'shadows': float(shadows),
        'highlights': float(highlights),
        'saturation': float(raw.get('Saturation', 0)),
        'clarity': float(raw.get('Clarity', 0)),
    }


def fetch_expert(con, expert_name: str):
    """\u8fd4\u56de [(image_name, raw_dict)] for one expert collection."""
    cur = con.cursor()
    cur.execute(
        '''
        SELECT f.baseName, f.extension, ds.text, i.copyName, m.id_local AS master_id
        FROM AgLibraryCollectionImage ci
        JOIN AgLibraryCollection c ON c.id_local = ci.collection
        JOIN Adobe_images i ON i.id_local = ci.image
        JOIN Adobe_images m ON m.id_local = i.masterImage
        JOIN AgLibraryFile f ON f.id_local = m.rootFile
        JOIN Adobe_imageDevelopSettings ds ON ds.image = ci.image
        WHERE c.name = ?
          AND ds.text IS NOT NULL AND length(ds.text) > 50
        ''', (expert_name,))
    out = []
    for baseName, ext, text, copyName, master_id in cur.fetchall():
        raw = parse_lua_settings(text)
        out.append({
            'image_name': f'{baseName}.{ext}',
            'copy_name': copyName,
            'master_id': master_id,
            'raw': raw,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lrcat',
                    default=r'E:\Data\dataset\fivek_dataset\raw_photos\fivek.lrcat')
    ap.add_argument('--output', default='data/fivek_expert_abcde_params.json')
    ap.add_argument('--experts', nargs='+', default=['A', 'B', 'C', 'D', 'E'])
    args = ap.parse_args()

    lrcat = Path(args.lrcat)
    if not lrcat.exists():
        print(f'[ERR] lrcat not found: {lrcat}')
        return 1
    print(f'Reading lrcat: {lrcat}  ({lrcat.stat().st_size / 1e9:.2f} GB)')

    con = sqlite3.connect(f'file:{lrcat}?mode=ro', uri=True)

    all_samples = []
    per_expert_stats = {}
    for ex in args.experts:
        print(f'\nExpert {ex}:')
        recs = fetch_expert(con, ex)
        print(f'  {len(recs)} records fetched')

        # \u62a4\u533a\u91cd\u590d: \u540c\u4e00 image \u53ef\u80fd\u67d0 expert \u51fa\u73b0\u591a\u4e2a record
        # (process version \u5347\u7ea7\u91cd\u590d\u4fdd\u5b58), \u53d6\u6700\u540e\u4e00\u4e2a (\u6700\u65b0 PV)
        by_image = {}
        for r in recs:
            by_image[r['image_name']] = r
        print(f'  -> {len(by_image)} unique images after dedup-by-image')

        # \u5b57\u6bb5\u8986\u76d6\u7edf\u8ba1
        field_count = Counter()
        for r in by_image.values():
            for k in r['raw']:
                field_count[k] += 1
        top = field_count.most_common(8)
        print(f'  top fields: {top}')

        # \u8f93\u51fa\u8bb0\u5f55
        n_with_temp = 0
        for r in by_image.values():
            params_7d = map_to_7d(r['raw'])
            if params_7d['white_balance'] is not None:
                n_with_temp += 1
            all_samples.append({
                'image_name': r['image_name'],
                'expert': ex,
                'lr_copy_name': r['copy_name'],
                'master_id': r['master_id'],
                'params': params_7d,
                'raw_lr': r['raw'],
            })
        print(f'  with Temperature non-null: {n_with_temp}/{len(by_image)}')

        per_expert_stats[ex] = {
            'n_records': len(recs),
            'n_unique_images': len(by_image),
            'n_with_temperature': n_with_temp,
        }

    con.close()

    # \u8f93\u51fa
    out = {
        'meta': {
            'pipeline': 'fivek.lrcat -> 5 expert virtual copies -> develop settings',
            'lrcat_path': str(lrcat),
            'experts': args.experts,
            'per_expert': per_expert_stats,
            'total_samples': len(all_samples),
            'created_at': datetime.now().isoformat(),
            'param_schema_7d': [
                'white_balance (Kelvin)',
                'brightness (LR Brightness or Exposure)',
                'contrast (LR Contrast)',
                'shadows (LR FillLight PV2010 or Shadows PV2012)',
                'highlights (-LR HighlightRecovery PV2010 or Highlights PV2012)',
                'saturation (LR Saturation)',
                'clarity (LR Clarity)',
            ],
            'note': 'raw_lr 字段保留 LR 原始 develop settings 完整内容',
        },
        'samples': all_samples,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f'\n[OK] saved {len(all_samples)} samples -> {out_path} '
          f'({out_path.stat().st_size/1e6:.1f} MB)')

    # \u7b80\u5355\u5404\u53c2\u6570 stat
    print('\nPer-expert 7D param stats (mean):')
    print(f'  {"expert":<8s}{"n":>5s}'
          f'{"wb_mean":>10s}{"bright":>8s}{"contr":>8s}'
          f'{"shad":>8s}{"high":>8s}{"sat":>8s}{"clar":>8s}')
    for ex in args.experts:
        rows = [s for s in all_samples if s['expert'] == ex]
        if not rows:
            continue
        avg = lambda k: sum(s['params'][k] or 0 for s in rows) / len(rows)
        wbs = [s['params']['white_balance'] for s in rows if s['params']['white_balance'] is not None]
        wb_mean = sum(wbs) / len(wbs) if wbs else 0
        print(f'  {ex:<8s}{len(rows):>5d}'
              f'{wb_mean:>10.0f}{avg("brightness"):>8.2f}{avg("contrast"):>8.2f}'
              f'{avg("shadows"):>8.2f}{avg("highlights"):>8.2f}'
              f'{avg("saturation"):>8.2f}{avg("clarity"):>8.2f}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
