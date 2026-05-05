"""生成 IP2P 测试结果对比 HTML 页面."""
import os
import re
from pathlib import Path

OUT_DIR = Path('outputs/ip2p_test')

html_parts = ["""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>IP2P Edit Comparison</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 20px; background: #1a1a1a; color: #e0e0e0; }
h1 { color: #fff; }
.sample { margin: 30px 0; padding: 20px; background: #2a2a2a; border-radius: 8px; }
.sample h2 { margin: 0 0 10px 0; color: #4af; }
.images { display: flex; gap: 20px; margin: 15px 0; }
.col { flex: 1; }
.col img { width: 100%; border: 2px solid #555; border-radius: 4px; }
.col .label { text-align: center; padding: 5px; background: #444; font-weight: bold; }
.prompt { background: #1e3a5f; padding: 12px; border-left: 4px solid #4af; margin: 10px 0; font-family: monospace; }
.suggestion { background: #2e2e2e; padding: 12px; border-left: 4px solid #888; margin: 10px 0; color: #aaa; font-size: 0.9em; }
.metric { color: #fa0; font-weight: bold; }
</style>
</head>
<body>
<h1>InstructPix2Pix 编辑测试结果</h1>
<p>每个样本: 原图 vs IP2P 编辑后, 使用 Venus AesGuide 提取的 prompt</p>
"""]

# 找出所有样本
samples = sorted(set(int(re.match(r'(\d+)_', f.name).group(1))
                     for f in OUT_DIR.glob('*_orig.png')))

for idx in samples:
    orig_path = f'{idx:04d}_orig.png'
    edit_path = f'{idx:04d}_edit.png'
    prompt_path = OUT_DIR / f'{idx:04d}_prompt.txt'

    info = {}
    if prompt_path.exists():
        with open(prompt_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                    info[k.strip()] = v.strip()

    html_parts.append(f"""
<div class="sample">
  <h2>Sample {idx}: {info.get('Image', '')}</h2>
  <div class="prompt"><b>Edit Prompt:</b> {info.get('Prompt', '')}</div>
  <div class="suggestion"><b>Original full suggestion:</b> {info.get('Full suggestion', '')}</div>
  <div class="images">
    <div class="col">
      <div class="label">原图 (Original)</div>
      <img src="{orig_path}" />
    </div>
    <div class="col">
      <div class="label">IP2P 编辑后 (Edited)</div>
      <img src="{edit_path}" />
    </div>
  </div>
</div>
""")

html_parts.append("""
</body>
</html>
""")

out_html = OUT_DIR / 'compare.html'
out_html.write_text(''.join(html_parts), encoding='utf-8')
print(f'Saved: {out_html.resolve()}')
print(f'打开浏览器查看: file:///{str(out_html.resolve()).replace(chr(92), "/")}')
