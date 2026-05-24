"""生成 outputs/firered_compare.html 让用户直接对比 4 个 baseline viewer."""
from pathlib import Path

html = """<html><head><meta charset=utf-8>
<title>FireRed Baseline v1 - v4 对比</title>
<style>
body{font-family:sans-serif;background:#222;color:#eee;padding:16px;max-width:1400px;margin:auto}
a{color:#9cf;font-size:18px;text-decoration:none}
a:hover{text-decoration:underline}
h2{margin-top:24px}
.stats{background:#333;padding:12px;border-radius:6px;margin:12px 0}
.win{color:#9f9;font-weight:bold}.lose{color:#f99}.neutral{color:#ddd}
table{border-collapse:collapse;font-family:monospace;margin:8px 0}
th,td{border:1px solid #555;padding:6px 12px;text-align:center}
.viewer-links a{display:inline-block;margin:8px 12px 8px 0;padding:8px 16px;background:#345;border-radius:6px}
</style></head><body>
<h1>FireRed Baseline 演进 (v1 → v2 → v3 → v4)</h1>

<div class=stats>
<h2>4-way per-action primary param R (Pearson, 越高越好)</h2>
<table>
<tr><th>action</th><th>v1 (base)</th><th>v2 (+ pixel loss)</th><th>v3 (action-aware)</th><th>v4 (wb global×3)</th><th>胜者</th></tr>
<tr><td>contrast</td><td>+0.47</td><td class=lose>+0.08</td><td class=win>+0.71</td><td>+0.49</td><td>v3</td></tr>
<tr><td>saturation</td><td class=win>+0.60</td><td>+0.37</td><td>+0.47</td><td>+0.39</td><td>v1</td></tr>
<tr><td>shadows</td><td class=win>+0.55</td><td>+0.16</td><td class=lose>-0.41</td><td class=lose>-0.29</td><td>v1</td></tr>
<tr><td>highlights</td><td class=win>+0.64</td><td>+0.30</td><td>+0.43</td><td>+0.37</td><td>v1</td></tr>
<tr><td>wb</td><td class=lose>-0.17</td><td class=lose>-0.12</td><td class=win>+0.80</td><td>+0.18</td><td>v3</td></tr>
</table>
<p>结论: <b>无单一胜者</b>. v1 在 contrast/sat/shadows/highlights 上保持均衡, v3 在 wb 上完胜但 shadows 反向; v4 是中庸.</p>
</div>

<div class=stats>
<h2>整体像素 L1 (model_render 离 FireRed_edit, 越小越好)</h2>
<table>
<tr><th>baseline</th><th>L1(model, FR_edit)</th><th>L1(orig, model) 改动幅度</th><th>val_loss</th><th>best Ep</th></tr>
<tr><td>v1 (param-only)</td><td class=win>0.0732</td><td>0.0488</td><td>0.0492</td><td>22</td></tr>
<tr><td>v2 (+ pixel loss)</td><td class=win>0.0713</td><td>0.0493</td><td>0.1038</td><td>48</td></tr>
<tr><td>v3 (action-aware × 5)</td><td class=lose>0.0810</td><td>0.0387</td><td>0.0427</td><td>13</td></tr>
<tr><td>v4 (wb global × 3)</td><td>0.0783</td><td>0.0414</td><td>0.0764</td><td>9</td></tr>
</table>
<p>v1 和 v2 像素 L1 最佳, 但 v2 用 pixel loss 直接优化也只微胜 v1. v3/v4 加大 wb 后整体像素反而退步, 因 wb 是双刃剑.</p>
</div>

<div class=stats>
<h2>实验进化路径</h2>
<ul>
<li><b>v1 (base)</b>: 朴素 weighted MSE + tier weighting → 4/5 action 学到 (R≥+0.47), wb 学不出 (-0.17)</li>
<li><b>v2 (pixel loss)</b>: 加 diff_isp 渲染重建 L1 → <span class=lose>失败</span>, 7D ISP 非单射, 模型找到"渲染相似但参数偏离"局部解, |dP| +40%</li>
<li><b>v3 (action-aware × 5)</b>: 主参数 ×5, 其他 ×0.2 → wb +0.80 突破 ✓, contrast +0.71 ✓, 但 shadows 反向 (-0.41), 整体 L1 退步</li>
<li><b>v4 (wb global × 3)</b>: 所有 action 上 wb param 权重 = 3 → wb 仅+0.18 (未达 v3), 其他维度跟 v1 持平, 折中</li>
</ul>
</div>

<div class="stats viewer-links">
<h2>视觉验证 viewer (打开看各模型 5×4-panel)</h2>
<a href="firered_v1_val_compare/">→ v1 viewer (推荐, 综合最强)</a>
<a href="firered_v3_val_compare/">→ v3 viewer (wb 最准但 shadows 错)</a>
<a href="firered_v4_val_compare/">→ v4 viewer (折中)</a>
<a href="firered_v2_val_compare/">→ v2 viewer (失败实验)</a>
</div>

<div class=stats>
<h2>下一步建议</h2>
<ol>
<li><b>G1. 扩大数据 N=499→2000</b>: 跟 FireRed 多跑 1500 张, 重点补 wb action (从 28 train → 200). 数据量是最直接 win, 可能让 v1 wb 也学起来</li>
<li><b>G2. 双头模型</b>: wb 单独 head (因物理 scale 100×), 其他 6D 共享 head. 避免 wb 影响其他 loss balance</li>
<li><b>G5. Expert C + FireRed 联合训练</b>: 4498 + 372 = 4870 samples, condition 编码 ∈ {action_5, expert_c}. 大数据量减轻过拟合, 各取所长</li>
<li><b>Qwen3-VL-4B LoRA SFT</b>: 项目主线最终目标, MLLM 端到端 (orig+caption) → 7D JSON. MobileViT baseline 到此为止, 切换主线</li>
</ol>
</div>

</body></html>"""

Path('outputs').mkdir(exist_ok=True)
Path('outputs/firered_compare.html').write_text(html, encoding='utf-8')
print('wrote outputs/firered_compare.html')
