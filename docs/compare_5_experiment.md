# compare_5: 5 张图片编辑模型对比实验

> 本实验是 IntelligenceCamera 项目中用于对比 **LongCat-Image-Edit-Turbo** vs **FireRed-Image-Edit-1.1-Lightning** 两大编辑模型, 以及 **prompt 风格**(场景描述 vs 编辑指令) 与 **prompt rewrite** 对结果的影响.
>
> 产物位于 `outputs/compare_5/`, 本文为其 README / 实验说明.

## 1. 实验目的

回答 3 个问题:

1. **两大编辑模型谁强**: LongCat-Turbo (美团, ~30GB, 本地 diffusers) vs FireRed-Lightning (小红书, 在线 ModelScope Studio)?
2. **Prompt 风格影响**: 给 LongCat 喂"场景描述 (sceneA)"还是"编辑指令 (editB)" 哪个分高?
3. **LLM rewrite 有没有用**: 开启 FireRed 内置 rewrite, 或用 Qwen3-VL 把 editB 重写成场景描述后喂 LongCat, 能不能提分?

## 2. 测试样本 (5 张图)

挑自 `outputs/ip2p_pilot_100/` 的 100 张候选, 覆盖不同语义+光照+挑战类型:

| idx | 源文件 | 场景 | 编辑挑战 |
|:---:|-------|-----|---------|
| 0071 | `5806.png` | 手机+杂乱建筑背景 | 背景模糊聚焦主体 |
| 0110 | `5845.png` | 黑白海面+船 | 加强对比+保持 B&W |
| 0194 | `5929.png` | 剪影+大片空白右侧 | 戏剧性侧光+填空 |
| 0448 | `6185.png` | 山景+经幡 | 金色时分光+保旗帜鲜艳 |
| 0808 | `6547.png` | 枯枝+蓝天 | 暖阳光+保持原彩色 |

## 3. 实验条件

### 两类 caption (`data/compare_5_captions*.json`)

| 组 | 文件 | 风格 | 示例 |
|----|------|------|------|
| **sceneA** | `compare_5_captions.json` | 场景描述 | "A cluttered phone scene with emphasized screen, background blurred into soft bokeh" |
| **editB** | `compare_5_captions_edit.json` | 编辑指令 | "Blur the cluttered building background into soft bokeh to emphasize the smartphone screen." |

### 5 个条件 (生成的 5 组输出)

| 条件名 | 模型 | Caption | Rewrite | 所在目录 |
|-------|------|---------|---------|---------|
| `longcat_sceneA` | LongCat-Turbo | sceneA | — | `longcat_sceneA/` |
| `longcat_editB` | LongCat-Turbo | editB | — | `longcat_editB/` |
| `firered_editB` | FireRed-Lightning | editB | ❌ (off) | `firered_editB/` |
| `firered_editB_rewrite` | FireRed-Lightning | editB | ✅ (on) | `firered_editB_rewrite/` |
| `longcat_editB_rewritten` | LongCat-Turbo | **editB → Qwen3-VL 重写后的场景 prompt** | — | `longcat_editB_rewritten/` *(待生成)* |

## 4. 目录结构

```
outputs/compare_5/
├── originals/                 5 张原图 (去重, 共享)
│   └── <idx>.png              (0071, 0110, 0194, 0448, 0808)
├── longcat_sceneA/            5 张编辑后图, 文件名 <idx>.png
├── longcat_editB/             同上
├── firered_editB/             同上
├── firered_editB_rewrite/     同上
├── longcat_editB_rewritten/   同上 (待生成)
├── scores/                    Qwen3-VL 1-10 评分 JSON
│   └── <condition>_10pt.json
├── meta/                      推理 metadata (seed/cfg/runtime)
│   └── <condition>.json
├── legacy_1to5/               旧 1-5 分评分归档
│   └── <condition>.json
└── viewer.html                4 栏对比 HTML (浏览器直开)
```

**文件命名约定**: 只用 `<idx>.png` (4 位数字), 不带后缀. 用目录名区分实验条件.

## 5. 评分方法 (Qwen3-VL 1-10)

脚本: `tools/eval/score_longcat_edits.py`
模型: `Qwen/Qwen3-VL-4B-Instruct` (本地 AutoDL)

每张图评 4 维度 + 1 汇总 + reason:

| 维度 | 含义 |
|------|-----|
| `instruction_follow` [1-10] | 是否实现 caption 要求的修改 |
| `aesthetic` [1-10] | 美学质量 (光影/色彩/构图) |
| `identity_preserve` [1-10] | 主体保留 (与原图一致性) |
| `realism` [1-10] | 真实感 (是否伪影/失真) |
| `overall` [1-10] | 4 项均值四舍五入 |

分段参考: 10=完美 / 8-9=优秀 / 6-7=良好 / 4-5=中等 / 2-3=较差 / 1=失败

## 6. 当前结果 (2026-05-05 截止)

4 组已评分 (第 5 组待 Qwen3-VL 重写后跑):

| 条件 | IF | Aes | IdP | Real | **Overall** |
|-----|:--:|:--:|:--:|:--:|:--:|
| `longcat_sceneA` | 9.0 | **9.2** | 9.8 | **10.0** | **9.2** 🥇 |
| `firered_editB` | **9.0** | 8.8 | **10.0** | **10.0** | **9.0** 🥈 |
| `longcat_editB` | 8.6 | 8.4 | **10.0** | 9.8 | **8.8** 🥉 |
| `firered_editB_rewrite` | *待评* | — | — | — | — |
| `longcat_editB_rewritten` | *待生成+评* | — | — | — | — |

**关键发现** (1-10 分):
1. **LongCat + 场景描述 最高** (9.2): 整图重绘路线
2. **FireRed Lightning 追平指令路线** (9.0, +0.2 vs LongCat-editB 的 8.8)
3. **三家底线 IdP + Real 都 ≥ 9.8**: 区分度全在 IF + Aes
4. **FireRed 5 张方差=0 最稳**: 全 9, LongCat 波动 8-10

## 7. 如何复现 (端到端)

### 0. 原图就位
```powershell
# 5 张原图应在 outputs/compare_5/originals/<idx>.png
# 若缺失, 从 outputs/ip2p_pilot_100/<idx>_orig.png 复制 + rename
```

### 1. LongCat sceneA (长款场景描述)
```powershell
python tools/data/local_run_longcat_turbo.py `
  --captions data/compare_5_captions.json `
  --out_dir outputs/compare_5/longcat_sceneA `
  --skip_download --use_4bit --steps 4 --offload sequential
```

### 2. LongCat editB (编辑指令)
```powershell
python tools/data/local_run_longcat_turbo.py `
  --captions data/compare_5_captions_edit.json `
  --out_dir outputs/compare_5/longcat_editB `
  --skip_download --use_4bit --steps 4 --offload sequential
```

### 3. FireRed editB (rewrite off)
```powershell
python tools/data/run_firered_online.py `
  --captions data/compare_5_captions_edit.json `
  --input_dir outputs/compare_5/originals `
  --out_dir outputs/compare_5/firered_editB `
  --lora Lightning
```

### 4. FireRed editB (rewrite on)
```powershell
python tools/data/run_firered_online.py `
  --captions data/compare_5_captions_edit.json `
  --input_dir outputs/compare_5/originals `
  --out_dir outputs/compare_5/firered_editB_rewrite `
  --lora Lightning --rewrite_prompt
```

### 5. AutoDL 评分 4 组
```bash
ssh autodl "bash /root/autodl-tmp/IntelligenceCamera/scripts/autodl/autodl_rescore_all_10pt.sh"
scp -r root@autodl:/root/autodl-tmp/IntelligenceCamera/outputs/compare_5/scores/ outputs/compare_5/
```

### 6. Qwen3-VL 重写 + 第 5 组
```bash
# AutoDL Phase 1: Qwen3-VL 把 editB 重写 + 4 组评分
ssh autodl "bash scripts/autodl/autodl_phase1_rewrite_and_rescore.sh"
# 把重写后 captions 拉回
scp autodl:.../data/compare_5_captions_edit_rewritten.json data/
```

```powershell
# 本地用重写 caption 跑 LongCat
python tools/data/local_run_longcat_turbo.py `
  --captions data/compare_5_captions_edit_rewritten.json `
  --out_dir outputs/compare_5/longcat_editB_rewritten `
  --skip_download --use_4bit --steps 4

# 上传给 AutoDL 评分
scp -r outputs/compare_5/longcat_editB_rewritten autodl:.../outputs/compare_5/
ssh autodl "bash scripts/autodl/autodl_phase2_score_longcat_rewritten.sh"
scp autodl:.../outputs/compare_5/scores/longcat_editB_rewritten_10pt.json outputs/compare_5/scores/
```

### 7. 视觉对比 HTML + 表格展示
```powershell
python tools/eval/build_compare_html.py
# → outputs/compare_5/viewer.html

python tools/eval/present_3way_10pt.py
# → 控制台打印 5-way 均值表
```

## 8. 延伸阅读

- 整体项目结构: `docs/PROJECT_STRUCTURE.md`
- 实验日志: `docs/experiment_log.md`
- 数据准备: `docs/data_preparation.md`
- scripts/ 索引: `scripts/README.md`
