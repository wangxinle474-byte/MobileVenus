# 本地数据资源参考

> 本文件供 AI 助手快速了解本地 E 盘上的数据集和资源分布。
> 当上下文丢失时，读取此文件即可恢复对本地环境的认知。
>
> 最后更新: 2026-05-01

---

## 项目路径

- **项目根目录**: `E:\智能相机\Venus_CVPR2026-main\IntelligenceCamera\`
- **GitHub**: `https://github.com/wangxinle474-byte/MobileVenus`
- **Git 代理**: `127.0.0.1:7897`

---

## 本地数据集 (`E:\Data\dataset\`)

### 核心训练数据

| 数据集 | 路径 | 文件数 | 大小 | 说明 |
|--------|------|--------|------|------|
| **FiveK RAW** | `fivek_dataset/raw_photos/` | 5,243 | 47.3 GB | DNG 原始 RAW 文件 (仅本地) |
| **FiveK JPEG** | `fivek_jpeg/` | 5,121 | 216 MB | RAW→JPEG 转换后 (AutoDL 也有) |
| **FiveK Expert 标注** | `fivek_expert/` | 1 | 37 MB | `fivek_expert_settings.json` |

### 美学评分数据

| 数据集 | 路径 | 文件数 | 大小 | 说明 |
|--------|------|--------|------|------|
| **AADB** | `AADB/` | 20,969 | 2.4 GB | 美学评分数据集 (仅本地) |
| **AADB HF 缓存** | `E:\cache\huggingface\datasets--Iceclear--AADB\` | 3 | <1 MB | HuggingFace 缓存索引 |

AADB 子目录:
- `datasetImages_originalSize/` — 原始分辨率图片
- `datasetImages_warp256/` — 256×256 缩放图片
- `AADB_newtest_originalSize/` — 测试集原始图
- `imgListFiles_label/` — 标签文件
- `AllinAll.csv` — 全部标注
- `AADBinfo.mat` — MATLAB 格式标注

### 人像修图数据

| 数据集 | 路径 | 文件数 | 大小 | 说明 |
|--------|------|--------|------|------|
| **PPR10K** | `PPR10K/` | 7 | 13.1 GB | XMP 标注包 (AutoDL 有 111G 全量版) |

PPR10K 子目录:
- `xmp_source.zip` — 源 XMP
- `xmp_target_a/b/c.zip` — 三位修图师的目标 XMP
- `train_val_images_tif_360p/` — 360p 预览图

### Venus 原始数据

| 数据集 | 路径 | 文件数 | 大小 | 说明 |
|--------|------|--------|------|------|
| **Venus 全量** | `Venus_data/` | 83,736 | 53.5 GB | 仅本地 |

Venus_data 子目录:
- `Stage1/` — 第一阶段数据
- `Stage2/` — 第二阶段数据
- `Benchmark_AesGuide/` — 美学引导 benchmark
- `Benchmark_FLMS/` — FLMS benchmark

### 打包/上传用

| 路径 | 文件数 | 大小 | 说明 |
|------|--------|------|------|
| `autodl_fivek_package/` | 4 | 216 MB | 上传到 AutoDL 的打包文件 (fivek_jpeg.zip, fivek_data.zip, code.zip) |
| `stage1_upload/` | 5,379 | 12.9 GB | venus_text_embeddings + 图片 + 训练包 |

---

## 数据在 AutoDL 和本地的对应关系

| 数据 | 本地 E 盘 | AutoDL | 备注 |
|------|:---------:|:------:|------|
| FiveK RAW (DNG) | ✅ 47 GB | ❌ | 原始文件，仅本地 |
| FiveK JPEG | ✅ 216 MB | ✅ 227 MB | 两端一致 |
| FiveK Expert C GT | ❌ | ✅ 289 MB | 仅 AutoDL (由 Expert 设定渲染) |
| FiveK Expert 标注 | ✅ 37 MB | ✅ (data/) | 两端都有 |
| PPR10K | ✅ 13 GB (XMP) | ✅ 111 GB (全量) | AutoDL 是完整版 |
| AADB | ✅ 2.4 GB | ❌ | 仅本地 (需上传才能训练) |
| Venus 原始数据 | ✅ 53.5 GB | ❌ (部分 embedding) | 原始数据仅本地 |
| 训练 Embedding | ✅ (stage1_upload) | ✅ (data/) | 两端都有 |
| 大模型/HF缓存 | ❌ | ✅ 83 GB | 仅 AutoDL |

---

## 各阶段训练所需数据

| 训练阶段 | 需要的数据 | AutoDL 就绪? |
|----------|-----------|:---:|
| Stage A 语义对齐 | FiveK JPEG + MiniLM text embedding | ✅ |
| Stage B 参数预测 | FiveK JPEG + Expert C GT + 参数标注 | ✅ |
| Stage C 文本条件 | FiveK JPEG + instruction_data.json | ✅ |
| RefinementNet | FiveK JPEG (512) + Expert C GT (512) | ✅ |
| PPR10K 扩展训练 | PPR10K 全量 | ✅ |
| AADB 美学评分器 | AADB 图片 + 标签 | ❌ 需上传 |
