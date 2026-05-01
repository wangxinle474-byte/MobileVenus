# MobileVenus 小程序版 PRD

> 版本: v2.1 | 日期: 2026-03-29 | 产品阶段: MVP

---

## 1. 产品概述

### 1.1 产品定位

MobileVenus 小程序版是 **MobileVenus 研究项目的可交互演示入口**。

用户上传或拍摄一张照片后，后端调用 MobileVenus Distill 模型（PSNR=33.15 dB，SSIM=0.9326）自动预测 8 个 Lightroom 参数，通过 Gamma-aware ISP Pipeline 渲染增强图，并将参数结果可视化展示。

核心价值在于让人**直观感受**"语义蒸馏"的效果：同一张照片，AI 理解了光线语义后给出的调参比纯回归更协调、更自然。

### 1.2 产品目标

本版本的首要目标是 **CVPR 演示 + 效果验证**，其次是轻量用户体验验证：

- **演示目标**：让评委、老师、面试官在手机上直接体验模型效果，不需要跑 Python 脚本
- **效果验证**：前后对比是否足够直观（对应 PSNR +1.10 dB 的感知差距）
- **可解释性验证**：用户是否认可"参数可读 + AI 给出理由"的展示形式
- **闭环验证**：小程序端能否顺畅完成选图 → 推理 → 对比 → 保存的完整链路

### 1.3 版本范围

本版本聚焦 **最小可演示闭环**，不追求完整修图能力，不追求商业级产品体验。

### 1.4 与研究项目的关系

| 维度 | 研究版（MobileVenus） | 小程序版（本 PRD） |
|---|---|---|
| 目标 | CVPR 论文发表 | 可交互演示 + 用户验证 |
| 模型 | Distill v2 / v4（.pt 权重） | 后端加载同一套权重 |
| 输入 | FiveK JPEG (224×224 标准化) | 用户手机随拍图片（任意分辨率） |
| 输出 | PSNR/SSIM 指标 | 增强图 + 可视化参数 |
| 代码基础 | `models/`、`training/`、`tools/` | 后端直接复用推理链路 |

---

## 2. 目标用户

### 2.1 核心用户

- **演示对象**：老师、评委、面试官（需要快速看到效果，无需理解技术细节）
- **AI/图像方向关注者**：对语义蒸馏原理和参数可解释性感兴趣的研究者
- **轻度体验用户**：希望快速优化照片的普通用户

### 2.2 用户核心诉求

- 操作简单，一键出图
- 能明显看到前后差异（对应实验中 +1.10 dB 的感知提升）
- 能知道"改了什么、为什么这样改"（参数可解释性）
- 能保存增强后的结果图

---

## 3. 核心功能列表

### 3.1 图片输入

- 相册选图（P0）
- 拍照上传（P0）
- 更换图片重新处理（P0）

**输入约束**：
- 支持格式：JPEG / PNG
- 建议大小：≤ 10MB（超出给出提示）
- 后端会将图片 resize 到 224×224 做推理，原图保留用于 ISP 渲染

### 3.2 云端推理

后端依托现有推理链路，**不从零实现模型推理**：

```
用户图片上传（携带 model_version 参数）
    ↓
图片预处理（resize 224×224 + 标准化）
    → 对应：training/semantic_distill/model.py 的预处理逻辑
    ↓
按用户所选模型版本推理
    → ModelRegistry 根据 model_version 路由到对应已预加载权重
    → 模型组件：models/vision_encoder.py + models/semantic_bridge.py
    ↓
输出 8 个 Lightroom 参数
    ↓
Gamma-aware ISP Pipeline 渲染增强图
    → 对应：models/isp_pipeline.py（已有完整实现）
    ↓
返回：增强图 URL + 参数 JSON + 元信息（含 model_name）
```

**模型预加载要求**：Django 服务启动时在 `AppConfig.ready()` 阶段将模型加载到内存（CPU 或 GPU），请求到来时直接推理，**禁止每次请求重新加载权重**。

**模型选择说明**：

| 可用权重 | PSNR | SSIM | 推荐场景 |
|---|---|---|---|
| `checkpoints/semantic_distill_v2/` | **33.15 dB** | 0.9311 | 演示像素精度优先 |
| `checkpoints/distill_v4/` | 33.11 dB | **0.9326** | 演示感知质量优先（默认推荐） |

### 3.3 结果展示

- 原图 / 增强图滑动对比（核心交互，体现 +1.10 dB 的视觉感知）
- "原图 / 增强"状态标识
- 增强效果主视觉展示

### 3.4 参数面板展示

展示 8 个 Lightroom 参数的预测结果，**参数分组和置信度标注**基于项目已有的专家一致性数据（`data/fivek_expert_consensus.json`）：

| 参数 | 显示名 | 物理范围 | 专家一致性 | 置信度标注 |
|---|---|---|---|---|
| `ev_compensation` | 曝光补偿 | −3.0 ~ +3.0 EV | 17.5% | 中等 |
| `white_balance` | 色温 | 2000 ~ 10000 K | 6.9% | ⚠ 仅参考 |
| `contrast` | 对比度 | −100 ~ +100 | 14.5% | 中等 |
| `brightness` | 亮度 | −100 ~ +100 | 19.8% | 中等 |
| `shadows` | 阴影 | −100 ~ +100 | 8.3% | 较高 |
| `highlights` | 高光 | −100 ~ +100 | 18.3% | 中等 |
| `saturation` | 饱和度 | −100 ~ +100 | **2.2%** | ✅ 高置信 |
| `vibrance` | 自然饱和度 | −100 ~ +100 | 7.0% | 较高 |

> **色温（WB）特别说明**：Distill v4 的 WB MAE 为 618K（高于 Baseline 502K），这是已知的模型局限，展示时需加"仅供参考"标注，避免用户对色温结果产生过高期望。

### 3.5 推理元信息展示（P1）

后端返回 `meta_info`，前端可选展示：

```json
{
  "model_name": "MobileVenus Distill v4",
  "backbone": "MobileViT-Small (1.93M params)",
  "pipeline": "SemanticDistill + Gamma-aware ISP",
  "inference_time_ms": "<实际耗时>",
  "image_width": "<原图宽>",
  "image_height": "<原图高>",
  "psnr_reference": "33.11 dB (FiveK val 500张)"
}
```

推理耗时展示有助于体现"轻量"这一研究卖点。

### 3.6 Venus 美学评分（P1）

推理完成后，后台异步调用 Venus-Q-Stage1（Qwen-VL-Chat 微调）对原图和增强图进行美学评分，前端通过轮询接口获取结果。

**评分维度**：构图 / 光线 / 色彩 / 清晰度 / 主体 + 综合分（均 1–10 分）

**交互设计**：
- 推理完成后立即显示评分卡（显示“Venus 正在评分，请稍候…”）
- 评分完成后自动切换为对比表格，展示各维度分数和就增量
- 模型不可用时显示“评分暂不可用”而不是空白

**前后端协议**：
- enhance 接口返回 `aesthetic_status: "running"`, `aesthetic_before/after: null`
- 前端每 6s 轮询 `GET /api/v1/inference/aesthetic/<request_id>/`
- 轮询到终态（`done` / `failed` / `unavailable`）后停止

### 3.7 结果保存

- 保存增强图到手机相册（P0）
- 失败时给出权限提示或错误提示（P0）

### 3.8 状态反馈

| 状态 | 触发时机 |
|---|---|
| `idle` | 初始进入页面 |
| `uploading` | 图片上传中 |
| `processing` | 后端推理中 |
| `success` | 推理完成、结果可展示 |
| `error` | 任意异常（含超时） |

### 3.9 模型选择

用户可在发起推理前选择使用哪个已训练的模型版本，以便对比不同模型的增强效果。

**可选模型**（基于 `checkpoints/` 下已有权重）：

| 显示名 | 权重路径 | PSNR | SSIM | 标签 |
|---|---|---|---|---|
| Baseline | `checkpoints/fivek_8param/best.pt` | 32.05 dB | 0.9269 | 对照基准（无蒸馏） |
| Distill v2 | `checkpoints/semantic_distill_v2/stage_b/best.pt` | **33.15 dB** | 0.9311 | 像素精度最优 |
| Distill v4 | `checkpoints/distill_v4/best.pt` | 33.11 dB | **0.9326** | 感知质量最优（默认） |

**交互设计**：
- 位置：首页选图区下方，推理按钮上方
- 形式：横向单选 Chip 组（3 项），当前选中高亮，默认选中 Distill v4
- 每个 Chip 显示：模型名 + 一句话标签（如"感知最优" / "精度最优" / "对照基准"）
- 选中状态存入 Pinia store，页面切换后不丢失
- 结果页 meta_info 面板回显当前使用的模型名与 PSNR 参考值

**前后端协议**：
- 前端在 `POST /api/v1/inference/enhance/` 请求体中携带 `model_version` 字段（`baseline` / `distill_v2` / `distill_v4`，缺省时默认 `distill_v4`）
- 后端 `ModelRegistry` 在 Django 启动时**全部预加载**三个版本，请求到来时按 `model_version` 路由，切换无需等待模型重载

**演示价值**：
- 同一张照片依次切换 Baseline → Distill v4，可现场展示语义蒸馏带来的 +1.10 dB 提升
- 适合 CVPR 演示：评委无需看论文图表，在手机上直接感受蒸馏效果差异

---

## 4. 功能优先级

| 优先级 | 功能 | 说明 |
|---|---|---|
| P0 | 相册选图 / 拍照 | 入口 |
| P0 | 图片上传到后端 | 打通推理链路 |
| P0 | 后端推理并返回增强图 | 核心结果 |
| P0 | 原图 / 增强图滑动对比 | 直观体现模型价值 |
| P0 | 8 参数面板展示（含置信度分组） | 体现可解释性 |
| P0 | 保存到相册 | 形成闭环 |
| P0 | 加载中 / 失败态提示 | 保证可用性 |
| P1 | 推理耗时展示 | 体现模型轻量化优势 |
| P1 | 参数折叠 / 展开 | 优化结果页信息层级 |
| P1 | 最近一次结果缓存 | 避免误退丢失 |
| P1 | 模型选择（Baseline / Distill v2 / v4） | CVPR 演示核心，同图对比蒸馏效果 |
| P1 | Venus 美学评分（异步评分 + 轮询展示） | 第三方模型评价增强效果，强化演示说服力 |
| P2 | 增强强度滑杆 | 后续版本考虑 |
| P2 | 参数手动微调 | 后续版本考虑 |
| P2 | 历史记录列表 | 后续版本考虑 |

---

## 5. 技术栈

### 5.1 小程序前端

**方案：uni-app + Vue 3 + TypeScript + Pinia**

- 页面框架：uni-app
- 状态管理：Pinia
- 请求封装：`uni.request` 二次封装（统一超时、错误码处理）
- UI：自定义轻量组件，不引入重型 UI 库
- 图片对比：自定义滑块对比组件（`CompareSlider`）

### 5.2 后端推理服务

**方案：Django + DRF**（详见 `DESIGN.md`）

后端的**核心职责**是将现有推理链路包装为 HTTP 接口，**不重新实现推理逻辑**：

| 职责 | 对应现有代码 |
|---|---|
| 图片预处理 | 参考 `tools/demo_8param.py` 的 transform 逻辑 |
| 模型推理 | `training/semantic_distill/model.py` → `DistillParamModel` |
| ISP 渲染 | `models/isp_pipeline.py` → `LightroomISP` |
| 模型权重 | `checkpoints/distill_v4/best.pt`（默认）|
| 参数范围转换 | `training/fivek_8param/config.py` → `PARAM_RANGES` |

**模型加载策略**（关键，不可忽略）：

```python
# apps/inference/apps.py
class InferenceConfig(AppConfig):
    def ready(self):
        from .services.predictor import ModelPreloader
        ModelPreloader.load()  # 服务启动时加载，后续请求复用
```

### 5.3 文件存储

- 开发环境：本地 `MEDIA_ROOT`（`media/uploads/` + `media/outputs/`）
- 生产环境：MinIO / OSS / COS + CDN

---

## 6. 代码风格与架构

### 6.1 前端

- 使用 TypeScript + Composition API
- 页面层不写复杂业务逻辑
- 接口字段统一类型定义（对齐后端返回的参数字段名）
- 样式使用主题变量，不硬编码颜色

**命名约定**：

- 页面：`pages/result/result.vue`
- 组件：`components/CompareSlider/CompareSlider.vue`
- 服务：`services/inference.ts`
- 类型：`types/inference.ts`（字段名与后端 JSON 对齐）
- 状态：`stores/resultStore.ts`

### 6.2 后端

采用 **View / Serializer / Service / Model** 四层拆分（详见 `DESIGN.md`）：

- `views.py`：接请求、调 Service、返响应
- `serializers.py`：校验输入（文件格式、大小）、序列化输出
- `services/predictor.py`：封装模型推理，**复用现有推理代码**
- `services/isp.py`：封装 `models/isp_pipeline.py`
- `models.py`：`InferenceRecord` 落库

### 6.3 建议目录结构

```
前端 src/
  pages/
    index/          index.vue       # 选图/拍照入口
    processing/     processing.vue  # 推理等待页
    result/         result.vue      # 对比 + 参数展示
  components/
    CompareSlider/  CompareSlider.vue
    ParamPanel/     ParamPanel.vue  # 含置信度标注
    UploadEntry/    UploadEntry.vue
    LoadingState/   LoadingState.vue
    ErrorState/     ErrorState.vue
  services/
    api.ts          # 请求基础封装
    inference.ts    # 推理接口
    upload.ts       # 上传封装
  stores/
    app.ts
    resultStore.ts
  types/
    api.ts
    inference.ts    # 8 参数字段类型（与后端字段名 1:1 对齐）
  utils/
    image.ts
    permission.ts
    toast.ts
    error.ts
```

---

## 7. 限制条件与边界场景

### 7.1 模型能力边界

本版本使用的模型（Distill v4）在以下方面存在已知局限，**前端展示和用户文案需对齐**：

| 限制项 | 说明 | 前端处理建议 |
|---|---|---|
| 色温（WB）预测精度偏低 | MAE=618K，高于 Baseline 502K | 参数面板加"仅供参考"标注 |
| 训练域为 DSLR 照片 | 模型在 FiveK（DSLR RAW 转 JPEG）上训练，手机随拍照片存在域偏移 | 结果页加说明文案 |
| 部分图片增强效果不明显 | 已接近曝光/色彩最优的图片，参数预测趋近于 0 | 显示"当前照片已较理想"提示 |
| 专家一致性低的参数 | 高光/曝光/亮度专家分歧大（17-20%方差） | UI 用颜色区分置信度 |

### 7.2 技术限制

**小程序端**：
- 不支持端侧模型推理（PyTorch Mobile / ONNX Runtime 不可用于小程序）
- 权限受微信平台限制：保存相册、拍照均需明确授权
- 图片上传体积受小程序限制，建议前端压缩后上传

**网络**：
- 图片上传依赖网络质量，建议设置 30s 超时
- 弱网环境需提供超时提示和重试入口
- 云端推理延迟包含：上传 + 预处理 + 模型推理 + ISP 渲染 + 返回

### 7.3 边界场景处理

| 场景 | 处理方式 |
|---|---|
| 图片格式不支持（非 JPEG/PNG） | 前端选图时过滤，后端返回错误码 1002 |
| 图片超过 10MB | 前端压缩到 ≤ 5MB 后上传 |
| 图片尺寸过小（< 64px） | 后端校验，返回错误码 1004 |
| 模型推理失败 | 后端返回错误码 2001，前端显示"增强失败，请换张照片重试" |
| ISP 渲染失败 | 后端返回错误码 2002 |
| 用户拒绝相册权限 | 弹出引导说明，提示在系统设置中开启 |
| 用户中途退出 | Pinia 状态持久化最近一次成功结果（P1） |

### 7.4 结果预期边界

本产品输出的是 **AI 增强推荐结果**，需在 UI 文案中体现：

- 增强效果基于 MIT-Adobe FiveK 专业摄影师调参数据训练
- 饱和度参数置信度最高（专家一致性 97.8%），曝光/高光分歧较大
- 色温仅供参考，如有明显偏差建议手动微调
- 模型未针对夜景、强曝光过度、HDR 图片优化

---

## 8. 总结

### V1 最小闭环

**选图/拍照 → 上传 → 后端推理（Distill v4 + ISP Pipeline）→ 前后对比 → 8参数面板（含置信度）→ 保存结果**

### 实现原则

- **复用优先**：后端推理直接封装 `models/`、`training/` 和 `checkpoints/` 下的现有代码，不重写推理逻辑
- **模型预加载**：Django 启动时加载，禁止请求时动态加载权重
- **前端轻量**：页面少而清晰，优先保证闭环可跑通
- **展示诚实**：参数置信度、色温局限、域偏移风险，在 UI 文案中如实体现
- **演示导向**：推理耗时、模型版本、PSNR 参考值等信息保留在元信息面板，服务演示场景

