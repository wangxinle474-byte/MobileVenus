# MobileVenus 小程序版技术设计文档

> 版本: v2.1 | 日期: 2026-03-29 | 对应 PRD: v2.1

---

## 1. 技术栈

### 1.1 后端主框架

- **Django**：负责项目骨架、ORM、数据库迁移、后台管理（`makemigrations` / `migrate` 管理 schema，Django admin 用于推理记录查询和异常排查）
- **Django REST framework（DRF）**：负责小程序 API，提供 `Serializer` / `APIView` / `ViewSet` / Router，适合前后端分离接口开发

### 1.2 数据库

- **开发环境**：SQLite（内置，无需安装）
- **生产环境**：PostgreSQL（推荐，具备更好的 JSON 查询和并发能力，与 `parameters` / `meta_info` JSONField 配合更优）

### 1.3 推理与图像处理

- **PyTorch**：加载并运行 MobileVenus Distill 模型（`.pt` 权重）
- **Pillow**：图片读取、格式校验、尺寸检查、结果图保存
- **torchvision.transforms**：图片预处理（resize 224×224 + ImageNet 标准化）

> 核心原则：不在 view 层写任何推理逻辑，全部封装在 `services/` 层。

### 1.4 文件与对象存储

- **开发环境**：本地 `MEDIA_ROOT`（`media/uploads/` + `media/outputs/`）
- **生产环境**：MinIO / OSS / COS + CDN（通过 Django storage 后端切换，无需改业务代码）

### 1.5 前端小程序

- **uni-app + Vue 3 + TypeScript + Pinia**
- 前端只负责：图片输入、状态展示、结果展示与保存，不承载模型推理

---

## 2. 项目结构

采用 **"一个 Django Project + 多个业务 App"** 的结构，接口、推理、记录、用户职责解耦。

```
mobilevenus_backend/
  manage.py
  config/
    __init__.py
    settings/
      __init__.py
      base.py          # 公共配置（INSTALLED_APPS、媒体文件、CORS 等）
      dev.py           # 开发环境（SQLite、DEBUG=True、本地媒体）
      prod.py          # 生产环境（PostgreSQL、DEBUG=False、对象存储）
    urls.py
    wsgi.py
    asgi.py

  apps/
    inference/         # ★ 核心业务模块
      apps.py          # AppConfig.ready() 负责模型预加载
      models.py        # InferenceRecord
      admin.py
      serializers.py   # EnhanceImageRequestSerializer / EnhanceImageResponseSerializer
      views.py         # EnhanceImageAPIView / LatestResultAPIView / HealthAPIView / AestheticScoreAPIView
      urls.py
      services/
        predictor.py   # 封装 MobileVenus 模型推理（复用现有代码）
        isp.py         # 封装 models/isp_pipeline.py（ISP 渲染）
        preprocess.py  # 图片预处理（resize + 标准化）
        storage.py     # 文件保存与路径管理
        exceptions.py  # 业务异常定义
        aesthetic.py   # Venus-Q-Stage1 异步美学评分（后台线程，首次调用加载模型）

    users/             # 轻量用户模块（V1 可仅建表，不做完整登录）
      models.py        # MiniProgramUser
      admin.py
      serializers.py
      views.py
      urls.py

    common/            # 公共基础模块
      responses.py     # 统一响应构造 success() / error()
      exceptions.py    # 全局异常处理器
      permissions.py
      pagination.py
      utils.py

  media/
    uploads/           # 原图落盘目录
    outputs/           # 增强图落盘目录

  static/
    css/               # 自定义 CSS 文件（STATICFILES_DIRS 已配置）
    js/                # 自定义 JS 文件

  staticfiles/         # collectstatic 输出目录（生产部署用）

  requirements/
    base.txt           # django、djangorestframework、torch、Pillow、django-cors-headers
    dev.txt            # -r base.txt（SQLite 内置，无需额外依赖）
    prod.txt           # -r base.txt、psycopg2-binary、gunicorn
```

---

## 3. 模型组件与复用策略

后端推理服务**不重新实现模型**，直接封装研究项目中已有的推理链路。

### 3.1 现有代码映射

| service 层文件 | 复用的现有代码 | 说明 |
|---|---|---|
| `services/predictor.py` | `training/semantic_distill/model.py` → `DistillParamModel` | 模型定义与前向推理 |
| `services/aesthetic.py` | `pretrained_weights/Venus-Q-Stage1`（Qwen-VL-Chat 微调） | Venus 美学评分，后台线程异步执行 |
| `services/predictor.py` | `checkpoints/distill_v4/best.pt`（默认） | 模型权重 |
| `services/isp.py` | `models/isp_pipeline.py` → `LightroomISP` | 参数→增强图渲染 |
| `services/preprocess.py` | `tools/demo_8param.py` 中的 transform 逻辑 | resize 224×224 + ImageNet 标准化 |
| `services/predictor.py` | `training/fivek_8param/config.py` → `PARAM_RANGES` | 模型输出反归一化到物理范围 |

### 3.2 可用模型版本

三个版本在 Django 启动时**全部预加载**，请求到来时按 `model_version` 字段路由，切换无需等待模型重载。

| `model_version` 枚举值 | 权重路径 | PSNR | SSIM | 说明 |
|---|---|---|---|---|
| `baseline` | `checkpoints/fivek_8param/best.pt` | 32.05 dB | 0.9269 | 对照基准（无蒸馏） |
| `distill_v2` | `checkpoints/semantic_distill_v2/stage_b/best.pt` | **33.15 dB** | 0.9311 | 像素精度最优 |
| `distill_v4` | `checkpoints/distill_v4/best.pt` | 33.11 dB | **0.9326** | 感知质量最优（**默认**） |

`model_version` 缺省时默认 `distill_v4`，由 `EnhanceImageRequestSerializer` 在反序列化时填入默认值。

### 3.3 模型预加载（关键设计）

**禁止在每次 HTTP 请求时动态加载模型权重**，必须在服务启动时预加载：

```python
# apps/inference/apps.py
class InferenceConfig(AppConfig):
    name = "apps.inference"

    def ready(self):
        from .services.predictor import ModelRegistry
        ModelRegistry.load()   # Django 启动时执行，后续请求直接复用
```

```python
# apps/inference/services/predictor.py
CHECKPOINT_MAP = {
    "baseline":   "checkpoints/fivek_8param/best.pt",
    "distill_v2": "checkpoints/semantic_distill_v2/stage_b/best.pt",
    "distill_v4": "checkpoints/distill_v4/best.pt",
}

class ModelRegistry:
    _models = {}   # { model_version: model_instance }

    @classmethod
    def load(cls):
        from training.semantic_distill.model import DistillParamModel
        from training.fivek_8param.model import BaselineParamModel
        from django.conf import settings
        device = settings.INFERENCE_DEVICE
        for version, ckpt_path in CHECKPOINT_MAP.items():
            ModelClass = BaselineParamModel if version == "baseline" else DistillParamModel
            model = ModelClass(...)
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state["model_state_dict"])
            model.eval()
            cls._models[version] = model

    @classmethod
    def get(cls, model_version: str = "distill_v4"):
        if model_version not in cls._models:
            raise RuntimeError(f"Model '{model_version}' not loaded. Valid: {list(CHECKPOINT_MAP)}")
        return cls._models[model_version]

    @classmethod
    def loaded_versions(cls) -> list:
        return list(cls._models.keys())
```

---

## 4. 模块设计

### 4.1 inference 模块（核心）

#### 4.1.1 上传与校验

- 接收 `multipart/form-data` 的 `file` 字段（`request.FILES['file']`）
- 在 `EnhanceImageRequestSerializer` 中完成：格式校验（JPEG/PNG）、文件大小（≤ 10MB）、Pillow 解析验证
- 图片尺寸过小（< 64px）在 `preprocess.py` 中捕获

#### 4.1.2 推理服务（service 层核心）

五个 service 文件各司其职：

| 文件 | 职责 |
|---|---|
| `preprocess.py` | Pillow 读图 → resize 224×224 → 转 tensor → ImageNet 标准化 |
| `predictor.py` | 调用 `ModelRegistry.get()` 推理 → 输出归一化参数 → 反归一化到物理范围 |
| `isp.py` | 调用 `LightroomISP`，将原图（全分辨率）+8参数 → 渲染增强图 |
| `storage.py` | 原图/增强图落盘，返回相对路径；生产环境切换为对象存储 |
| `exceptions.py` | 定义 `InferenceFailedException` / `ISPRenderException` / `StorageException` |
| `aesthetic.py` | Venus-Q-Stage1 异步美学评分 |

#### 4.1.3 记录管理

`InferenceRecord` 记录每次推理的完整生命周期，支持 Django admin 按 `status` / `created_at` 筛选和异常排查。

#### 4.1.4 管理后台

Django admin 注册 `InferenceRecord`，支持：
- 按 `status=failed` 筛选异常记录
- 查看 `error_code`、`error_message`、原图、创建时间
- 定位问题来源（输入校验 / 模型推理 / ISP 渲染 / 存储）

---

### 4.2 common 模块

- `responses.py`：统一构造 `{"code": 0, "message": "success", "data": ...}` 和错误响应
- `exceptions.py`：DRF 全局异常处理器，将业务异常统一转换为标准响应格式
- `utils.py`：请求 ID 生成（`uuid4`）、时间戳工具等

---

### 4.3 users 模块（V1 轻量）

V1 只建 `MiniProgramUser` 表，用于关联推理记录，不做完整登录体系。`client_id` 字段可作为临时用户标识，后续接入微信 openid 时无需改数据模型。

---

## 5. 接口设计

V1 采用**同步接口**，闭环跑通优先。若推理耗时超出可接受范围，再升级为异步任务模型。

### 5.1 上传并增强图片

**POST** `/api/v1/inference/enhance/`

**请求**：`multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | File | ✅ | 图片文件（JPEG/PNG，≤ 10MB） |
| `model_version` | string | ❌ | 模型版本：`baseline` / `distill_v2` / `distill_v4`，缺省默认 `distill_v4` |
| `client_id` | string | ❌ | 客户端标识，用于关联最近结果 |

**成功响应**：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "request_id": "req_20260328_a1b2c3",
    "original_image_url": "/media/uploads/2026/03/28/xxx.jpg",
    "enhanced_image_url": "/media/outputs/2026/03/28/xxx_enhanced.jpg",
    "parameters": {
      "ev_compensation": -0.14,
      "white_balance": 5376,
      "contrast": 7.22,
      "brightness": 3.43,
      "shadows": 6.30,
      "highlights": 8.13,
      "saturation": 1.74,
      "vibrance": 6.97
    },
    "parameter_confidence": {
      "ev_compensation": "medium",
      "white_balance": "reference_only",
      "contrast": "medium",
      "brightness": "medium",
      "shadows": "high",
      "highlights": "medium",
      "saturation": "high",
      "vibrance": "high"
    },
    "meta_info": {
      "model_name": "MobileVenus Distill v4",
      "backbone": "MobileViT-Small (1.93M params)",
      "pipeline": "SemanticDistill + Gamma-aware ISP",
      "inference_time_ms": 412,
      "image_width": 1280,
      "image_height": 1940,
      "psnr_reference": "33.11 dB (FiveK val 500张)"
    },
    "aesthetic_status": "running",
    "aesthetic_before": null,
    "aesthetic_after": null
  }
}
```

> **字段说明**：
> - `parameters` 字段名与研究项目 `training/fivek_8param/config.py` 中 `PARAM_NAMES` 保持一致
> - `parameter_confidence` 基于 `data/fivek_expert_consensus.json` 的专家一致性数据静态生成，由 service 层附加，不需要每次推理重新计算
> - `white_balance` 的 `confidence` 固定为 `reference_only`（已知 MAE=618K 偏高）

**实现**：`EnhanceImageAPIView(APIView)`，手写 `post()` 方法，不走 ViewSet。

---

### 5.2 Venus 美学评分轮询

**GET** `/api/v1/inference/aesthetic/<request_id>/`

前端在推理完成后每 6s 轮询此端点，直到 `aesthetic_status` 进入终态。

| 字段 | 类型 | 说明 |
|---|---|---|
| `aesthetic_status` | string | `pending` / `running` / `done` / `failed` / `unavailable` |
| `aesthetic_before` | object\|null | 原图 Venus 评分 `{overall, dimensions:{composition,lighting,color,clarity,subject}}` |
| `aesthetic_after` | object\|null | 增强图 Venus 评分（结构同上） |

**实现**：`AestheticScoreAPIView(APIView)`，后台 `threading.Thread` 在 enhance 完成后异步运行 Venus-Q-Stage1 打分，结果写回 `InferenceRecord`；轮询端点直接读 DB。

---

### 5.3 照片结果库

> 对 `InferenceRecord` 中 `status=success` 的记录提供独立浏览接口；图片二进制直接从 DB 返回，无需文件系统。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/photos/` | 成功推理列表，支持 `client_id` / `limit` / `offset` |
| GET | `/api/v1/photos/<pk>/` | 单条详情（含参数、meta_info、aesthetic 字段） |
| GET | `/api/v1/photos/<pk>/original/` | 原图二进制（`image/jpeg`，直接从 DB 读取） |
| GET | `/api/v1/photos/<pk>/enhanced/` | 增强图二进制（`image/jpeg`，直接从 DB 读取） |

**实现**：`PhotoListAPIView` / `PhotoDetailAPIView` / `PhotoImageAPIView`，均查询 `InferenceRecord`，无中间表。

---

### 5.4 查询最近一次推理结果

**GET** `/api/v1/inference/latest/`

| 参数 | 类型 | 说明 |
|---|---|---|
| `client_id` | query string | 按客户端标识查询 |

- 返回该 `client_id` 最近一次 `status=success` 的记录
- 用于小程序刷新结果页时恢复数据（对应 PRD P1 缓存需求）

**实现**：`LatestResultAPIView(APIView)`


### 5.5 健康检查

**GET** `/api/v1/health/`

```json
{
  "code": 0,
  "message": "ok",
  "data": {
    "models_loaded": ["baseline", "distill_v2", "distill_v4"],
    "default_version": "distill_v4"
  }
}
```

`models_loaded` 由 `ModelRegistry.loaded_versions()` 实时返回，可验证三个版本是否全部预加载成功，便于部署时快速排查启动异常。

---

## 6. 数据模型

### 6.1 用户表

```python
class MiniProgramUser(models.Model):
    client_id = models.CharField(max_length=64, unique=True)   # 前端设备标识
    openid = models.CharField(max_length=64, blank=True, default="")  # 微信 openid（V2 接入）
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "mini_program_user"
```

### 6.2 推理记录表

```python
class InferenceRecord(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("success", "Success"),
        ("failed", "Failed"),
    ]

    AESTHETIC_STATUS = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("done", "Done"),
        ("failed", "Failed"),
        ("unavailable", "Unavailable"),
    ]

    request_id = models.CharField(max_length=64, unique=True)
    user = models.ForeignKey(
        MiniProgramUser,
        null=True, blank=True,
        on_delete=models.SET_NULL,
        related_name="records"
    )

    original_image = models.ImageField(upload_to="uploads/%Y/%m/%d/")
    enhanced_image = models.ImageField(upload_to="outputs/%Y/%m/%d/", null=True, blank=True)

    original_image_data = models.BinaryField(null=True, blank=True)   # 原图二进制（入库，不依赖文件系统）
    enhanced_image_data = models.BinaryField(null=True, blank=True)   # 增强图二进制

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="pending")

    parameters = models.JSONField(null=True, blank=True)         # 8 个 Lightroom 参数
    meta_info = models.JSONField(null=True, blank=True)          # 模型版本、推理耗时等
    model_version = models.CharField(max_length=32, default="distill_v4")  # 记录使用的模型版本

    aesthetic_status = models.CharField(
        max_length=16, choices=AESTHETIC_STATUS, default="pending"
    )
    aesthetic_before = models.JSONField(null=True, blank=True)   # Venus 原图评分
    aesthetic_after = models.JSONField(null=True, blank=True)    # Venus 增强图评分

    error_code = models.CharField(max_length=16, blank=True, default="")
    error_message = models.CharField(max_length=255, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "inference_record"
        ordering = ["-created_at"]
```

> **说明**：
> - `model_version` 字段记录实际使用的权重版本，便于日后对比 v2 / v4 效果差异
> - `aesthetic_status` 跟踪 Venus 异步评分状态：enhance 接口返回时为 `running`，后台线程完成后更新为 `done` / `failed` / `unavailable`
> - `aesthetic_before/after` 存储 Venus-Q-Stage1 的 6 维评分（overall + composition/lighting/color/clarity/subject）
> - `parameters` 的 key 与接口响应中 `parameters` 字段名完全一致
> - `meta_info` 存储推理耗时、图片尺寸等，直接用于接口响应组装
> - `original_image_data` / `enhanced_image_data`：图片二进制直接入库，enhance 成功时同步写入，使图片可脱离文件系统访问

---

## 7. 状态管理

### 7.1 后端任务状态

| 状态 | 触发时机 |
|---|---|
| `pending` | `InferenceRecord` 创建时 |
| `success` | ISP 渲染完成、增强图和参数写入后 |
| `failed` | 任意步骤异常（校验 / 推理 / ISP / 存储） |

即使 V1 是同步接口，状态字段提前建好，后续升级异步任务时数据模型无需改动。

### 7.2 前端页面状态

| 状态 | 触发时机 |
|---|---|
| `idle` | 初始进入页面 |
| `uploading` | 前端开始上传图片 |
| `processing` | 请求已发出，等待后端响应 |
| `success` | 接口返回 `code=0` |
| `error` | 接口返回非 0 错误码 / 网络超时 |

---

## 8. 关键流程

### 8.1 主流程：选图 → 推理 → 展示

```
[前端]                          [后端 Django]
  │                                  │
  ├─ 选图/拍照                        │
  ├─ multipart/form-data 上传 ──────► │
  │                                  ├─ EnhanceImageRequestSerializer 校验
  │                                  ├─ 创建 InferenceRecord(status='pending')
  │                                  ├─ storage.save_original(file) → 原图路径
  │                                  ├─ preprocess.run(image_path) → tensor
  │                                  │     └─ 复用 demo_8param.py transform 逻辑
  │                                  ├─ predictor.run(tensor) → 8 参数（归一化）
  │                                  │     └─ ModelRegistry.get().forward()
  │                                  │     └─ PARAM_RANGES 反归一化
  │                                  ├─ isp.render(original_image, params) → 增强图
  │                                  │     └─ LightroomISP（models/isp_pipeline.py）
  │                                  ├─ storage.save_enhanced(enhanced_image)
  │                                  ├─ 更新 InferenceRecord(status='success', ...)
  │                                  ├─ 附加 parameter_confidence（静态 JSON）
  │                                  └─ 返回标准响应 JSON
  │ ◄──────────────────────────────── │
  ├─ 进入 result 页
  ├─ 展示滑动对比 + 参数面板
```

### 8.2 保存到相册流程

1. 前端拿到 `enhanced_image_url`
2. `uni.downloadFile()` 下载到临时路径
3. 调用 `uni.authorize({ scope: 'scope.writePhotosAlbum' })` 申请权限
4. `uni.saveImageToPhotosAlbum()` 保存
5. 成功/失败给出 Toast 提示

后端只需保证 `enhanced_image_url` 可公开访问（开发环境 Django 的 `MEDIA_URL` 直接服务）。

### 8.3 后台排查流程

1. 登录 Django admin → `InferenceRecord`
2. 按 `status=failed` 筛选
3. 查看 `error_code` + `error_message` + 原图
4. 定位是哪一层出错（错误码前缀 1xxx=输入，2xxx=推理，3xxx=存储）

---

## 9. 异常处理

### 9.1 错误码体系

| 错误码 | 含义 | 触发层 |
|---|---|---|
| `1001` | 文件缺失 | Serializer |
| `1002` | 格式非法（非 JPEG/PNG） | Serializer |
| `1003` | 文件过大（> 10MB） | Serializer |
| `1004` | 图片解析失败（Pillow 无法读取） | preprocess.py |
| `1005` | 图片尺寸过小（< 64px） | preprocess.py |
| `2001` | 模型推理失败 | predictor.py |
| `2002` | ISP 渲染失败 | isp.py |
| `3001` | 文件存储失败 | storage.py |

### 9.2 各层处理策略

**输入层（Serializer）**：
- 在 `validate_file()` 方法中完成格式、大小校验
- 抛出 `ValidationError`，DRF 自动转换为 400 响应

**Service 层**：
- 各 service 抛出 `exceptions.py` 中定义的业务异常
- 不在 service 中 try/except 后静默处理，让 view 层统一决策

**View 层**：
- `try/except` 捕获业务异常
- 更新 `InferenceRecord.status='failed'`，写入 `error_code` / `error_message`
- 调用 `common.responses.error()` 返回标准错误响应

**全局**：
- `common/exceptions.py` 注册为 DRF `EXCEPTION_HANDLER`，兜底处理未预期异常

### 9.3 统一响应格式

成功：
```json
{"code": 0, "message": "success", "data": { ... }}
```

失败：
```json
{"code": 2001, "message": "model inference failed", "data": null}
```

### 9.4 事务一致性

"创建记录 + 更新结果" 使用 `django.db.transaction.atomic()` 包裹，确保要么完整成功，要么回滚到 `pending`：

```python
with transaction.atomic():
    record = InferenceRecord.objects.create(...)
    # ... 推理 ...
    record.status = "success"
    record.save()
```

---

## 10. 代码风格与命名规范

### 10.1 架构模式

**View / Serializer / Service / Model 四层拆分**：

| 层 | 职责 | 不应做的事 |
|---|---|---|
| View / APIView | 接请求、调 Service、返响应 | 不写推理细节，不拼复杂返回结构 |
| Serializer | 校验输入、序列化输出 | 不写耗时业务逻辑 |
| Service | 模型推理、ISP 渲染、文件存储 | 不直接操作 HTTP request/response |
| Model | 数据落库、状态流转 | 不写业务逻辑 |

### 10.2 DRF 选型

- **推理接口** → `APIView`（非标准 CRUD，手写 `post()`）
- **最近结果接口** → `APIView`
- **历史记录接口（V2）** → `ReadOnlyModelViewSet` + Router

### 10.3 命名规范

| 类型 | 命名示例 |
|---|---|
| Model | `InferenceRecord` |
| Serializer（请求） | `EnhanceImageRequestSerializer` |
| Serializer（响应） | `EnhanceImageResponseSerializer` |
| View | `EnhanceImageAPIView` |
| Service | `ModelRegistry`、`ISPRenderer`、`ImagePreprocessor` |
| Exception | `InferenceFailedException`、`ISPRenderException` |