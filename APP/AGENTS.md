# AGENTS.md

> 版本: v2.1 | 日期: 2026-03-29 | 对应 PRD: v2.1 / DESIGN: v2.1

## 项目说明
这是一个微信小程序版 AI 图像增强项目（MobileVenus），当前只做 MVP。
目标是在小程序内完成最小闭环：

**选择模型 → 拍照/选图 → 上传 → 云端推理 → 前后对比 → 参数展示 → 保存结果**

当前版本定位是**CVPR 演示入口**，核心价值是让评委在手机上直接感受语义蒸馏带来的 +1.10 dB 画质提升，以及在不同模型版本间即时对比。不是完整修图 App，也不是实时相机系统。

---

## 技术基线

### 前端
- uni-app
- Vue 3
- TypeScript
- Pinia

### 后端
- Django
- Django REST framework（DRF）
- 开发环境数据库：SQLite
- 生产环境数据库：PostgreSQL
- 模型推理：PyTorch（`.pt` 权重，不引入 ONNX Runtime）
- 图像预处理：Pillow + torchvision.transforms（不引入 OpenCV）

### 文件存储
- 开发环境：本地 `MEDIA_ROOT`
- 生产环境：MinIO / OSS / COS + CDN

---

## 命令说明

### 前端
- 安装依赖：`npm install`
- 本地开发：优先按项目现有脚本执行；若无统一脚本，使用 HBuilderX / 微信开发者工具运行 uni-app 小程序项目
- 小程序调试：使用微信开发者工具打开编译后的项目

### 后端
- 安装依赖：`pip install -r requirements/dev.txt`
- 数据库迁移：
  - `python manage.py makemigrations`
  - `python manage.py migrate`
- 启动开发服务：`python manage.py runserver 127.0.0.1:8002 --settings=config.settings.dev`
- 创建后台管理员（如需要）：`python manage.py createsuperuser`

### 其他
- 若项目中已存在脚本或 Makefile，优先复用现有命令
- 不要擅自引入新的运行方式，除非当前方式不可用

---

## 当前 MVP 范围

P0 功能（必须完成）：

- 拍照 / 相册选图
- 图片上传到后端
- 云端推理（后端复用现有模型代码，不重新实现）
- 返回增强图并展示
- 原图 / 增强图滑动对比
- 8 参数面板展示（含置信度分组）
- 保存到相册
- 加载中 / 失败态提示

P1 功能（明确要求时实现）：

- 推理耗时展示
- 参数折叠 / 展开
- 最近一次结果缓存
- **模型选择（Baseline / Distill v2 / Distill v4）**

P2 功能（默认不做）：

- 增强强度滑杆
- 参数手动微调
- 历史记录列表

---

## 编码约束

### 总原则
- 优先最小修改
- 不要重构无关代码
- 不要改动未被本轮需求覆盖的模块
- 命名清晰、语义明确
- 优先复用现有结构与工具函数
- 所有改动都要服务于当前 MVP 闭环

### 前端约束
- 使用 TypeScript
- 使用 Composition API
- 页面层不写复杂业务逻辑，页面展示、流程控制、接口调用分层
- 组件命名清晰，避免单字母命名
- 接口字段必须统一类型定义（`types/api.ts`）
- 样式优先复用主题变量，不写硬编码颜色
- 图片对比、参数面板、上传入口、模型选择器、加载态、错误态拆成独立组件
- 模型选择状态存入 Pinia store，跨页面不丢失，请求时随 `model_version` 字段一起上传

### 后端约束
- 采用 **View / Serializer / Service / Model 四层拆分**：
  - View / APIView：接请求、调服务、返响应
  - Serializer：校验输入、序列化输出
  - Service：业务逻辑与模型推理（`predictor.py` / `isp.py` / `preprocess.py` / `storage.py`）
  - Model：数据落库与状态管理
- 不在 View 中写模型推理细节
- 不在 Serializer 中写耗时业务逻辑
- 不直接在 View 中拼复杂返回结构
- 业务异常统一在 `services/exceptions.py` 定义
- **模型必须在 `AppConfig.ready()` 阶段预加载，禁止每次请求重新加载权重**
- 后端不重新实现模型推理，直接复用：
  - `training/semantic_distill/model.py` → `DistillParamModel`
  - `models/isp_pipeline.py` → `LightroomISP`
  - `tools/demo_8param.py` 中的 transform 逻辑
  - `training/fivek_8param/config.py` → `PARAM_RANGES`
- 统一响应格式：
  - 成功：`{ "code": 0, "message": "success", "data": ... }`
  - 失败：`{ "code": <error_code>, "message": <error_message>, "data": null }`

### 数据与状态约束
- 推理记录状态统一使用：`pending / success / failed`
- 美学评分状态统一使用：`pending / running / done / failed / unavailable`
- 前端页面状态统一使用：`idle / uploading / processing / success / error`
- `InferenceRecord` 中的 `model_version` 字段必须记录实际使用的模型版本
- `parameters` 字段的 key 使用：`ev_compensation / white_balance / contrast / brightness / shadows / highlights / saturation / vibrance`
- 涉及"创建记录 + 更新结果"的关键链路，使用 `transaction.atomic()` 保证事务一致性

---

## 推荐项目结构

### 前端
```text
src/
  pages/
    index/
    processing/
    result/
  components/
    CompareSlider/
    ParamPanel/
    ModelSelector/      ← 模型选择器组件（P1）
    UploadEntry/
    LoadingState/
    ErrorState/
  services/
    api.ts
    inference.ts
    upload.ts
  stores/
    app.ts
    resultStore.ts
    modelStore.ts       ← 模型选择状态（P1）
  types/
    api.ts
    inference.ts
  utils/
    image.ts
    permission.ts
    toast.ts
    error.ts
```

### 后端
```text
mobilevenus_backend/
  manage.py
  config/
    settings/
      base.py
      dev.py
      prod.py
    urls.py
    wsgi.py
    asgi.py

  apps/
    inference/
      apps.py           ← AppConfig.ready() 预加载模型
      models.py         ← InferenceRecord
      admin.py
      serializers.py
      views.py
      urls.py
      services/
        preprocess.py   ← 复用 demo_8param.py transform 逻辑
        predictor.py    ← ModelRegistry，封装三版本模型
        isp.py          ← 封装 models/isp_pipeline.py
        storage.py
        exceptions.py
        aesthetic.py    ← Venus-Q-Stage1 异步美学评分服务

    users/
      models.py
      admin.py
      serializers.py
      views.py
      urls.py

    common/
      responses.py
      exceptions.py
      permissions.py
      pagination.py
      utils.py

  static/
    css/              ← 自定义 CSS（STATICFILES_DIRS 已配置）
    js/               ← 自定义 JS

  media/
    uploads/
    outputs/

  requirements/
    base.txt
    dev.txt
    prod.txt
```

---

## 接口与流程约束

### 当前接口列表

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/inference/enhance/` | 上传并推理增强 |
| GET | `/api/v1/inference/latest/` | 查询最近一次结果 |
| GET | `/api/v1/inference/aesthetic/<request_id>/` | 轮询 Venus 美学评分状态与结果 |
| GET | `/api/v1/photos/` | 成功推理结果列表（支持 client_id / limit / offset） |
| GET | `/api/v1/photos/<pk>/` | 单条推理结果详情 |
| GET | `/api/v1/photos/<pk>/original/` | 原图二进制（image/jpeg，直接从 DB 返回） |
| GET | `/api/v1/photos/<pk>/enhanced/` | 增强图二进制（image/jpeg，直接从 DB 返回） |
| GET | `/api/v1/health/` | 服务与模型健康检查 |

### 主接口参数（`POST /api/v1/inference/enhance/`）

- 请求类型：`multipart/form-data`
- `file`：必填，图片文件（JPEG/PNG，≤ 10MB）
- `model_version`：选填，枚举值 `baseline` / `distill_v2` / `distill_v4`，缺省默认 `distill_v4`
- `client_id`：选填，客户端标识

### 可用模型版本（`model_version` 枚举）

| 值 | 权重路径 | PSNR | SSIM |
|---|---|---|---|
| `baseline` | `checkpoints/fivek_8param/best.pt` | 32.05 dB | 0.9269 |
| `distill_v2` | `checkpoints/semantic_distill_v2/stage_b/best.pt` | 33.15 dB | 0.9311 |
| `distill_v4` | `checkpoints/distill_v4/best.pt` | 33.11 dB | 0.9326 |

三个版本必须在 Django 启动时**全部预加载**，请求按 `model_version` 路由，不重新加载。

### 主流程要求
1. 小程序端选择模型版本（默认 Distill v4）
2. 拍照或从相册选图
3. 前端携带 `model_version` 上传图片
4. Django 接收文件，`EnhanceImageRequestSerializer` 校验
5. 创建 `InferenceRecord(status='pending', model_version=...)`
6. `storage.save_original()` 保存原图
7. `preprocess.run()` → tensor（resize 224×224 + 标准化）
8. `predictor.run(tensor, model_version)` → 8 个归一化参数 → 反归一化
9. `isp.render(original_image, params)` → 增强图（`LightroomISP`）
10. `storage.save_enhanced()` 保存增强图
11. 更新 `InferenceRecord(status='success', ...)`
12. 返回增强图 URL + 8 参数 + `parameter_confidence` + `meta_info`
13. 前端进入结果页，展示滑动对比 + 参数面板

### 保存结果要求
- 结果页负责通过 `uni.saveImageToPhotosAlbum()` 保存到相册
- 需要处理 `scope.writePhotosAlbum` 权限申请与失败提示
- 后端只需保证增强图 URL 可访问

---

## 范围边界
本轮不要擅自增加以下内容：

- 登录 / 注册 / 用户中心
- 完整账号体系
- 多端同步
- 数据统计后台
- **端侧本地模型推理**（小程序不支持 PyTorch Mobile，严禁引入）
- 复杂实时滤镜预览
- 专业级相机参数接管
- 批量处理
- 完整修图工作台
- 与当前 MVP 无关的页面美化重构

如果需求没有明确要求，默认也不要新增：
- 历史记录列表
- 参数手动微调
- 多风格模式
- 增强强度滑杆（P2，明确要求前不做）
- 社区分享
- 消息通知

---

## 异常处理要求

### 前端必须处理
- 用户取消选图
- 拍照 / 相册授权被拒绝
- 上传超时
- 推理失败（含模型版本不支持）
- 图片保存失败
- 页面中途退出导致状态丢失

### 后端必须处理
- 文件缺失
- 文件格式非法（非 JPEG/PNG）
- 图片解析失败（Pillow 无法读取）
- 图片过大（> 10MB）
- 图片尺寸过小（< 64px）
- 模型未预加载 / 加载失败
- 推理报错
- ISP 渲染失败
- 文件存储失败
- 数据库写入失败

### 错误码（完整列表）

| 错误码 | 含义 | 触发层 |
|---|---|---|
| `1001` | 文件缺失 | Serializer |
| `1002` | 格式非法 | Serializer |
| `1003` | 文件过大 | Serializer |
| `1004` | 图片解析失败 | preprocess.py |
| `1005` | 图片尺寸过小 | preprocess.py |
| `2001` | 模型推理失败 | predictor.py |
| `2002` | ISP 渲染失败 | isp.py |
| `3001` | 存储失败 | storage.py |

---

## 测试要求
修改后至少检查：

### 前端
- 能否正常拍照或从相册选图
- 模型选择器切换是否正确更新 store 并在请求中携带 `model_version`
- 上传中、处理中、成功、失败状态是否正常切换
- 原图 / 增强图对比是否正常
- 参数面板是否正常展示（含置信度标注、WB"仅参考"标识）
- 保存到相册是否正常
- 权限拒绝时是否有明确提示

### 后端
- `enhance` 接口能否正常接收 `multipart/form-data`
- `model_version` 缺省时是否默认 `distill_v4`
- `model_version` 传入非法值时是否返回正确错误
- Django 启动后 `/api/v1/health/` 的 `models_loaded` 是否返回三个版本
- 三个模型版本各自推理是否正常
- 推理成功时是否正确写入 `InferenceRecord` 并返回结果
- 推理失败时是否正确写入 `status=failed`
- 统一响应结构是否一致
- 数据库迁移后 `model_version` 字段是否符合设计

### 联调
- 小程序与 Django 接口联调是否正常
- 切换模型版本后推理结果是否有可感知差异（Baseline vs Distill v4 最明显）
- 弱网 / 超时 / 错图输入时页面是否可恢复
- 结果页刷新后是否符合当前设计预期

---

## 交付格式
每次完成修改后，请说明：

- 改了哪些文件
- 做了什么修改
- 为什么这样改
- 是否涉及接口、数据模型、状态流变化
- 可能影响哪些地方
- 建议如何测试

如果有未完成项，也必须明确说明：
- 还差什么
- 为什么没做
- 下一步建议是什么

---

## 文档同步
每次完成修改后，如果改动影响接口、状态、数据模型、目录结构或范围边界，需同步更新：
- `APP/PRD.md`
- `APP/DESIGN.md`
- `APP/AGENTS.md`（本文件）

---

## 决策优先级
当实现方案有冲突时，按以下优先级决策：

1. 当前 MVP 范围（P0 优先）
2. 小程序主闭环是否能跑通
3. 不破坏现有研究代码结构（`models/`、`training/`、`checkpoints/`）
4. 代码清晰与可维护性
5. 后续扩展性

默认优先选择更简单、更稳定、更容易联调、更符合当前设计文档的方案。
