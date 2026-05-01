# TASK.md

> 每次完成修改后必须填写本文件，记录变更上下文，便于追溯和交接。

---

## 交付格式

每次完成修改后，请说明：

- **改了哪些文件**
- **做了什么修改**
- **为什么这样改**
- **是否涉及接口、数据模型、状态流变化**
- **可能影响哪些地方**
- **建议如何测试**

如果有未完成项，也必须明确说明：

- **还差什么**
- **为什么没做**
- **下一步建议是什么**

---

## 变更记录

---

### [2026-03-29] Venus 异步美学评分集成

#### 改了哪些文件

**后端**
- `apps/inference/models.py` — 新增 `aesthetic_status`、`aesthetic_before`、`aesthetic_after` 字段
- `apps/inference/migrations/0002_inferencerecord_aesthetic_after_and_more.py` — 对应数据库迁移
- `apps/inference/views.py` — 移除同步 Venus 打分，改为后台线程；新增 `AestheticScoreAPIView`
- `apps/inference/urls.py` — 注册 `/inference/aesthetic/<request_id>/` 路由
- `apps/inference/services/aesthetic.py` — 重写为 Venus-Q-Stage1 打分逻辑（替换原 AADB 方案）

**前端**
- `src/types/api.ts` — `EnhanceResult` 新增 `aesthetic_status/before/after`；新增 `AestheticScore`、`AestheticDimensions` 类型
- `src/services/api.ts` — 新增 `AestheticPollResult` 接口和 `getAestheticScore()` 函数
- `src/stores/resultStore.ts` — 新增 `patchAesthetic()` 方法（直接属性赋值，避免 spread 兼容问题）
- `src/pages/result/index.vue` — 新增 Venus 美学评分卡片、轮询逻辑、loading/done/failed 状态展示

**文档**
- `APP/PRD.md`、`APP/DESIGN.md`、`APP/AGENTS.md` — 同步更新至 v2.1

#### 做了什么修改

- `enhance` 接口现在立即返回（不等 Venus 打分），响应中 `aesthetic_status` 初始为 `"running"`
- 后台 `threading.Thread` 异步执行 Venus-Q-Stage1 打分，完成后将结果写入 `InferenceRecord`
- 前端结果页每 6s 轮询 `GET /inference/aesthetic/<id>/`，评分就绪后自动更新卡片

#### 为什么这样改

Venus-Q-Stage1 模型约 20GB，同步打分会阻塞 enhance 接口数十秒；改为异步后接口响应不受影响，评分结果通过轮询独立获取。

#### 涉及接口、数据模型、状态流变化

- **新增接口**：`GET /api/v1/inference/aesthetic/<request_id>/`
- **数据模型变化**：`InferenceRecord` 新增三字段（已迁移）
- **状态流新增**：`aesthetic_status`: `pending → running → done / failed / unavailable`
- **enhance 响应体变化**：新增 `aesthetic_status`、`aesthetic_before`、`aesthetic_after` 三字段

#### 可能影响哪些地方

- Django admin 展示 `InferenceRecord` 时会多出三列（无破坏性）
- `/inference/latest/` 返回的记录也含新字段（前端需兼容）
- 若 Venus 模型不可用，`aesthetic_status` 最终为 `"unavailable"`，前端显示"评分暂不可用"

#### 建议如何测试

1. 发起推理，DevTools Network 确认 enhance 响应含 `aesthetic_status: "running"`
2. 观察结果页 Venus 评分卡显示"正在评分…"
3. 等待约 6s 后轮询响应，确认状态切换到 `done`（有 Venus 模型时）或 `unavailable`（无模型时）
4. 后端日志确认 aesthetic 线程启动和完成记录

#### 未完成项

- **Venus 打分本机不可用**：本地内存不足以加载 20GB 模型，`aesthetic_status` 最终为 `"unavailable"`；需部署到 GPU 服务器才能验证 `done` 状态完整流程
- **下一步建议**：服务器部署后设置 `VENUS_MODEL_PATH` 环境变量，指向 `pretrained_weights/Venus-Q-Stage1`，重启 Django 后重新测试完整评分流程

---

### [2026-03-29] 照片数据库（PhotoRecord）

#### 改了哪些文件

- `apps/inference/models.py` — 新增 `PhotoRecord` 模型（OneToOne → `InferenceRecord`，含 `title`、`saved_at`）
- `apps/inference/migrations/0003_photo_record.py` — 数据库迁移（自动生成并已应用）
- `apps/inference/admin.py` — 注册 `PhotoRecordAdmin`，支持按 `model_version` 筛选、按 `request_id` 搜索
- `apps/inference/views.py` — enhance 成功事务内自动 `get_or_create` PhotoRecord；新增 `PhotoListAPIView`、`PhotoDetailAPIView`、`_serialize_photo()`
- `apps/inference/urls.py` — 注册 `GET /photos/` 和 `GET /photos/<pk>/` 路由

#### 做了什么修改

- 每次推理成功时，在同一 `transaction.atomic()` 内自动创建 `PhotoRecord`，与 `InferenceRecord` 一对一关联
- `GET /api/v1/photos/` 支持 `client_id` / `limit` / `offset` 分页参数，返回 `{total, results: [...]}`
- `GET /api/v1/photos/<pk>/` 返回单条详情，字段含原图/增强图 URL、参数、meta_info、aesthetic 评分

#### 为什么这样改

`InferenceRecord` 承载完整推理过程（含失败记录），`PhotoRecord` 专门对应成功结果，作为照片库独立入口，便于后续扩展（收藏、备注、历史列表页）。

#### 涉及接口、数据模型、状态流变化

- **新增数据表**：`photo_record`（字段：`id`、`inference_record_id`、`title`、`saved_at`）
- **新增接口**：`GET /api/v1/photos/` 和 `GET /api/v1/photos/<pk>/`
- **enhance 流程变化**：成功事务中额外插入 `photo_record`，对响应体无影响

#### 可能影响哪些地方

- 历史成功推理记录**不会**自动补录 `PhotoRecord`，仅新增推理才触发；如需补全可在 shell 中手动执行
- Django admin 新增 `Photo Records` 管理面板

#### 建议如何测试

1. 发起一次推理，完成后 `GET /api/v1/photos/` 确认 `total >= 1`
2. 用返回的 `id` 访问 `GET /api/v1/photos/<id>/` 确认字段完整
3. 登录 `http://127.0.0.1:8000/admin/` → `Photo Records` 确认记录可见且关联正确

#### 未完成项

无

---

### [2026-03-29] 图片二进制迁移至 InferenceRecord，删除 PhotoRecord

#### 改了哪些文件

- `apps/inference/models.py` — `InferenceRecord` 新增 `original_image_data`、`enhanced_image_data`（`BinaryField`）；删除 `PhotoRecord` 类
- `apps/inference/migrations/0004_photo_record_image_data.py` — 为 `PhotoRecord` 添加字段（已废弃，随 0005 一并清理）
- `apps/inference/migrations/0005_remove_photo_record_add_image_data.py` — 向 `InferenceRecord` 添加两个 `BinaryField`，删除 `photo_record` 表
- `apps/inference/admin.py` — 移除 `PhotoRecordAdmin`，只保留 `InferenceRecordAdmin`
- `apps/inference/views.py` — enhance 成功时将图片二进制写入 `InferenceRecord`；`PhotoListAPIView`/`PhotoDetailAPIView`/`PhotoImageAPIView` 改为直接查询 `InferenceRecord`；`_serialize_photo` 重命名为 `_serialize_record`
- `apps/inference/urls.py` — 移除 `PhotoImageAPIView` 单独导入（已合并）

#### 做了什么修改

- `photo_record` 表已彻底删除
- 图片原始二进制（原图 + 增强图）直接存入 `InferenceRecord.original_image_data` / `enhanced_image_data`
- `GET /api/v1/photos/` 直接查询 `InferenceRecord.objects.filter(status='success')`，无中间表
- `GET /api/v1/photos/<id>/original/` 和 `/enhanced/` 从 `InferenceRecord` 读取二进制返回 `image/jpeg`

#### 为什么这样改

用户确认 `InferenceRecord` 就是照片存储的目标，`PhotoRecord` 作为中间层反而增加复杂度。合并后结构更简单，无需跨表查询。

#### 涉及接口、数据模型、状态流变化

- **删除数据表**：`photo_record`
- **InferenceRecord 新增字段**：`original_image_data`（BinaryField）、`enhanced_image_data`（BinaryField）
- **接口无变化**：路由路径保持不变，仅底层查询对象从 `PhotoRecord` 改为 `InferenceRecord`

#### 可能影响哪些地方

- 历史 15 条成功记录已从文件系统补录二进制数据
- 新推理会在 enhance 成功事务内同步写入二进制
- Django admin 中 `Photo Records` 面板已消失，所有记录统一在 `Inference Records` 下管理

#### 建议如何测试

1. `GET http://127.0.0.1:8002/api/v1/photos/` 确认 `total=15`
2. 浏览器访问 `http://127.0.0.1:8002/api/v1/photos/18/original/` 确认图片正常显示
3. 浏览器访问 `http://127.0.0.1:8002/api/v1/photos/18/enhanced/` 确认增强图正常显示

#### 未完成项

无

---

### [2026-03-29] 后端访问端口改为 8002

#### 改了哪些文件

- `config/settings/dev.py` — `MEDIA_BASE_URL` 从 `http://10.101.12.251:8000` 改为 `http://127.0.0.1:8002`
- `frontend/src/services/api.ts` — `BASE_URL` 从 `http://127.0.0.1:8000/api/v1` 改为 `http://127.0.0.1:8002/api/v1`

#### 做了什么修改

Django 开发服务器改在 `127.0.0.1:8002` 启动，前后端地址统一。

#### 为什么这样改

用户要求。

#### 涉及接口、数据模型、状态流变化

无，仅端口变更。

#### 可能影响哪些地方

- 前端需重新 `npm run build:mp-weixin` 使新 `BASE_URL` 生效
- 所有 API 调用和图片 URL 均需通过 8002 端口

#### 建议如何测试

访问 `http://127.0.0.1:8002/api/v1/health/` 确认返回 `code=0`。

#### 未完成项

无

---

### [2026-03-29] 修复结果页前端展示问题

#### 改了哪些文件

- `frontend/src/components/CompareSlider/index.vue` — `sliderX` 初始值 `0` 改为 `50`
- `frontend/src/components/ParamPanel/index.vue` — `.param-value` 加 `white-space: nowrap; flex-shrink: 0; margin-left: 16rpx`；`.param-left` 加 `min-width: 0; overflow: hidden`
- `frontend/src/pages/result/index.vue` — `.model-name` 加 `flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-right: 16rpx`；`.meta-pills` 加 `flex-shrink: 0`；`.pill` 加 `white-space: nowrap`

#### 做了什么修改

- 对比滑块默认停在 50%，用户进入结果页即可看到原图与增强图混合效果
- 参数面板右侧数值（如 `3769.4834K`）不再被截断
- meta-bar 右侧 PSNR / 推理耗时 pill 不再被截断，模型名过长时自动省略号

#### 为什么这样改

截图中对比滑块停在最左（增强图不可见）、参数数值和 pill 被屏幕右边缘裁切，影响演示体验。

#### 涉及接口、数据模型、状态流变化

无，仅前端样式与初始状态修改。

#### 可能影响哪些地方

无负面影响，均为纯展示层调整。

#### 建议如何测试

1. 进入结果页，对比滑块应默认显示在 50% 位置，图像为原图与增强图混合
2. 参数面板所有数值完整显示，不被截断
3. meta-bar 右侧两个 pill（PSNR + 耗时）完整显示

#### 未完成项

无

---

### [2026-03-29] 修复结果页横向溢出（右侧内容被截断）

#### 改了哪些文件

- `frontend/src/pages/result/index.vue` — `.page` 新增 `width: 100%; box-sizing: border-box; overflow-x: hidden`
- `frontend/src/pages/index/index.vue` — 同上

#### 做了什么修改

两个页面的 `scroll-view.page` 容器均补充了：
```css
width: 100%;
box-sizing: border-box;
overflow-x: hidden;
```

#### 为什么这样改

微信小程序中 `scroll-view` 默认 `box-sizing: content-box`，`padding: 32rpx` 两侧合计 64rpx 被叠加到宽度之外，导致整个页面内容向右偏移、右侧内容（数值、pill 文字）超出屏幕边缘被截断。

#### 涉及接口、数据模型、状态流变化

无，仅 CSS 布局修正。

#### 可能影响哪些地方

无负面影响，对所有子组件均生效（宽度自动收敛至视口内）。

#### 建议如何测试

进入结果页，meta-bar 的 `282.7 ms` 和 `33.11 dB` 完整显示；参数面板所有数值不被截断。

#### 未完成项

无

---

### [2026-03-29] 整体界面改为白色主题

#### 改了哪些文件

- `frontend/src/pages.json` — 导航栏改为白底黑字，全局背景改为 `#f5f6fa`
- `frontend/src/uni.scss` — 主题变量全部改为亮色
- `frontend/src/pages/index/index.vue` — 页面背景、标题颜色、禁用按钮色
- `frontend/src/pages/result/index.vue` — 页面背景、卡片背景、文字颜色、分割线
- `frontend/src/components/CompareSlider/index.vue` — 图像区占位背景、轨道背景
- `frontend/src/components/ParamPanel/index.vue` — 卡片背景、标题、参数名颜色
- `frontend/src/components/ModelSelector/index.vue` — 选项卡背景、边框、文字
- `frontend/src/components/UploadEntry/index.vue` — 虚线框、背景、提示文字
- `frontend/src/components/LoadingState/index.vue` — Spinner 轨道、说明文字
- `frontend/src/components/ErrorState/index.vue` — 错误说明文字

#### 做了什么修改

全局颜色替换：

| 旧（暗色） | 新（亮色） | 用途 |
|---|---|---|
| `#0f0f1a` | `#f5f6fa` | 页面背景 |
| `#1a1a2e` | `#ffffff` | 卡片 / 导航栏背景 |
| `#2a2a3e` / `#2a2a4e` | `#e8e8f0` | 边框 / 分割线 / Spinner 轨道 |
| `#12122a` | `#f0f0f8` / `#f8f8ff` | 图像占位背景 |
| `#ffffff` | `#1a1a2e` | 主要文字 |
| `#a0a0b8` / `#c0c0d8` | `#6a6a8a` / `#4a4a6a` | 次要文字 |
| `#606078` | `#9090a8` | 静音文字 |

强调色 `#6c63ff`（紫色）保留不变。卡片新增 `box-shadow: 0 2rpx 12rpx rgba(0,0,0,0.06)` 提升层次感。

#### 为什么这样改

用户要求。

#### 涉及接口、数据模型、状态流变化

无，仅样式改动。

#### 可能影响哪些地方

全部页面和组件视觉风格改变；强调色、状态色（success / error / warning）均保持不变。

#### 建议如何测试

1. 首页：背景白色，上传区域浅色虚线框，模型选项白色卡片
2. 结果页：参数面板、美学评分卡、元信息卡均为白色卡片，导航栏黑色文字
3. 加载状态：Spinner 轨道为浅灰色

#### 未完成项

无

---
