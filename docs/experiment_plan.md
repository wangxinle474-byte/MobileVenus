# MobileVenus 对比实验方案

## 1. 问题定位：我们的方法 vs SOTA

### 1.1 关键区别

我们的方法与传统 image retouching SOTA 有本质区别：

| 维度 | 传统 SOTA (3D LUT, CSRNet 等) | Ours (MobileVenus) |
|------|-------------------------------|---------------------|
| 输出 | 像素级增强图像 | **可解释的 Lightroom 参数** |
| 自由度 | 高（每像素独立调整） | 低（8 个全局参数） |
| 可解释性 | ❌ 黑盒 | ✅ 用户可调 |
| 端侧部署 | 部分可（LUT），部分不可 | ✅ MobileViT 极轻量 |
| 语义理解 | ❌ 无 | ✅ Venus 语义蒸馏 |

**结论：直接对比 PSNR/SSIM 不公平，但可以做，需要加上可解释性和效率维度。**

### 1.2 实验目标

1. **证明语义蒸馏有效**：Distill > Baseline（内部消融实验）
2. **和 SOTA 对比不落后太多**：在轻量+可解释约束下接近 SOTA
3. **展示独特优势**：可解释性 + 跨数据集泛化 + 端侧推理速度

---

## 2. 评估协议

### 2.1 数据集

| 数据集 | 用途 | 划分 | GT |
|--------|------|------|-----|
| **MIT-Adobe FiveK** | 主实验 | 4,500 train / 500 test (标准) | Expert C |
| **PPR10K** | 跨数据集泛化 | 直接测试(zero-shot) | Expert A/B/C |

### 2.2 评估指标

| 指标 | 说明 | 类型 |
|------|------|------|
| **PSNR** (dB) | 峰值信噪比 | 图像质量 |
| **SSIM** | 结构相似度 | 图像质量 |
| **LPIPS** | 感知距离(可选) | 感知质量 |
| **Parameter MAE** | 8参数平均绝对误差 | 参数准确度 |
| **#Params** (K) | 模型参数量 | 效率 |
| **FLOPs** (G) | 计算量 | 效率 |
| **Latency** (ms) | 端侧推理时间 | 效率 |

### 2.3 评估流程

```
Input JPEG (500 test) → Model → 8 Lightroom 参数 → ISP 渲染 → Output
                                                           ↓
                                                    vs Expert C GT
                                                    → PSNR / SSIM
```

**注意事项：**
- 所有方法使用相同的 500 张测试集
- Expert C 作为 ground truth（FiveK 标准）
- 图片统一 resize 到 480p 再计算指标（与文献一致）
- PSNR/SSIM 计算在 sRGB 空间

---

## 3. 对比方法

### 3.1 SOTA 方法（引用论文数据）

以下数据来自各论文在 **MIT-Adobe FiveK (480p, Expert C)** 上的报告值：

| 方法 | 年份 | 类型 | PSNR ↑ | SSIM ↑ | #Params | 可解释 |
|------|------|------|--------|--------|---------|--------|
| HDRNet | SIGGRAPH'17 | Bilateral grid | 23.93 | 0.897 | 482K | ❌ |
| White-Box | CVPR'18 | White-box filters | 24.55 | 0.920 | - | ✅ 部分 |
| DeepLPF | CVPR'20 | Local filters | 24.48 | 0.911 | 1.7M | ❌ |
| 3D LUT | TPAMI'20 | Lookup table | 25.29 | 0.927 | 593K | ❌ |
| CSRNet | ECCV'22 | Curve estimation | 25.56 | 0.929 | 37K | ✅ 部分 |
| CLUT-Net | MM'22 | Compressed LUT | 25.64 | 0.931 | 198K | ❌ |
| SepLUT | ECCV'22 | Separable LUT | 25.70 | 0.933 | 58K | ❌ |
| eLIR-Net | WACV'25 | Param regression | ~26.0 | ~0.935 | 120K | ✅ 部分 |

> **说明**：以上数据从各论文原文摘录。部分方法有 480p/1080p 两个版本，统一用 480p。

### 3.2 我们的方法（需要实测）

| 变体 | 描述 | 预期 PSNR | 预期 SSIM |
|------|------|-----------|-----------|
| **Baseline** | MobileViT + 直接回归 | ~23.5-24.5* | ~0.91* |
| **Distill v2** (3K) | + 语义蒸馏 (Stage A 3K) | ~24.5-25.5* | ~0.92* |
| **Distill v3** (5.3K) | + 语义蒸馏 (Stage A 5.3K) | ~25.0-26.0* | ~0.93* |

> *注：当前 ISP 渲染的 PSNR=32-33dB 是在 **我们自己的渲染** vs **Expert C** 下测量的。
> 需要统一到与 SOTA 相同的协议（原图→增强图 vs Expert C 目标图），PSNR 可能会降低。

---

## 4. 消融实验（Ablation Study）

### 4.1 语义蒸馏消融

| 实验 | Stage A 数据量 | Stage A 方法 | Stage B |
|------|---------------|-------------|---------|
| w/o distill | - | - | 直接训练 |
| + 语义对齐 (3K) | 3,000 | MiniLM align | 微调 |
| + 语义对齐 (5.3K) | 5,377 | MiniLM align | 微调 |
| + 语义对齐 (full) | ~14K | MiniLM align | 微调 |

### 4.2 损失函数消融

| 实验 | align_loss | uniform_loss | param_loss | consensus |
|------|-----------|-------------|-----------|-----------|
| base | ✅ | ✅ (0.1) | ✅ | ❌ |
| + low uniform | ✅ | ✅ (0.03) | ✅ | ❌ |
| + consensus | ✅ | ✅ (0.1) | ✅ | ✅ |

### 4.3 骨干网络消融

| Backbone | #Params | FLOPs | PSNR | Latency |
|----------|---------|-------|------|---------|
| MobileViT-XS | 2.3M | 0.7G | ? | ? ms |
| MobileViT-S | 5.6M | 1.8G | ? | ? ms |
| MobileViT-XXS | 1.3M | 0.4G | ? | ? ms |
| EfficientNet-B0 | 5.3M | 0.4G | ? | ? ms |

---

## 5. 需要补充的实验

### 5.1 高优先级（论文必需）

1. **统一评估协议**
   - [ ] 确认我们的 PSNR 计算与 SOTA 一致
   - [ ] 检查：输入是 original image 还是 camera-processed JPEG
   - [ ] 确保 resize 到 480p、Expert C 作为 GT

2. **重新评估 Baseline 和 Distill**
   - [ ] 在标准 480p 协议下重跑 eval_psnr_ssim.py
   - [ ] 如果当前 ISP 渲染过于简化导致 PSNR 偏低，考虑用 Adobe Lightroom CLI 渲染

3. **效率指标**
   - [ ] 统计 #Params 和 FLOPs (用 thop 或 fvcore)
   - [ ] 测端侧推理延迟 (CPU + GPU, 不同分辨率)

4. **Distill v3 (5.3K) 训练**
   - [ ] AutoDL 数据就绪后立即训练
   - [ ] 对比 v2 (3K) 的提升

### 5.2 中优先级

5. **PPR10K 跨数据集验证**
   - [ ] 下载 PPR10K 数据集
   - [ ] Zero-shot 评估（不 fine-tune）
   - [ ] 和 3D LUT / CSRNet 的跨数据集结果对比

6. **可解释性展示**
   - [ ] 制作参数可视化对比图（每个参数单独调节的效果）
   - [ ] 展示语义 embedding 的 t-SNE 聚类（不同场景类型）
   - [ ] 展示用户可编辑性 demo（预测参数 + 用户微调）

### 5.3 低优先级（加分项）

7. **用户研究**
   - [ ] A/B 测试：Ours vs SOTA 的主观偏好
   - [ ] 评估可解释性的实用价值

8. **端侧部署 Demo**
   - [ ] ONNX 导出
   - [ ] Android/iOS 推理延迟测试

---

## 6. 评估代码更新计划

### 6.1 eval_psnr_ssim.py 改进

```
需要更新：
1. 添加 480p resize 选项
2. 添加 LPIPS 指标
3. 支持批量对比多个 checkpoint
4. 输出标准格式的 LaTeX 表格
5. 添加 per-image 结果保存（用于分析 failure case）
```

### 6.2 新增 eval_efficiency.py

```
需要实现：
1. #Params 统计
2. FLOPs 计算 (thop)
3. CPU/GPU 推理延迟 (torch.utils.benchmark)
4. 输出效率对比表
```

### 6.3 新增 eval_interpretability.py

```
需要实现：
1. 参数敏感度分析（每个参数独立扰动）
2. 语义 embedding t-SNE 可视化
3. 场景类型 vs 参数分布分析
```

---

## 7. 论文表格模板

### Table 1: 与 SOTA 对比 (MIT-Adobe FiveK, 480p)

| Method | PSNR ↑ | SSIM ↑ | #Params ↓ | Interpretable |
|--------|--------|--------|-----------|---------------|
| HDRNet | 23.93 | 0.897 | 482K | ❌ |
| 3D LUT | 25.29 | 0.927 | 593K | ❌ |
| CSRNet | 25.56 | 0.929 | 37K | Partial |
| SepLUT | 25.70 | 0.933 | 58K | ❌ |
| **Ours (Baseline)** | **TBD** | **TBD** | **~6M** | ✅ Full |
| **Ours (+ Semantic)** | **TBD** | **TBD** | **~6M** | ✅ Full |

### Table 2: 消融实验

| Stage A Data | PSNR ↑ | SSIM ↑ | Δ PSNR |
|-------------|--------|--------|--------|
| None (Baseline) | 32.05 | 0.9269 | - |
| 3K (Distill v2) | 33.15 | 0.9311 | +1.10 |
| 5.3K (Distill v3) | TBD | TBD | TBD |

> 注：此表的 PSNR 基于当前 ISP 渲染协议，后续需统一到标准协议。

---

## 8. 关键风险和应对

| 风险 | 影响 | 应对方案 |
|------|------|----------|
| ISP 渲染精度不足，PSNR 偏低 | 和 SOTA 差距过大 | 改进 ISP 或用 Lightroom SDK 渲染 |
| 8参数自由度太低 | PSNR 上限受限 | 强调可解释性+效率优势 |
| PSNR 不如 SOTA | 审稿人质疑 | 加入效率维度(Pareto曲线)，强调不同赛道 |
| 跨数据集泛化差 | 方法通用性不足 | PPR10K fine-tune 版本 |

---

## 9. 时间线

| 时间 | 任务 |
|------|------|
| **本周** | Distill v3 (5.3K) 训练 + 效率指标统计 |
| **下周** | 统一评估协议 + 重新跑所有模型评估 |
| **第3周** | PPR10K 验证 + 可解释性实验 |
| **第4周** | 整理论文表格 + 补充实验 |
