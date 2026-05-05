# scripts/ 索引

> 按"在哪里运行"分了三类子目录，找脚本先看下表。

## 目录速查

| 子目录 | 运行位置 | 文件数 | 用途 |
|--------|----------|:-:|------|
| [`autodl/`](#autodl) | AutoDL 远端 GPU 实例 | 21 | 模型下载/安装/训练/推理/服务/评分/监控 |
| [`local/`](#local) | 本地 Windows (.ps1) 或 Linux/WSL (.sh) | 10 | 本地↔远端同步、HF 权重下载、日志监控 |

> 旧训练入口 (`train_v*.py` / `training/main/train_stage_c.py` 等) 已迁到 `training/legacy/`, 不再在 scripts/。

---

## autodl/

### 下载模型权重 (5)
| 脚本 | 下载目标 |
|------|---------|
| `autodl_download_aesexpert.sh` | AesExpert (AesMMIT_LLaVA_v1.5_7b) |
| `autodl_download_artedit.sh` | ArtEdit |
| `autodl_download_jarvisevo.sh` | JarvisEvo |
| `autodl_download_longcat.sh` | LongCat-Image-Edit-Turbo |
| `autodl_download_qwen3vl_4b.sh` | Qwen3-VL-4B-Instruct |

### 环境安装 (2)
| 脚本 | 装什么 |
|------|-------|
| `autodl_install_llama_factory.sh` | LlamaFactory (LoRA SFT 训练框架) |
| `autodl_install_vllm.sh` | vLLM (推理加速) |

### LongCat/FireRed 对比主流水线 (5)
| 脚本 | 作用 |
|------|------|
| `autodl_phase1_rewrite_and_rescore.sh` | **Phase 1**: Qwen3-VL 重写 editB 指令 → 4 组评分 |
| `autodl_phase2_score_longcat_rewritten.sh` | **Phase 2**: 评分第 5 组 (LongCat×重写prompt) |
| `autodl_rescore_all_10pt.sh` | 4 组 1-10 分评分（独立跑） |
| `autodl_run_longcat_compare.sh` | LongCat sceneA + editB 两组推理 + 评分 |
| `autodl_launch_compare_bg.sh` | 后台调用上者, setsid + nohup 防 SSH 断 |

### 训练 / 推理服务 (4)
| 脚本 | 作用 |
|------|------|
| `autodl_run_lora_sft.sh` | LoRA SFT 训练 (LlamaFactory + Qwen3-VL) |
| `autodl_serve_jarvisevo_v2.sh` | 部署 JarvisEvo v2 服务 |
| `autodl_serve_jarvisevo_vllm.sh` | vLLM 部署 JarvisEvo |
| `autodl_resume_jarvisevo_retry.py` | JarvisEvo 重试恢复 |

### 监控 + 验证 + 其他 (5)
| 脚本 | 作用 |
|------|------|
| `autodl_watch_pipeline.sh` | 实时监控 LoRA 流水线（watch -n 3 ...） |
| `autodl_rescore_longcat.sh` | 老版 LongCat 评分（1-5 分） |
| `run_validate_on_autodl.sh` | 验证 `diff_isp_vs_lr` |
| `run_validate_nisp_on_autodl.sh` | 验证 Neural ISP |
| `inspect_artedit.sh` | 简查 ArtEdit 数据 |

---

## local/

### 本地↔AutoDL 同步 (2)
| 脚本 | 作用 |
|------|------|
| `sync_to_autodl.sh` | Linux/WSL 同步（rsync） |
| `sync_to_autodl.ps1` | Windows PowerShell 同步（scp） |

### HF 断点续传 (3)
| 脚本 | 作用 |
|------|------|
| `local_resume_aesexpert.ps1` | AesExpert HF 镜像续传 |
| `local_resume_aesexpert_v2.ps1` | v2 加强版（稳定性改进） |
| `local_resume_longcat.ps1` | LongCat HF 镜像续传 |

### 日志监控 / 数据拉取 (5)
| 脚本 | 作用 |
|------|------|
| `tail_autodl_log.ps1` | tail -f 远端 AutoDL 日志 |
| `watch_download.ps1` | 监视本地下载进度 |
| `check_dl_progress.sh` | 查 AutoDL 下载进度（ssh wrapper） |
| `fetch_pseudo_labels.ps1` | 从 AutoDL 拉伪标签 JSON |
| `upload_compare_5.ps1` | 上传 5 张对比图到 AutoDL |

---

## 旧训练入口

已迁移到 `training/legacy/`. 具体:

| 文件 | 版本 | 说明 |
|------|------|------|
| `training/legacy/train_v5_clean.py` | v5 | 最早的 clean version |
| `training/legacy/train_v6_stage_a.py` | v6 | Stage A 语义对齐 (FiveK) |
| `training/legacy/train_v6_stage_b.py` | v6 | Stage B 参数预测（Expert C 单专家） |
| `training/legacy/train_v7_stage_b.py` | v7 | 退化增强 + 对比学习 |
| `training/legacy/train_v8_stage_b_root.py` | v8 | Stage B 根目录版 |
| `training/legacy/train_stage_c.py` | v8 | Stage C 文本条件 (TextEncoder + FiLM) |
| `training/legacy/train_v13_multiscale.py` | v13 | 多尺度精修 (含 EMA + WarmRestarts) |

> 主流活动训练在根目录 `training/main/train_v10_e2e.py` / `training/main/train_v11_refine.py` / `training/main/train_v12_refine_hd.py` 。这里只放"已完成演进"的版本。

---

## 常见用法快照

```bash
# [本地] 同步代码到 AutoDL
.\scripts\local\sync_to_autodl.ps1 -Host "root@connect.xxx.seetacloud.com" -Port 12345

# [AutoDL] 跑 LongCat+FireRed 对比全流程
ssh autodl "bash /root/autodl-tmp/IntelligenceCamera/scripts/autodl/autodl_phase1_rewrite_and_rescore.sh"

# [AutoDL] 4 组独立评分
ssh autodl "bash /root/autodl-tmp/IntelligenceCamera/scripts/autodl/autodl_rescore_all_10pt.sh"

# [AutoDL] 跑旧训练入口 (现在在 training/legacy/)
ssh autodl "cd /root/autodl-tmp/IntelligenceCamera && python training/legacy/train_v6_stage_a.py ..."

# [本地] tail 远端日志
.\scripts\local\tail_autodl_log.ps1 -Log "compare_full.log"
