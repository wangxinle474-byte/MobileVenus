"""生成工作汇报 Word 文档 (docs/reports/工作汇报_2026-05-11.docx)。

两条主线：
  1. 完善已上线的 RPA Coze 分布式 AI 运维平台
  2. 继续推进 IntelligenceCamera (CVPR 2026) 智能相机实验

运行:
  python docs/reports/gen_work_report_2026-05-11.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


OUT_PATH = Path(__file__).resolve().parent / "工作汇报_2026-05-11.docx"


# ---------------------------------------------------------------------------
# 样式工具
# ---------------------------------------------------------------------------
def set_cn_font(run, name: str = "微软雅黑", size_pt: float | None = None,
                bold: bool | None = None, color: tuple[int, int, int] | None = None):
    """统一设置中英文字体 + 字号/加粗/颜色。"""
    run.font.name = name
    r = run._element
    rpr = r.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        from docx.oxml import OxmlElement
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for k in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(k), name)
    if size_pt is not None:
        run.font.size = Pt(size_pt)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = RGBColor(*color)


def add_heading(doc: Document, text: str, level: int = 1):
    """添加中文友好的标题。"""
    sizes = {0: 22, 1: 16, 2: 14, 3: 12}
    colors = {0: (30, 30, 30), 1: (31, 73, 125), 2: (31, 73, 125), 3: (68, 68, 68)}
    p = doc.add_paragraph()
    if level == 0:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    set_cn_font(run, size_pt=sizes.get(level, 12), bold=True, color=colors.get(level))
    p.paragraph_format.space_before = Pt(6 if level > 0 else 0)
    p.paragraph_format.space_after = Pt(4)
    return p


def add_para(doc: Document, text: str, *, bold: bool = False, size_pt: float = 10.5,
             indent_cm: float | None = None, color: tuple[int, int, int] | None = None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    set_cn_font(run, size_pt=size_pt, bold=bold, color=color)
    p.paragraph_format.line_spacing = 1.3
    p.paragraph_format.space_after = Pt(2)
    if indent_cm is not None:
        p.paragraph_format.left_indent = Cm(indent_cm)
    return p


def add_bullet(doc: Document, text: str, *, bold_prefix: str | None = None, size_pt: float = 10.5):
    """以 "• " 为符号的普通项目符号（避免依赖 list 样式）。"""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.first_line_indent = Cm(-0.4)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.3
    bullet = p.add_run("• ")
    set_cn_font(bullet, size_pt=size_pt)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        set_cn_font(rb, size_pt=size_pt, bold=True)
    rt = p.add_run(text)
    set_cn_font(rt, size_pt=size_pt)
    return p


def add_table(doc: Document, header: list[str], rows: list[list[str]],
              col_widths_cm: list[float] | None = None):
    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.style = "Light Grid Accent 1"
    table.autofit = False

    # 表头
    for i, h in enumerate(header):
        cell = table.rows[0].cells[i]
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        set_cn_font(run, size_pt=10, bold=True, color=(255, 255, 255))
        # 表头底色
        tc_pr = cell._tc.get_or_add_tcPr()
        from docx.oxml import OxmlElement
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), "1F497D")
        tc_pr.append(shd)

    # 数据行
    for r, row in enumerate(rows, start=1):
        for c, txt in enumerate(row):
            cell = table.rows[r].cells[c]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(txt))
            set_cn_font(run, size_pt=10)

    if col_widths_cm:
        for i, w in enumerate(col_widths_cm):
            for row in table.rows:
                row.cells[i].width = Cm(w)
    return table


# ---------------------------------------------------------------------------
# 正文
# ---------------------------------------------------------------------------
def build() -> Document:
    doc = Document()

    # 页面 & 默认段落
    section = doc.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)

    # 默认样式字体
    style = doc.styles["Normal"]
    style.font.name = "微软雅黑"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")

    # ======= 标题 =======
    add_heading(doc, "工作汇报", level=0)

    # 基本信息表
    info_tbl = doc.add_table(rows=1, cols=4)
    info_tbl.autofit = True
    row = info_tbl.rows[0].cells
    for i, (k, v) in enumerate([("汇报人", "Xinle Wang"),
                                 ("汇报日期", "2026-05-11")]):
        cell_k = row[i * 2]
        cell_v = row[i * 2 + 1]
        cell_k.text = ""
        rk = cell_k.paragraphs[0].add_run(k)
        set_cn_font(rk, size_pt=10, bold=True)
        cell_v.text = ""
        rv = cell_v.paragraphs[0].add_run(v)
        set_cn_font(rv, size_pt=10)
    doc.add_paragraph()

    # ======= 一、工作概述 =======
    add_heading(doc, "一、工作概述", level=1)
    add_para(doc,
             "本期工作围绕两条主线推进：第一条是对已上线的企业级分布式 RPA 平台"
             "（RPA Coze）进行持续完善，重点在 AIOps 自愈闭环、故障知识库召回、"
             "以及多节点治理与 OTA 升级；第二条是继续推进智能相机项目"
             "（IntelligenceCamera，目标 CVPR 2026 投稿）的数据管线与模型实验，"
             "近期集中在 Qwen3-VL-4B LoRA SFT v1 评测、FireRed-Edit 教师编辑样本"
             "生成，以及三阶段蒸馏/精修训练框架的工程化整理。")

    # ======= 二、RPA Coze =======
    add_heading(doc, "二、RPA Coze 平台完善工作", level=1)

    add_heading(doc, "2.1 项目定位", level=2)
    add_para(doc,
             "RPA Coze 是一套由对话 AI（Coze Bot + Function Calling）驱动的企业级"
             "分布式 RPA 平台，Master-Worker 架构承载 N 台 Worker × 40+ 业务流程，"
             "覆盖财务关账、经销商对账、汇率下载、跨公司报告、银行对账、费用与凭证"
             "批量入账等场景。线上链路：Coze Bot → Coze 代理 → Master REST → "
             "Worker WebSocket → 浪潮 GS (pywinauto) / 抖音链客 (Playwright) / "
             "OCR 兜底，结果通过分块上传回传 Master 并以 Markdown 多链接回推至 "
             "Coze 会话。项目主页：https://daisyle.cn/projects/rpa-coze-ai-rpa")

    add_heading(doc, "2.2 平台七类核心能力", level=2)
    capabilities = [
        ("对话式任务触发", "Coze AI Bot 做统一入口，自然语言映射到 40+ RPA 任务；"
                         "HTTP Tools + Function Calling 调 Master REST API；"
                         "Prompt JSON Schema + 后端 task_mapping.json 双层校验拦截 LLM 幻觉。"),
        ("分布式任务调度", "单 Master 同时维护 N 路 WebSocket 双向 + M 路 SSE 单向连接；"
                         "SQLAlchemy 2.0 async + asyncpg (PostgreSQL) + Redis 分布式锁，"
                         "可 fallback 到 SQLite + 内存缓存；按心跳活跃度 + 历史任务数加权分配。"),
        ("企业 ERP + Web 自动化", "pywinauto UIA backend 深度控制浪潮 GS (.NET WinForms MDI)；"
                                "Playwright 处理抖音链客等浏览器场景；pytesseract + Pillow OCR 兜底；"
                                "PyInstaller 打包 exe + PyQt5 沙盘监控悬浮窗。"),
        ("文件全链路回流", "Worker → Master 分块上传（SHA-256 校验 + 临时文件原子 rename），"
                         "/files/{id}/download 对外直出；Nginx client_max_body_size=200M 支持大文件。"),
        ("全栈监控与可观测性", "dashboard.html 五 Tab 监控面板（HTTPBasic 鉴权）；"
                             "Worker 健康检查守护线程（60s 心跳熔断 + 任务自动重派）；"
                             "资源心跳 + 事件告警（severity 分级）+ 三源日志聚合；零外部依赖。"),
        ("AI 运维助手 + 自愈测试闭环", "独立子系统 rpa-aiops/（基于 SuperBizAgent 扩展，端口 :9900）："
                                     "RAG 问答 (Milvus + qwen-embedding-v4) + LangGraph 1.1 "
                                     "Plan-Execute-Replan StateGraph + MCP 协议三工具 + "
                                     "双层故障知识库召回 + APScheduler 定时自动测试。"),
        ("Worker OTA 热更新与多节点治理", "Master 侧 update_manager.py + /updates 管理台；"
                                        "Worker 侧 watchdog.py 看门狗 + restart 自重启；"
                                        "主仓 → 副本仓 → 阿里云生产节点 → 客户内网四地同步；"
                                        "部署 30min → 3min。"),
    ]
    for title, body in capabilities:
        add_bullet(doc, body, bold_prefix=f"{title} — ")

    add_heading(doc, "2.3 本期完善重点", level=2)
    add_para(doc,
             "围绕 AIOps 自愈闭环与平台稳定性，本期主要完善工作包括：", bold=False)
    focus = [
        ("AIOps 双层故障知识库召回",
         "这是本项目对 SuperBizAgent 的核心扩展。第一层在 Planner 起步阶段以"
         "「用户任务描述」为 query 召回潜在风险案例（top_k=2），写入 "
         "state.fault_kb_hints，让 LLM 制定计划时主动避坑；第二层在 Replanner "
         "失败评估阶段以「past_steps 末尾失败信号」为 query 召回历史修复经验，"
         "写入 state.fault_kb_recalled，形成「执行→失败→召回→修复→沉淀」"
         "自学习闭环。任一召回异常均安全降级为空列表，不阻断主流程。"),
        ("MCP 工具层扩展 + 新增 RPA 自动测试 Server",
         "基于 fastmcp 3.2 + langchain-mcp-adapters 0.2 + mcp 1.27 构建 "
         "MultiServerMCPClient，新增 rpa server（:8005）作为 streamable-http "
         "transport，可反向驱动 RPA Master 跑 E2E 验证修复效果；配合原有 "
         "cls server（日志 :8003）、monitor server（监控 :8004），全局 "
         "retry_interceptor 拦截失败自动重试。"),
        ("LangGraph 状态机护栏",
         "落地三条关键约束防止无限循环：past_steps ≥ 8 强制 respond；"
         "past_steps ≥ 5 禁 replan 只能 respond；replan 新步骤数 ≤ 当前"
         "剩余步骤数；plan 失败/异常安全降级到默认 3 步计划。"),
        ("Phase 3-c 定时自动测试",
         "APScheduler 3.11 cron（默认 0 3 * * * Asia/Shanghai）拉 "
         "test_cases/safe_subset.yaml，AIOps Agent 串行跑巡检验证 + 修复回归 + "
         "全链路冒烟（通过 mcp_rpa 触发 RPA Master 跑真任务），配合 "
         "LangSmith 做 LLM 全链路追踪。"),
        ("对话式触发链路稳定性",
         "Coze 代理优化 SSE 流式转发、multipart 上传、conversation_id 自动"
         "提取保存；task_mapping.json 增补校验字段拦截 LLM 幻觉参数。"),
        ("多节点治理脚本化",
         "主仓（Gitee monorepo）→ 副本仓 → 阿里云生产节点 → 客户内网节点的 "
         "SCP + Git 同步流程脚本化，部署时长由 30 分钟降至 3 分钟。"),
    ]
    for title, body in focus:
        add_bullet(doc, body, bold_prefix=f"{title}：")

    add_heading(doc, "2.4 业务价值", level=2)
    values = [
        "月末关账：财务 ERP 手工操作由小时级降至分钟级；",
        "7×24 无人值守：员工下班/离职不影响任务继续运行；",
        "对话降门槛：非技术人员在聊天框一句话触发，无需懂 ERP 菜单；",
        "故障自愈：AIOps 自动诊断 + 沉淀知识库，减少运维人工介入；",
        "规模：覆盖 N 台 Worker、M 类业务流程、K 家公司；",
        "可靠性：任务成功率 > 99.X%，端到端 P99 延迟可控。",
    ]
    for v in values:
        add_bullet(doc, v)

    # ======= 三、IntelligenceCamera =======
    add_heading(doc, "三、IntelligenceCamera 智能相机实验工作", level=1)

    add_heading(doc, "3.1 项目定位", level=2)
    add_para(doc,
             "IntelligenceCamera 面向移动端智能相机场景，目标是把 Venus 7B VLM 的"
             "自然语言美学理解能力蒸馏到约 1.93M 参数的轻量模型（FP16 约 4MB，"
             "推理延迟 < 80ms），端到端从图像 / 自然语言指令预测 6 个核心 "
             "Lightroom 参数（EV / WB / Contrast / Shadows / Highlights / Saturation），"
             "并通过 RefinementNet 做像素级精修。目标投稿：CVPR 2026。")

    add_heading(doc, "3.2 已有核心实验结果", level=2)
    add_table(
        doc,
        header=["版本", "核心思路", "关键指标", "定位"],
        rows=[
            ["Baseline (FiveK 8param)", "5 专家共识加权 8 参数回归", "PSNR 32.05 / SSIM 0.9269", "参考基线"],
            ["Distill v4", "FiveK Stage A 语义对齐", "PSNR 33.11 / SSIM 0.9326", "语义桥接验证"],
            ["Distill v5", "Stage B Expert C 单专家精准监督", "PSNR 34.11 / SSIM 0.9449", "★ 技术最优"],
            ["Distill v6", "Venus NL + 8→6 参数精简", "PSNR 32.05 / SSIM 0.9289", "NL 蒸馏"],
            ["Distill v7", "退化增强 + 对比学习", "PSNR 25.97 / Venus 美学 5.38", "★ 美学最优"],
            ["RefineNet V3", "8.93M, 3 级编解码器", "val_MUSIQ 4.20 (best@ep111)", "突破 V2 4.15 天花板"],
            ["v14 (当前主力)", "三阶段蒸馏 + AesExpert 高质量混训", "进行中", "端到端训练框架"],
        ],
        col_widths_cm=[3.2, 5.4, 4.6, 2.8],
    )

    add_heading(doc, "3.3 近期重点 A：Qwen3-VL-4B LoRA SFT v1", level=2)
    add_para(doc,
             "基于 LLaMA-Factory 在 RTX 5090 上完成端侧 VLM LoRA 微调，用于从"
             "「原图 + 编辑指令」直接预测 Lightroom 参数 JSON。")
    add_table(
        doc,
        header=["配置项", "取值"],
        rows=[
            ["Base model", "Qwen3-VL-4B-Instruct (8.3 GB)"],
            ["LoRA rank / alpha", "16 / 32, target = all linear, freeze vision tower"],
            ["Template / cutoff_len", "qwen3_vl_nothink / 12288"],
            ["Batch × accum / lr / epochs", "1 × 8 / 2e-4 / 3.0 (cosine, warmup 5%)"],
            ["数据", "ArtEdit_LoRA_train 682 / val 75"],
            ["训练开销", "258 steps / 9 min 35 s"],
            ["train_loss / eval_loss", "0.5077 / 0.5559"],
            ["产物", "adapter_model.safetensors (127 MB) 等"],
        ],
        col_widths_cm=[5.0, 10.5],
    )

    add_para(doc, "评测结果（75 val samples）：", bold=True)
    add_table(
        doc,
        header=["key", "n_paired", "MAE", "评价"],
        rows=[
            ["exposure", "60", "0.13", "极准"],
            ["dehaze", "46", "0.93", "极准（但 std=0.00 存在 mode collapse）"],
            ["clarity", "20", "2.55", "不错"],
            ["vibrance", "62", "3.79", "不错"],
            ["contrast", "75", "5.76", "可接受"],
            ["saturation", "33", "6.82", "可接受"],
            ["blacks", "52", "8.02", "一般"],
            ["shadows", "67", "12.70", "一般"],
            ["temp", "5", "812.80", "❌ Kelvin 与 offset 量纲混用"],
            ["whites", "1", "110.00", "❌ paired sample 过少"],
        ],
        col_widths_cm=[3.0, 2.4, 2.4, 7.7],
    )
    add_para(doc, "亮点：格式合规率 75/75 = 100%（CN 35/35、EN 40/40 全部正确"
                  "输出 <think> + <tool_call> 结构）。", bold=False)
    add_para(doc, "发现的关键问题：", bold=True)
    issues = [
        "稀有 key 的 mode collapse：dehaze / tint / texture / whites 预测 std=0.00，"
        "模型把它们学成了固定值 10。",
        "temp 量纲混乱：训练数据里有的样本用 Kelvin（例如 4000），有的用 offset"
        "（例如 5），模型学得混乱，需要在数据生成阶段统一。",
        "highlights / whites / texture 在训练样本中覆盖不足，预测几乎不出现。",
    ]
    for it in issues:
        add_bullet(doc, it)
    add_para(doc, "v2 改进方向：", bold=True)
    plans = [
        "数据清洗：temp 统一为 offset (-100~+100) 或 Kelvin，二选一；",
        "数据增强：对 highlights / whites / texture 加权采样；",
        "训练：lora_alpha 32 → 64，多跑 1-2 epoch 观察 eval_loss 是否继续下降；",
        "Loss 设计：对 mode-collapse 的 key 加 KL 正则鼓励输出多样性。",
    ]
    for p in plans:
        add_bullet(doc, p)

    add_heading(doc, "3.4 近期重点 B：FireRed-Edit 教师编辑数据管线", level=2)
    add_para(doc,
             "为 LoRA v2 补充高质量「原图 — 专业编辑」对，搭建了基于 "
             "FireRed-Image-Edit-1.0 的教师推理候选筛选与生成管线。")
    pipeline_pts = [
        "tools/data/data_prep/select_teacher_edit_candidates.py："
        "从 data/aug_ip2p_full.json 按 score_delta 降序取 top N 作为候选，"
        "排除 compare_5 已用的 5 个 idx（71/448/808/110/194），输出 caption "
        "JSON 和去 _orig 后缀的原图复制。",
        "已产出的 caption / run_meta 数据：teacher_edits_top20.json、"
        "teacher_edits_firered_top20.json、_instr.json、_instr_retry.json、"
        "_styleC.json、_toneonly.json，以及针对 0110/0600 的小规模 pilot / retry 扩展。",
        "已产出的图像数据：outputs/teacher_edits/firered_top20/（20 组原图 + "
        "编辑图 + run_meta.json + viewer.html），覆盖 idx 0078 / 0090 / 0141 / "
        "0312 / 0347 / 0412 / 0427 / 0469 / 0472 / 0497 / 0520 / 0554 / 0558 / "
        "0588 / 0600 / 0685 / 0745 / 0754 / 0818 / 0819 共 20 个样本。",
        "配套生成 viewer.html（左右对照原图/编辑图 + caption 元信息）便于人工"
        "筛选与标注。",
    ]
    for pt in pipeline_pts:
        add_bullet(doc, pt)

    add_heading(doc, "3.5 工程化整理", level=2)
    eng_pts = [
        "training/ 重构：legacy/ 归档早期脚本，main/ 汇总当前主力脚本 "
        "（train_neural_isp / train_stage_c / train_v10_e2e / train_v11_refine / "
        "train_v12_refine_hd / train_v8_stage_b），semantic_distill / "
        "text_condition / fivek_8param 三个可导入子包。",
        "tools/data/ 重构为 6 个功能子目录（analysis / data_prep / editor_models 等）。",
        "scripts/ 拆分为 autodl/（远程训练与下载）、local/（本地同步与 resume）"
        "、legacy_training/（历史脚本归档）。",
        "建立 TRAINING_LOG.md：LoRA SFT 从 install → 四次训练失败到最终成功"
        "的完整时间线、错误与修复记录，沉淀为后续调试参考。",
        "建立 .windsurf/ 三份上下文文档：project_overview.md（架构/组件/版本线）"
        "、autodl_resources.md（远端文件结构与命令）、local_resources.md"
        "（本地数据集清单），在 IDE 上下文丢失时可快速恢复认知。",
    ]
    for pt in eng_pts:
        add_bullet(doc, pt)

    # ======= 四、下一步计划 =======
    add_heading(doc, "四、下一步计划", level=1)

    add_heading(doc, "4.1 RPA Coze", level=2)
    rpa_plan = [
        "持续维护客户内网节点的同步与 OTA 升级；",
        "按业务方需求扩展新的 RPA 任务与故障知识库案例；",
        "AIOps 巡检 cron 结果接入告警（企微 / 钉钉）。",
    ]
    for p in rpa_plan:
        add_bullet(doc, p)

    add_heading(doc, "4.2 IntelligenceCamera", level=2)
    ic_plan = [
        "LoRA v2：temp 量纲统一 + 稀有 key 加权采样 + KL 正则 + "
        "lora_alpha 64；目标消除 mode collapse 并降低 temp MAE；",
        "v14 AesExpert 高质量数据混训 ParamModel（计划中）；",
        "FireRed 教师编辑样本扩量至 top 100 并进入 LoRA v2 训练集；",
        "SOTA 方法对比（HDRNet / 3D LUT / CSRNet）；",
        "移动端部署（ONNX / CoreML）；",
        "论文撰写（CVPR 2026 投稿）。",
    ]
    for p in ic_plan:
        add_bullet(doc, p)

    # ======= 附录 =======
    add_heading(doc, "五、附录：关键产物清单", level=1)
    add_table(
        doc,
        header=["类别", "位置", "说明"],
        rows=[
            ["LoRA v1 权重", "/root/autodl-tmp/checkpoints/intelligence_camera/lora_v1/",
             "adapter_model.safetensors 127 MB + training_loss.png + trainer_log.jsonl"],
            ["LoRA v1 报告", "docs/lora_v1/EVAL_REPORT.md", "75 val samples 完整评测"],
            ["训练时间线", "TRAINING_LOG.md", "LoRA SFT install → 训练 → eval 全记录"],
            ["教师编辑 caption", "data/teacher_edits_firered_top20_*.json 等 8 份",
             "top20 原图 + 4 种指令风格 + 扩展 pilot/retry"],
            ["教师编辑图像", "outputs/teacher_edits/firered_top20/（20 图 + viewer.html）",
             "FireRed-Image-Edit 推理产物，含 run_meta.json"],
            ["项目概览", ".windsurf/project_overview.md", "三阶段架构 + 组件 + 版本线"],
        ],
        col_widths_cm=[3.0, 6.0, 6.5],
    )

    return doc


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = build()
    doc.save(OUT_PATH)
    print(f"[OK] work report saved to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
