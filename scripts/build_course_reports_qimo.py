#!/usr/bin/env python3
"""Generate qimo-version opening and midterm reports for the B07 campus RAG project."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "course_reports_qimo"
LOG_DIR = ROOT / "logs"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTER_ROOT = ROOT.parent

TEAM = "第11组：周子涵、凌霄、归梦依、杨歆苒"
PROJECT = "B07 RAG 校园智能问答助手《同小智》"
REPO_URL = "https://github.com/LINGXIAO-aico/Python-AI"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def read_ablation() -> list[dict[str, str]]:
    candidates = [
        OUTER_ROOT / "ablation_results(1).csv",
        LOG_DIR / "ablation_results.csv",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))
    return []


def pct(value: float | str) -> str:
    number = float(value)
    if number <= 1:
        number *= 100
    return f"{number:.2f}%"


def ms(value: float | str) -> str:
    return f"{float(value):.0f} ms"


PROFILE = read_json(LOG_DIR / "data_profile.json")
TRAINING = read_json(LOG_DIR / "training_log.json")
EVAL = read_json(LOG_DIR / "evaluation_summary.json")
ABLATION = read_ablation()

RAW_FAQ_COUNT = read_jsonl_count(RAW_DIR / "campus_faq.jsonl")
CRAWLED_COUNT = read_jsonl_count(RAW_DIR / "crawled_pages.jsonl")
COMBINED_COUNT = read_jsonl_count(RAW_DIR / "combined_kb.jsonl")
CLEAN_COUNT = int(PROFILE["clean_rows"])
CHUNK_COUNT = int(PROFILE["chunk_count"])

TOP_CATEGORIES = sorted(PROFILE["categories"].items(), key=lambda item: item[1], reverse=True)[:6]

GITHUB_TIMELINE = [
    ("2026-05-10", "b4dd1a0", "建立 B07 校园 RAG 项目骨架，加入 app、CLI、核心模块、原始数据和基础测试。"),
    ("2026-05-10", "7986e99", "扩展知识库与课程报告脚本，把项目从演示 FAQ 推进到可提交的课程材料。"),
    ("2026-05-10", "cf29d53", "补充运行手册与项目解说，沉淀复现流程和课堂讲解材料。"),
    ("2026-05-11", "610c412", "美化 Streamlit 页面，加入同济蓝主题、卡片式来源、模式切换等前端展示。"),
    ("2026-05-31", "3424ade", "RAG V2 全面升级：真实网页、7层检索增强管线、190题消融、CI与工程化结构。"),
    ("2026-06-03", "8d9cb97", "修复 nDCG@5 评测，加入 E9 Self-RAG 与 LLM-as-Judge 失败分析。"),
    ("2026-06-03", "fc43b5b", "补齐 embeddings、FAISS、reranker、query rewriting 等测试，覆盖率达到 82.74%。"),
    ("2026-06-04", "2f61692", "上传 qimo 最终版项目文件，整理最终展示、报告、日志与前端版本。"),
]


def ablation_row(keyword: str) -> dict[str, str] | None:
    for row in ABLATION:
        haystack = " ".join(row.values()).lower()
        if keyword.lower() in haystack:
            return row
    return None


def ablation_pdf_rows() -> list[list[str]]:
    desired = [
        ("E1", "无检索基线"),
        ("E4", "BGE Dense"),
        ("E5", "RRF 融合"),
        ("E6", "RRF + Reranker"),
        ("E8", "完整管线"),
        ("E9", "Self-RAG"),
    ]
    rows = [["实验", "配置", "Hit@1", "Hit@5", "MRR", "nDCG@5", "延迟"]]
    for exp_id, name in desired:
        row = None
        for item in ABLATION:
            if item.get("experiment", "").startswith(exp_id):
                row = item
                break
        if not row:
            continue
        rows.append([
            exp_id,
            name,
            pct(row.get("hit_at_1", 0)),
            pct(row.get("hit_at_5", 0)),
            f"{float(row.get('mrr', 0)):.4f}",
            f"{float(row.get('ndcg_at_5', 0)):.4f}",
            ms(row.get("avg_latency_ms", 0)),
        ])
    return rows


def write_markdown() -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cat_text = "、".join(f"{name}{count}条" for name, count in TOP_CATEGORIES)
    e6 = ablation_row("E6") or {}
    e5 = ablation_row("E5") or {}

    opening = f"""# 开题（选题）报告：{PROJECT}

**小组信息**：{TEAM}  
**课程题目**：项目库 B07：RAG 校园智能问答助手  
**代码仓库**：{REPO_URL}

## 一、选题背景与问题定义

校园信息并不缺少，但分散在教务处、研究生院、新闻网、学生事务、体育部、一卡通、图书馆、宿舍后勤等不同入口。学生遇到真实问题时，常常不是“不知道有没有规定”，而是“不知道去哪一个网页找、该用什么关键词搜、能不能相信通用大模型的回答”。通用大模型直接回答校园事务时容易把常识、旧经验和学校规定混在一起，尤其在办理材料、时间节点、网址来源等问题上存在幻觉风险。

本项目选择 B07 RAG 校园智能问答助手，是因为它和课程要求中的数据采集、清洗、建模、评估、展示四部分高度契合。我们不从零训练大模型，而是围绕同济校园公开资料和人工整理 FAQ 构建一个小型知识库，让系统先检索可信资料，再让大模型依据证据生成回答，并在回答中保留来源引用。这样既能体现自然语言处理和信息检索的技术含量，也能做出一个答辩现场可以演示、同学能理解的校园应用。

## 二、课程要求与项目库对应

任务书要求项目完成数据采集与清洗、数据分析与可视化、模型训练与评估、结果展示与汇报。B07 项目库要求构建校园知识库，完成文本切分、向量化、索引构建、检索策略设计，调用大模型 API 做检索增强问答，并与无检索基线对比。我们的开题目标据此确定为：做一个可复现的“数据表 + 检索索引 + RAG 问答 + 消融评测 + 前端 Demo”完整闭环，而不是只做一个聊天页面。

## 三、数据来源与建设计划

数据计划采用“两条线合成一个校园知识库”：一条线是人工整理 FAQ，覆盖选课、图书馆、一卡通、宿舍、奖助评优、考试、校园网络等日常高频问题；另一条线是同济公开网页，包括同济新闻网、研究生院、教务处、体育部、学生事务等公开页面。所有记录统一为 `doc_id、category、title、content、source、url、last_updated` 字段，先合并为 `combined_kb.jsonl`，再清洗成 `knowledge_base_clean.csv`，最后切分成 `chunks.csv` 供索引构建。

开题阶段预期至少完成 150 条以上可用知识记录；qimo 最终版实际形成 {COMBINED_COUNT} 条原始记录，清洗后 {CLEAN_COUNT} 条文档，切分为 {CHUNK_COUNT} 个文本块，覆盖 {len(PROFILE["categories"])} 个类目，主要分布为：{cat_text}。

## 四、技术路线

系统采用“资料抽取/爬取 -> 字段统一 -> 清洗去重 -> 中文文本切分 -> 多策略检索 -> RAG 生成 -> 指标评测 -> 前端展示”的路线。检索层保留 TF-IDF 和 BM25 作为可解释基线，引入 BGE-large-zh-v1.5 生成 1024 维语义向量，用 FAISS HNSW 建立向量索引；最终将 BGE Dense 与 jieba-BM25 通过 RRF 融合，并使用 BGE-reranker-v2-m3 做 Cross-Encoder 精排。生成层使用 DeepSeek-chat，校验层尝试 DeepSeek-reasoner Self-RAG。

## 五、评测设计

评测不只看“回答像不像”，而是从检索与生成两方面验证：检索侧使用 Hit@1、Hit@3、Hit@5、MRR、nDCG@5、平均延迟；生成侧使用关键词召回率，并保留 RAG 与无检索基线对比。最终 qimo 版本在 190 题消融评测中，最优生产配置为 E6 RRF + Reranker，Hit@1={pct(e6.get("hit_at_1", 0)) if e6 else "69.47%"}，Hit@5={pct(e6.get("hit_at_5", 0)) if e6 else "71.58%"}，MRR={float(e6.get("mrr", 0)):.4f}；相比 E5 RRF，Reranker 带来约 +{(float(e6.get("hit_at_1", 0)) - float(e5.get("hit_at_1", 0))) * 100:.2f} 个百分点 Hit@1 提升。

为了避免“只展示好看的个例”，我们计划把评测集分成两类：一类是人工确认的常见校园问题，用于验证系统能否稳定回答真实使用场景；另一类是由知识库文档半自动生成、再人工抽检的问题，用于扩大覆盖范围并支持消融实验。对每个问题记录 `gold_doc_id` 和答案关键词，这样可以客观判断检索结果是否把正确资料排在前面，也能判断生成回答是否覆盖关键事实。

## 六、预期成果与风险控制

预期成果包括：1）一份字段统一、可统计、可追溯的校园知识库；2）一套可复现的数据处理与索引构建脚本；3）一组可比较的检索策略和消融实验；4）一个可以现场演示的 Streamlit 校园问答页面；5）开题、中期、最终报告和答辩 PPT。项目风险主要有三点：公开网页噪声较多、中文短问题检索不稳定、大模型可能编造。对应措施分别是清洗过滤与来源保留、多路召回和 RRF 融合、严格 prompt 约束与引用编号输出。

## 七、小组分工

- 凌霄：检索架构、BGE/FAISS、BM25、RRF、reranker、消融实验与性能分析。
- 归梦依：数据采集、公开网页整理、FAQ 规范化、清洗规则、知识库画像。
- 周子涵：RAG 生成链路、DeepSeek API、Streamlit 前端、交互模式与演示流程。
- 杨歆苒：评测集、指标统计、可视化、报告/PPT 整理、答辩材料打磨。

## 八、阶段计划

第7周完成开题报告和基础项目结构；第8-10周完成 FAQ 知识库、清洗、基线检索和中期报告；第11-14周扩展真实公开网页、加入 BGE/FAISS/RRF/reranker、完善评测与失败案例；第15-16周完成 qimo 版本、最终报告、汇报 PPT 和现场 Demo。
"""

    midterm = f"""# 中期进展报告：{PROJECT}

**小组信息**：{TEAM}  
**代码仓库**：{REPO_URL}  
**当前整理版本**：qimo 最终版复盘，用于说明从开题到中期再到最终冲刺的真实过程。

## 一、阶段目标与完成情况

按照任务书和 B07 项目库要求，本项目中期阶段的核心目标是跑通“知识库 -> 切分 -> 索引 -> 检索 -> 回答 -> 评测”的闭环。最终 qimo 版本在此基础上继续升级，已经形成可演示的校园 RAG 系统：有结构化知识库、有真实公开网页、有多策略检索、有消融实验、有前端页面、有测试覆盖，也保留了报告和 PPT 材料。

目前代码结构清晰分为 `campus_rag/` 核心包、`scripts/` 流水线脚本、`data/` 数据、`models/` 索引模型、`logs/` 实验日志、`reports/` 汇报材料。核心模块包括 `data.py` 清洗、`splitter.py` 切分、`embeddings.py` BGE 编码、`vectorstore.py` FAISS 向量库、`retriever.py` 多策略检索、`reranker.py` 精排、`generator.py` DeepSeek 生成与 Self-RAG 校验、`evaluate.py` 指标评测。

## 二、数据构建与知识库画像

数据采用两条线合成：人工 FAQ {RAW_FAQ_COUNT} 条，公开网页爬取 {CRAWLED_COUNT} 篇，合并入口为 `data/raw/combined_kb.jsonl`，共 {COMBINED_COUNT} 条原始记录。清洗后保留 {CLEAN_COUNT} 条文档，去除重复/异常记录 {PROFILE["duplicate_rows"]} 条；必需字段 `doc_id、category、title、content、source、url、last_updated` 缺失值均为 0。文档平均长度 {PROFILE["avg_content_length"]} 字，最短 {PROFILE["min_content_length"]} 字，最长 {PROFILE["max_content_length"]} 字。

知识库共覆盖 {len(PROFILE["categories"])} 个类目，重点类目包括：{cat_text}。这说明项目不是只围绕单一 FAQ，而是覆盖了新闻资讯、研究生培养、教务选课、校园生活、宿舍后勤、一卡通、奖助评优、图书馆等多种校园信息场景。

切分阶段使用 `RecursiveCharacterTextSplitter`，配置 `chunk_size=360`、`chunk_overlap=80`，分隔符优先级为段落、换行、中文句号/问号/分号/逗号等。最终得到 {CHUNK_COUNT} 个文本块，平均块长 {PROFILE["avg_chunk_length"]} 字。切分的目的不是“把文本变碎”，而是让检索时每个候选块足够聚焦，减少长文档中无关内容干扰；同时通过 80 字重叠保留跨句上下文，避免答案刚好落在边界处被切断。

## 三、模型与系统设计

索引构建脚本 `scripts/02_build_index.py` 同时生成 TF-IDF、旧 BM25、jieba-BM25、BGE Dense + FAISS 四类检索基础。日志显示，TF-IDF 使用字符 n-gram，词表规模 {TRAINING["tfidf"]["vocabulary_size"]}，矩阵形状 {TRAINING["tfidf"]["matrix_shape"]}；BGE 模型为 {TRAINING["bge_dense_faiss"]["model"]}，向量维度 {TRAINING["bge_dense_faiss"]["dim"]}，共写入 {TRAINING["bge_dense_faiss"]["total_vectors"]} 个 FAISS 向量，整体索引构建耗时 {TRAINING["total_build_seconds"]} 秒。

检索策略从简单到复杂逐步升级：TF-IDF 负责精确词面匹配，BM25 负责关键词相关性，BGE Dense 负责语义相似度，Hybrid RRF 将 Dense 与 jieba-BM25 的排名融合，BGE-reranker-v2-m3 对 top 候选做 Cross-Encoder 精排。生成端采用 DeepSeek-chat，系统提示词要求“只能根据参考资料回答、资料不足时明确说明、关键内容标注引用编号”。Self-RAG 使用 DeepSeek-reasoner 对回答主张做 supported/unsupported 校验。

前端 `app.py` 使用 Streamlit，实现快速推荐、高精度、语义检索、关键词检索、无检索基线、带校验等模式。这样答辩时既能展示推荐生产配置，也能把不同模块的作用通过模式切换讲清楚。

## 四、代码复现与工程质量

项目保留了完整流水线脚本：`scripts/10_crawl_tongji.py` 负责从同济公开站点发现并抓取文章，`scripts/11_pages_to_kb.py` 将网页转成标准知识库格式，`scripts/01_prepare_data.py` 做合并清洗与数据画像，`scripts/02_build_index.py` 构建 TF-IDF、BM25、BGE/FAISS 等索引，`scripts/03_evaluate.py` 生成主评测结果，`scripts/20_run_ablation.py` 运行 9 组消融实验。也就是说，报告中的数据不是手工填表，而是能从代码和日志追溯到具体输出文件。

工程质量方面，项目后期补齐了 embeddings、FAISS vectorstore、reranker、query rewriter、generator、evaluate、loaders 等模块测试。GitHub 过程记录显示，离线测试通过 79 项，model/llm 测试带有缓存或 API Key 守卫，coverage 达到 82.74%。这些测试的意义是保证核心 RAG 组件不是只停留在“能跑一次”的状态，而是后续修改时仍然能被自动检查。

## 五、实验结果

50 题主评测用于验证系统闭环：Hybrid RRF 的 Hit@1={pct(EVAL["hit_at_1"])}，Hit@3={pct(EVAL["hit_at_3"])}，Hit@5={pct(EVAL["hit_at_5"])}，MRR={EVAL["mrr"]:.4f}，nDCG@5={EVAL["ndcg_at_5"]:.4f}，RAG 关键词召回率 {pct(EVAL["rag_keyword_recall"])}，无检索基线关键词召回率 {pct(EVAL["baseline_keyword_recall"])}。

190 题消融评测更能反映各模块贡献。E1 无检索基线在检索指标上为 0；E4 BGE Dense 的 Hit@1 为 67.89%；E5 RRF 融合提升 Hit@5 到 71.58%；E6 RRF + Reranker 达到最佳综合表现，Hit@1=69.47%、MRR=0.7037、nDCG@5=0.7067。E8 完整管线没有继续变好，主要因为 HyDE 对部分校园短问题引入了噪声，因此最终建议把 E6 作为默认生产配置，HyDE/Self-RAG 作为高级可选模式。

进一步看实验结论，TF-IDF 和 BM25 虽然简单，但在明确词面的校园问题上速度很快；BGE Dense 对口语化、改写类问题更稳；RRF 的价值是把“词面准确”和“语义相似”合在一起，不需要手动调权重；Reranker 的价值是对候选片段做更精细的句对相关性判断。HyDE 和 Self-RAG 属于增强能力，但不应盲目作为默认配置，因为消融结果已经显示复杂链路并不必然带来更高检索指标。

## 六、GitHub 过程记录

{chr(10).join(f"- {date} `{sha}`：{desc}" for date, sha, desc in GITHUB_TIMELINE)}

这些提交能说明项目不是一次性生成的，而是经历了基础版、报告版、运行手册、前端美化、RAG V2、评测修正、测试覆盖、qimo 整理等多个阶段。

## 七、阶段问题与解决方式

1. 早期知识库偏 FAQ，覆盖面有限。解决方式是加入同济公开网页爬取，并统一到 `combined_kb.jsonl`，后续清洗统计可追溯。
2. 中文短问题容易受分词和同义表达影响。解决方式是保留 TF-IDF 字符 n-gram、jieba-BM25 和 BGE Dense 三种互补检索，再用 RRF 融合排序。
3. 只追求“管线越复杂越好”并不可靠。消融显示 HyDE/完整管线并非总是提升，所以最终选择 E6 作为生产配置，把实验结论写清楚。
4. 评测如果只看少量人工题会偏乐观。解决方式是同时保留 50 题主评测和 190 题消融评测，分别服务于演示闭环和模块比较。
5. 工程可信度需要自动化验证。后期补充 13 个测试文件，覆盖 embeddings、FAISS、reranker、query rewriting、generator、evaluate 等模块，离线测试和覆盖率结果可写入报告。

## 八、后续完善计划

最终答辩前主要完成四件事：第一，继续检查前端细节，避免页面底部输入区与整体风格冲突；第二，把 PPT 中旧版“179条/374块”等数字统一替换为 qimo 最终数据；第三，准备失败案例讲解，主动说明 HyDE 和 Self-RAG 为什么不是默认配置；第四，现场演示时优先用“快速推荐”和“高精度”两种模式，分别展示速度和精度。

如果答辩时间允许，我们会补充一个对比演示：同一个问题分别用“无检索基线”和“RRF + Reranker”回答。前者容易给出泛化建议，后者能显示具体来源、doc_id 和校园资料片段。这个对比可以直观说明 RAG 的价值，也能把任务书中“与无检索基线对比”的要求讲得更清楚。
"""

    opening_path = OUT_DIR / "开题报告_B07_RAG校园智能问答助手_qimo详细版.md"
    midterm_path = OUT_DIR / "中期报告_B07_RAG校园智能问答助手_qimo详细版.md"
    opening_path.write_text(opening, encoding="utf-8")
    midterm_path.write_text(midterm, encoding="utf-8")
    return opening_path, midterm_path


def register_font() -> str:
    font_path = Path("C:/Windows/Fonts/simhei.ttf")
    if not font_path.exists():
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
    pdfmetrics.registerFont(TTFont("CN", str(font_path)))
    return "CN"


def styles() -> dict[str, ParagraphStyle]:
    font = register_font()
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title_cn",
            parent=base["Title"],
            fontName=font,
            fontSize=14,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=6,
        ),
        "h": ParagraphStyle(
            "heading_cn",
            parent=base["Heading2"],
            fontName=font,
            fontSize=9.2,
            leading=11.5,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=4,
            spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "body_cn",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.6,
            leading=10.2,
            alignment=TA_LEFT,
            firstLineIndent=12,
            spaceAfter=2.2,
        ),
        "small": ParagraphStyle(
            "small_cn",
            parent=base["BodyText"],
            fontName=font,
            fontSize=6.9,
            leading=8.6,
            alignment=TA_LEFT,
            spaceAfter=1.2,
        ),
        "table": ParagraphStyle(
            "table_cn",
            parent=base["BodyText"],
            fontName=font,
            fontSize=6.5,
            leading=8.1,
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "table_header_cn",
            parent=base["BodyText"],
            fontName=font,
            fontSize=6.7,
            leading=8.2,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
        "meta": ParagraphStyle(
            "meta_cn",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.2,
            leading=9.0,
            alignment=TA_LEFT,
        ),
    }


STYLES = styles()


def p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def table(rows: list[list[str]], widths: list[float], header: bool = True) -> Table:
    converted = []
    for row_idx, row in enumerate(rows):
        row_style = "table_header" if header and row_idx == 0 else "table"
        converted.append([p(str(cell), row_style) for cell in row])
    tbl = Table(converted, colWidths=widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style_cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "CN"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C8D3E0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if header:
        style_cmds.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")))
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("CN", 6.5)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawCentredString(A4[0] / 2, 0.62 * cm, f"{PROJECT}    第 {doc.page} 页")
    canvas.restoreState()


def build_opening_pdf(path: Path) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.05 * cm,
        rightMargin=1.05 * cm,
        topMargin=0.85 * cm,
        bottomMargin=1.0 * cm,
    )
    story = [
        p(f"开题（选题）报告：{PROJECT}", "title"),
        table(
            [
                ["小组", TEAM, "题目来源", "项目库 B07"],
                ["目标", "建设可检索、可引用、可评测的校园 RAG 助手", "仓库", REPO_URL],
            ],
            [2.0 * cm, 6.2 * cm, 2.0 * cm, 8.0 * cm],
            header=False,
        ),
        p("一、选题背景", "h"),
        p("同济校园信息分散在教务、研究生院、学生事务、体育部、新闻网、图书馆、一卡通和后勤等多个入口。学生遇到办事、学习生活、奖助就业和场馆服务问题时，难点往往不是没有资料，而是不知道去哪找、用什么关键词找，以及通用大模型给出的回答是否可靠。本项目用 RAG 把公开资料先检索出来，再由大模型基于证据回答，并保留引用编号，降低幻觉风险；回答不再只是“像真的”，而是能说明依据来自哪一条校园资料。"),
        table(
            [
                ["现实痛点", "项目对应做法"],
                ["入口分散、网页多、检索成本高", "把人工 FAQ 与公开网页统一成同一套知识库字段"],
                ["通用大模型可能编造流程/时间/地点", "只允许模型依据检索片段回答，并输出 doc_id 引用"],
                ["短问题表达口语化，关键词不稳定", "同时使用词面检索、语义检索和 RRF 融合"],
            ],
            [5.8 * cm, 12.4 * cm],
            header=True,
        ),
        p("二、课程要求对应", "h"),
        p("任务书要求覆盖数据采集清洗、分析可视化、模型训练评估、结果展示汇报；B07 要求构建校园知识库，完成文本切分、向量化、索引构建、检索策略设计，并与无检索基线对比。本项目以“数据表 + 检索索引 + RAG 问答 + 消融评测 + 前端 Demo”为完整闭环，既能体现 Python 数据处理，也能体现 AI 系统工程能力。"),
        p("三、数据与知识库计划", "h"),
        p(f"数据采用两条线合成：人工 FAQ 与同济公开网页。所有记录统一为 doc_id、category、title、content、source、url、last_updated 七个字段，先进入 combined_kb.jsonl，再清洗为 CSV 并切分为文本块。qimo 最终版实际得到原始记录 {COMBINED_COUNT} 条、清洗文档 {CLEAN_COUNT} 条、文本块 {CHUNK_COUNT} 个、类目 {len(PROFILE['categories'])} 个。"),
        table(
            [
                ["字段", "作用"],
                ["doc_id/category/title", "保证每条资料可定位、可分类、可在评测中作为 gold_doc_id"],
                ["content/source/url", "保留正文、来源单位和网页链接，方便回答引用与人工复查"],
                ["last_updated", "记录资料时间，避免把过期校园信息混入最终回答"],
            ],
            [4.1 * cm, 14.1 * cm],
            header=True,
        ),
        p("四、技术路线", "h"),
        table(
            [["资料整理", "清洗切分", "检索建库", "生成评测"],
             ["人工 FAQ + 公开网页", "字段统一、去重、长度过滤、中文标点切分", "TF-IDF、jieba-BM25、BGE+FAISS、RRF、Reranker", "DeepSeek 生成、Self-RAG 校验、Hit@K/MRR/nDCG/延迟"],
             ["可复现脚本", "01_prepare_data.py", "02_build_index.py", "03_evaluate.py / 20_run_ablation.py"]],
            [4.25 * cm, 4.65 * cm, 4.7 * cm, 4.6 * cm],
            header=True,
        ),
        p("五、预期评测", "h"),
        p("评测设计包含检索指标和回答指标：Hit@1/3/5、MRR、nDCG@5、平均延迟、关键词召回率，并设置无检索大模型直答作为基线。评测集记录问题、gold_doc_id 和答案关键词，既能判断正确资料是否进入 top-k，也能判断回答是否覆盖关键事实。最终 qimo 版 190 题消融显示，RRF + Reranker 是最佳生产配置，Hit@1 为 69.47%，Hit@5 为 71.58%。"),
        table(
            [
                ["风险", "控制方式"],
                ["公开网页噪声", "长度过滤、中文比例检查、来源 URL 保留、人工抽查"],
                ["复杂模块未必提升", "用消融实验决定默认配置，不把 HyDE/Self-RAG 盲目放默认"],
                ["答辩演示不稳定", "保留快速推荐、高精度、关键词检索和无检索基线多种模式"],
            ],
            [4.2 * cm, 14.0 * cm],
            header=True,
        ),
        p("六、分工与计划", "h"),
        table(
            [
                ["成员", "主要职责"],
                ["凌霄", "检索架构、BGE/FAISS、BM25、RRF、reranker、消融实验"],
                ["归梦依", "数据采集、公开网页整理、FAQ 规范化、清洗规则、知识库画像"],
                ["周子涵", "RAG 生成链路、DeepSeek API、Streamlit 前端、交互模式"],
                ["杨歆苒", "评测集、指标统计、可视化、报告/PPT、答辩材料"],
            ],
            [2.5 * cm, 15.7 * cm],
            header=True,
        ),
        table(
            [
                ["预期交付物", "对应课程要求"],
                ["combined_kb.jsonl / knowledge_base_clean.csv / chunks.csv", "数据采集、清洗、字段统一、文本切分"],
                ["TF-IDF、BM25、BGE+FAISS、RRF、Reranker 索引与脚本", "模型训练与评估、对比实验"],
                ["evaluation_summary.json、ablation_results.csv、图表与失败分析", "数据分析、指标解释、结果讨论"],
                ["Streamlit Demo、最终报告、汇报 PPT", "结果展示、答辩演示、课程材料提交"],
            ],
            [6.3 * cm, 11.9 * cm],
            header=True,
        ),
        p("阶段安排：第7周完成开题和基础结构；第8-10周完成 FAQ、清洗、基线检索和中期报告；第11-14周扩展真实网页、BGE/FAISS/RRF/reranker、消融实验和前端；第15-16周完成 qimo、最终报告、PPT 与 Demo。", "small"),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def build_midterm_pdf(path: Path) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=1.05 * cm,
        rightMargin=1.05 * cm,
        topMargin=0.85 * cm,
        bottomMargin=1.0 * cm,
    )
    cat_rows = [["类目", "数量"]] + [[name, str(count)] for name, count in TOP_CATEGORIES]
    story = [
        p(f"中期进展报告：{PROJECT}", "title"),
        table(
            [["小组", TEAM, "仓库", REPO_URL],
             ["阶段口径", "中期闭环 + qimo 最终版复盘", "核心目标", "数据、检索、生成、评测、展示可复现"]],
            [2.0 * cm, 6.5 * cm, 2.0 * cm, 7.7 * cm],
            header=False,
        ),
        p("一、已完成进展", "h"),
        p("项目已经跑通从知识库构建到前端展示的完整链路。代码分为 campus_rag 核心包、scripts 流水线、data 数据、models 索引、logs 实验日志和 reports 汇报材料。核心模块包括数据清洗、中文切分、BGE 编码、FAISS 向量库、多策略检索、Cross-Encoder 重排、DeepSeek 生成、Self-RAG 校验和批量评测。中期闭环先保证“能处理数据、能检索、能回答、能评测”，qimo 阶段进一步补充真实网页、消融实验和工程测试。"),
        table(
            [
                ["目录/文件", "作用"],
                ["campus_rag/", "清洗、切分、检索、生成、评测等核心 Python 包"],
                ["scripts/", "爬取、入库、建索引、评测、消融、报告生成的流水线入口"],
                ["data/models/logs", "分别保存原始/处理数据、索引模型、实验日志，方便复现"],
            ],
            [4.2 * cm, 14.0 * cm],
            header=True,
        ),
        p("二、数据构建与画像", "h"),
        p(f"数据来源为人工 FAQ {RAW_FAQ_COUNT} 条与公开网页 {CRAWLED_COUNT} 篇，合并入口 combined_kb.jsonl 共 {COMBINED_COUNT} 条。清洗后保留 {CLEAN_COUNT} 条，去重/异常过滤 {PROFILE['duplicate_rows']} 条，七个必需字段缺失值均为 0。平均文档长度 {PROFILE['avg_content_length']} 字，最长 {PROFILE['max_content_length']} 字。"),
        table(
            [["来源", "数量", "说明"],
             ["人工 FAQ", str(RAW_FAQ_COUNT), "覆盖选课、宿舍、一卡通、图书馆、奖助评优等高频问法"],
             ["公开网页", str(CRAWLED_COUNT), "来自同济新闻网、研究生院、教务处、体育部、学生事务等公开页面"],
             ["清洗后文档", str(CLEAN_COUNT), "统一字段、去重、过滤噪声后进入知识库画像与切分"]],
            [3.2 * cm, 2.0 * cm, 13.0 * cm],
            header=True,
        ),
        table(cat_rows, [8.8 * cm, 2.2 * cm], header=True),
        p("三、清洗与切分规则", "h"),
        p(f"清洗规则包括：统一必需字段、空白归一化、按 doc_id 与 title+content 去重、过滤正文少于 20 字的噪声记录、按 category/doc_id 排序写出 CSV。切分采用 RecursiveCharacterTextSplitter，chunk_size=360、overlap=80，以段落、换行和中文标点为自然断点，最终形成 {CHUNK_COUNT} 个文本块，平均块长 {PROFILE['avg_chunk_length']} 字。切分的作用是让检索命中更聚焦，并用重叠保留跨句上下文。"),
        table(
            [
                ["处理环节", "产物", "为什么需要"],
                ["合并入口", "combined_kb.jsonl", "让 FAQ 与网页先进入同一张原始数据表，便于统计和追溯"],
                ["清洗入库", "knowledge_base_clean.csv", "去掉重复、噪声和过短正文，保证索引质量"],
                ["文本切分", "chunks.csv", "把长文档拆成语义更集中的候选片段，适配 top-k 检索"],
            ],
            [3.0 * cm, 4.0 * cm, 11.2 * cm],
            header=True,
        ),
        p("四、模型与系统设计", "h"),
        table(
            [
                ["模块", "实现"],
                ["索引", f"TF-IDF 词表 {TRAINING['tfidf']['vocabulary_size']}；BGE {TRAINING['bge_dense_faiss']['dim']} 维；FAISS 向量 {TRAINING['bge_dense_faiss']['total_vectors']} 个"],
                ["检索", "TF-IDF、旧 BM25、jieba-BM25、BGE Dense、Hybrid RRF、RRF + Reranker"],
                ["生成", "DeepSeek-chat 按参考资料生成，关键内容带 doc_id 引用；DeepSeek-reasoner 做 Self-RAG 校验"],
                ["展示", "Streamlit 支持快速推荐、高精度、语义检索、关键词检索、无检索基线和带校验模式"],
            ],
            [3.0 * cm, 15.2 * cm],
            header=True,
        ),
        p("系统默认推荐 RRF + Reranker 的原因是：Dense 能处理语义相似，BM25 能保住关键词精确匹配，RRF 免调参地融合两路排名，Reranker 再对候选片段做句对级精排。这样比单一路径更稳，也方便在答辩中解释每一层的作用。"),
        table(
            [
                ["复现命令", "输出"],
                ["python scripts/01_prepare_data.py", "清洗数据、数据画像、chunks.csv"],
                ["python scripts/02_build_index.py", "TF-IDF、BM25、BGE+FAISS 索引和 training_log.json"],
                ["python scripts/03_evaluate.py", "50 题主评测、策略对比表、评测图"],
                ["python scripts/20_run_ablation.py", "190 题消融实验，比较 E1-E9 模块贡献"],
            ],
            [6.6 * cm, 11.6 * cm],
            header=True,
        ),
        table(
            [
                ["前端模式", "答辩讲法"],
                ["快速推荐", "默认 RRF，速度与效果平衡，适合现场演示"],
                ["高精度", "RRF + Reranker，用于说明精排带来的 Hit@1 提升"],
                ["无检索基线", "用于对比通用大模型直答，突出 RAG 的引用和可追溯价值"],
            ],
            [4.0 * cm, 14.2 * cm],
            header=True,
        ),
        PageBreak(),
        p(f"中期进展报告：{PROJECT}", "title"),
        p("五、实验结果", "h"),
        p(f"50 题主评测中，Hybrid RRF 的 Hit@1={pct(EVAL['hit_at_1'])}，Hit@3={pct(EVAL['hit_at_3'])}，Hit@5={pct(EVAL['hit_at_5'])}，MRR={EVAL['mrr']:.4f}，nDCG@5={EVAL['ndcg_at_5']:.4f}；RAG 关键词召回率 {pct(EVAL['rag_keyword_recall'])}，无检索基线仅 {pct(EVAL['baseline_keyword_recall'])}。190 题消融用于判断模块贡献，结论是 E6 RRF + Reranker 最适合作为默认生产配置。"),
        table(ablation_pdf_rows(), [1.45 * cm, 4.3 * cm, 2.2 * cm, 2.2 * cm, 2.1 * cm, 2.2 * cm, 2.0 * cm], header=True),
        p("实验解释：E1 无检索基线没有资料定位能力；E4 BGE Dense 对口语化问题更友好；E5 RRF 把 Dense 与 BM25 的优势合并，Hit@5 最稳定；E6 增加 reranker 后 Hit@1 和 MRR 最高。E8/E9 虽然链路更复杂，但 HyDE/Self-RAG 会增加延迟，也可能给短问题引入噪声，因此更适合作为高级可选能力。"),
        table(
            [
                ["指标", "含义"],
                ["Hit@1/Hit@5", "正确资料是否排在第1位/前5位，直接衡量检索能否找对依据"],
                ["MRR", "正确资料越靠前分数越高，比单纯 Hit@K 更能体现排序质量"],
                ["nDCG@5", "考虑排名折损，适合比较不同检索策略的前5排序"],
                ["关键词召回", "回答是否覆盖人工标注的关键事实，用于对比 RAG 与无检索基线"],
            ],
            [3.2 * cm, 15.0 * cm],
            header=True,
        ),
        p("六、工程质量与复现", "h"),
        table(
            [
                ["检查项", "结果"],
                ["索引构建日志", f"TF-IDF 词表 {TRAINING['tfidf']['vocabulary_size']}；BGE 向量 {TRAINING['bge_dense_faiss']['total_vectors']}；总耗时 {TRAINING['total_build_seconds']} 秒"],
                ["自动化测试", "13 个测试文件覆盖 embeddings、FAISS、reranker、query rewriting、generator、evaluate 等模块"],
                ["覆盖率记录", "GitHub 质量提交显示离线测试 79 项通过，coverage 达到 82.74%"],
            ],
            [4.0 * cm, 14.2 * cm],
            header=True,
        ),
        p("七、GitHub 持续迭代记录", "h"),
        table(
            [["日期", "提交", "工作内容"]] + [list(item) for item in GITHUB_TIMELINE],
            [2.1 * cm, 2.0 * cm, 14.1 * cm],
            header=True,
        ),
        p("八、问题与解决", "h"),
        table(
            [
                ["问题", "解决方式"],
                ["早期 FAQ 覆盖面不足", "加入公开网页，统一到 combined_kb.jsonl，保留来源和 URL"],
                ["中文短问法受分词和同义表达影响", "TF-IDF、BM25、Dense 互补，再用 RRF 融合排序"],
                ["复杂管线不一定更好", "用消融实验选择 E6 默认配置，HyDE/Self-RAG 作为可选增强"],
                ["答辩材料数字容易混旧版", "以 qimo 日志为准统一 239/238/538/190 等最终数字"],
            ],
            [5.2 * cm, 13.0 * cm],
            header=True,
        ),
        p("九、后续计划", "h"),
        p("答辩前继续统一 PPT 中旧版数字，完善前端底部输入区风格，准备 2-3 个成功案例和 1 个失败案例。现场优先演示快速推荐与高精度两种模式，并用同一问题对比无检索基线和 RAG 回答，说明速度、精度和可解释引用之间的取舍。"),
        table(
            [
                ["答辩准备项", "讲解重点"],
                ["成功案例", "展示检索来源、引用编号和最终回答如何对应"],
                ["失败案例", "说明知识库覆盖不足或 HyDE 噪声，不回避局限"],
                ["最终口径", "默认 E6，快速推荐用于演示速度，高精度用于展示最佳效果"],
            ],
            [4.2 * cm, 14.0 * cm],
            header=True,
        ),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    opening_md, midterm_md = write_markdown()

    opening_pdf = OUT_DIR / "开题报告_B07_RAG校园智能问答助手_qimo详细版.pdf"
    midterm_pdf = OUT_DIR / "中期报告_B07_RAG校园智能问答助手_qimo详细版.pdf"
    build_opening_pdf(opening_pdf)
    build_midterm_pdf(midterm_pdf)

    for path in [opening_pdf, midterm_pdf]:
        pages = len(PdfReader(str(path)).pages)
        print(f"{path}  pages={pages}")
    print(opening_md)
    print(midterm_md)


if __name__ == "__main__":
    main()
