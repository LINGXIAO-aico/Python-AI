# 同小智 — 基于 RAG 的同济大学校园智能问答助手

<p align="center">
  <img src="assets/tongji-logo.png" alt="同济大学" height="80">
</p>

<p align="center">
  <strong>B07 小组 · 同济大学《Python人工智能程序设计实践》课程项目</strong><br>
  凌霄· 归梦依 · 周子涵 · 杨歆苒
</p>

<p align="center">
  <a href="#-快速开始"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/LINGXIAO-aico/Python-AI/actions"><img src="https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white" alt="CI"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
</p>

---

## 目录

- [项目简介](#项目简介)
- [效果演示](#效果演示)
- [系统架构](#系统架构)
- [知识库](#知识库)
- [评测结果](#评测结果)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [参考文献](#参考文献)

---

## 项目简介

**同小智**是一个基于 RAG（检索增强生成，Retrieval-Augmented Generation）的校园智能问答系统，专为同济大学校园场景设计。

### 它解决什么问题？

高校校园信息分散在教务、图书、体育、后勤等十多个网站中。学生面对"一卡通丢了怎么补办""两门课时间冲突怎么办"等问题时，往往要在多个网站间反复查找，效率极低。直接问通用大模型（如 ChatGPT）又可能得到编造的流程和政策细节。

### 它是怎么工作的？

同小智采用 **"先查资料再回答"** 的 RAG 范式：

1. **检索**：从校园知识库中检索与问题最相关的文档片段
2. **生成**：基于检索到的资料，由大模型组织答案
3. **引用**：每条关键信息标注来源编号，确保答案可追溯

这避免了纯大模型"凭空编造"的幻觉问题，使回答始终有据可循。

---

## 效果演示

| 输入问题 | 系统回答 |
|---|---|
| 图书馆周末几点开门？ | 图书馆阅览区工作日开放时间为 8:00 至 22:00，周末开放时间为 9:00 至 21:00。法定节假日和寒暑假开放安排以图书馆官网通知为准。[KB006] |
| 两门课时间冲突怎么办？ | 若两门课程上课时间冲突，系统会自动拦截选课。确需修读的学生应先联系任课教师和所在学院，提交课程冲突修读申请，经学院审核通过后由教务老师手工处理。[KB002] |
| 校园卡丢了第一步应该做什么？ | 校园卡遗失后应尽快在校园卡服务平台或自助终端办理挂失，避免余额被盗刷。补办需携带身份证或学生证到校园卡服务中心办理。[KB009] |

> 每个回答末尾的 `[KBxxx]` 为引用编号，可追溯到知识库中的原始文档。

---

## 系统架构

### 七层 RAG 管线

```
用户问题
  │
  ▼
┌─────────────────────────────────────────────────┐
│ L1  查询改写      HyDE 假设文档生成               │
│                   DeepSeek-chat, 100-200字假想答案 │
├─────────────────────────────────────────────────┤
│ L2  双路召回      BGE Dense (FAISS HNSW)          │
│                   + jieba-BM25 稀疏检索           │
│                   各召回 Top-20                    │
├─────────────────────────────────────────────────┤
│ L3  RRF 融合      Reciprocal Rank Fusion (K=60)  │
│                   免调参排名融合                   │
├─────────────────────────────────────────────────┤
│ L4  重排序        BGE-reranker-v2-m3 Cross-Encoder│
│                   Top-20 → Top-5                  │
├─────────────────────────────────────────────────┤
│ L5  上下文压缩    截断去重，控制 prompt 长度        │
├─────────────────────────────────────────────────┤
│ L6  LLM 生成      DeepSeek-chat 流式输出           │
│                   引用感知 prompt，标注 [doc_id]    │
├─────────────────────────────────────────────────┤
│ L7  答案校验      DeepSeek-reasoner Self-RAG      │
│                   逐条主张审查，判断是否有据可循    │
└─────────────────────────────────────────────────┘
  │
  ▼
带引用来源的回答 + 校验结论
```

### 检索策略

系统提供 **4 种检索器**，可灵活组合：

| 检索器 | 类型 | 适用场景 |
|---|---|---|
| `TfidfRetriever` | 稀疏向量（字符 n-gram） | 精确词面匹配，保留用于消融对比 |
| `Bm25JiebaRetriever` | 稀疏检索（jieba 分词 + BM25） | 关键词匹配，中文分词准确 |
| `DenseRetriever` | 密集向量（BGE + FAISS） | 语义理解，口语化查询 |
| `HybridRRFRetriever` | RRF 融合 Dense + BM25 | 综合最优，免调参 |

### Web 应用（6 种回答模式）

| 模式 | 检索策略 | 特点 | 典型延迟 |
|---|---|---|---|
| ⚡ 快速推荐 | RRF 混合检索 | 速度与精度均衡 | ~158ms |
| 🎯 高精度 | RRF + Reranker | Cross-Encoder 精排，最优配置 | ~1354ms |
| 🧠 语义检索 | BGE Dense | 适合口语化、同义改写问题 | ~159ms |
| 🔤 关键词检索 | jieba-BM25 | 适合明确术语、名称查询 | ~2ms |
| 🤖 无检索基线 | LLM 直答 | 对照实验，不推荐日常使用 | ~2964ms |
| 🔍 带校验 | RRF + Self-RAG | 回答后校验依据 | ~1775ms |

---

## 知识库

### 数据来源

通过 `scripts/10_crawl_tongji.py` 对同济大学 **7 个公开二级网站**进行合规爬取，并辅以人工编写的 FAQ：

| 来源 | 文档数 | 覆盖内容 |
|---|---|---|
| 同济大学新闻网 | 50 篇 | 学校动态、重要通知 |
| 研究生院 | 49 篇 | 招生、培养、学位管理 |
| 教务处、体育部、学生事务等 | 20 篇 | 选课政策、体测标准、奖助贷 |
| 人工编写 FAQ | 120 条 | 高频办事流程问答 |
| **合计** | **239 篇原始** | **20 个类目** |

### 数据处理流水线

```
爬取 (crawled_pages.jsonl, 119篇)
  → 页面清洗 (campus_kb.jsonl, 119篇)
  → 合并 FAQ (campus_faq.jsonl, 120条)
  → 去重清洗 (combined_kb.jsonl, 239→238篇)
  → 中文感知切分 (chunk_size=360, overlap=80)
  → 538 个 chunks
```

### 知识库统计

| 指标 | 数值 |
|---|---|
| 原始文档 | 239 篇 |
| 清洗后文档 | 238 篇 |
| 文本块 (Chunks) | 538 个 |
| 类目数 | 20 个 |
| 平均文档长度 | ~360 字符 |
| 平均 chunk 长度 | ~181 字符 |
| 嵌入维度 | 1024 (BGE-large-zh-v1.5) |

### 类目分布

教务选课 · 考试管理 · 图书馆 · 一卡通 · 宿舍后勤 · 奖助评优 · 校园生活 · 生活服务 · 校园网络 · 体育场馆 · 社团活动 · 医疗健康 · 学生事务 · 学籍管理 · 就业双创 · 就业指导 · 校园活动 · 学校概况 · 同济新闻 · 研究生培养

---

## 评测结果

### 消融实验（190 题评测集，`eval_150.jsonl`）

| 实验 | 配置 | Hit@1 | Hit@5 | MRR | nDCG@5 | 延迟(ms) |
|---|---|---|---|---|---|---|
| E1 | 无检索基线（LLM 直答） | 0.00% | 0.00% | 0.0000 | 0.0000 | 2964 |
| E2 | TF-IDF | 67.37% | 69.47% | 0.6829 | 0.6859 | 6.2 |
| E3 | BM25(jieba) | 64.21% | 69.47% | 0.6607 | 0.6692 | 2.3 |
| E4 | BGE Dense | 67.89% | 70.00% | 0.6877 | 0.6909 | 159 |
| E5 | RRF 融合 | 67.89% | **71.58%** | 0.6927 | 0.6985 | 158 |
| **E6** | **RRF + Reranker** ✨ | **69.47%** | **71.58%** | **0.7037** | **0.7067** | 1354 |
| E7 | RRF + HyDE | 68.42% | 70.53% | 0.6939 | 0.6968 | 547 |
| E8 | 完整管线 | 68.95% | 69.47% | 0.6912 | 0.6921 | 1701 |
| E9 | + Self-RAG | 68.95% | 69.47% | 0.6921 | 0.6928 | 1775 |

**关键结论**：
- **E6 (RRF + Reranker) 是最优生产配置**，Hit@1 达 69.47%
- Reranker 是单模块增益最大的环节（E5→E6: Hit@1 +1.58pp, MRR +1.1pp）
- RAG 相比无检索基线，关键词召回率从 0.2531 提升至 0.6491+（**提升 157%**）
- 完整管线（E8）因 HyDE 引入噪声，反而出现性能退化

### 主评测（50 题，Hybrid RRF）

| 指标 | 数值 |
|---|---|
| Hit@1 | **0.92** |
| Hit@3 | **1.00** |
| Hit@5 | **1.00** |
| MRR | **0.96** |
| nDCG@5 | **0.9705** |

> 注：50 题评测集的 gold_doc_id 主要对应 FAQ，指标偏高。**190 题消融评测集的结果更真实可信**。

---

## 快速开始

### 环境要求

- Python **3.10+**
- 建议 **8GB+** 内存（首次运行需加载 BGE 模型，约 **1.3GB**）
- Windows / macOS / Linux

### 安装

```bash
# 1. 克隆项目
git clone https://github.com/LINGXIAO-aico/Python-AI.git
cd Python-AI/Python-AI-qimo

# 2. 创建虚拟环境（推荐）
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 配置 API

在项目**上级目录**创建 `deepseek.env` 文件：

```ini
DEEPSEEK_API_KEY=你的API密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_CHAT_MODEL=deepseek-chat
DEEPSEEK_REASONER_MODEL=deepseek-reasoner
```

> API 密钥可向项目负责人凌霄获取。不配置密钥仍可使用抽取式回答模式（离线运行）。

### 验证安装

```bash
python -c "import sentence_transformers, faiss, openai; print('环境就绪!')"
```

### 一步启动

```bash
streamlit run app.py
```

浏览器打开 **http://localhost:8501** 即可使用。

---

## 使用指南

### Streamlit Web 界面（推荐）

```bash
streamlit run app.py
```

- 底部输入框输入问题，回车发送
- 左侧栏切换 6 种回答模式
- 点击 **"检索来源"** 展开查看检索依据
- 支持多轮对话、亮色/暗色主题、对话导出

### 命令行

```bash
# 基础问答
python cli.py "图书馆周末几点开门"

# 指定检索策略
python cli.py --strategy rrf "研究生选课流程"
python cli.py --strategy dense "一卡通怎么充值"

# 开启 Self-RAG 校验
python cli.py --verify "申请缓考需要什么材料"

# 查看全部选项
python cli.py --help
```

### 运行评测

```bash
# 快速消融（仅检索指标，约 2 分钟，不调 API）
python scripts/20_run_ablation.py --quick --skip-baseline

# 完整消融（含 LLM 生成，约 30 分钟，需 API）
python scripts/20_run_ablation.py
```

### 运行测试

```bash
# 全部测试
pytest tests/ -v

# 含覆盖率
pytest tests/ -v --cov=campus_rag --cov-report=term
```

### 从头构建数据

```bash
# 1. 爬取同济网站
python scripts/10_crawl_tongji.py

# 2. 清洗页面
python scripts/11_pages_to_kb.py

# 3. 合并 FAQ 与爬取数据（需手动执行 Python）：
python -c "
import json
from pathlib import Path
def load(f): return [json.loads(l) for l in open(f, 'r', encoding='utf-8') if l.strip()]
kb = load('data/raw/campus_kb.jsonl')
faq = load('data/raw/campus_faq.jsonl')
combined = kb + faq
Path('data/raw/combined_kb.jsonl').write_text(
    '\n'.join(json.dumps(x, ensure_ascii=False) for x in combined),
    encoding='utf-8'
)
print(f'合并完成: {len(combined)} 条')
"

# 4. 数据清洗 + 分块
python scripts/01_prepare_data.py

# 5. 构建索引
python scripts/02_build_index.py
```

---

## 项目结构

```
Python-AI-qimo/
│
├── app.py                     # Streamlit 主应用（6 种模式，多轮对话，流式输出）
├── cli.py                     # 命令行问答工具
├── pyproject.toml             # 项目配置（依赖 / lint / 类型检查 / 测试）
├── requirements.txt           # Python 依赖列表
│
├── campus_rag/                # 核心代码库（10 个模块）
│   ├── config.py              # 路径、API 密钥、模型参数配置
│   ├── data.py                # JSONL 读写、数据清洗、去重
│   ├── embeddings.py          # BGE-large-zh-v1.5 嵌入模型封装
│   ├── vectorstore.py         # FAISS HNSW 向量存储与检索
│   ├── retriever.py           # 4 类检索器（TF-IDF / BM25 / Dense / RRF）
│   ├── reranker.py            # BGE-reranker-v2-m3 Cross-Encoder 重排
│   ├── splitter.py            # 中文标点感知的递归文本切分
│   ├── generator.py           # DeepSeek 流式生成 + Self-RAG 校验
│   ├── query_rewriter.py      # HyDE 假设文档 / Multi-Query 多视角改写
│   ├── evaluate.py            # Hit@K / MRR / nDCG / LLM-as-Judge
│   └── memory.py              # 多轮对话记忆（滑动窗口）
│
├── scripts/                   # 工具脚本
│   ├── 01_prepare_data.py     # 数据清洗 → 分块 → 数据剖析报告
│   ├── 02_build_index.py      # 构建全部索引（TF-IDF / BGE+FAISS / BM25）
│   ├── 03_evaluate.py         # 多策略评测 + LLM-as-Judge + 图表生成
│   ├── 10_crawl_tongji.py     # 爬虫：同济 7 大网站合规爬取
│   ├── 11_pages_to_kb.py      # 爬取页面清洗为知识库格式
│   ├── 12_build_eval.py       # 基于知识库自动生成评测集
│   ├── 20_run_ablation.py     # 消融实验（9 组配置，逐层对比）
│   └── build_reports.py       # 报告生成
│
├── tests/                     # 单元测试（13 个文件，49 个用例）
│   ├── test_config.py         # 配置与路径
│   ├── test_data.py           # 数据读写与清洗
│   ├── test_splitter.py       # 文本切分
│   ├── test_retriever.py      # 检索器
│   ├── test_embeddings.py     # 嵌入模型
│   ├── test_vectorstore.py    # 向量存储
│   ├── test_reranker.py       # 重排序
│   ├── test_generator.py      # 答案生成与校验
│   ├── test_evaluate_metrics.py # 评测指标
│   ├── test_query_rewriter.py # 查询改写
│   ├── test_memory.py         # 对话记忆
│   ├── test_loaders.py        # 文件加载
│   └── test_pipeline.py       # 端到端管线
│
├── data/
│   ├── raw/                   # 原始数据
│   │   ├── campus_faq.jsonl       # 人工 FAQ（120 条）
│   │   ├── crawled_pages.jsonl    # 爬虫原始输出
│   │   ├── campus_kb.jsonl        # 清洗后的爬取知识库
│   │   ├── combined_kb.jsonl      # 合并知识库（FAQ + 爬取，239 篇）
│   │   ├── eval_questions.jsonl   # 评测集（50 题）
│   │   └── eval_150.jsonl         # 评测集（190 题，含陷阱题）
│   └── processed/              # 清洗后数据
│       ├── knowledge_base_clean.csv  # 去重清洗后（238 篇）
│       └── chunks.csv               # 分块结果（538 chunks）
│
├── models/                    # 模型与索引
│   ├── tfidf_vector_store.joblib  # TF-IDF 索引
│   ├── faiss_index.bin            # FAISS HNSW 索引
│   ├── chunk_meta.parquet         # Chunk 元数据
│   ├── bge_cache/                 # BGE 模型缓存（约 1.3GB）
│   └── reranker_cache/            # Reranker 模型缓存
│
├── logs/                      # 评测日志与实验结果
│   ├── data_profile.json          # 数据剖析报告
│   ├── training_log.json          # 索引构建日志
│   ├── evaluation_summary.json    # 评测汇总
│   ├── evaluation_detail.csv      # 逐题评测明细
│   ├── retrieval_strategy_comparison.csv  # 多策略对比
│   ├── ablation_results.csv       # 9 组消融结果
│   └── llm_judge_scores.csv       # LLM-as-Judge 评分
│
├── reports/                   # 报告文档
│   ├── 运行手册.md
│   └── 汇报ppt.pptx
│
├── assets/
│   └── tongji-logo.png
│
└── .github/workflows/
    └── ci.yml                 # CI/CD（ruff + mypy + pytest）
```

---

## 技术栈

| 层次 | 组件 | 技术选型 |
|---|---|---|
| **嵌入模型** | BAAI/bge-large-zh-v1.5 | C-MTEB 中文基准 SOTA，1024 维 |
| **向量存储** | FAISS HNSW | 内存高效，内积度量，hnsw_m=32 |
| **稀疏检索** | jieba 分词 + rank-bm25 | 中文分词准确，BM25 标准化实现 |
| **重排序** | BAAI/bge-reranker-v2-m3 | Cross-Encoder 全注意力交互 |
| **文本切分** | RecursiveCharacterTextSplitter | 中文标点感知，chunk_size=360, overlap=80 |
| **大模型** | DeepSeek-chat / deepseek-reasoner | API 稳定，成本低，reasoner 支持推理链 |
| **前端** | Streamlit + Gradio | 快速构建交互式 Web UI |
| **爬虫** | httpx + BeautifulSoup + lxml | 合规爬取，遵守 robots.txt |
| **工程化** | ruff + mypy + pytest + GitHub Actions | 代码规范、类型检查、自动测试 |

### 依赖清单

```
pandas, numpy, scikit-learn, joblib        # 数据处理
matplotlib, seaborn                         # 可视化
streamlit, gradio                            # Web UI
openai                                       # DeepSeek API
sentence-transformers, faiss-cpu             # 嵌入 & 向量检索
rank-bm25, jieba                             # 稀疏检索
langchain-text-splitters                     # 文档切分
httpx, beautifulsoup4, lxml                  # 爬虫
pytest                                       # 测试
python-dotenv                                # 环境变量
```

---

## 参考文献

[1] Lewis, P., et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS*, 2020.  
[2] Karpukhin, V., et al. "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP*, 2020.  
[3] Gao, L., et al. "Precise Zero-Shot Dense Retrieval without Relevance Labels." *arXiv:2212.10496*, 2022.  
[4] Es, S., et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." *arXiv:2309.15217*, 2023.  
[5] Xiao, S., et al. "C-Pack: Packaged Resources To Advance General Chinese Embedding." *SIGIR*, 2024.  
[6] Robertson, S., & Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in IR*, 2009.  
[7] Cormack, G.V., et al. "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods." *SIGIR*, 2009.  

---

<p align="center">
  <sub>B07 小组 · 同济大学《Python人工智能程序设计实践》课程项目 · 2026</sub>
</p>
