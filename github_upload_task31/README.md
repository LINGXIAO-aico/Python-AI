# B07 RAG 校园智能问答助手

本项目面向《Python 人工智能程序设计实践》B07 题目，构建一个可离线复现、可接入大模型 API 的校园智能问答助手。系统围绕校园手册、课程问答、办事指南等资料构建知识库，完成数据清洗、文本切分、向量化、索引构建、检索增强问答、无检索基线对比和可视化评估。

## 小组分工

| 成员 | 主要职责 | 对应文件 |
|---|---|---|
| 归梦依 | 数据收集、PDF/Word 文本抽取、清洗规则、知识库搭建 | `campus_rag/loaders.py`, `scripts/00_extract_documents.py`, `scripts/01_prepare_data.py`, `data/` |
| 凌霄 | 检索策略设计与优化，完成向量检索、BM25、混合检索对比 | `campus_rag/retriever.py`, `scripts/02_build_index.py` |
| 周子涵 | RAG 链路、大模型 API 调用、CLI/Streamlit/Gradio 系统开发 | `campus_rag/generator.py`, `cli.py`, `app.py`, `gradio_app.py` |
| 杨歆苒 | 评测问题集、指标统计、可视化、报告撰写 | `campus_rag/evaluate.py`, `scripts/03_evaluate.py`, `reports/` |

## 项目结构

```text
B07_RAG_campus_assistant/
  app.py                         # Streamlit 演示界面
  gradio_app.py                  # Gradio 演示界面
  cli.py                         # 命令行问答入口
  campus_rag/                    # RAG 核心模块
  data/source_docs/              # 可放入真实 PDF/Word 原始文件
  data/raw/                      # 原始 FAQ 与评测问题
  data/processed/                # 清洗数据与文本块
  models/                        # 向量索引模型
  logs/                          # 清洗、训练、评估日志
  reports/                       # 选题报告、中期报告、图表
  scripts/                       # 项目流水线脚本
```

## 环境安装

建议使用 Python 3.10 及以上版本。

```bash
cd E:\python\B07_RAG_campus_assistant
python -m pip install -r requirements.txt
```

如后续要使用 LangChain + FAISS + 中文 Embedding，可额外安装：

```bash
python -m pip install -r requirements-advanced.txt
```

## 一键复现流程

当前仓库已经内置课程演示用校园 FAQ 数据，可直接运行完整流水线。当前 `campus_faq.jsonl` 为 120 条 FAQ，合并知识库清洗后约 238 条记录、538 个 chunks，评测集为 50 条问题。

```bash
python scripts/01_prepare_data.py
python scripts/02_build_index.py
python scripts/03_evaluate.py
```

运行后生成：

- `data/processed/knowledge_base_clean.csv`
- `data/processed/chunks.csv`
- `models/tfidf_vector_store.joblib`
- `logs/data_profile.json`
- `logs/training_log.json`
- `logs/retrieval_strategy_comparison.csv`
- `logs/evaluation_detail.csv`
- `logs/evaluation_summary.json`
- `reports/figures/*.png`

## 抽取真实校园文档

把学生手册、教务规定、图书馆指南等 PDF/DOCX/TXT/MD 文件放入：

```text
data/source_docs/
```

然后运行：

```bash
python scripts/00_extract_documents.py
```

脚本会输出 `data/raw/extracted_documents.jsonl`。当前中期版本仍使用结构化 FAQ 作为主知识库，后续可把真实文档抽取结果整理为同样的知识库字段。

## 命令行问答

默认使用混合检索（TF-IDF 向量 + BM25 关键词），不需要 API Key。

```bash
python cli.py "图书馆周末几点开门？"
python cli.py "校园卡丢了怎么办？" --strategy hybrid
python cli.py "忘记统一身份认证密码怎么办？" --strategy bm25
```

可选检索策略：

- `--strategy hybrid`：混合检索，默认策略
- `--strategy tfidf`：TF-IDF 向量检索
- `--strategy bm25`：BM25 关键词检索

## 接入通义千问或其他 OpenAI 兼容 API

无 API Key 时，系统使用离线抽取式回答，保证可复现。有通义千问 Key 后可运行：

```powershell
$env:DASHSCOPE_API_KEY="你的API_KEY"
python cli.py "学生证丢了怎么补办？" --backend qwen
```

也支持其他 OpenAI 兼容服务：

```powershell
$env:OPENAI_API_KEY="你的API_KEY"
$env:OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
$env:OPENAI_MODEL="your-model-name"
python cli.py "宿舍空调坏了从哪里报修？" --backend openai
```

## Web 演示

Streamlit：

```bash
streamlit run app.py
```

Gradio：

```bash
python gradio_app.py
```

## 对比实验

当前评测包含 50 条校园问题，记录以下对比：

1. 检索策略对比：TF-IDF 向量检索、BM25 关键词检索、混合检索。
2. 系统回答对比：RAG 系统回答 vs 无检索基线回答。
3. 指标：Hit@1、Hit@3、MRR、关键词召回率、平均检索耗时。

结果文件：

- `logs/retrieval_strategy_comparison.csv`
- `logs/evaluation_summary.json`
- `reports/figures/retrieval_strategy_comparison.png`
- `reports/figures/evaluation_comparison.png`

## 测试

```bash
python -m pytest -q
```

## 参考资料

- scikit-learn `TfidfVectorizer` 文档：https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Streamlit 文档：https://docs.streamlit.io/
- Gradio 文档：https://www.gradio.app/docs
- OpenAI Python SDK：https://github.com/openai/openai-python
- Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, NeurIPS 2020.
