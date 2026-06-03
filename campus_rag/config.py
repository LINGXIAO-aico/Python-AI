from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# 自动加载项目根目录的 deepseek.env（支持多级目录查找）
_ENV_PATH = Path(__file__).resolve().parents[1] / "deepseek.env"
if not _ENV_PATH.exists():
    _ENV_PATH = Path(__file__).resolve().parents[2] / "deepseek.env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SOURCE_DOCS_DIR = PROJECT_ROOT / "data" / "source_docs"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

# === 原始数据路径 ===
RAW_KB_PATH = RAW_DIR / "combined_kb.jsonl"
RAW_FAQ_PATH = RAW_DIR / "campus_faq.jsonl"  # 保留旧FAQ路径
RAW_EVAL_PATH = RAW_DIR / "eval_questions.jsonl"
CRAWLED_PAGES_PATH = RAW_DIR / "crawled_pages.jsonl"
CAMPUS_KB_PATH = RAW_DIR / "campus_kb.jsonl"
EVAL_150_PATH = RAW_DIR / "eval_150.jsonl"

# === 处理后数据 ===
CLEAN_KB_PATH = PROCESSED_DIR / "knowledge_base_clean.csv"
CHUNK_PATH = PROCESSED_DIR / "chunks.csv"

# === 模型与索引 ===
INDEX_PATH = MODEL_DIR / "tfidf_vector_store.joblib"
BGE_CACHE_DIR = MODEL_DIR / "bge_cache"
FAISS_INDEX_PATH = MODEL_DIR / "faiss_index.bin"
CHUNK_META_PATH = MODEL_DIR / "chunk_meta.parquet"
RERANKER_CACHE_DIR = MODEL_DIR / "reranker_cache"

# === 日志 ===
DATA_PROFILE_PATH = LOG_DIR / "data_profile.json"
TRAINING_LOG_PATH = LOG_DIR / "training_log.json"
EVAL_DETAIL_PATH = LOG_DIR / "evaluation_detail.csv"
RETRIEVAL_STRATEGY_PATH = LOG_DIR / "retrieval_strategy_comparison.csv"
EVAL_SUMMARY_PATH = LOG_DIR / "evaluation_summary.json"
ABLATION_RESULTS_PATH = LOG_DIR / "ablation_results.csv"
LLM_JUDGE_PATH = LOG_DIR / "llm_judge_scores.csv"

# === DeepSeek API 配置 ===
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
DEEPSEEK_REASONER_MODEL = os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner")

# === 嵌入模型配置 ===
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# === Reranker 模型配置 ===
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# === 检索配置 ===
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
DENSE_RECALL_K = int(os.getenv("DENSE_RECALL_K", "20"))
BM25_RECALL_K = int(os.getenv("BM25_RECALL_K", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))

# === 切分配置 ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# === 多轮对话配置 ===
MAX_TURNS = int(os.getenv("MAX_TURNS", "3"))

# === 目录初始化 ===
for _dir in [MODEL_DIR, LOG_DIR, FIGURE_DIR, BGE_CACHE_DIR, RERANKER_CACHE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
