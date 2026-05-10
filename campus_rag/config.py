from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RAW_KB_PATH = RAW_DIR / "campus_faq.jsonl"
RAW_EVAL_PATH = RAW_DIR / "eval_questions.jsonl"
CLEAN_KB_PATH = PROCESSED_DIR / "knowledge_base_clean.csv"
CHUNK_PATH = PROCESSED_DIR / "chunks.csv"
INDEX_PATH = MODEL_DIR / "tfidf_vector_store.joblib"
DATA_PROFILE_PATH = LOG_DIR / "data_profile.json"
TRAINING_LOG_PATH = LOG_DIR / "training_log.json"
EVAL_DETAIL_PATH = LOG_DIR / "evaluation_detail.csv"
RETRIEVAL_STRATEGY_PATH = LOG_DIR / "retrieval_strategy_comparison.csv"
EVAL_SUMMARY_PATH = LOG_DIR / "evaluation_summary.json"
