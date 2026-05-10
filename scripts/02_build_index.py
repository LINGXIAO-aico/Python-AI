from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_rag.config import CHUNK_PATH, INDEX_PATH, LOG_DIR, MODEL_DIR, TRAINING_LOG_PATH
from campus_rag.data import ensure_dirs
from campus_rag.retriever import BM25Retriever, TfidfRetriever


def main() -> None:
    ensure_dirs(MODEL_DIR, LOG_DIR)
    chunks = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")
    started = time.perf_counter()
    retriever = TfidfRetriever.fit(chunks)
    bm25 = BM25Retriever.fit(chunks)
    retriever.save(INDEX_PATH)
    elapsed = time.perf_counter() - started
    log = {
        "model": "TF-IDF char_wb ngram(2,4) + custom BM25 + hybrid ensemble",
        "chunk_count": int(len(chunks)),
        "vocabulary_size": int(len(retriever.vectorizer.vocabulary_)),
        "bm25_vocabulary_size": int(len(bm25.idf)),
        "matrix_shape": list(retriever.matrix.shape),
        "train_seconds": round(elapsed, 4),
        "index_path": str(INDEX_PATH),
        "retrieval_strategies": ["tfidf_vector", "bm25_keyword", "hybrid_50_50"],
    }
    with TRAINING_LOG_PATH.open("w", encoding="utf-8") as file:
        json.dump(log, file, ensure_ascii=False, indent=2)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
