#!/usr/bin/env python3
"""构建全部索引：TF-IDF + BGE Dense (FAISS) + BM25，写入 models/。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_rag.config import (
    BGE_CACHE_DIR,
    CHUNK_PATH,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
    INDEX_PATH,
    LOG_DIR,
    MODEL_DIR,
    TRAINING_LOG_PATH,
)
from campus_rag.data import ensure_dirs
from campus_rag.embeddings import BGEEmbedder
from campus_rag.retriever import Bm25JiebaRetriever, BM25Retriever, DenseRetriever, TfidfRetriever


def main() -> None:
    ensure_dirs(MODEL_DIR, LOG_DIR, BGE_CACHE_DIR)
    chunks = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")
    print(f"知识库 chunks: {len(chunks)}")

    # 1. TF-IDF
    t0 = time.perf_counter()
    tfidf = TfidfRetriever.fit(chunks)
    tfidf.save(INDEX_PATH)
    tfidf_elapsed = time.perf_counter() - t0
    print(f"TF-IDF: 词表 {len(tfidf.vectorizer.vocabulary_)} ({tfidf_elapsed:.1f}s)")

    # 2. BM25 (旧)
    t0 = time.perf_counter()
    bm25_old = BM25Retriever.fit(chunks)
    bm25_elapsed = time.perf_counter() - t0
    print(f"BM25(旧): 词表 {len(bm25_old.idf)} ({bm25_elapsed:.1f}s)")

    # 3. jieba-BM25
    t0 = time.perf_counter()
    Bm25JiebaRetriever.fit(chunks)
    jieba_elapsed = time.perf_counter() - t0
    print(f"jieba-BM25: ({jieba_elapsed:.1f}s)")

    # 4. BGE Dense + FAISS
    t0 = time.perf_counter()
    embedder = BGEEmbedder()
    print(f"BGE 模型: {embedder.model_name}")
    dense = DenseRetriever.build(embedder, chunks)
    dense.save()
    dense_elapsed = time.perf_counter() - t0
    print(f"FAISS 索引: {dense.store.size} 向量 ({dense_elapsed:.1f}s)")

    total_elapsed = tfidf_elapsed + bm25_elapsed + jieba_elapsed + dense_elapsed

    log = {
        "build_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_count": int(len(chunks)),
        "retrieval_strategies": [
            "tfidf_vector",
            "bm25_old",
            "bm25_jieba",
            "bge_dense_faiss",
            "hybrid_rrf",
        ],
        "tfidf": {
            "vocabulary_size": int(len(tfidf.vectorizer.vocabulary_)),
            "matrix_shape": list(tfidf.matrix.shape),
            "build_seconds": round(tfidf_elapsed, 3),
            "index_path": str(INDEX_PATH),
        },
        "bm25_old": {
            "vocabulary_size": int(len(bm25_old.idf)),
            "build_seconds": round(bm25_elapsed, 3),
        },
        "bm25_jieba": {
            "build_seconds": round(jieba_elapsed, 3),
        },
        "bge_dense_faiss": {
            "model": embedder.model_name,
            "dim": EMBEDDING_DIM,
            "total_vectors": dense.store.size,
            "build_seconds": round(dense_elapsed, 3),
            "faiss_path": str(FAISS_INDEX_PATH),
        },
        "total_build_seconds": round(total_elapsed, 3),
    }

    with TRAINING_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"\n训练日志: {TRAINING_LOG_PATH}")


if __name__ == "__main__":
    main()
