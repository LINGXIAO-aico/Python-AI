from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import BM25_RECALL_K, DENSE_RECALL_K, RETRIEVAL_TOP_K, RRF_K
from .embeddings import BGEEmbedder
from .vectorstore import FAISSStore


@dataclass
class RetrievedChunk:
    rank: int
    score: float
    chunk_id: str
    doc_id: str
    category: str
    title: str
    content: str
    source: str
    url: str

    @classmethod
    def from_row(cls, rank: int, score: float, row: pd.Series) -> "RetrievedChunk":
        return cls(
            rank=rank,
            score=round(float(score), 4),
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            category=str(row["category"]),
            title=str(row["title"]),
            content=str(row["content"]),
            source=str(row["source"]),
            url=str(row["url"]),
        )

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "category": self.category,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "url": self.url,
        }


# ============================================================
# 通用工具
# ============================================================

def _minmax(scores: np.ndarray) -> np.ndarray:
    if not len(scores):
        return scores
    mn, mx = float(scores.min()), float(scores.max())
    if mx == mn:
        return np.zeros_like(scores, dtype=float)
    return (scores - mn) / (mx - mn)


def _rank_chunks(
    chunks: pd.DataFrame,
    scores: np.ndarray,
    top_k: int = 3,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    if not len(scores):
        return []
    candidate_ids = np.argsort(scores)[::-1][:top_k]
    results: list[RetrievedChunk] = []
    for rank, row_idx in enumerate(candidate_ids, start=1):
        score = float(scores[row_idx])
        if score < min_score:
            continue
        results.append(RetrievedChunk.from_row(rank, score, chunks.iloc[row_idx]))
    return results


def _rrf_fusion(
    rank_lists: list[list[tuple[int, float]]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion：合并多路检索排名。"""
    scores: dict[int, float] = {}
    for ranked in rank_lists:
        for rank, (idx, _score) in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# BGE Dense Retriever（FAISS HNSW）
# ============================================================

class DenseRetriever:
    def __init__(self, embedder: BGEEmbedder, store: FAISSStore):
        self.embedder = embedder
        self.store = store

    @classmethod
    def build(
        cls,
        embedder: BGEEmbedder,
        chunks: pd.DataFrame,
    ) -> "DenseRetriever":
        texts = chunks["content"].fillna("").astype(str).tolist()
        embeddings = embedder.encode_documents(texts)
        store = FAISSStore(dim=embedder.dim)
        store.build(embeddings, chunks)
        return cls(embedder, store)

    @classmethod
    def load(cls, embedder: BGEEmbedder, dim: int) -> "DenseRetriever":
        store = FAISSStore.load(dim)
        return cls(embedder, store)

    def save(self) -> None:
        self.store.save()

    def retrieve(
        self,
        query: str,
        top_k: int = DENSE_RECALL_K,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        q_vec = self.embedder.encode_query(query)
        scores, indices = self.store.search(q_vec, top_k)
        chunks = self.store.chunks
        results: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(chunks):
                continue
            if float(score) < min_score:
                continue
            results.append(RetrievedChunk.from_row(
                rank, float(score), chunks.iloc[int(idx)]
            ))
        return results

    def retrieve_with_indices(
        self, query: str, top_k: int = DENSE_RECALL_K
    ) -> list[tuple[int, float]]:
        q_vec = self.embedder.encode_query(query)
        scores, indices = self.store.search(q_vec, top_k)
        return [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]


# ============================================================
# TF-IDF Retriever（保留用于消融）
# ============================================================

class TfidfRetriever:
    def __init__(self, vectorizer: TfidfVectorizer, matrix, chunks: pd.DataFrame):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks.reset_index(drop=True)

    @classmethod
    def fit(cls, chunks: pd.DataFrame) -> "TfidfRetriever":
        texts = chunks["content"].fillna("").astype(str).tolist()
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm="l2",
        )
        matrix = vectorizer.fit_transform(texts)
        return cls(vectorizer=vectorizer, matrix=matrix, chunks=chunks.copy())

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "matrix": self.matrix, "chunks": self.chunks},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "TfidfRetriever":
        payload = joblib.load(path)
        return cls(payload["vectorizer"], payload["matrix"], payload["chunks"])

    def score_all(self, query: str) -> np.ndarray:
        query_vec = self.vectorizer.transform([query])
        return cosine_similarity(query_vec, self.matrix).ravel()

    def retrieve(
        self, query: str, top_k: int = RETRIEVAL_TOP_K, min_score: float = 0.0
    ) -> list[RetrievedChunk]:
        scores = self.score_all(query)
        return _rank_chunks(self.chunks, scores, top_k=top_k, min_score=min_score)


# ============================================================
# 旧 BM25（保留用于消融）
# ============================================================

def tokenize_for_bm25(text: str) -> list[str]:
    text = str(text).lower()
    tokens: list[str] = []
    tokens.extend(re.findall(r"[a-z0-9_@.]+", text))
    for seq in re.findall(r"[一-鿿]+", text):
        tokens.extend(seq)
        tokens.extend(seq[i : i + 2] for i in range(max(0, len(seq) - 1)))
    return tokens


class BM25Retriever:
    def __init__(
        self,
        chunks: pd.DataFrame,
        doc_token_counts: list[Counter],
        doc_lengths: np.ndarray,
        idf: dict[str, float],
        avgdl: float,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.chunks = chunks.reset_index(drop=True)
        self.doc_token_counts = doc_token_counts
        self.doc_lengths = doc_lengths
        self.idf = idf
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b

    @classmethod
    def fit(cls, chunks: pd.DataFrame, k1: float = 1.5, b: float = 0.75) -> "BM25Retriever":
        texts = chunks["content"].fillna("").astype(str).tolist()
        tokenized = [tokenize_for_bm25(text) for text in texts]
        doc_token_counts = [Counter(tokens) for tokens in tokenized]
        doc_lengths = np.array([len(tokens) for tokens in tokenized], dtype=float)
        avgdl = float(doc_lengths.mean()) if len(doc_lengths) else 0.0
        doc_freq: Counter = Counter()
        for tokens in tokenized:
            doc_freq.update(set(tokens))
        total_docs = len(tokenized)
        idf = {
            token: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }
        return cls(chunks.copy(), doc_token_counts, doc_lengths, idf, avgdl, k1=k1, b=b)

    def score_all(self, query: str) -> np.ndarray:
        query_terms = tokenize_for_bm25(query)
        scores = np.zeros(len(self.chunks), dtype=float)
        if not query_terms or not len(self.chunks):
            return scores
        query_counts = Counter(query_terms)
        for idx, doc_counts in enumerate(self.doc_token_counts):
            dl = self.doc_lengths[idx]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            score = 0.0
            for term, qf in query_counts.items():
                tf = doc_counts.get(term, 0)
                if not tf:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + denom_norm
                score += self.idf.get(term, 0.0) * numerator / denominator * qf
            scores[idx] = score
        return scores

    def retrieve(
        self, query: str, top_k: int = RETRIEVAL_TOP_K, min_score: float = 0.0
    ) -> list[RetrievedChunk]:
        scores = self.score_all(query)
        return _rank_chunks(self.chunks, scores, top_k=top_k, min_score=min_score)


# ============================================================
# jieba-BM25 Retriever（新）
# ============================================================

class Bm25JiebaRetriever:
    def __init__(self, chunks: pd.DataFrame, bm25: BM25Okapi):
        self.chunks = chunks.reset_index(drop=True)
        self.bm25 = bm25

    @classmethod
    def fit(cls, chunks: pd.DataFrame) -> "Bm25JiebaRetriever":
        import jieba
        texts = chunks["content"].fillna("").astype(str).tolist()
        tokenized = [list(jieba.cut(text)) for text in texts]
        bm25 = BM25Okapi(tokenized)
        return cls(chunks.copy(), bm25)

    def retrieve(
        self,
        query: str,
        top_k: int = BM25_RECALL_K,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        import jieba
        tokens = list(jieba.cut(str(query)))
        scores = self.bm25.get_scores(tokens)
        return _rank_chunks(self.chunks, np.array(scores), top_k=top_k, min_score=min_score)

    def retrieve_with_indices(
        self, query: str, top_k: int = BM25_RECALL_K
    ) -> list[tuple[int, float]]:
        import jieba
        tokens = list(jieba.cut(str(query)))
        scores = self.bm25.get_scores(tokens)
        indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in indices]


# ============================================================
# 旧 Hybrid（加权和，保留用于消融）
# ============================================================

class HybridRetriever:
    def __init__(self, tfidf: TfidfRetriever, bm25: BM25Retriever, dense_weight: float = 0.5):
        if len(tfidf.chunks) != len(bm25.chunks):
            raise ValueError("TF-IDF and BM25 must use the same chunk table.")
        self.tfidf = tfidf
        self.bm25 = bm25
        self.dense_weight = dense_weight
        self.chunks = tfidf.chunks

    def score_all(self, query: str) -> np.ndarray:
        return self.dense_weight * _minmax(self.tfidf.score_all(query)) + \
               (1 - self.dense_weight) * _minmax(self.bm25.score_all(query))

    def retrieve(
        self, query: str, top_k: int = RETRIEVAL_TOP_K, min_score: float = 0.0
    ) -> list[RetrievedChunk]:
        scores = self.score_all(query)
        return _rank_chunks(self.chunks, scores, top_k=top_k, min_score=min_score)


# ============================================================
# Hybrid RRF Retriever（新）
# ============================================================

class HybridRRFRetriever:
    """RRF 融合 BGE Dense + jieba-BM25，免调参、更鲁棒。"""

    def __init__(self, dense: DenseRetriever, bm25: Bm25JiebaRetriever, k: int = RRF_K):
        if len(dense.store.chunks) != len(bm25.chunks):
            raise ValueError("Dense and BM25 retrievers must use the same chunk table.")
        self.dense = dense
        self.bm25 = bm25
        self.k = k
        self.chunks = bm25.chunks

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        dense_k: int = DENSE_RECALL_K,
        bm25_k: int = BM25_RECALL_K,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        dense_ranks = self.dense.retrieve_with_indices(query, top_k=dense_k)
        bm25_ranks = self.bm25.retrieve_with_indices(query, top_k=bm25_k)
        fused = _rrf_fusion([dense_ranks, bm25_ranks], k=self.k)

        top_fused = fused[:top_k]
        results: list[RetrievedChunk] = []
        for rank, (chunk_idx, score) in enumerate(top_fused, start=1):
            results.append(RetrievedChunk.from_row(
                rank, score, self.chunks.iloc[chunk_idx]
            ))
        return [r for r in results if r.score >= min_score]
