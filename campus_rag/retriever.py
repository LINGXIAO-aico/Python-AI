from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[RetrievedChunk]:
        scores = self.score_all(query)
        return _rank_chunks(self.chunks, scores, top_k=top_k, min_score=min_score)


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize Chinese campus text without requiring jieba.

    We keep English/number spans and add Chinese characters plus bigrams.
    This gives BM25 enough lexical signal for short Chinese questions.
    """
    text = str(text).lower()
    tokens: list[str] = []
    tokens.extend(re.findall(r"[a-z0-9_@.]+", text))
    for seq in re.findall(r"[\u4e00-\u9fff]+", text):
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

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[RetrievedChunk]:
        scores = self.score_all(query)
        return _rank_chunks(self.chunks, scores, top_k=top_k, min_score=min_score)


class HybridRetriever:
    def __init__(self, tfidf: TfidfRetriever, bm25: BM25Retriever, dense_weight: float = 0.5):
        if len(tfidf.chunks) != len(bm25.chunks):
            raise ValueError("TF-IDF and BM25 retrievers must use the same chunk table.")
        self.tfidf = tfidf
        self.bm25 = bm25
        self.dense_weight = dense_weight
        self.chunks = tfidf.chunks

    def score_all(self, query: str) -> np.ndarray:
        dense_scores = _minmax(self.tfidf.score_all(query))
        bm25_scores = _minmax(self.bm25.score_all(query))
        return self.dense_weight * dense_scores + (1 - self.dense_weight) * bm25_scores

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[RetrievedChunk]:
        scores = self.score_all(query)
        return _rank_chunks(self.chunks, scores, top_k=top_k, min_score=min_score)


def _minmax(scores: np.ndarray) -> np.ndarray:
    if not len(scores):
        return scores
    min_score = float(scores.min())
    max_score = float(scores.max())
    if max_score == min_score:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_score) / (max_score - min_score)


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
