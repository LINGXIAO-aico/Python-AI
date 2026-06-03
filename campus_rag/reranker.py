from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from .config import RERANKER_CACHE_DIR, RERANKER_MODEL_NAME
from .retriever import RetrievedChunk


class BGEReranker:
    """BGE Cross-Encoder 重排序器：对 Top-N 召回结果精排。"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or RERANKER_MODEL_NAME
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                cache_folder=str(RERANKER_CACHE_DIR),
            )
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return chunks

        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)
        scores = np.array(scores, dtype=float)

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results: list[RetrievedChunk] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = chunks[int(idx)]
            results.append(RetrievedChunk(
                rank=rank,
                score=round(float(scores[int(idx)]), 4),
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                category=chunk.category,
                title=chunk.title,
                content=chunk.content,
                source=chunk.source,
                url=chunk.url,
            ))
        return results

    def rerank_with_scores(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[tuple[RetrievedChunk, float]]:
        if not chunks:
            return []
        pairs = [(query, chunk.content) for chunk in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(chunk, round(float(score), 4)) for chunk, score in ranked]
