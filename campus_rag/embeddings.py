from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import BGE_CACHE_DIR, EMBEDDING_BATCH_SIZE, EMBEDDING_DIM, EMBEDDING_MODEL_NAME


class BGEEmbedder:
    """封装 BGE-large-zh-v1.5，批量编码 + L2 归一化 + 本地缓存。"""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or EMBEDDING_MODEL_NAME
        self.dim = EMBEDDING_DIM
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(BGE_CACHE_DIR),
            )
        return self._model

    def encode(
        self,
        texts: list[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """为查询添加 instruction 前缀（BGE 推荐）。"""
        if not queries:
            return np.empty((0, self.dim), dtype=np.float32)
        prefixed = [f"为这个句子生成表示以用于检索相关文章：{q}" for q in queries]
        return self.encode(prefixed, show_progress=False)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode_queries([query])[0]

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        return self.encode(documents)

    def get_cache_dir(self) -> Path:
        return BGE_CACHE_DIR
