from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from .config import CHUNK_META_PATH, FAISS_INDEX_PATH


class FAISSStore:
    """FAISS HNSW 索引封装：build / save / load / search。"""

    def __init__(self, dim: int):
        self.dim = dim
        self.index: faiss.Index | None = None
        self._chunks: pd.DataFrame | None = None

    @property
    def chunks(self) -> pd.DataFrame:
        if self._chunks is None:
            raise RuntimeError("FAISSStore 未加载或未构建。")
        return self._chunks

    def build(
        self,
        embeddings: np.ndarray,
        chunks: pd.DataFrame,
        hnsw_m: int = 32,
    ) -> None:
        """从嵌入向量构建 HNSW 索引。"""
        embeddings = embeddings.astype(np.float32)
        self._chunks = chunks.reset_index(drop=True)

        self.index = faiss.IndexHNSWFlat(self.dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.index.train(embeddings)
        self.index.add(embeddings)

    def search(self, query_vec: np.ndarray, top_k: int = 20) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("索引未加载。请先 build 或 load。")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        return self.index.search(query_vec.astype(np.float32), min(top_k, self.index.ntotal))

    def save(self, index_path: Path | None = None, meta_path: Path | None = None) -> None:
        index_path = index_path or FAISS_INDEX_PATH
        meta_path = meta_path or CHUNK_META_PATH
        if self.index is None:
            raise RuntimeError("索引未构建，无法保存。")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        if self._chunks is not None:
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            self._chunks.to_parquet(meta_path, index=False)

    @classmethod
    def load(
        cls,
        dim: int,
        index_path: Path | None = None,
        meta_path: Path | None = None,
    ) -> "FAISSStore":
        index_path = index_path or FAISS_INDEX_PATH
        meta_path = meta_path or CHUNK_META_PATH
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS 索引文件不存在: {index_path}")

        store = cls(dim)
        store.index = faiss.read_index(str(index_path))
        if meta_path.exists():
            store._chunks = pd.read_parquet(meta_path)
        return store

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "total_vectors": self.size,
            "has_meta": self._chunks is not None,
            "chunk_count": len(self._chunks) if self._chunks is not None else 0,
        }
