"""FAISS 向量库测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from campus_rag.vectorstore import FAISSStore


def _make_chunks() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": "001_C01",
                "doc_id": "001",
                "category": "教务",
                "title": "选课",
                "content": "学生在教务系统完成选课。",
                "source": "教务处",
                "url": "https://example.edu/1",
            },
            {
                "chunk_id": "002_C01",
                "doc_id": "002",
                "category": "图书馆",
                "title": "开放时间",
                "content": "图书馆工作日开放至晚上十点。",
                "source": "图书馆",
                "url": "https://example.edu/2",
            },
            {
                "chunk_id": "003_C01",
                "doc_id": "003",
                "category": "后勤",
                "title": "宿舍报修",
                "content": "宿舍设施损坏可在线提交报修。",
                "source": "后勤处",
                "url": "https://example.edu/3",
            },
        ]
    )


def _make_embeddings() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.6, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def test_build_creates_index_with_all_vectors() -> None:
    store = FAISSStore(dim=4)

    store.build(_make_embeddings(), _make_chunks(), hnsw_m=8)

    assert store.index is not None
    assert store.index.ntotal == 3
    assert store.size == 3


def test_search_returns_top_k_results_in_descending_score_order() -> None:
    store = FAISSStore(dim=4)
    store.build(_make_embeddings(), _make_chunks(), hnsw_m=8)

    scores, indices = store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=2)

    assert scores.shape == (1, 2)
    assert indices.shape == (1, 2)
    assert list(indices[0]) == [0, 1]
    assert scores[0][0] >= scores[0][1]


def test_save_and_load_round_trip(tmp_path) -> None:
    index_path = tmp_path / "faiss_index.bin"
    meta_path = tmp_path / "chunk_meta.parquet"
    store = FAISSStore(dim=4)
    store.build(_make_embeddings(), _make_chunks(), hnsw_m=8)

    store.save(index_path=index_path, meta_path=meta_path)
    loaded = FAISSStore.load(dim=4, index_path=index_path, meta_path=meta_path)
    scores, indices = loaded.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=2)

    assert loaded.size == 3
    assert len(loaded.chunks) == 3
    assert list(indices[0]) == [0, 1]
    assert scores[0][0] >= scores[0][1]


def test_search_before_build_raises() -> None:
    store = FAISSStore(dim=4)

    with pytest.raises(RuntimeError, match="索引未加载"):
        store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=1)


@pytest.mark.model
def test_vectorstore_accepts_1024_dimensional_embeddings() -> None:
    store = FAISSStore(dim=1024)
    embeddings = np.zeros((2, 1024), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 1] = 1.0

    store.build(embeddings, _make_chunks().iloc[:2], hnsw_m=8)
    scores, indices = store.search(embeddings[0], top_k=1)

    assert store.index is not None
    assert store.index.ntotal == 2
    assert indices[0][0] == 0
    assert scores[0][0] == pytest.approx(1.0)
