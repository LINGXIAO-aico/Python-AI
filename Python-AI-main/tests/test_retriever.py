"""检索器模块测试（无需模型的轻量检索器）。"""

import tempfile
from pathlib import Path

import pandas as pd

from campus_rag.retriever import (
    BM25Retriever,
    Bm25JiebaRetriever,
    RetrievedChunk,
    TfidfRetriever,
    _minmax,
)


def _make_chunks() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "chunk_id": "001_C01", "doc_id": "001", "category": "教务",
            "title": "选课通知", "content": "每学期第2-3周为选课周，学生登录教务系统进行选课操作。",
            "source": "教务处", "url": "http://test.com/1", "last_updated": "2026-01-01",
            "char_length": 30,
        },
        {
            "chunk_id": "002_C01", "doc_id": "002", "category": "图书馆",
            "title": "开放时间", "content": "图书馆周一至周五 8:00-22:00 开放，周末 9:00-21:00。",
            "source": "图书馆", "url": "http://test.com/2", "last_updated": "2026-01-01",
            "char_length": 32,
        },
        {
            "chunk_id": "003_C01", "doc_id": "003", "category": "校园生活",
            "title": "宿舍报修", "content": "宿舍空调损坏可通过后勤报修平台申报，维修人员24小时内上门。",
            "source": "后勤处", "url": "http://test.com/3", "last_updated": "2026-01-01",
            "char_length": 32,
        },
    ])


class TestMinmax:
    def test_normal(self):
        import numpy as np
        result = _minmax(np.array([1.0, 2.0, 3.0], dtype=float))
        assert result[0] == 0.0
        assert result[2] == 1.0

    def test_all_same(self):
        import numpy as np
        result = _minmax(np.array([5.0, 5.0, 5.0], dtype=float))
        assert all(r == 0.0 for r in result)


class TestRetrievedChunk:
    def test_from_row(self):
        chunks = _make_chunks()
        chunk = RetrievedChunk.from_row(1, 0.8523, chunks.iloc[0])
        assert chunk.rank == 1
        assert chunk.score == 0.8523
        assert chunk.doc_id == "001"

    def test_to_dict(self):
        chunks = _make_chunks()
        chunk = RetrievedChunk.from_row(1, 0.9, chunks.iloc[0])
        d = chunk.to_dict()
        assert d["rank"] == 1
        assert d["doc_id"] == "001"
        assert "content" in d


class TestTfidfRetriever:
    def test_fit_and_retrieve(self):
        chunks = _make_chunks()
        retriever = TfidfRetriever.fit(chunks)
        results = retriever.retrieve("选课", top_k=3)
        assert len(results) > 0
        assert results[0].doc_id == "001"

    def test_save_and_load(self):
        chunks = _make_chunks()
        retriever = TfidfRetriever.fit(chunks)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = Path(f.name)
        retriever.save(path)
        loaded = TfidfRetriever.load(path)
        assert len(loaded.chunks) == len(chunks)
        path.unlink()


class TestBM25Retriever:
    def test_fit_and_retrieve(self):
        chunks = _make_chunks()
        retriever = BM25Retriever.fit(chunks)
        results = retriever.retrieve("图书馆开放时间", top_k=3)
        assert len(results) > 0
        assert results[0].doc_id == "002"


class TestBm25JiebaRetriever:
    def test_fit_and_retrieve(self):
        chunks = _make_chunks()
        retriever = Bm25JiebaRetriever.fit(chunks)
        results = retriever.retrieve("宿舍空调坏了怎么办", top_k=3)
        assert len(results) > 0
        assert results[0].doc_id == "003"

    def test_retrieve_with_indices(self):
        chunks = _make_chunks()
        retriever = Bm25JiebaRetriever.fit(chunks)
        indices = retriever.retrieve_with_indices("选课", top_k=3)
        assert len(indices) > 0
        assert isinstance(indices[0], tuple)
        assert isinstance(indices[0][0], int)
