"""BGE 重排序器测试。"""

from __future__ import annotations

import pytest

from campus_rag.config import RERANKER_CACHE_DIR
from campus_rag.reranker import BGEReranker
from campus_rag.retriever import RetrievedChunk


class DummyCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.seen_pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]], show_progress_bar: bool) -> list[float]:
        self.seen_pairs = pairs
        return self.scores[: len(pairs)]


def _chunk(idx: int, content: str) -> RetrievedChunk:
    return RetrievedChunk(
        rank=idx,
        score=0.0,
        chunk_id=f"{idx:03d}_C01",
        doc_id=f"{idx:03d}",
        category="校园事务",
        title=f"资料{idx}",
        content=content,
        source="测试数据",
        url=f"https://example.edu/{idx}",
    )


def _chunks() -> list[RetrievedChunk]:
    return [
        _chunk(1, "图书馆开放时间为工作日八点到二十二点。"),
        _chunk(2, "宿舍空调损坏可以通过后勤平台报修。"),
        _chunk(3, "奖学金申请需要关注学院通知。"),
    ]


def _has_cached_model_files() -> bool:
    return any(RERANKER_CACHE_DIR.rglob("modules.json")) or any(RERANKER_CACHE_DIR.rglob("config.json"))


def test_reranker_is_lazy_loaded() -> None:
    reranker = BGEReranker(model_name="local-reranker")

    assert reranker.model_name == "local-reranker"
    assert reranker._model is None


def test_rerank_returns_top_k_sorted_by_score() -> None:
    reranker = BGEReranker(model_name="local-reranker")
    reranker._model = DummyCrossEncoder([0.2, 0.9, 0.5])

    results = reranker.rerank("宿舍怎么报修？", _chunks(), top_k=2)

    assert len(results) == 2
    assert [item.doc_id for item in results] == ["002", "003"]
    assert [item.rank for item in results] == [1, 2]
    assert results[0].score >= results[1].score


def test_rerank_with_scores_returns_ranked_pairs() -> None:
    reranker = BGEReranker(model_name="local-reranker")
    dummy = DummyCrossEncoder([0.3, 0.1, 0.8])
    reranker._model = dummy
    chunks = _chunks()

    ranked = reranker.rerank_with_scores("奖学金", chunks)

    assert [(chunk.doc_id, score) for chunk, score in ranked] == [
        ("003", 0.8),
        ("001", 0.3),
        ("002", 0.1),
    ]
    assert dummy.seen_pairs[0] == ("奖学金", chunks[0].content)


def test_rerank_empty_chunks_returns_empty_list() -> None:
    reranker = BGEReranker(model_name="local-reranker")

    assert reranker.rerank("任意问题", [], top_k=3) == []
    assert reranker.rerank_with_scores("任意问题", []) == []


@pytest.mark.model
def test_real_reranker_returns_top_k_scores() -> None:
    if not _has_cached_model_files():
        pytest.skip("本地 reranker 模型缓存缺失，跳过真实模型集成测试。")

    reranker = BGEReranker()

    results = reranker.rerank("图书馆开放时间？", _chunks(), top_k=2)

    assert len(results) == 2
    assert all(isinstance(item.score, float) for item in results)
    assert results[0].score >= results[1].score
