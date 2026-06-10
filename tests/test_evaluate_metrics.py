"""评测指标（纯数学部分）测试。"""

from __future__ import annotations

import json

from campus_rag import evaluate
from campus_rag.evaluate import (
    evaluate_retriever,
    evaluate_retrievers,
    keyword_recall,
    ndcg_at_k,
    reciprocal_rank,
)
from campus_rag.retriever import RetrievedChunk


class TestKeywordRecall:
    def test_all_matched(self):
        assert keyword_recall("校园卡丢失补办流程", ["校园卡", "补办"]) == 1.0

    def test_partial_match(self):
        assert keyword_recall("校园卡丢失", ["校园卡", "补办"]) == 0.5

    def test_no_match(self):
        assert keyword_recall("图书馆开放时间", ["校园卡", "补办"]) == 0.0

    def test_empty_keywords(self):
        assert keyword_recall("任何回答", []) == 0.0


class TestReciprocalRank:
    def test_rank_one(self):
        assert reciprocal_rank("A", ["A", "B", "C"]) == 1.0

    def test_rank_two(self):
        assert reciprocal_rank("A", ["B", "A", "C"]) == 0.5

    def test_not_found(self):
        assert reciprocal_rank("A", ["B", "C", "D"]) == 0.0

    def test_empty_list(self):
        assert reciprocal_rank("A", []) == 0.0


class TestNdcgAtK:
    def test_gold_at_rank_one(self):
        score = ndcg_at_k("A", ["A", "B", "C", "D", "E"], k=5)
        assert score > 0.9  # 接近 1.0

    def test_gold_not_found(self):
        assert ndcg_at_k("A", ["B", "C", "D", "E", "F"], k=5) == 0.0

    def test_gold_at_rank_three(self):
        score = ndcg_at_k("A", ["B", "C", "A", "D", "E"], k=5)
        assert 0 < score < 0.7


def _chunk(doc_id: str, content: str = "学生登录教务系统选课。") -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        score=0.9,
        chunk_id=f"{doc_id}_C01",
        doc_id=doc_id,
        category="教务",
        title="选课通知",
        content=content,
        source="教务处",
        url="https://example.edu",
    )


class _FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.chunks[:top_k]


class _FakeReranker:
    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        return list(reversed(chunks))[:top_k]


def _write_eval(path) -> None:
    rows = [
        {
            "question": "如何选课？",
            "gold_doc_id": "001",
            "answer_keywords": ["教务系统", "选课"],
        },
        {
            "question": "图书馆开放时间？",
            "gold_doc_id": "002",
            "answer_keywords": ["图书馆"],
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def test_judge_helpers_parse_json(monkeypatch) -> None:
    monkeypatch.setattr(evaluate, "_judge", lambda template, **kwargs: {"score": 5})

    assert evaluate.judge_faithfulness("q", "a", "c")["score"] == 5
    assert evaluate.judge_relevancy("q", "a")["score"] == 5
    assert evaluate.judge_context_precision("q", "c")["score"] == 5


def test_evaluate_retrievers_writes_detail_summary_and_strategy_files(tmp_path, monkeypatch) -> None:
    eval_path = tmp_path / "eval.jsonl"
    detail_path = tmp_path / "detail.csv"
    summary_path = tmp_path / "summary.json"
    strategy_path = tmp_path / "strategy.csv"
    _write_eval(eval_path)
    monkeypatch.setattr(evaluate, "judge_faithfulness", lambda *args, **kwargs: {"score": 5})
    monkeypatch.setattr(evaluate, "judge_relevancy", lambda *args, **kwargs: {"score": 4})
    monkeypatch.setattr(evaluate, "judge_context_precision", lambda *args, **kwargs: {"score": 3})
    retrievers = {
        "hybrid": _FakeRetriever([_chunk("001"), _chunk("002", "图书馆开放")]),
        "bm25": _FakeRetriever([_chunk("002", "图书馆开放"), _chunk("001")]),
    }

    detail_df, summary = evaluate_retrievers(
        retrievers,
        eval_path=eval_path,
        detail_path=detail_path,
        summary_path=summary_path,
        strategy_detail_path=strategy_path,
        selected_strategy="missing",
        top_k=2,
        use_llm=True,
        reranker=None,
    )

    assert len(detail_df) == 2
    assert summary["selected_strategy"] == "hybrid"
    assert summary["llm_judge"]["faithfulness_mean"] == 5.0
    assert detail_path.exists()
    assert strategy_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["question_count"] == 2


def test_evaluate_retriever_wrapper(tmp_path) -> None:
    eval_path = tmp_path / "eval.jsonl"
    _write_eval(eval_path)

    detail_df, summary = evaluate_retriever(
        _FakeRetriever([_chunk("001"), _chunk("002", "图书馆开放")]),
        eval_path=eval_path,
        detail_path=tmp_path / "detail.csv",
        summary_path=tmp_path / "summary.json",
        top_k=2,
    )

    assert len(detail_df) == 2
    assert summary["selected_strategy"] == "default"
