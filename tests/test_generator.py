"""生成模块测试（离线抽取式回答）。"""

from __future__ import annotations

import pandas as pd

from campus_rag import generator
from campus_rag.generator import answer_question, extractive_answer, no_retrieval_baseline
from campus_rag.retriever import RetrievedChunk


def _make_chunks() -> list[RetrievedChunk]:
    chunks_df = pd.DataFrame([{
        "chunk_id": "001_C01", "doc_id": "001", "category": "教务",
        "title": "选课通知",
        "content": "每学期第2-3周为选课周，学生登录教务系统进行选课操作。",
        "source": "教务处", "url": "http://test.com",
    }])
    return [RetrievedChunk.from_row(1, 0.95, chunks_df.iloc[0])]


class TestExtractiveAnswer:
    def test_returns_answer_with_citation(self):
        chunks = _make_chunks()
        answer = extractive_answer("如何选课？", chunks)
        assert "选课" in answer
        assert "[001]" in answer
        assert "教务处" in answer

    def test_empty_chunks_returns_fallback(self):
        answer = extractive_answer("任何问题", [])
        assert "暂未检索" in answer or "建议" in answer


class TestNoRetrievalBaseline:
    def test_extractive_backend(self):
        answer = no_retrieval_baseline("校园卡丢了怎么办", backend="extractive")
        assert len(answer) > 0
        assert "知识库" in answer or "学校" in answer or "官网" in answer


class _FakeDelta:
    content = "流式片段"


class _FakeChoice:
    delta = _FakeDelta()


class _FakeStreamChunk:
    choices = [_FakeChoice()]


class _FakeMessage:
    content = '{"verdict": "fully_supported", "explanation": "证据充分"}'


class _FakeNonStreamChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeNonStreamChoice()]


class _FakeCompletions:
    def create(self, **kwargs):
        if kwargs.get("stream"):
            return [_FakeStreamChunk()]
        return _FakeResponse()


class _FakeChat:
    completions = _FakeCompletions()


class _FakeClient:
    chat = _FakeChat()


class _FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.chunks[:top_k]


class _FakeReranker:
    def rerank(
        self, question: str, chunks: list[RetrievedChunk], top_k: int = 5
    ) -> list[RetrievedChunk]:
        return chunks[:top_k]


def test_llm_answer_stream_yields_content(monkeypatch) -> None:
    monkeypatch.setattr(generator, "_get_client", lambda: _FakeClient())

    parts = list(generator.llm_answer_stream("如何选课？", _make_chunks()))

    assert parts == ["流式片段"]


def test_self_rag_verify_parses_json(monkeypatch) -> None:
    monkeypatch.setattr(generator, "_get_client", lambda: _FakeClient())

    result = generator.self_rag_verify("如何选课？", "根据资料回答", _make_chunks())

    assert result["verdict"] == "fully_supported"


def test_self_rag_verify_returns_parse_error(monkeypatch) -> None:
    class BadMessage:
        content = "不是 JSON"

    class BadChoice:
        message = BadMessage()

    class BadResponse:
        choices = [BadChoice()]

    class BadCompletions:
        def create(self, **kwargs):
            return BadResponse()

    class BadChat:
        completions = BadCompletions()

    class BadClient:
        chat = BadChat()

    monkeypatch.setattr(generator, "_get_client", lambda: BadClient())

    result = generator.self_rag_verify("如何选课？", "根据资料回答", _make_chunks())

    assert result["verdict"] == "parse_error"


def test_no_retrieval_baseline_deepseek_backend(monkeypatch) -> None:
    monkeypatch.setattr(generator, "_get_client", lambda: _FakeClient())

    answer = no_retrieval_baseline("校园卡丢了怎么办？", backend="deepseek")

    assert "fully_supported" in answer


def test_answer_question_with_rerank_and_verify(monkeypatch) -> None:
    monkeypatch.setattr(generator, "llm_answer", lambda *args, **kwargs: "LLM回答")
    monkeypatch.setattr(generator, "self_rag_verify", lambda *args, **kwargs: {"verdict": "ok"})

    result = answer_question(
        "如何选课？",
        _FakeRetriever(_make_chunks()),
        backend="deepseek",
        reranker=_FakeReranker(),
        verify=True,
    )

    assert result["answer"] == "LLM回答"
    assert result["citations"] == ["001"]
    assert result["verification"]["verdict"] == "ok"
