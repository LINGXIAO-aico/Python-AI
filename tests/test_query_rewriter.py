"""查询改写模块测试。"""

from __future__ import annotations

import pytest

from campus_rag import query_rewriter
from campus_rag.config import DEEPSEEK_API_KEY


def test_hyde_rewrite_returns_non_empty_string(monkeypatch) -> None:
    monkeypatch.setattr(query_rewriter, "_call_deepseek", lambda *args, **kwargs: "一段假设性校园指南")

    rewritten = query_rewriter.hyde_rewrite("校园卡丢了怎么办？")

    assert rewritten == "一段假设性校园指南"


def test_multi_query_expand_returns_requested_count(monkeypatch) -> None:
    monkeypatch.setattr(
        query_rewriter,
        "_call_deepseek",
        lambda *args, **kwargs: "校园卡挂失流程\n校园卡补办地点\n学生卡遗失处理",
    )

    queries = query_rewriter.multi_query_expand("校园卡丢了怎么办？", n=3)

    assert queries == ["校园卡挂失流程", "校园卡补办地点", "学生卡遗失处理"]


def test_multi_query_expand_falls_back_to_original_question(monkeypatch) -> None:
    monkeypatch.setattr(query_rewriter, "_call_deepseek", lambda *args, **kwargs: "")

    queries = query_rewriter.multi_query_expand("考试缓考如何申请？", n=3)

    assert queries == ["考试缓考如何申请？"]


def test_rewrite_query_dispatches_all_modes(monkeypatch) -> None:
    monkeypatch.setattr(query_rewriter, "hyde_rewrite", lambda question: f"HyDE:{question}")
    monkeypatch.setattr(
        query_rewriter,
        "multi_query_expand",
        lambda question: [f"Q1:{question}", f"Q2:{question}", f"Q3:{question}"],
    )

    assert query_rewriter.rewrite_query("选课时间？", method="hyde") == "HyDE:选课时间？"
    assert query_rewriter.rewrite_query("选课时间？", method="multi_query") == [
        "Q1:选课时间？",
        "Q2:选课时间？",
        "Q3:选课时间？",
    ]
    assert query_rewriter.rewrite_query("选课时间？", method="both") == [
        "HyDE:选课时间？",
        "Q1:选课时间？",
        "Q2:选课时间？",
        "Q3:选课时间？",
    ]
    assert query_rewriter.rewrite_query("选课时间？", method="none") == "选课时间？"


@pytest.mark.llm
def test_real_hyde_rewrite_returns_non_empty_string() -> None:
    if not DEEPSEEK_API_KEY:
        pytest.skip("未配置 DeepSeek API Key，跳过真实 LLM 集成测试。")

    rewritten = query_rewriter.hyde_rewrite("同济大学图书馆开放时间？")

    assert isinstance(rewritten, str)
    assert rewritten.strip()


@pytest.mark.llm
def test_real_multi_query_expand_returns_three_rewrites() -> None:
    if not DEEPSEEK_API_KEY:
        pytest.skip("未配置 DeepSeek API Key，跳过真实 LLM 集成测试。")

    queries = query_rewriter.multi_query_expand("校园卡丢了怎么办？", n=3)

    assert len(queries) == 3
    assert all(query.strip() for query in queries)
