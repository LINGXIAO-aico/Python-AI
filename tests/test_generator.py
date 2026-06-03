"""生成模块测试（离线抽取式回答）。"""

import pandas as pd

from campus_rag.generator import extractive_answer, no_retrieval_baseline
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
