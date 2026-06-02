from campus_rag.generator import extractive_answer
from campus_rag.retriever import BM25Retriever, HybridRetriever, RetrievedChunk, TfidfRetriever
import pandas as pd


def test_extractive_answer_includes_citation():
    chunk = RetrievedChunk(
        rank=1,
        score=0.9,
        chunk_id="KB001_C01",
        doc_id="KB001",
        category="教务选课",
        title="选课时间与补退选",
        content="补退选在第2周开放，学生需登录教务系统处理。",
        source="测试来源",
        url="https://example.edu",
    )
    answer = extractive_answer("还能补选吗？", [chunk])
    assert "KB001" in answer
    assert "补退选" in answer


def test_hybrid_retriever_returns_expected_doc():
    chunks = pd.DataFrame(
        [
            {
                "chunk_id": "KB001_C01",
                "doc_id": "KB001",
                "category": "图书馆",
                "title": "图书馆开放时间",
                "content": "图书馆周末开放时间为9:00至21:00。",
                "source": "测试来源",
                "url": "https://example.edu/library",
            },
            {
                "chunk_id": "KB002_C01",
                "doc_id": "KB002",
                "category": "宿舍",
                "title": "宿舍报修",
                "content": "宿舍空调损坏可通过后勤服务平台提交报修单。",
                "source": "测试来源",
                "url": "https://example.edu/dorm",
            },
        ]
    )
    tfidf = TfidfRetriever.fit(chunks)
    bm25 = BM25Retriever.fit(chunks)
    hybrid = HybridRetriever(tfidf, bm25)
    result = hybrid.retrieve("图书馆周末几点开门", top_k=1)
    assert result[0].doc_id == "KB001"
