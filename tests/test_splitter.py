"""文本切分模块测试。"""

import tempfile
from pathlib import Path

import pandas as pd

from campus_rag.splitter import build_chunks


def _make_kb_csv(texts: list[dict]) -> Path:
    """创建临时知识库 CSV。"""
    df = pd.DataFrame(texts)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8-sig"
    ) as tmp:
        df.to_csv(tmp.name, index=False, encoding="utf-8-sig")
        return Path(tmp.name)


class TestBuildChunks:
    def test_single_short_doc(self):
        kb_path = _make_kb_csv([{
            "doc_id": "001",
            "category": "测试",
            "title": "校园卡办理",
            "content": "校园卡丢失后，请携带身份证前往一卡通中心补办。",
            "source": "一卡通中心",
            "url": "http://test.com",
            "last_updated": "2026-01-01",
        }])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8-sig") as f:
            chunk_path = Path(f.name)

        chunks = build_chunks(kb_path, chunk_path, chunk_size=200, overlap=40)

        assert len(chunks) >= 1
        assert chunks.iloc[0]["doc_id"] == "001"
        assert "校园卡办理" in chunks.iloc[0]["content"]

        kb_path.unlink()
        chunk_path.unlink()

    def test_long_doc_splits(self):
        long_text = "第一段。" * 30 + "第二段。" * 30 + "第三段。" * 30
        kb_path = _make_kb_csv([{
            "doc_id": "001",
            "category": "测试",
            "title": "长文档",
            "content": long_text,
            "source": "测试",
            "url": "http://test.com",
            "last_updated": "2026-01-01",
        }])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8-sig") as f:
            chunk_path = Path(f.name)

        chunks = build_chunks(kb_path, chunk_path, chunk_size=100, overlap=20)
        # 长文本应该被切分为多个 chunk
        assert len(chunks) > 1
        assert all(c["char_length"] > 0 for _, c in chunks.iterrows())

        kb_path.unlink()
        chunk_path.unlink()

    def test_multiple_docs(self):
        kb_path = _make_kb_csv([
            {
                "doc_id": "001", "category": "教务", "title": "选课",
                "content": "每学期第2-3周为选课周。",
                "source": "教务处", "url": "http://a.com", "last_updated": "2026-01-01",
            },
            {
                "doc_id": "002", "category": "图书馆", "title": "开放时间",
                "content": "周一至周五 8:00-22:00，周末 9:00-21:00。",
                "source": "图书馆", "url": "http://b.com", "last_updated": "2026-01-01",
            },
        ])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8-sig") as f:
            chunk_path = Path(f.name)

        chunks = build_chunks(kb_path, chunk_path, chunk_size=200, overlap=40)
        doc_ids = chunks["doc_id"].unique()
        assert "001" in doc_ids
        assert "002" in doc_ids

        kb_path.unlink()
        chunk_path.unlink()
