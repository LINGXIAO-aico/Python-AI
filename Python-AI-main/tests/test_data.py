"""数据处理模块测试。"""

import tempfile
from pathlib import Path

import pandas as pd

from campus_rag.data import (
    clean_knowledge_base,
    normalize_text,
    read_jsonl,
    write_json,
)


class TestNormalizeText:
    def test_fullwidth_space_to_halfwidth(self):
        assert normalize_text("测试　文本") == "测试 文本"

    def test_multiple_spaces_collapsed(self):
        assert normalize_text("a   b  c") == "a b c"

    def test_strips_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestReadJsonl:
    def test_read_valid_jsonl(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"a": 1}\n{"a": 2}\n')
            tmp = f.name
        rows = read_jsonl(Path(tmp))
        assert len(rows) == 2
        assert rows[0]["a"] == 1
        Path(tmp).unlink()

    def test_skip_empty_lines(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"a": 1}\n\n{"a": 2}\n')
            tmp = f.name
        rows = read_jsonl(Path(tmp))
        assert len(rows) == 2
        Path(tmp).unlink()


class TestCleanKnowledgeBase:
    def test_clean_basic(self):
        import json
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            json.dump({
                "doc_id": "KB001", "category": "测试", "title": "标题",
                "content": "这是正文内容，需要至少二十个字才能通过最小长度过滤",
                "source": "测试来源", "url": "http://test.com", "last_updated": "2026-01-01",
            }, f, ensure_ascii=False)
            f.write("\n")
            raw = Path(f.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8-sig"
        ) as f:
            out = Path(f.name)

        profile = clean_knowledge_base(raw, out)

        assert profile.raw_rows == 1
        assert profile.clean_rows == 1
        assert profile.duplicate_rows == 0
        assert "测试" in profile.categories

        df = pd.read_csv(out, encoding="utf-8-sig")
        assert len(df) == 1
        assert df.iloc[0]["doc_id"] == "KB001"

        raw.unlink()
        out.unlink()

    def test_filters_short_content(self):
        import json
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            json.dump({
                "doc_id": "001", "category": "测试", "title": "短",
                "content": "太短", "source": "s", "url": "http://t.com", "last_updated": "2026-01-01",
            }, f, ensure_ascii=False)
            f.write("\n")
            raw = Path(f.name)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8-sig"
        ) as f:
            out = Path(f.name)

        profile = clean_knowledge_base(raw, out)
        assert profile.clean_rows == 0  # 太短被过滤
        raw.unlink()
        out.unlink()


class TestWriteJson:
    def test_write_and_read(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            tmp = Path(f.name)
        write_json(tmp, {"key": "值"})
        import json
        data = json.loads(tmp.read_text(encoding="utf-8"))
        assert data["key"] == "值"
        tmp.unlink()
