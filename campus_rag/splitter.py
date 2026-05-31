from __future__ import annotations

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .data import iter_records

# 中文标点 + 换行作为自然断点
_CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    "、",
    " ",
    "",
]


def build_chunks(
    clean_kb_path,
    chunk_path,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> pd.DataFrame:
    """使用 RecursiveCharacterTextSplitter 对知识库做中文感知切分。"""
    kb_df = pd.read_csv(clean_kb_path, encoding="utf-8-sig", dtype={"doc_id": str})

    splitter = RecursiveCharacterTextSplitter(
        separators=_CHINESE_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunk_rows: list[dict] = []
    for record in iter_records(kb_df):
        full_text = f"{record['title']}。{record['content']}"
        chunks = splitter.split_text(full_text)
        for idx, chunk in enumerate(chunks):
            chunk_rows.append({
                "chunk_id": f"{record['doc_id']}_C{idx + 1:02d}",
                "doc_id": record["doc_id"],
                "category": record["category"],
                "title": record["title"],
                "content": chunk,
                "source": record["source"],
                "url": record["url"],
                "last_updated": record["last_updated"],
                "char_length": len(chunk),
            })

    chunks_df = pd.DataFrame(chunk_rows)
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(chunk_path, index=False, encoding="utf-8-sig")
    return chunks_df
