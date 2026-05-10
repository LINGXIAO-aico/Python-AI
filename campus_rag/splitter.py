from __future__ import annotations

import pandas as pd

from .data import iter_records


def split_text(text: str, chunk_size: int = 260, overlap: int = 60) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]


def build_chunks(clean_kb_path, chunk_path, chunk_size: int = 260, overlap: int = 60) -> pd.DataFrame:
    kb_df = pd.read_csv(clean_kb_path, encoding="utf-8-sig")
    chunk_rows: list[dict] = []
    for record in iter_records(kb_df):
        full_text = f"{record['title']}。{record['content']}"
        for idx, chunk in enumerate(split_text(full_text, chunk_size=chunk_size, overlap=overlap)):
            chunk_rows.append(
                {
                    "chunk_id": f"{record['doc_id']}_C{idx + 1:02d}",
                    "doc_id": record["doc_id"],
                    "category": record["category"],
                    "title": record["title"],
                    "content": chunk,
                    "source": record["source"],
                    "url": record["url"],
                    "last_updated": record["last_updated"],
                    "char_length": len(chunk),
                }
            )
    chunks_df = pd.DataFrame(chunk_rows)
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_df.to_csv(chunk_path, index=False, encoding="utf-8-sig")
    return chunks_df
