from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_FIELDS = [
    "doc_id",
    "category",
    "title",
    "content",
    "source",
    "url",
    "last_updated",
]


@dataclass
class DataProfile:
    raw_rows: int
    clean_rows: int
    duplicate_rows: int
    categories: dict[str, int]
    avg_content_length: float
    min_content_length: int
    max_content_length: int
    missing_values: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "raw_rows": self.raw_rows,
            "clean_rows": self.clean_rows,
            "duplicate_rows": self.duplicate_rows,
            "categories": self.categories,
            "avg_content_length": self.avg_content_length,
            "min_content_length": self.min_content_length,
            "max_content_length": self.max_content_length,
            "missing_values": self.missing_values,
        }


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    text = str(text).replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_knowledge_base(raw_path: Path, output_path: Path) -> DataProfile:
    rows = read_jsonl(raw_path)
    raw_df = pd.DataFrame(rows)
    missing_columns = [col for col in REQUIRED_FIELDS if col not in raw_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = raw_df[REQUIRED_FIELDS].copy()
    for column in REQUIRED_FIELDS:
        df[column] = df[column].map(normalize_text)

    missing_values = df.eq("").sum().to_dict()
    df = df.drop_duplicates(subset=["doc_id"], keep="first")
    df = df.drop_duplicates(subset=["title", "content"], keep="first")
    df = df[df["content"].str.len() >= 20].copy()
    df = df.sort_values(["category", "doc_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    lengths = df["content"].str.len()
    categories = Counter(df["category"])
    has_data = len(df) > 0
    return DataProfile(
        raw_rows=len(raw_df),
        clean_rows=len(df),
        duplicate_rows=len(raw_df) - len(df),
        categories=dict(categories),
        avg_content_length=round(float(lengths.mean()), 2) if has_data else 0.0,
        min_content_length=int(lengths.min()) if has_data else 0,
        max_content_length=int(lengths.max()) if has_data else 0,
        missing_values={str(k): int(v) for k, v in missing_values.items()},
    )


def iter_records(df: pd.DataFrame) -> Iterable[dict]:
    for record in df.to_dict(orient="records"):
        yield {key: normalize_text(value) for key, value in record.items()}
