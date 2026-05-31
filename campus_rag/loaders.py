from __future__ import annotations

import json
import re
from pathlib import Path


def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", str(text))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"第\s*\d+\s*页", "", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("请先安装 pdfplumber：pip install pdfplumber") from exc

    text_parts: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return clean_text("\n".join(text_parts))


def extract_text_from_docx(docx_path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("请先安装 python-docx：pip install python-docx") from exc

    doc = Document(str(docx_path))
    return clean_text("\n".join(p.text for p in doc.paragraphs if p.text.strip()))


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)
    if suffix in {".txt", ".md"}:
        return clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    raise ValueError(f"暂不支持的文档类型：{path.suffix}")


def export_extracted_jsonl(source_dir: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    supported = sorted(
        [
            *source_dir.glob("*.pdf"),
            *source_dir.glob("*.docx"),
            *source_dir.glob("*.txt"),
            *source_dir.glob("*.md"),
        ]
    )
    count = 0
    with output_path.open("w", encoding="utf-8") as file:
        for idx, path in enumerate(supported, start=1):
            text = extract_text(path)
            if not text:
                continue
            row = {
                "doc_id": f"SRC{idx:03d}",
                "title": path.stem,
                "source_file": path.name,
                "content": text,
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
