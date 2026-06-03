from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_rag.loaders import export_extracted_jsonl


def main() -> None:
    source_dir = ROOT / "data" / "source_docs"
    source_dir.mkdir(parents=True, exist_ok=True)
    output_path = ROOT / "data" / "raw" / "extracted_documents.jsonl"
    count = export_extracted_jsonl(source_dir, output_path)
    if count == 0:
        print(f"未在 {source_dir} 发现 PDF/DOCX/TXT/MD 文档，跳过抽取。")
        print("把学生手册、办事指南等原始文件放入该目录后，可重新运行本脚本。")
    else:
        print(f"已抽取 {count} 个文档，输出：{output_path}")


if __name__ == "__main__":
    main()
