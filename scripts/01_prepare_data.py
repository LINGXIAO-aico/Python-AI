from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_rag.config import CHUNK_PATH, CLEAN_KB_PATH, DATA_PROFILE_PATH, LOG_DIR, PROCESSED_DIR, RAW_KB_PATH
from campus_rag.data import clean_knowledge_base, ensure_dirs, write_json
from campus_rag.splitter import build_chunks


def main() -> None:
    ensure_dirs(PROCESSED_DIR, LOG_DIR)
    profile = clean_knowledge_base(RAW_KB_PATH, CLEAN_KB_PATH)
    chunks = build_chunks(CLEAN_KB_PATH, CHUNK_PATH)
    payload = profile.to_dict()
    payload["chunk_count"] = int(len(chunks))
    payload["avg_chunk_length"] = round(float(chunks["char_length"].mean()), 2)
    write_json(DATA_PROFILE_PATH, payload)
    print(f"Clean rows: {payload['clean_rows']}, chunks: {payload['chunk_count']}")
    print(f"Wrote {CLEAN_KB_PATH}")
    print(f"Wrote {CHUNK_PATH}")
    print(f"Wrote {DATA_PROFILE_PATH}")


if __name__ == "__main__":
    main()
