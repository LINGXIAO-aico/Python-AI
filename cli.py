from __future__ import annotations

import argparse

import pandas as pd

from campus_rag.config import CHUNK_PATH, INDEX_PATH
from campus_rag.generator import answer_question, no_retrieval_baseline
from campus_rag.retriever import BM25Retriever, HybridRetriever, TfidfRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Campus RAG question answering CLI")
    parser.add_argument("question", help="校园办事或学习生活问题")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--backend", choices=["extractive", "openai", "qwen", "llm"], default="extractive")
    parser.add_argument(
        "--strategy",
        choices=["hybrid", "tfidf", "bm25"],
        default="hybrid",
        help="hybrid=混合检索，tfidf=向量检索，bm25=关键词检索",
    )
    args = parser.parse_args()

    tfidf = TfidfRetriever.load(INDEX_PATH)
    chunks = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")
    bm25 = BM25Retriever.fit(chunks)
    retriever = {
        "tfidf": tfidf,
        "bm25": bm25,
        "hybrid": HybridRetriever(tfidf, bm25, dense_weight=0.5),
    }[args.strategy]
    result = answer_question(args.question, retriever, top_k=args.top_k, backend=args.backend)
    print("RAG系统回答：")
    print(result["answer"])
    print("\n无检索基线回答：")
    print(no_retrieval_baseline(args.question))
    print("\n检索片段：")
    for chunk in result["retrieved"]:
        print(f"- {chunk['doc_id']} | {chunk['title']} | score={chunk['score']}")


if __name__ == "__main__":
    main()
