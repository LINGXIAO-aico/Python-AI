from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from campus_rag.config import CHUNK_PATH, EMBEDDING_DIM, INDEX_PATH, RETRIEVAL_TOP_K
from campus_rag.embeddings import BGEEmbedder
from campus_rag.generator import answer_question, llm_answer, no_retrieval_baseline
from campus_rag.reranker import BGEReranker
from campus_rag.retriever import (
    Bm25JiebaRetriever,
    BM25Retriever,
    DenseRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
)
from campus_rag.vectorstore import FAISSStore


def main() -> None:
    parser = argparse.ArgumentParser(description="同小智 RAG 校园问答 CLI")
    parser.add_argument("question", help="校园办事或学习生活问题")
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument(
        "--backend",
        choices=["deepseek", "extractive", "qwen", "openai", "llm"],
        default="deepseek",
        help="deepseek=DeepSeek大模型生成，extractive=抽取式离线回答",
    )
    parser.add_argument(
        "--strategy",
        choices=["rrf", "dense", "bm25", "tfidf", "hybrid"],
        default="rrf",
        help="检索策略",
    )
    parser.add_argument("--no-rerank", action="store_true", help="禁用重排序")
    parser.add_argument("--verify", action="store_true", help="启用 Self-RAG 校验")
    args = parser.parse_args()

    chunks = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")

    # 构建检索器
    if args.strategy == "rrf":
        embedder = BGEEmbedder()
        dense = DenseRetriever.load(embedder, EMBEDDING_DIM)
        bm25j = Bm25JiebaRetriever.fit(chunks)
        retriever = HybridRRFRetriever(dense, bm25j)
        label = "RRF Hybrid (BGE + jieba-BM25)"
    elif args.strategy == "dense":
        embedder = BGEEmbedder()
        retriever = DenseRetriever.load(embedder, EMBEDDING_DIM)
        label = "BGE Dense (FAISS HNSW)"
    elif args.strategy == "tfidf":
        retriever = TfidfRetriever.load(INDEX_PATH)
        label = "TF-IDF"
    elif args.strategy == "bm25":
        retriever = Bm25JiebaRetriever.fit(chunks)
        label = "jieba-BM25"
    else:
        tfidf = TfidfRetriever.load(INDEX_PATH)
        bm25_old = BM25Retriever.fit(chunks)
        from campus_rag.retriever import HybridRetriever
        retriever = HybridRetriever(tfidf, bm25_old)
        label = "Hybrid (旧)"

    print(f"策略: {label} | top-k={args.top_k}")

    # 检索
    chunks_retrieved = retriever.retrieve(args.question, top_k=args.top_k)

    # 重排
    if not args.no_rerank and len(chunks_retrieved) > 1 and args.strategy != "tfidf":
        reranker = BGEReranker()
        chunks_retrieved = reranker.rerank(args.question, chunks_retrieved, top_k=min(5, len(chunks_retrieved)))
        print("已启用 Cross-Encoder 重排")

    # 生成
    result = answer_question(
        args.question,
        retriever,
        top_k=args.top_k,
        backend=args.backend,
        verify=args.verify,
    )

    print("\n" + "=" * 60)
    print("RAG 系统回答：")
    print(result["answer"])
    print("\n" + "-" * 40)
    print("无检索基线回答：")
    print(no_retrieval_baseline(args.question, backend=args.backend))
    print("\n" + "-" * 40)
    print("检索片段：")
    for chunk in result["retrieved"]:
        print(f"  [{chunk['doc_id']}] {chunk['title']} | score={chunk['score']:.4f}")

    if args.verify and "verification" in result:
        print("\n" + "-" * 40)
        v = result["verification"]
        print(f"Self-RAG 校验: {v.get('verdict', '?')} — {v.get('explanation', '')}")


if __name__ == "__main__":
    main()
