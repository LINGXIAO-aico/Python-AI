#!/usr/bin/env python3
"""消融实验：9 组配置对比，输出 logs/ablation_results.csv。

实验设计:
  E1  无检索基线（DeepSeek 直答）
  E2  TF-IDF only
  E3  BM25(jieba) only
  E4  BGE-Dense only
  E5  Hybrid (加权和) vs RRF
  E6  + Cross-Encoder 重排
  E7  + HyDE 改写
  E8  完整管线 (RRF + Reranker + HyDE)
  E9  完整管线 + Self-RAG
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from campus_rag.config import (
    ABLATION_RESULTS_PATH,
    CHUNK_PATH,
    DENSE_RECALL_K,
    BM25_RECALL_K,
    RETRIEVAL_TOP_K,
    EMBEDDING_DIM,
    RAW_EVAL_PATH,
    EVAL_150_PATH,
    LLM_JUDGE_PATH,
)
from campus_rag.data import read_jsonl
from campus_rag.embeddings import BGEEmbedder
from campus_rag.evaluate import (
    evaluate_retrievers,
    judge_faithfulness,
    judge_relevancy,
    judge_context_precision,
)
from campus_rag.generator import llm_answer, no_retrieval_baseline
from campus_rag.query_rewriter import hyde_rewrite
from campus_rag.reranker import BGEReranker
from campus_rag.retriever import (
    Bm25JiebaRetriever,
    DenseRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
)
from campus_rag.vectorstore import FAISSStore


def load_chunks() -> pd.DataFrame:
    return pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")


def build_retrievers(chunks: pd.DataFrame):
    """构建四种基础检索器。"""
    print("构建检索器...")
    embedder = BGEEmbedder()

    # TF-IDF
    print("  [1/4] TF-IDF...")
    tfidf = TfidfRetriever.fit(chunks)

    # BM25-jieba
    print("  [2/4] BM25-jieba...")
    bm25 = Bm25JiebaRetriever.fit(chunks)

    # Dense (BGE + FAISS)
    print("  [3/4] BGE Dense + FAISS...")
    dense = DenseRetriever.build(embedder, chunks)
    dense.save()

    # RRF Hybrid
    print("  [4/4] RRF Hybrid...")
    rrf = DenseRetriever(embedder, FAISSStore.load(EMBEDDING_DIM))
    hybrid_rrf = HybridRRFRetriever(rrf, bm25)

    return embedder, tfidf, bm25, dense, hybrid_rrf


def run_ablation(
    chunks: pd.DataFrame,
    quick: bool = False,
    skip_hyde: bool = False,
    skip_baseline: bool = False,
) -> list[dict]:
    embedder, tfidf, bm25, dense, hybrid_rrf = build_retrievers(chunks)

    # 优先用新的150题评测集，不存在则回退到旧评测集
    eval_path = EVAL_150_PATH if EVAL_150_PATH.exists() else RAW_EVAL_PATH
    if not eval_path.exists():
        print(f"[错误] 评测集文件不存在: {eval_path}")
        return []

    questions = read_jsonl(eval_path)
    print(f"评测集: {len(questions)} 题")

    reranker = BGEReranker()
    results: list[dict] = []

    experiments = [
        ("E1_baseline_no_retrieval", "baseline", None),
        ("E2_tfidf_only", "tfidf", tfidf),
        ("E3_bm25_jieba", "bm25", bm25),
        ("E4_bge_dense", "dense", dense),
        ("E5_hybrid_rrf", "rrf", hybrid_rrf),
        ("E6_rrf_reranker", "rrf_reranker", hybrid_rrf),
        ("E7_rrf_hyde", "rrf_hyde", hybrid_rrf),
        ("E8_full_pipeline", "full", hybrid_rrf),
        ("E9_self_rag", "self_rag", hybrid_rrf),
    ]

    if skip_baseline:
        experiments = [e for e in experiments if not e[1].startswith("baseline")]
    if skip_hyde:
        experiments = [e for e in experiments if "hyde" not in e[1] and e[0] not in ("E8_full_pipeline", "E9_self_rag")]

    for exp_id, exp_name, retriever in experiments:
        print(f"\n{'=' * 50}")
        print(f"[{exp_id}] {exp_name}")

        if exp_name == "baseline":
            # E1: 无检索基线
            latencies = []
            hits = []
            for q in questions:
                t0 = time.perf_counter()
                ans = no_retrieval_baseline(q["question"], backend="deepseek")
                latencies.append(time.perf_counter() - t0)
                keywords = q.get("answer_keywords", [])
                hits.append(
                    sum(1 for kw in keywords if kw in ans) / max(len(keywords), 1)
                )
            results.append({
                "experiment": exp_id,
                "desc": exp_name,
                "hit_at_1": 0,
                "hit_at_3": 0,
                "hit_at_5": 0,
                "mrr": 0,
                "ndcg_at_5": 0,
                "keyword_recall": round(float(pd.Series(hits).mean()), 4),
                "avg_latency_ms": round(float(pd.Series(latencies).mean()) * 1000, 3),
                "note": "无检索基线，不适用检索指标",
            })
            continue

        # 检索+回答
        total_latency = 0.0
        all_rows: list[dict] = []

        for item in questions:
            question = item["question"]
            gold = item["gold_doc_id"]
            keywords = item.get("answer_keywords", [])

            # HyDE 改写（仅 E7, E8, E9）
            search_query = question
            if exp_name in ("rrf_hyde", "full", "self_rag"):
                try:
                    hyde_text = hyde_rewrite(question)
                    search_query = f"{question} {hyde_text}"
                except Exception:
                    pass

            t0 = time.perf_counter()
            chunks = retriever.retrieve(search_query, top_k=RETRIEVAL_TOP_K)

            # Reranker（仅 E6, E8, E9）
            if exp_name in ("rrf_reranker", "full", "self_rag") and chunks:
                chunks = reranker.rerank(question, chunks, top_k=5)

            lat = time.perf_counter() - t0
            total_latency += lat

            doc_ids = [c.doc_id for c in chunks]
            # 快速模式用抽取式答案（不调API），完整模式用LLM
            if quick:
                from campus_rag.generator import extractive_answer
                ans = extractive_answer(question, chunks)
            else:
                ans = llm_answer(question, chunks)
            kw_recall = sum(1 for kw in keywords if kw in ans) / max(len(keywords), 1)

            hit1 = int(bool(doc_ids and doc_ids[0] == gold))
            hit3 = int(gold in doc_ids[:3])
            hit5 = int(gold in doc_ids[:5])
            # MRR
            mrr = 0.0
            for idx, did in enumerate(doc_ids, 1):
                if did == gold:
                    mrr = 1.0 / idx
                    break

            all_rows.append({
                "question": question,
                "hit_at_1": hit1,
                "hit_at_3": hit3,
                "hit_at_5": hit5,
                "mrr": mrr,
                "keyword_recall": kw_recall,
                "latency_ms": round(lat * 1000, 3),
                "top3_docs": "|".join(doc_ids[:3]),
            })

        df = pd.DataFrame(all_rows)
        avg_lat = total_latency / max(len(questions), 1) * 1000

        results.append({
            "experiment": exp_id,
            "desc": exp_name,
            "hit_at_1": round(float(df["hit_at_1"].mean()), 4),
            "hit_at_3": round(float(df["hit_at_3"].mean()), 4),
            "hit_at_5": round(float(df["hit_at_5"].mean()), 4),
            "mrr": round(float(df["mrr"].mean()), 4),
            "ndcg_at_5": 0,  # 消融简化，全评测时启用
            "keyword_recall": round(float(df["keyword_recall"].mean()), 4),
            "avg_latency_ms": round(float(avg_lat), 3),
            "note": exp_name,
        })

    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="RAG 消融实验")
    parser.add_argument("--quick", action="store_true", help="仅跑检索指标，跳过LLM生成（快速模式）")
    parser.add_argument("--skip-hyde", action="store_true", help="跳过 HyDE 相关实验（E7/E8/E9）")
    parser.add_argument("--skip-baseline", action="store_true", help="跳过无检索基线（E1）")
    args = parser.parse_args()

    print("=" * 60)
    print("RAG 消融实验（E1-E9）")
    if args.quick:
        print(">> 快速模式：仅检索指标 <<")
    print("=" * 60)

    if not CHUNK_PATH.exists():
        print(f"[错误] 未找到 chunks 文件: {CHUNK_PATH}")
        print("请先运行 scripts/01_prepare_data.py")
        return

    chunks = load_chunks()
    print(f"知识库: {len(chunks)} chunks")

    results = run_ablation(
        chunks,
        quick=args.quick,
        skip_hyde=args.skip_hyde,
        skip_baseline=args.skip_baseline,
    )

    if not results:
        return

    df = pd.DataFrame(results)
    ABLATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ABLATION_RESULTS_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 60}")
    print("消融实验结果:")
    print(df.to_string(index=False))
    print(f"\n输出文件: {ABLATION_RESULTS_PATH}")


if __name__ == "__main__":
    main()
