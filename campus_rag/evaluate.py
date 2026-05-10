from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .data import read_jsonl
from .generator import extractive_answer, no_retrieval_baseline
from .retriever import TfidfRetriever


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for keyword in keywords if keyword in answer)
    return hits / len(keywords)


def reciprocal_rank(gold_doc_id: str, retrieved_doc_ids: list[str]) -> float:
    for idx, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == gold_doc_id:
            return 1.0 / idx
    return 0.0


def evaluate_retriever(
    retriever: TfidfRetriever,
    eval_path: Path,
    detail_path: Path,
    summary_path: Path,
    top_k: int = 3,
) -> tuple[pd.DataFrame, dict]:
    return evaluate_retrievers(
        {"tfidf_vector": retriever},
        eval_path=eval_path,
        detail_path=detail_path,
        summary_path=summary_path,
        strategy_detail_path=None,
        selected_strategy="tfidf_vector",
        top_k=top_k,
    )


def evaluate_retrievers(
    retrievers: dict[str, object],
    eval_path: Path,
    detail_path: Path,
    summary_path: Path,
    strategy_detail_path: Path | None = None,
    selected_strategy: str = "hybrid",
    top_k: int = 3,
) -> tuple[pd.DataFrame, dict]:
    questions = read_jsonl(eval_path)
    if selected_strategy not in retrievers:
        selected_strategy = next(iter(retrievers.keys()))

    strategy_rows: list[dict] = []
    selected_lookup: dict[str, dict] = {}
    for strategy_name, retriever in retrievers.items():
        rows_for_strategy: list[dict] = []
        total_latency = 0.0
        for item in questions:
            started = time.perf_counter()
            retrieved = retriever.retrieve(item["question"], top_k=top_k)
            total_latency += time.perf_counter() - started
            retrieved_doc_ids = [chunk.doc_id for chunk in retrieved]
            row = {
                "strategy": strategy_name,
                "question": item["question"],
                "gold_doc_id": item["gold_doc_id"],
                "top1_doc_id": retrieved_doc_ids[0] if retrieved_doc_ids else "",
                "top3_doc_ids": "|".join(retrieved_doc_ids),
                "hit_at_1": int(bool(retrieved_doc_ids and retrieved_doc_ids[0] == item["gold_doc_id"])),
                "hit_at_3": int(item["gold_doc_id"] in retrieved_doc_ids[:3]),
                "mrr": reciprocal_rank(item["gold_doc_id"], retrieved_doc_ids),
                "retrieved_chunks": retrieved,
            }
            rows_for_strategy.append(row)
            if strategy_name == selected_strategy:
                selected_lookup[item["question"]] = row

        strategy_df = pd.DataFrame([{k: v for k, v in row.items() if k != "retrieved_chunks"} for row in rows_for_strategy])
        strategy_rows.append(
            {
                "strategy": strategy_name,
                "top_k": top_k,
                "question_count": int(len(strategy_df)),
                "hit_at_1": round(float(strategy_df["hit_at_1"].mean()), 4),
                "hit_at_3": round(float(strategy_df["hit_at_3"].mean()), 4),
                "mrr": round(float(strategy_df["mrr"].mean()), 4),
                "avg_latency_ms": round(total_latency / max(len(questions), 1) * 1000, 3),
            }
        )

    rows: list[dict] = []
    for item in questions:
        question = item["question"]
        gold_doc_id = item["gold_doc_id"]
        keywords = item.get("answer_keywords", [])
        selected = selected_lookup[question]
        retrieved = selected["retrieved_chunks"]
        retrieved_doc_ids = [chunk.doc_id for chunk in retrieved]
        rag_answer = extractive_answer(question, retrieved)
        baseline_answer = no_retrieval_baseline(question)
        rows.append(
            {
                "strategy": selected_strategy,
                "question": question,
                "gold_doc_id": gold_doc_id,
                "top1_doc_id": retrieved_doc_ids[0] if retrieved_doc_ids else "",
                "top3_doc_ids": "|".join(retrieved_doc_ids),
                "hit_at_1": int(bool(retrieved_doc_ids and retrieved_doc_ids[0] == gold_doc_id)),
                "hit_at_3": int(gold_doc_id in retrieved_doc_ids[:3]),
                "mrr": reciprocal_rank(gold_doc_id, retrieved_doc_ids),
                "rag_keyword_recall": keyword_recall(rag_answer, keywords),
                "baseline_keyword_recall": keyword_recall(baseline_answer, keywords),
                "rag_answer": rag_answer,
                "baseline_answer": baseline_answer,
            }
        )

    detail_df = pd.DataFrame(rows)
    strategy_df = pd.DataFrame(strategy_rows)
    selected_metrics = strategy_df[strategy_df["strategy"] == selected_strategy].iloc[0].to_dict()
    summary = {
        "question_count": int(len(detail_df)),
        "selected_strategy": selected_strategy,
        "hit_at_1": round(float(selected_metrics["hit_at_1"]), 4),
        "hit_at_3": round(float(selected_metrics["hit_at_3"]), 4),
        "mrr": round(float(selected_metrics["mrr"]), 4),
        "avg_latency_ms": round(float(selected_metrics["avg_latency_ms"]), 3),
        "rag_keyword_recall": round(float(detail_df["rag_keyword_recall"].mean()), 4),
        "baseline_keyword_recall": round(float(detail_df["baseline_keyword_recall"].mean()), 4),
        "retrieval_strategies": {
            row["strategy"]: {
                "top_k": int(row["top_k"]),
                "hit_at_1": round(float(row["hit_at_1"]), 4),
                "hit_at_3": round(float(row["hit_at_3"]), 4),
                "mrr": round(float(row["mrr"]), 4),
                "avg_latency_ms": round(float(row["avg_latency_ms"]), 3),
            }
            for row in strategy_rows
        },
    }
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    if strategy_detail_path:
        strategy_detail_path.parent.mkdir(parents=True, exist_ok=True)
        strategy_df.to_csv(strategy_detail_path, index=False, encoding="utf-8-sig")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return detail_df, summary
