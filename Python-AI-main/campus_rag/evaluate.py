from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from openai import OpenAI

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_REASONER_MODEL,
)
from .data import read_jsonl
from .generator import extractive_answer, llm_answer, no_retrieval_baseline


class RetrieverProtocol(Protocol):
    def retrieve(self, query: str, top_k: int = 5):
        ...


# ============================================================
# 传统检索指标
# ============================================================

def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw in answer)
    return hits / len(keywords)


def reciprocal_rank(gold_doc_id: str, retrieved_doc_ids: list[str]) -> float:
    for idx, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == gold_doc_id:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(gold_doc_id: str, retrieved_doc_ids: list[str], k: int = 5) -> float:
    """nDCG@k：gold doc 相关性=1，其余=0。"""
    for idx, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id == gold_doc_id:
            dcg = 1.0 / math.log2(idx + 1)
            idcg = 1.0 / math.log2(2)  # 理想排序：gold rank=1
            return dcg / idcg if idcg > 0 else 0.0
    return 0.0


# ============================================================
# LLM-as-Judge 指标
# ============================================================

FAITHFULNESS_TEMPLATE = """你是一个严格的评测员。请判断以下AI回答是否完全基于【参考资料】。

评分规则：
- 5分：回答中每条主张都能在参考资料中找到依据
- 4分：大部分主张有依据，极少数次要细节未提及
- 3分：主要主张有依据，但部分细节无法验证
- 2分：回答中明显混入了参考资料未提的信息
- 1分：回答与参考资料无关或严重编造

参考资料：
{context}

问题：{question}
AI回答：{answer}

请输出JSON：{{"score": 整数1-5, "reason": "一句话理由"}}"""

RELEVANCY_TEMPLATE = """你是一个评测员。请判断AI回答是否切题、完整地回答了用户问题。

评分规则：
- 5分：回答完全切题，覆盖了问题的所有要点
- 4分：回答大部分切题，遗漏了少量相关信息
- 3分：回答基本切题，但信息不完整或部分偏题
- 2分：回答部分相关，但遗漏了重要信息
- 1分：回答与问题基本无关

问题：{question}
AI回答：{answer}

请输出JSON：{{"score": 整数1-5, "reason": "一句话理由"}}"""

CONTEXT_PRECISION_TEMPLATE = """你是一个评测员。请判断用于回答问题的【检索结果】中，真正有用的比例。

评分规则：
- 5分：所有检索片段都对回答问题有帮助
- 4分：大部分检索片段有帮助
- 3分：约一半检索片段有帮助
- 2分：只有少数片段有用
- 1分：检索片段基本无用

问题：{question}
检索到的前5个片段：
{context}

请输出JSON：{{"score": 整数1-5, "reason": "一句话理由"}}"""


def _get_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _judge(template: str, **kwargs) -> dict:
    """调用 DeepSeek-reasoner 做 LLM 评测。"""
    import json as _json
    client = _get_client()
    prompt = template.format(**kwargs)
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_REASONER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content or "{}"
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return _json.loads(raw.strip())
    except Exception:
        return {"score": 0, "reason": "judge failed"}


def judge_faithfulness(question: str, answer: str, context: str) -> dict:
    return _judge(FAITHFULNESS_TEMPLATE, question=question, answer=answer, context=context)


def judge_relevancy(question: str, answer: str) -> dict:
    return _judge(RELEVANCY_TEMPLATE, question=question, answer=answer)


def judge_context_precision(question: str, context: str) -> dict:
    return _judge(CONTEXT_PRECISION_TEMPLATE, question=question, context=context)


# ============================================================
# 批量评测
# ============================================================

def evaluate_retrievers(
    retrievers: dict[str, RetrieverProtocol],
    eval_path: Path,
    detail_path: Path,
    summary_path: Path,
    strategy_detail_path: Path | None = None,
    selected_strategy: str = "hybrid",
    top_k: int = 5,
    use_llm: bool = False,
    use_llm_answer: bool = False,
    reranker=None,
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
            raw_chunks = retriever.retrieve(item["question"], top_k=top_k)
            if reranker and raw_chunks:
                raw_chunks = reranker.rerank(item["question"], raw_chunks, top_k=min(5, len(raw_chunks)))
            total_latency += time.perf_counter() - started

            retrieved_doc_ids = [chunk.doc_id for chunk in raw_chunks]
            row = {
                "strategy": strategy_name,
                "question": item["question"],
                "gold_doc_id": item["gold_doc_id"],
                "top1_doc_id": retrieved_doc_ids[0] if retrieved_doc_ids else "",
                "top5_doc_ids": "|".join(retrieved_doc_ids[:5]),
                "hit_at_1": int(bool(retrieved_doc_ids and retrieved_doc_ids[0] == item["gold_doc_id"])),
                "hit_at_3": int(item["gold_doc_id"] in retrieved_doc_ids[:3]),
                "hit_at_5": int(item["gold_doc_id"] in retrieved_doc_ids[:5]),
                "mrr": reciprocal_rank(item["gold_doc_id"], retrieved_doc_ids),
                "ndcg_at_5": ndcg_at_k(item["gold_doc_id"], retrieved_doc_ids, k=5),
                "retrieved_chunks": raw_chunks,
            }
            rows_for_strategy.append(row)
            if strategy_name == selected_strategy:
                selected_lookup[item["question"]] = row

        strategy_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "retrieved_chunks"}
            for r in rows_for_strategy
        ])
        strategy_rows.append({
            "strategy": strategy_name,
            "top_k": top_k,
            "question_count": int(len(strategy_df)),
            "hit_at_1": round(float(strategy_df["hit_at_1"].mean()), 4),
            "hit_at_3": round(float(strategy_df["hit_at_3"].mean()), 4),
            "hit_at_5": round(float(strategy_df["hit_at_5"].mean()), 4),
            "mrr": round(float(strategy_df["mrr"].mean()), 4),
            "ndcg_at_5": round(float(strategy_df["ndcg_at_5"].mean()), 4),
            "avg_latency_ms": round(total_latency / max(len(questions), 1) * 1000, 3),
        })

    # === 详细评测（selected_strategy） ===
    rows: list[dict] = []
    llm_scores: list[dict] = []
    count = 0

    for item in questions:
        question = item["question"]
        gold_doc_id = item["gold_doc_id"]
        keywords = item.get("answer_keywords", [])
        selected = selected_lookup[question]
        retrieved = selected["retrieved_chunks"]
        retrieved_doc_ids = [chunk.doc_id for chunk in retrieved]

        if use_llm_answer:
            rag_answer = llm_answer(question, retrieved)
            baseline_answer = no_retrieval_baseline(question, backend="deepseek")
        else:
            rag_answer = extractive_answer(question, retrieved)
            baseline_answer = no_retrieval_baseline(question)

        row = {
            "strategy": selected_strategy,
            "question": question,
            "gold_doc_id": gold_doc_id,
            "top1_doc_id": retrieved_doc_ids[0] if retrieved_doc_ids else "",
            "top5_doc_ids": "|".join(retrieved_doc_ids[:5]),
            "hit_at_1": int(bool(retrieved_doc_ids and retrieved_doc_ids[0] == gold_doc_id)),
            "hit_at_3": int(gold_doc_id in retrieved_doc_ids[:3]),
            "hit_at_5": int(gold_doc_id in retrieved_doc_ids[:5]),
            "mrr": reciprocal_rank(gold_doc_id, retrieved_doc_ids),
            "ndcg_at_5": ndcg_at_k(gold_doc_id, retrieved_doc_ids, k=5),
            "rag_keyword_recall": keyword_recall(rag_answer, keywords),
            "baseline_keyword_recall": keyword_recall(baseline_answer, keywords),
            "rag_answer": rag_answer,
            "baseline_answer": baseline_answer,
        }

        # LLM-as-judge（仅对前 N 题做，控制时间/费用）
        if use_llm and count < 150:
            context_text = "\n".join(
                f"[{c.doc_id}] {c.content}" for c in retrieved[:5]
            )
            faith = judge_faithfulness(question, rag_answer, context_text)
            relev = judge_relevancy(question, rag_answer)
            cprec = judge_context_precision(question, context_text)
            row["faithfulness"] = faith.get("score", 0)
            row["answer_relevancy"] = relev.get("score", 0)
            row["context_precision"] = cprec.get("score", 0)
            llm_scores.append({
                "question": question,
                "faithfulness": faith,
                "answer_relevancy": relev,
                "context_precision": cprec,
            })
            count += 1

        rows.append(row)

    detail_df = pd.DataFrame(rows)
    strategy_summary_df = pd.DataFrame(strategy_rows)
    selected_metrics = strategy_summary_df[
        strategy_summary_df["strategy"] == selected_strategy
    ].iloc[0].to_dict()

    summary = {
        "question_count": int(len(detail_df)),
        "selected_strategy": selected_strategy,
        "hit_at_1": round(float(selected_metrics["hit_at_1"]), 4),
        "hit_at_3": round(float(selected_metrics["hit_at_3"]), 4),
        "hit_at_5": round(float(selected_metrics["hit_at_5"]), 4),
        "mrr": round(float(selected_metrics["mrr"]), 4),
        "ndcg_at_5": round(float(selected_metrics["ndcg_at_5"]), 4),
        "avg_latency_ms": round(float(selected_metrics["avg_latency_ms"]), 3),
        "rag_keyword_recall": round(float(detail_df["rag_keyword_recall"].mean()), 4),
        "baseline_keyword_recall": round(float(detail_df["baseline_keyword_recall"].mean()), 4),
        "retrieval_strategies": {
            row["strategy"]: {
                "top_k": int(row["top_k"]),
                "hit_at_1": round(float(row["hit_at_1"]), 4),
                "hit_at_3": round(float(row["hit_at_3"]), 4),
                "hit_at_5": round(float(row["hit_at_5"]), 4),
                "mrr": round(float(row["mrr"]), 4),
                "ndcg_at_5": round(float(row["ndcg_at_5"]), 4),
                "avg_latency_ms": round(float(row["avg_latency_ms"]), 3),
            }
            for row in strategy_rows
        },
    }

    if llm_scores:
        summary["llm_judge"] = {
            "faithfulness_mean": round(
                float(np.mean([s["faithfulness"].get("score", 0) for s in llm_scores])), 2
            ),
            "answer_relevancy_mean": round(
                float(np.mean([s["answer_relevancy"].get("score", 0) for s in llm_scores])), 2
            ),
            "context_precision_mean": round(
                float(np.mean([s["context_precision"].get("score", 0) for s in llm_scores])), 2
            ),
            "judged_count": len(llm_scores),
        }

    # 写盘
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    if strategy_detail_path:
        strategy_detail_path.parent.mkdir(parents=True, exist_ok=True)
        strategy_summary_df.to_csv(strategy_detail_path, index=False, encoding="utf-8-sig")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # LLM judge 明细
    if llm_scores:
        from .config import LLM_JUDGE_PATH
        LLM_JUDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(llm_scores).to_csv(LLM_JUDGE_PATH, index=False, encoding="utf-8-sig")

    return detail_df, summary


def evaluate_retriever(
    retriever,
    eval_path: Path,
    detail_path: Path,
    summary_path: Path,
    top_k: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """单策略评测（向后兼容）。"""
    return evaluate_retrievers(
        {"default": retriever},
        eval_path=eval_path,
        detail_path=detail_path,
        summary_path=summary_path,
        selected_strategy="default",
        top_k=top_k,
    )
