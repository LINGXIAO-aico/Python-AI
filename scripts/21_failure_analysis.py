#!/usr/bin/env python3
"""失败案例分析：从全量评测结果中提取最差题目，分类诊断。

运行前需先执行:
    python scripts/03_evaluate.py

输出: logs/failure_analysis.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from campus_rag.config import EVAL_DETAIL_PATH


def classify_failure(row: pd.Series) -> str:
    """根据指标诊断失败类别。"""
    if row["hit_at_5"] == 0:
        return "检索失败 — gold doc 未进入检索 top-5"
    if row["rag_keyword_recall"] < 0.3:
        return "知识缺漏 — 检索到的内容不足以回答问题"
    # 无法自动判断幻觉，标记为需人工审查
    return "需人工审查 — 可能有幻觉或题目歧义"


def main() -> None:
    if not EVAL_DETAIL_PATH.exists():
        print(f"[错误] 评测详情文件不存在: {EVAL_DETAIL_PATH}")
        print("请先运行: python scripts/03_evaluate.py")
        return

    df = pd.read_csv(EVAL_DETAIL_PATH, encoding="utf-8-sig")
    print(f"评测集: {len(df)} 题")

    lines = [
        "# 失败案例分析报告",
        "",
        f"**评测策略**: hybrid_rrf（完整管线 + Reranker）",
        f"**题目总数**: {len(df)}",
        f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
    ]

    # 整体概览
    lines.append("## 整体指标")
    lines.append("")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    for col in ["hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5"]:
        if col in df.columns:
            lines.append(f"| {col} | {df[col].mean():.4f} |")
    if "rag_keyword_recall" in df.columns:
        lines.append(f"| rag_keyword_recall | {df['rag_keyword_recall'].mean():.4f} |")
    if "baseline_keyword_recall" in df.columns:
        lines.append(f"| baseline_keyword_recall | {df['baseline_keyword_recall'].mean():.4f} |")
    lines.append("")

    # LLM Judge 概览（如果存在）
    judge_cols = ["faithfulness", "answer_relevancy", "context_precision"]
    if all(c in df.columns for c in judge_cols):
        lines.append("## LLM-as-Judge 评分")
        lines.append("")
        lines.append(f"| 维度 | 均值 |")
        lines.append(f"|------|------|")
        for col in judge_cols:
            valid = df[col].dropna()
            if len(valid) > 0:
                lines.append(f"| {col} | {valid.mean():.2f} (n={len(valid)}) |")
        lines.append("")

    # 最差检索案例 (hit@5 == 0)
    worst_retrieval = df[df["hit_at_5"] == 0].head(5)
    lines.append("---")
    lines.append("")
    lines.append("## 1. 检索失败案例 (hit@5 = 0)")
    lines.append("")
    if len(worst_retrieval) == 0:
        lines.append("> ✅ 所有题目的 gold doc 均在 top-5 中。")
    else:
        lines.append(f"共 {len(df[df['hit_at_5'] == 0])} 题检索完全失败，以下是最差 5 例：")
        lines.append("")
        for i, (_, row) in enumerate(worst_retrieval.iterrows(), 1):
            lines.append(f"### 案例 {i}: {row['question'][:100]}")
            lines.append("")
            lines.append(f"- **Gold Doc**: `{row['gold_doc_id']}`")
            lines.append(f"- **Top-5 检索**: `{row.get('top5_doc_ids', 'N/A')}`")
            lines.append(f"- **RAG 回答** (前200字): {str(row.get('rag_answer', ''))[:200]}")
            lines.append(f"- **诊断**: 检索失败 — gold doc 未进入 top-5，需优化检索策略或增加同义词/改写")
            if "faithfulness" in row:
                lines.append(f"- **Faithfulness**: {row['faithfulness']}")
            lines.append("")
    lines.append("")

    # 最差关键词召回案例
    worst_recall = df.nsmallest(5, "rag_keyword_recall")
    lines.append("---")
    lines.append("")
    lines.append("## 2. 低关键词召回案例")
    lines.append("")
    for i, (_, row) in enumerate(worst_recall.iterrows(), 1):
        lines.append(f"### 案例 {i}: {row['question'][:100]}")
        lines.append("")
        lines.append(f"- **RAG 关键词召回**: {row['rag_keyword_recall']:.2%}")
        lines.append(f"- **基线关键词召回**: {row.get('baseline_keyword_recall', 0):.2%}")
        lines.append(f"- **Gold Doc**: `{row['gold_doc_id']}`")
        lines.append(f"- **RAG 回答** (前300字): {str(row.get('rag_answer', ''))[:300]}")
        category = classify_failure(row)
        lines.append(f"- **诊断**: {category}")
        lines.append("")
    lines.append("")

    # 幻觉嫌疑案例（仅当有 LLM Judge 时）
    if "faithfulness" in df.columns:
        low_faith = df.nsmallest(5, "faithfulness")
        lines.append("---")
        lines.append("")
        lines.append("## 3. 低 Faithfulness 案例（幻觉嫌疑）")
        lines.append("")
        for i, (_, row) in enumerate(low_faith.iterrows(), 1):
            lines.append(f"### 案例 {i}: {row['question'][:100]}")
            lines.append("")
            lines.append(f"- **Faithfulness**: {row['faithfulness']}")
            lines.append(f"- **Answer Relevancy**: {row.get('answer_relevancy', 'N/A')}")
            lines.append(f"- **RAG 回答** (前300字): {str(row.get('rag_answer', ''))[:300]}")
            lines.append(f"- **诊断**: 回答可能包含检索片段外的编造内容，建议人工审查")
            lines.append("")
        lines.append("")

    # 总结与建议
    lines.append("---")
    lines.append("")
    lines.append("## 4. 改进建议")
    lines.append("")
    lines.append("| 失败类型 | 建议措施 |")
    lines.append("|----------|----------|")
    lines.append("| 检索失败 | 1) 增加 query 改写/扩展；2) 优化 chunk 大小；3) 检查 KB 覆盖率 |")
    lines.append("| 知识缺漏 | 1) 补充 KB 内容；2) 降低 chunk 粒度；3) 增加多跳检索 |")
    lines.append("| 幻觉 | 1) 增强 system prompt 约束；2) 启用 Self-RAG 校验；3) 降低 temperature |")
    lines.append("| 题目问题 | 1) 标记为 unanswerable；2) 增加 NONE 处理逻辑 |")
    lines.append("")

    out_path = PROJECT_ROOT / "logs" / "failure_analysis.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已生成: {out_path}")


if __name__ == "__main__":
    main()
