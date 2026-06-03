#!/usr/bin/env python3
"""全量评测：多策略对比 + LLM-as-judge + 图表生成。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campus_rag.config import (
    CHUNK_PATH,
    CLEAN_KB_PATH,
    EMBEDDING_DIM,
    EVAL_DETAIL_PATH,
    EVAL_SUMMARY_PATH,
    FIGURE_DIR,
    INDEX_PATH,
    RAW_EVAL_PATH,
    RETRIEVAL_STRATEGY_PATH,
)
from campus_rag.data import ensure_dirs
from campus_rag.embeddings import BGEEmbedder
from campus_rag.evaluate import evaluate_retrievers
from campus_rag.reranker import BGEReranker
from campus_rag.retriever import (
    Bm25JiebaRetriever,
    BM25Retriever,
    DenseRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
)
from campus_rag.vectorstore import FAISSStore


def configure_fonts() -> None:
    font_candidates = [
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    for fp in font_candidates:
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            name = font_manager.FontProperties(fname=fp).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["font.sans-serif"] = [
        *plt.rcParams.get("font.sans-serif", []),
        "Noto Sans CJK SC", "Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_figures(summary: dict, strategy_df: pd.DataFrame) -> None:
    ensure_dirs(FIGURE_DIR)
    sns.set_theme(style="whitegrid")
    configure_fonts()

    # 类别分布
    if CLEAN_KB_PATH.exists():
        kb_df = pd.read_csv(CLEAN_KB_PATH, encoding="utf-8-sig")
        plt.figure(figsize=(9, 5))
        order = kb_df["category"].value_counts().index
        sns.countplot(data=kb_df, y="category", order=order, color="#4C78A8")
        plt.title("知识库条目类别分布", fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "category_distribution.png", dpi=180)
        plt.close()

    # 文本块长度
    if CHUNK_PATH.exists():
        cdf = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")
        plt.figure(figsize=(9, 5))
        sns.histplot(cdf["char_length"], bins=20, color="#59A14F")
        plt.title("文本块长度分布", fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "chunk_length_distribution.png", dpi=180)
        plt.close()

    # 检索策略对比
    if not strategy_df.empty:
        plt.figure(figsize=(10, 5))
        melted = strategy_df.melt(
            id_vars=["strategy"],
            value_vars=["hit_at_1", "hit_at_3", "hit_at_5", "mrr", "ndcg_at_5"],
            var_name="指标", value_name="得分",
        )
        sns.barplot(data=melted, x="指标", y="得分", hue="strategy", palette="Set2")
        plt.ylim(0, 1.05)
        plt.title("多策略检索效果对比", fontsize=14)
        plt.legend(title="", bbox_to_anchor=(1.01, 1), borderaxespad=0)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "retrieval_strategy_comparison.png", dpi=180)
        plt.close()

    # RAG vs 基线
    plt.figure(figsize=(8, 5))
    metrics = pd.DataFrame([
        {"指标": "关键词召回", "系统": "RAG回答", "得分": summary.get("rag_keyword_recall", 0)},
        {"指标": "关键词召回", "系统": "无检索基线", "得分": summary.get("baseline_keyword_recall", 0)},
    ])
    sns.barplot(data=metrics, x="指标", y="得分", hue="系统", palette={"RAG回答": "#4C78A8", "无检索基线": "#F58518"})
    plt.ylim(0, 1.05)
    plt.title("RAG vs 无检索基线", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "evaluation_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    chunks = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")
    embedder = BGEEmbedder()

    # 构建检索器
    tfidf = TfidfRetriever.load(INDEX_PATH)
    bm25_old = BM25Retriever.fit(chunks)
    bm25_jieba = Bm25JiebaRetriever.fit(chunks)
    dense = DenseRetriever.load(embedder, EMBEDDING_DIM)
    hybrid_rrf = HybridRRFRetriever(dense, bm25_jieba)

    retrievers = {
        "tfidf_vector": tfidf,
        "bm25_old": bm25_old,
        "bm25_jieba": bm25_jieba,
        "bge_dense": dense,
        "hybrid_rrf": hybrid_rrf,
    }

    reranker = BGEReranker()

    print("=" * 60)
    print("全量评测（含 LLM-as-judge）")
    print(f"评测集: {RAW_EVAL_PATH}")
    print(f"知识库: {len(chunks)} chunks")
    print("=" * 60)

    detail_df, summary = evaluate_retrievers(
        retrievers,
        RAW_EVAL_PATH,
        EVAL_DETAIL_PATH,
        EVAL_SUMMARY_PATH,
        strategy_detail_path=RETRIEVAL_STRATEGY_PATH,
        selected_strategy="hybrid_rrf",
        top_k=5,
        use_llm=True,
        use_llm_answer=True,
        reranker=reranker,
    )

    # 图表
    strategy_summary = pd.DataFrame(summary["retrieval_strategies"]).T.reset_index()
    strategy_summary.columns = ["strategy"] + list(strategy_summary.columns[1:])

    save_figures(summary, strategy_summary)

    print(f"\n评测完成！")
    print(f"  Hit@1: {summary['hit_at_1']:.4f}")
    print(f"  Hit@5: {summary['hit_at_5']:.4f}")
    print(f"  MRR:    {summary['mrr']:.4f}")
    print(f"  nDCG@5: {summary['ndcg_at_5']:.4f}")
    print(f"\n输出文件:")
    print(f"  {EVAL_DETAIL_PATH}")
    print(f"  {RETRIEVAL_STRATEGY_PATH}")
    print(f"  {EVAL_SUMMARY_PATH}")
    print(f"  {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
