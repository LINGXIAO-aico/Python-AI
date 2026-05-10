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
    EVAL_DETAIL_PATH,
    EVAL_SUMMARY_PATH,
    FIGURE_DIR,
    INDEX_PATH,
    RAW_EVAL_PATH,
    RETRIEVAL_STRATEGY_PATH,
)
from campus_rag.data import ensure_dirs
from campus_rag.evaluate import evaluate_retrievers
from campus_rag.retriever import BM25Retriever, HybridRetriever, TfidfRetriever


def configure_fonts() -> None:
    font_candidates = [
        r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["font.sans-serif"] = [
        *plt.rcParams.get("font.sans-serif", []),
        "Noto Sans CJK SC",
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_figures(summary: dict) -> None:
    ensure_dirs(FIGURE_DIR)
    sns.set_theme(style="whitegrid")
    configure_fonts()

    kb_df = pd.read_csv(CLEAN_KB_PATH, encoding="utf-8-sig")
    plt.figure(figsize=(9, 4.8))
    order = kb_df["category"].value_counts().index
    sns.countplot(data=kb_df, y="category", order=order, color="#4C78A8")
    plt.title("知识库条目类别分布")
    plt.xlabel("条目数量")
    plt.ylabel("类别")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "category_distribution.png", dpi=180)
    plt.close()

    chunks_df = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")
    plt.figure(figsize=(9.4, 4.8))
    sns.histplot(chunks_df["char_length"], bins=10, color="#59A14F")
    plt.title("文本块长度分布")
    plt.xlabel("字符数")
    plt.ylabel("文本块数量")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "chunk_length_distribution.png", dpi=180)
    plt.close()

    metrics_df = pd.DataFrame(
        [
            {"指标": "Hit@1", "系统": "RAG检索", "得分": summary["hit_at_1"]},
            {"指标": "Hit@3", "系统": "RAG检索", "得分": summary["hit_at_3"]},
            {"指标": "关键词召回", "系统": "RAG回答", "得分": summary["rag_keyword_recall"]},
            {"指标": "关键词召回", "系统": "无检索基线", "得分": summary["baseline_keyword_recall"]},
        ]
    )
    plt.figure(figsize=(8, 4.8))
    sns.barplot(
        data=metrics_df,
        x="指标",
        y="得分",
        hue="系统",
        palette={"RAG检索": "#4C78A8", "RAG回答": "#59A14F", "无检索基线": "#F58518"},
    )
    plt.ylim(0, 1.05)
    plt.title("RAG 与无检索基线评估对比")
    plt.xlabel("指标")
    plt.ylabel("得分")
    plt.legend(title="", loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "evaluation_comparison.png", dpi=180)
    plt.close()

    label_map = {
        "tfidf_vector": "向量检索",
        "bm25_keyword": "BM25关键词",
        "hybrid_50_50": "混合检索",
    }
    strategy_rows = []
    for strategy, values in summary["retrieval_strategies"].items():
        for metric in ["hit_at_1", "hit_at_3", "mrr"]:
            strategy_rows.append(
                {
                    "检索策略": label_map.get(strategy, strategy),
                    "指标": {"hit_at_1": "Hit@1", "hit_at_3": "Hit@3", "mrr": "MRR"}[metric],
                    "得分": values[metric],
                }
            )
    strategy_df = pd.DataFrame(strategy_rows)
    plt.figure(figsize=(8.4, 4.8))
    sns.barplot(data=strategy_df, x="检索策略", y="得分", hue="指标", palette="Set2")
    plt.ylim(0, 1.05)
    plt.title("不同检索策略效果对比")
    plt.xlabel("检索策略")
    plt.ylabel("得分")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "retrieval_strategy_comparison.png", dpi=180)
    plt.close()


def main() -> None:
    chunks = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")
    tfidf = TfidfRetriever.load(INDEX_PATH)
    bm25 = BM25Retriever.fit(chunks)
    hybrid = HybridRetriever(tfidf, bm25, dense_weight=0.5)
    retrievers = {
        "tfidf_vector": tfidf,
        "bm25_keyword": bm25,
        "hybrid_50_50": hybrid,
    }
    _, summary = evaluate_retrievers(
        retrievers,
        RAW_EVAL_PATH,
        EVAL_DETAIL_PATH,
        EVAL_SUMMARY_PATH,
        strategy_detail_path=RETRIEVAL_STRATEGY_PATH,
        selected_strategy="hybrid_50_50",
    )
    save_figures(summary)
    print(summary)
    print(f"Wrote {EVAL_DETAIL_PATH}")
    print(f"Wrote {RETRIEVAL_STRATEGY_PATH}")
    print(f"Wrote {EVAL_SUMMARY_PATH}")
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
