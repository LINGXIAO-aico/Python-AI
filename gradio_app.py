from __future__ import annotations

import pandas as pd

from campus_rag.config import CHUNK_PATH, INDEX_PATH
from campus_rag.generator import answer_question, no_retrieval_baseline
from campus_rag.retriever import BM25Retriever, HybridRetriever, TfidfRetriever


def build_retriever():
    chunks = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")
    tfidf = TfidfRetriever.load(INDEX_PATH)
    bm25 = BM25Retriever.fit(chunks)
    return HybridRetriever(tfidf, bm25, dense_weight=0.5)


retriever = build_retriever()


def chat(question: str):
    result = answer_question(question, retriever, top_k=3)
    citations = "\n".join(
        f"{item['rank']}. {item['doc_id']} {item['title']} score={item['score']}"
        for item in result["retrieved"]
    )
    return result["answer"], no_retrieval_baseline(question), citations


def main() -> None:
    import gradio as gr

    demo = gr.Interface(
        fn=chat,
        inputs=gr.Textbox(label="请输入问题", placeholder="例如：图书馆周末几点开门？"),
        outputs=[
            gr.Textbox(label="RAG系统回答"),
            gr.Textbox(label="无检索基线回答"),
            gr.Textbox(label="检索依据"),
        ],
        title="校园智能问答助手",
        description="混合检索（TF-IDF + BM25）与无检索基线对比演示。",
    )
    demo.launch()


if __name__ == "__main__":
    main()
