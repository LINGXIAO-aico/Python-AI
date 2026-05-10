from __future__ import annotations

import streamlit as st

from campus_rag.config import CHUNK_PATH, EVAL_SUMMARY_PATH, INDEX_PATH
from campus_rag.generator import answer_question, no_retrieval_baseline
from campus_rag.retriever import BM25Retriever, HybridRetriever, TfidfRetriever
import pandas as pd


st.set_page_config(page_title="RAG 校园智能问答助手", page_icon="AI", layout="wide")

st.title("RAG 校园智能问答助手")
st.caption("基于校园 FAQ 知识库的检索增强问答演示")


@st.cache_resource
def load_retrievers() -> dict:
    chunks = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")
    tfidf = TfidfRetriever.load(INDEX_PATH)
    bm25 = BM25Retriever.fit(chunks)
    return {
        "混合检索": HybridRetriever(tfidf, bm25, dense_weight=0.5),
        "向量检索": tfidf,
        "BM25关键词": bm25,
    }


retrievers = load_retrievers()

with st.sidebar:
    st.subheader("系统状态")
    st.write(f"知识块数量：{len(next(iter(retrievers.values())).chunks)}")
    if EVAL_SUMMARY_PATH.exists():
        st.json(EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
    top_k = st.slider("检索片段数", min_value=1, max_value=5, value=3)
    strategy = st.selectbox("检索策略", list(retrievers.keys()))

examples = [
    "图书馆周末几点开门？",
    "校园卡丢了怎么办？",
    "宿舍空调坏了从哪里报修？",
    "忘记统一身份认证密码怎么办？",
]
question = st.text_input("请输入问题", value=examples[0])
cols = st.columns(len(examples))
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        question = example

if st.button("生成回答", type="primary"):
    result = answer_question(question, retrievers[strategy], top_k=top_k)
    st.subheader("RAG系统回答")
    st.write(result["answer"])
    st.subheader("无检索基线回答")
    st.write(no_retrieval_baseline(question))
    st.subheader("检索依据")
    for chunk in result["retrieved"]:
        with st.expander(f"{chunk['rank']}. {chunk['title']} | {chunk['doc_id']} | score={chunk['score']}"):
            st.write(chunk["content"])
            st.caption(f"{chunk['source']} | {chunk['url']}")
