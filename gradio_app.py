"""Gradio 版校园智能问答 — ChatInterface 多轮对话。"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campus_rag.config import CHUNK_PATH, EMBEDDING_DIM, RETRIEVAL_TOP_K
from campus_rag.embeddings import BGEEmbedder
from campus_rag.generator import llm_answer, no_retrieval_baseline
from campus_rag.memory import ConversationMemory
from campus_rag.reranker import BGEReranker
from campus_rag.retriever import Bm25JiebaRetriever, DenseRetriever, HybridRRFRetriever
from campus_rag.vectorstore import FAISSStore

# 全局加载
print("正在加载模型...")
chunks = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")
embedder = BGEEmbedder()
dense = DenseRetriever.load(embedder, EMBEDDING_DIM)
bm25 = Bm25JiebaRetriever.fit(chunks)
retriever = HybridRRFRetriever(dense, bm25)
reranker = BGEReranker()
print(f"加载完成！知识库: {len(chunks)} chunks")


def respond(message: str, history: list[dict]) -> str:
    memory = ConversationMemory(max_turns=3)
    for h in history[-6:]:
        if isinstance(h, dict):
            memory.add(h.get("content", ""), "")

    chunks_list = retriever.retrieve(message, top_k=RETRIEVAL_TOP_K)
    if chunks_list and len(chunks_list) > 1:
        chunks_list = reranker.rerank(message, chunks_list, top_k=min(5, len(chunks_list)))

    answer = llm_answer(message, chunks_list, memory)

    sources = "\n\n---\n📎 **检索来源**：\n"
    for c in chunks_list:
        sources += f"- [{c.doc_id}] **{c.title}** (score: {c.score:.3f}) — {c.source}\n"

    return answer + sources


TONGJI_BLUE = "#003F87"

with gr.Blocks(
    title="同小智 — RAG 校园问答",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="amber"),
    css=f"""
    .tongji-header {{ background: linear-gradient(90deg, #002B5C, {TONGJI_BLUE}); color: white; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }}
    footer {{ visibility: hidden; }}
    """,
) as demo:
    gr.HTML(f"""
    <div class="tongji-header">
        <h2>🏛️ 同小智 — RAG 校园智能问答助手</h2>
        <p style="opacity:0.85;">BGE + FAISS + RRF + DeepSeek · 同济大学 B07 小组</p>
    </div>
    """)

    chat_iface = gr.ChatInterface(
        fn=respond,
        type="messages",
        chatbot=gr.Chatbot(height=500, placeholder="输入校园问题..."),
        textbox=gr.Textbox(
            placeholder="试试：图书馆周末几点开门？",
            container=False,
            scale=7,
        ),
        title=None,
        description=None,
        examples=[
            "图书馆周末几点开门？",
            "校园卡丢了怎么补办？",
            "宿舍空调坏了从哪里报修？",
            "大学生医保报销流程是什么？",
        ],
        cache_examples=False,
        retry_btn="🔄 重新生成",
        undo_btn="↩️ 撤销",
        clear_btn="🗑️ 清除对话",
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
