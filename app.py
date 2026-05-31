from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from campus_rag.config import (
    BM25_RECALL_K,
    CHUNK_PATH,
    DENSE_RECALL_K,
    EMBEDDING_DIM,
    EVAL_SUMMARY_PATH,
    INDEX_PATH,
    RETRIEVAL_TOP_K,
)
from campus_rag.embeddings import BGEEmbedder
from campus_rag.generator import (
    answer_question,
    llm_answer_stream,
    no_retrieval_baseline,
    self_rag_verify,
)
from campus_rag.memory import ConversationMemory
from campus_rag.reranker import BGEReranker
from campus_rag.retriever import (
    BM25Retriever,
    Bm25JiebaRetriever,
    DenseRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
)
from campus_rag.vectorstore import FAISSStore

# ============================================================
# 同济蓝/金主题 CSS（保留自原版）
# ============================================================
TONGJI_BLUE = "#003F87"
TONGJI_BLUE_LIGHT = "#0055B8"
TONGJI_BLUE_DARK = "#002B5C"
TONGJI_GOLD = "#D4A84B"
TONGJI_BG = "#F5F5F5"
TONGJI_CARD_BG = "#FFFFFF"
TEXT_DARK = "#1A1A1A"
TEXT_GRAY = "#666666"
TONGJI_LOGO_PATH = Path(__file__).parent / "assets" / "tongji-logo.png"


def image_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


CUSTOM_CSS = f"""
[data-testid="stAppViewContainer"] {{
    background: radial-gradient(circle at top right, rgba(212,168,75,0.10), transparent 28rem),
                linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%);
}}
.block-container {{ padding-top: 1rem; max-width: 1180px; }}

.main-header {{
    position: relative; overflow: hidden;
    background: linear-gradient(90deg, rgba(0,43,92,0.94) 0%, rgba(0,63,135,0.94) 62%, rgba(0,85,184,0.92) 100%);
    padding: 1.1rem 1.4rem; border-radius: 8px; margin-bottom: 1rem;
    box-shadow: 0 14px 34px rgba(0,43,92,0.22); border-top: 4px solid {TONGJI_GOLD};
}}

.main-header::after {{
    content: ""; position: absolute; inset: 0;
    background: linear-gradient(120deg, transparent 0 58%, rgba(255,255,255,0.11) 58% 59%, transparent 59%),
                linear-gradient(120deg, transparent 0 72%, rgba(255,255,255,0.08) 72% 73%, transparent 73%);
    pointer-events: none;
}}

.brand-lockup {{
    position: relative; z-index: 1; display: flex; align-items: center;
    justify-content: space-between; gap: 1rem;
}}
.brand-left {{ display: flex; align-items: center; gap: 0.8rem; min-width: 0; }}
.tj-logo {{ width: 144px; max-width: 30vw; height: auto; display: block; filter: drop-shadow(0 5px 12px rgba(0,0,0,0.22)); }}
.brand-divider {{ width: 1px; height: 50px; background: rgba(255,255,255,0.35); flex: 0 0 auto; }}
.main-kicker {{ color: rgba(255,255,255,0.78); font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.15rem; }}
.brand-badge {{
    border: 1px solid rgba(255,255,255,0.28); color: rgba(255,255,255,0.92);
    border-radius: 999px; padding: 0.35rem 0.65rem; font-size: 0.78rem;
    white-space: nowrap; background: rgba(255,255,255,0.08);
}}
.main-title {{ color: white !important; font-size: 1.65rem !important; font-weight: 700 !important; margin: 0 !important; text-shadow: 0 2px 8px rgba(0,0,0,0.18); }}
.main-subtitle {{ color: rgba(255,255,255,0.86) !important; font-size: 0.9rem !important; margin: 0.3rem 0 0 !important; font-weight: 400; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(232,239,248,0.98) 0%, rgba(220,230,242,0.98) 100%);
    border-right: 1px solid rgba(0,43,92,0.20);
}}

.stChatMessage {{ border-radius: 12px; padding: 0.5rem; }}
div[data-testid="stVerticalBlock"] > div[data-testid="stChatMessage"]:nth-child(odd) {{
    background: rgba(0,63,135,0.03);
}}

[data-testid="stChatInput"] textarea:focus {{
    border-color: {TONGJI_BLUE} !important;
    box-shadow: 0 0 0 3px rgba(0,63,135,0.1) !important;
}}

.sidebar-header {{
    color: {TONGJI_BLUE}; font-weight: 700; font-size: 1.05rem;
    padding-bottom: 0.5rem; border-bottom: 2px solid rgba(212,168,75,0.75); margin-bottom: 1rem;
}}

.sidebar-stat {{
    background: linear-gradient(135deg, {TONGJI_BLUE} 0%, {TONGJI_BLUE_DARK} 100%);
    padding: 0.9rem; border-radius: 8px; color: white; margin-bottom: 1.2rem;
    border-top: 3px solid {TONGJI_GOLD}; box-shadow: 0 8px 18px rgba(0,43,92,0.14);
}}

.score-bar {{
    height: 6px; border-radius: 3px; background: {TONGJI_GOLD};
    margin-bottom: 0.5rem; transition: width 0.5s ease;
}}

.section-title {{
    display: flex; align-items: center; gap: 0.5rem; margin: 1rem 0 0.6rem;
    color: {TEXT_DARK}; font-size: 1rem; font-weight: 700;
}}
.section-title::before {{
    content: ""; width: 4px; height: 1rem; border-radius: 4px; background: {TONGJI_GOLD};
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{ background: transparent; }}
"""

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="同小智 · RAG 校园问答",
    page_icon="🏛️",
    layout="wide",
    menu_items=None,
)
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# ============================================================
# 顶部标题栏
# ============================================================
logo_data_uri = image_to_data_uri(TONGJI_LOGO_PATH)
logo_html = (
    f'<img class="tj-logo" src="{logo_data_uri}" alt="同济大学">'
    if logo_data_uri
    else '<div class="tj-logo" style="color:white;font-weight:700;">TONGJI</div>'
)

st.markdown(f"""
<div class="main-header">
    <div class="brand-lockup">
        <div class="brand-left">
            {logo_html}
            <div class="brand-divider"></div>
            <div>
                <div class="main-kicker">Campus RAG Assistant</div>
                <h1 class="main-title">同小智 · RAG 校园智能问答</h1>
                <p class="main-subtitle">BGE + FAISS + RRF + DeepSeek · 七层检索增强管线</p>
            </div>
        </div>
        <div class="brand-badge">RAG V2</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 懒加载模型（缓存资源）
# ============================================================

@st.cache_resource
def load_pipeline() -> dict:
    """加载所有检索器、嵌入模型、重排序器。"""
    chunks = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")

    embedder = BGEEmbedder()
    dense = DenseRetriever.load(embedder, EMBEDDING_DIM)
    bm25_jieba = Bm25JiebaRetriever.fit(chunks)
    hybrid_rrf = HybridRRFRetriever(dense, bm25_jieba)

    tfidf = TfidfRetriever.load(INDEX_PATH)
    bm25_old = BM25Retriever.fit(chunks)

    reranker = BGEReranker()

    return {
        "rrf": hybrid_rrf,
        "dense": dense,
        "bm25": bm25_jieba,
        "tfidf": tfidf,
        "bm25_old": bm25_old,
        "reranker": reranker,
        "embedder": embedder,
        "chunk_count": len(chunks),
    }


pipeline = load_pipeline()

# ============================================================
# Session State 初始化
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_turns=3)
if "mode" not in st.session_state:
    st.session_state.mode = "RAG (RRF only)"  # 默认不开重排，保证响应速度
if "pipeline" not in st.session_state:
    st.session_state.pipeline = pipeline

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown('<p class="sidebar-header">⚙️ 系统配置</p>', unsafe_allow_html=True)

    chunk_count = pipeline["chunk_count"]
    st.markdown(f"""
    <div class="sidebar-stat">
        <div style="font-size:0.8rem;opacity:0.9;">知识库规模</div>
        <div style="font-size:1.8rem;font-weight:700;">{chunk_count}</div>
        <div style="font-size:0.75rem;opacity:0.8;">条文档片段（chunk）</div>
    </div>
    """, unsafe_allow_html=True)

    # 模式切换
    st.markdown('<p style="color:#666;font-weight:600;margin-top:1rem;">🔄 回答模式</p>', unsafe_allow_html=True)
    mode = st.radio(
        "选择模式",
        ["RAG + Reranker", "RAG (RRF only)", "无检索基线", "RAG + Self-RAG 校验"],
        label_visibility="collapsed",
        key="mode_selector",
    )
    st.session_state.mode = mode

    # 检索参数
    with st.expander("🔧 检索参数", expanded=False):
        top_k = st.slider("Top-K", 1, 10, RETRIEVAL_TOP_K)
        st.caption(f"Dense 召回: {DENSE_RECALL_K} | BM25 召回: {BM25_RECALL_K}")

    # 评测摘要
    if EVAL_SUMMARY_PATH.exists():
        with st.expander("📊 评测摘要", expanded=False):
            try:
                import json
                eval_data = json.loads(EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
                st.metric("Hit@1", f"{eval_data.get('hit_at_1', 0):.1%}")
                st.metric("MRR", f"{eval_data.get('mrr', 0):.1%}")
            except Exception:
                pass

    # 清除对话
    if st.button("🗑️ 清除对话历史", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

# ============================================================
# 聊天界面（多轮对话）
# ============================================================

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 检索来源"):
                for s in msg["sources"]:
                    st.caption(f"[{s['doc_id']}] {s['title']} (score: {s['score']:.3f})")
        if msg.get("verification"):
            v = msg["verification"]
            verdict_color = {"fully_supported": "green", "partially_supported": "orange", "unsupported": "red"}
            color = verdict_color.get(v.get("verdict", ""), "gray")
            st.caption(f"🔍 Self-RAG 校验: :{color}[{v.get('verdict', '?')}] — {v.get('explanation', '')}")

# 输入框
if prompt := st.chat_input("输入你的校园问题，例如：图书馆周末几点开门？"):
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏛️"):
        mode = st.session_state.mode
        memory = st.session_state.memory

        if mode == "无检索基线":
            with st.spinner("思考中..."):
                answer = no_retrieval_baseline(prompt, backend="deepseek")
            st.markdown(answer)
            sources_info = []
            verify_info = None

        else:
            # 检索
            retriever = pipeline["rrf"]
            reranker = pipeline["reranker"] if "Reranker" in mode else None

            with st.spinner("🔍 检索相关知识..."):
                chunks = retriever.retrieve(prompt, top_k=top_k)
                if reranker and len(chunks) > 1:
                    chunks = reranker.rerank(prompt, chunks, top_k=min(5, len(chunks)))

            sources_info = [
                {"doc_id": c.doc_id, "title": c.title, "score": c.score, "source": c.source}
                for c in chunks
            ]

            # 流式生成
            placeholder = st.empty()
            full_answer = ""
            try:
                for token in llm_answer_stream(prompt, chunks, memory):
                    full_answer += token
                    placeholder.markdown(full_answer + "▌")
                placeholder.markdown(full_answer)
            except Exception as e:
                placeholder.error(f"生成失败: {e}")
                full_answer = f"[生成失败] {e}"

            answer = full_answer

            # 检索来源展示
            with st.expander(f"📎 检索来源 (Top {len(chunks)})"):
                for i, c in enumerate(chunks):
                    bar_width = min(100, int(c.score * 100)) if c.score <= 1 else 100
                    st.markdown(f"""
                    <div style="margin-bottom:0.6rem;padding:0.5rem;border-radius:6px;border-left:3px solid {TONGJI_BLUE};background:#F8FAFC;">
                        <b>#{i+1} [{c.doc_id}] {c.title}</b>
                        <span style="float:right;color:{TONGJI_GOLD};font-weight:600;">{c.score:.4f}</span>
                        <div class="score-bar" style="width:{bar_width}%;"></div>
                        <p style="font-size:0.85rem;color:#555;margin:0.3rem 0 0;">{c.content[:200]}{'...' if len(c.content)>200 else ''}</p>
                        <span style="font-size:0.75rem;color:#999;">{c.source}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # Self-RAG 校验
            verify_info = None
            if "Self-RAG" in mode:
                with st.spinner("🔍 Self-RAG 校验中..."):
                    try:
                        verify_info = self_rag_verify(prompt, answer, chunks)
                        verdict = verify_info.get("verdict", "?")
                        v_map = {"fully_supported": "✅ 完全有据可循", "partially_supported": "⚠️ 部分有据", "unsupported": "❌ 缺乏依据"}
                        st.caption(f"{v_map.get(verdict, verdict)} — {verify_info.get('explanation', '')}")
                    except Exception as e:
                        st.caption(f"校验失败: {e}")

            memory.add(prompt, answer)

        # 保存本轮消息
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "avatar": "🏛️",
            "sources": sources_info,
            "verification": verify_info,
        })

# ============================================================
# 底部
# ============================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;color:#999;font-size:0.78rem;padding:0.5rem;">
    <p>同小智 · 同济大学 RAG 校园智能问答助手 · BGE + FAISS + RRF + DeepSeek</p>
    <p>B07 小组 · 凌霄 归梦依 周子涵 杨歆苒</p>
</div>
""", unsafe_allow_html=True)
