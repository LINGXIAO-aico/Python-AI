from __future__ import annotations

import base64
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

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
    llm_answer_stream,
    no_retrieval_baseline,
    self_rag_verify,
)
from campus_rag.memory import ConversationMemory
from campus_rag.reranker import BGEReranker
from campus_rag.retriever import (
    Bm25JiebaRetriever,
    DenseRetriever,
    HybridRRFRetriever,
    TfidfRetriever,
)

# ═══════════════════════════════════════════════════════════
# 设计系统：明亮 · 现代 · 同济蓝金
# ═══════════════════════════════════════════════════════════
TONGJI_BLUE = "#1B4F8A"
TONGJI_BLUE_LIGHT = "#3A7BD5"
TONGJI_GOLD = "#C8963E"
PRIMARY_ACCENT = "#2563EB"
SURFACE = "#F8FAFC"
CARD_BG = "#FFFFFF"
TEXT_PRIMARY = "#1E293B"
TEXT_SECONDARY = "#64748B"
BORDER = "#E2E8F0"
SUCCESS = "#10B981"
WARNING = "#F59E0B"

TONGJI_LOGO_PATH = Path(__file__).parent / "assets" / "tongji-logo.png"


def image_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


CUSTOM_CSS = f"""
/* ── 全局 ── */
[data-testid="stAppViewContainer"] {{
    background: {SURFACE};
}}
.block-container {{
    padding-top: 0.8rem;
    max-width: 1120px;
}}

/* ── 强制所有文字深色 ── */
.stMarkdown, .stMarkdown p, .stMarkdown span,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
.stMarkdown li, .stMarkdown td, .stMarkdown th,
[data-testid="stCaptionContainer"], .stCaption,
[data-testid="stMarkdownContainer"] p,
.stExpander summary, .stExpander div {{
    color: {TEXT_PRIMARY} !important;
}}

/* ── 顶部横条 ── */
.main-header {{
    background: linear-gradient(135deg, {TONGJI_BLUE} 0%, #0F2D58 100%);
    padding: 0.9rem 1.3rem; border-radius: 14px;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 24px rgba(27,79,138,0.18);
    position: relative; overflow: hidden;
}}
.main-header::before {{
    content: ""; position: absolute; top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}}
.brand-lockup {{
    position: relative; z-index: 1;
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
}}
.brand-left {{ display: flex; align-items: center; gap: 0.7rem; }}
.tj-logo {{ width: 120px; height: auto; display: block; }}
.brand-divider {{ width: 1px; height: 42px; background: rgba(255,255,255,0.3); }}
.main-kicker {{
    color: rgba(255,255,255,0.72); font-size: 0.68rem;
    letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 2px;
}}
.main-title {{
    color: #FFFFFF !important; font-size: 1.55rem !important;
    font-weight: 700 !important; margin: 0 !important;
}}
.main-subtitle {{
    color: rgba(255,255,255,0.78) !important;
    font-size: 0.82rem !important; margin: 3px 0 0 !important; font-weight: 400;
}}
.brand-badge {{
    border: 1px solid rgba(255,255,255,0.25);
    color: rgba(255,255,255,0.88); border-radius: 20px;
    padding: 0.25rem 0.7rem; font-size: 0.72rem;
    background: rgba(255,255,255,0.08); white-space: nowrap;
}}

/* ── 示例问题按钮 ── */
.example-section {{
    display: flex; flex-wrap: wrap; gap: 0.4rem;
    margin: 0.3rem 0 0.8rem 0;
}}
.example-label {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.8rem; font-weight: 600;
    margin-bottom: 0.25rem;
}}

/* ── 侧边栏 ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #F1F5F9 0%, #E8EDF3 100%);
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY} !important; }}
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stSelectbox label p {{
    color: {TEXT_PRIMARY} !important; font-weight: 500;
}}

.sidebar-header {{
    color: {TONGJI_BLUE} !important; font-weight: 700;
    font-size: 1rem; padding-bottom: 0.4rem;
    border-bottom: 2px solid {TONGJI_GOLD}; margin-bottom: 0.8rem;
}}
.sidebar-stat {{
    background: linear-gradient(135deg, {TONGJI_BLUE} 0%, #0D3B6E 100%);
    padding: 0.8rem; border-radius: 12px;
    color: #FFFFFF; margin-bottom: 1rem;
    border-top: 3px solid {TONGJI_GOLD};
    box-shadow: 0 4px 16px rgba(27,79,138,0.12);
}}
.sidebar-stat * {{ color: #FFFFFF !important; }}

/* ── 聊天消息 ── */
[data-testid="stChatMessage"] {{
    border-radius: 14px; padding: 0.6rem 0.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}}
.stChatMessage * {{ color: {TEXT_PRIMARY} !important; }}

/* ── 输入框 ── */
[data-testid="stChatInput"] textarea {{
    color: {TEXT_PRIMARY} !important;
    border: 1.5px solid {BORDER} !important;
    border-radius: 14px !important;
    background: {CARD_BG} !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: {PRIMARY_ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.08) !important;
}}

/* ── 按钮 ── */
.stButton > button {{
    border-radius: 10px !important; border: 1.5px solid {BORDER} !important;
    background: {CARD_BG} !important; color: {TEXT_PRIMARY} !important;
    font-size: 0.8rem !important; padding: 0.35rem 0.7rem !important;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    border-color: {PRIMARY_ACCENT} !important;
    background: #EEF2FF !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.1);
}}

/* ── 检索来源卡片 ── */
.source-card {{
    background: {CARD_BG}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 0.6rem; margin-bottom: 0.5rem;
    border-left: 3px solid {PRIMARY_ACCENT};
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}}
.source-card b {{ color: {TEXT_PRIMARY} !important; }}
.source-card p {{ color: {TEXT_SECONDARY} !important; font-size: 0.82rem; }}
.score-bar {{
    height: 5px; border-radius: 3px;
    background: linear-gradient(90deg, {TONGJI_GOLD}, #E8C876);
    margin: 0.3rem 0;
}}

/* ── 对比模式 ── */
.compare-card {{
    border: 1px solid {BORDER}; border-radius: 12px;
    padding: 0.8rem; margin-bottom: 0.5rem;
    background: {CARD_BG}; box-shadow: 0 2px 6px rgba(0,0,0,0.03);
}}
.compare-card-title {{
    font-weight: 700; color: {TONGJI_BLUE} !important;
    font-size: 0.88rem; margin-bottom: 0.4rem;
    border-bottom: 2px solid {TONGJI_GOLD}; padding-bottom: 0.3rem;
}}
.compare-card-content {{ color: {TEXT_PRIMARY} !important; font-size: 0.88rem; line-height: 1.6; }}

/* ── 展开器 ── */
.streamlit-expanderHeader {{ color: {TEXT_PRIMARY} !important; font-weight: 600; }}
.streamlit-expanderContent {{ color: {TEXT_SECONDARY} !important; }}

/* ── 评分条 ── */
[data-testid="stMetricValue"] {{ color: {TONGJI_BLUE} !important; }}

/* 隐藏默认header */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{ background: transparent; }}
"""

# ═══════════════════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="同小智 · RAG 校园问答",
    page_icon="🏛️",
    layout="wide",
    menu_items=None,
)
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 顶部标题栏
# ═══════════════════════════════════════════════════════════
logo_data_uri = image_to_data_uri(TONGJI_LOGO_PATH)
logo_html = (
    f'<img class="tj-logo" src="{logo_data_uri}" alt="同济大学">'
    if logo_data_uri
    else ""
)

st.markdown(f"""
<div class="main-header">
    <div class="brand-lockup">
        <div class="brand-left">
            {logo_html}
            <div class="brand-divider"></div>
            <div>
                <div class="main-kicker">Campus RAG Assistant</div>
                <h1 class="main-title">同小智 · 校园智能问答</h1>
                <p class="main-subtitle">BGE + FAISS + RRF + DeepSeek &nbsp;|&nbsp; 七层检索增强管线</p>
            </div>
        </div>
        <div class="brand-badge">🏛️ RAG V2</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 懒加载模型
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_pipeline() -> dict:
    chunks = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")
    embedder = BGEEmbedder()
    dense = DenseRetriever.load(embedder, EMBEDDING_DIM)
    bm25_jieba = Bm25JiebaRetriever.fit(chunks)
    hybrid_rrf = HybridRRFRetriever(dense, bm25_jieba)
    tfidf = TfidfRetriever.load(INDEX_PATH)
    reranker = BGEReranker()
    return {
        "rrf": hybrid_rrf,
        "dense": dense,
        "bm25": bm25_jieba,
        "tfidf": tfidf,
        "reranker": reranker,
        "embedder": embedder,
        "chunk_count": len(chunks),
    }

pipeline = load_pipeline()

# ═══════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════
defaults = {
    "messages": [],
    "memory": ConversationMemory(max_turns=3),
    "mode": "RAG (RRF only)",
    "pipeline": pipeline,
    "compare_mode": False,
    "single_mode": "RAG (RRF only) — ⚡ 快速推荐",
    "top_k": RETRIEVAL_TOP_K,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="sidebar-header">⚙️ 系统配置</p>', unsafe_allow_html=True)

    chunk_count = pipeline["chunk_count"]
    st.markdown(f"""
    <div class="sidebar-stat">
        <div style="font-size:0.75rem;opacity:0.85;">知识库规模</div>
        <div style="font-size:1.6rem;font-weight:700;">{chunk_count}</div>
        <div style="font-size:0.7rem;opacity:0.8;">个文档片段 · 16个类目</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="color:#1E293B;font-weight:600;margin-top:0.8rem;font-size:0.85rem;">🔄 回答模式</p>', unsafe_allow_html=True)
    mode = st.radio(
        "选择回答模式",
        ["单模式回答", "对比模式（4种方法并排对比）"],
        label_visibility="collapsed",
        key="mode_selector",
    )
    st.session_state.compare_mode = ("对比" in mode)

    if not st.session_state.compare_mode:
        single_mode = st.selectbox(
            "具体方法",
            ["RAG (RRF only) — ⚡ 快速推荐",
             "RAG + Reranker — 🎯 精度最高",
             "Dense (BGE) — 🧠 语义检索",
             "BM25 (jieba) — 🔤 关键词检索",
             "无检索基线 — 🤖 大模型直答",
             "RAG + Self-RAG — 🔍 带校验"],
            label_visibility="collapsed",
        )
        st.session_state.single_mode = single_mode

    with st.expander("🔧 检索参数", expanded=False):
        st.session_state.top_k = st.slider("Top-K", 1, 10, st.session_state.top_k)
        st.caption(f"Dense 召回: {DENSE_RECALL_K}  |  BM25 召回: {BM25_RECALL_K}")

    if EVAL_SUMMARY_PATH.exists():
        with st.expander("📊 评测摘要", expanded=False):
            try:
                import json
                eval_data = json.loads(EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
                c1, c2 = st.columns(2)
                c1.metric("Hit@1", f"{eval_data.get('hit_at_1', 0):.1%}")
                c2.metric("MRR", f"{eval_data.get('mrr', 0):.1%}")
            except Exception:
                pass

    if st.button("🗑️ 清除对话历史", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════
# 示例问题
# ═══════════════════════════════════════════════════════════
EXAMPLES = [
    ("📚", "图书馆周末开放时间？"),
    ("🎓", "研究生学位论文答辩流程？"),
    ("🏃", "体育课体测包含哪些项目？"),
    ("📝", "本科生选课流程和时间安排？"),
    ("💳", "一卡通丢了怎么补办？"),
    ("🏥", "校医院就诊报销流程？"),
]

st.markdown('<p style="color:#64748B;font-size:0.78rem;margin-bottom:0.2rem;font-weight:600;">👇 试试点击示例问题：</p>', unsafe_allow_html=True)
cols = st.columns(len(EXAMPLES))
for i, (emoji, question) in enumerate(EXAMPLES):
    with cols[i]:
        if st.button(f"{emoji}  {question}", key=f"ex_{i}", use_container_width=True):
            st.session_state.example_prompt = question
            st.rerun()

# ═══════════════════════════════════════════════════════════
# 渲染历史消息
# ═══════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=msg.get("avatar")):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 检索来源"):
                for s in msg["sources"]:
                    st.caption(f"[{s['doc_id']}] {s['title']}  (score: {s['score']:.3f})")
        if msg.get("verification"):
            v = msg["verification"]
            labels = {"fully_supported": "green", "partially_supported": "orange", "unsupported": "red"}
            st.caption(f"🔍 Self-RAG: :{labels.get(v.get('verdict',''),'gray')}[{v.get('verdict','?')}] — {v.get('explanation','')}")


# ═══════════════════════════════════════════════════════════
# 核心：检索 + 生成
# ═══════════════════════════════════════════════════════════
def do_retrieve_and_generate(query: str, method: str):
    """执行检索+生成，返回 (answer, sources, verify_info)。"""
    reranker = pipeline["reranker"]
    memory = st.session_state.memory
    top_k = st.session_state.top_k

    if "无检索" in method or "直答" in method or "NONE" in method:
        with st.spinner("💭 大模型直答中（无检索）..."):
            answer = no_retrieval_baseline(query, backend="deepseek")
        return answer, [], None

    # 选检索器
    if ("Dense" in method and "语义" in method) or method == "DENSE":
        retriever = pipeline["dense"]
    elif ("BM25" in method and "关键词" in method) or method == "BM25":
        retriever = pipeline["bm25"]
    else:
        retriever = pipeline["rrf"]

    with st.spinner("🔍 检索相关知识..."):
        chunks = retriever.retrieve(query, top_k=top_k)
        if ("Reranker" in method or "精排" in method or "+RR" in method or method == "RRF+RR") and len(chunks) > 1:
            chunks = reranker.rerank(query, chunks, top_k=min(5, len(chunks)))

    sources_info = [
        {"doc_id": c.doc_id, "title": c.title, "score": c.score, "source": c.source}
        for c in chunks
    ]

    placeholder = st.empty()
    full_answer = ""
    try:
        for token in llm_answer_stream(query, chunks, memory):
            full_answer += token
            placeholder.markdown(full_answer + "▌")
        placeholder.markdown(full_answer)
    except Exception as e:
        placeholder.error(f"生成失败: {e}")
        full_answer = f"[生成失败] {e}"

    # Self-RAG
    verify_info = None
    if "Self-RAG" in method or "校验" in method:
        with st.spinner("🔍 Self-RAG 校验中..."):
            try:
                verify_info = self_rag_verify(query, full_answer, chunks)
                verdict = verify_info.get("verdict", "?")
                labels = {"fully_supported": "✅ 完全有据可循", "partially_supported": "⚠️ 部分有据", "unsupported": "❌ 缺乏依据"}
                st.caption(f"{labels.get(verdict, verdict)} — {verify_info.get('explanation', '')}")
            except Exception as e:
                st.caption(f"校验失败: {e}")

    return full_answer, sources_info, verify_info


# ═══════════════════════════════════════════════════════════
# 处理用户输入
# ═══════════════════════════════════════════════════════════
prompt = st.chat_input("输入你的校园问题，例如：图书馆周末几点开门？")

# 示例问题触发
if "example_prompt" in st.session_state and st.session_state.example_prompt:
    prompt = st.session_state.example_prompt
    st.session_state.example_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "👤"})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # ═══════ 对比模式 ═══════
    if st.session_state.compare_mode:
        methods = [
            ("NONE",     "🤖 无检索基线",     "大模型直答·无知识库"),
            ("BM25",     "🔤 BM25 稀疏检索",  "jieba分词·关键词匹配"),
            ("DENSE",    "🧠 BGE 密集检索",   "语义向量·深度理解"),
            ("RRF+RR",   "🎯 RRF混合精排",    "双路融合+重排序"),
        ]

        with st.chat_message("assistant", avatar="🏛️"):
            st.markdown("#### 🔬 对比模式 — 四种方法同时回答")

            tabs = st.tabs([f"{emoji} {name}" for emoji, name, _ in methods])
            all_answers = {}

            for (key, _name, desc), tab in zip(methods, tabs, strict=True):
                with tab:
                    st.caption(f"*{desc}*")
                    ans, srcs, vfy = do_retrieve_and_generate(prompt, key)
                    all_answers[key] = (ans, srcs, vfy)
                    if srcs:
                        with st.expander("📎 检索来源"):
                            for s in srcs:
                                st.caption(f"[{s['doc_id']}] {s['title']}  (score: {s['score']:.3f})")

        all_sources = []
        for _, (_ans, srcs, _vfy) in all_answers.items():
            all_sources.extend(srcs)

        st.session_state.messages.append({
            "role": "assistant", "content": "> 🔬 对比模式 · 4种方法，请查看上方标签页对比差异",
            "avatar": "🏛️", "sources": all_sources, "verification": None,
        })
        st.session_state.memory.add(prompt, all_answers.get("RRF+RR", ("",))[0] or "")

    # ═══════ 单模式 ═══════
    else:
        method = st.session_state.get("single_mode", "RAG (RRF only) — ⚡ 快速推荐")
        with st.chat_message("assistant", avatar="🏛️"):
            ans, srcs, vfy = do_retrieve_and_generate(prompt, method)

            if srcs:
                with st.expander(f"📎 检索来源 (Top {len(srcs)})"):
                    for i, s in enumerate(srcs):
                        bar_width = min(100, int(s['score'] * 100)) if s['score'] <= 1 else 100
                        st.markdown(f"""
                        <div class="source-card">
                            <b>#{i+1} [{s['doc_id']}] {s['title']}</b>
                            <span style="float:right;color:{TONGJI_GOLD};font-weight:600;">{s['score']:.4f}</span>
                            <div class="score-bar" style="width:{bar_width}%;"></div>
                            <span style="font-size:0.72rem;color:{TEXT_SECONDARY};">{s['source']}</span>
                        </div>
                        """, unsafe_allow_html=True)

            st.session_state.memory.add(prompt, ans)

        st.session_state.messages.append({
            "role": "assistant", "content": ans, "avatar": "🏛️",
            "sources": srcs, "verification": vfy,
        })

# ═══════════════════════════════════════════════════════════
# 底部
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:0.4rem;">
    <p style="color:{TEXT_SECONDARY} !important;font-size:0.75rem;margin:0;">
        🏛️ 同小智 · 同济大学 RAG 校园智能问答助手 &nbsp;|&nbsp; B07 小组 · 凌霄 归梦依 周子涵 杨歆苒
    </p>
</div>
""", unsafe_allow_html=True)
