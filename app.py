from __future__ import annotations

import base64
import html
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campus_rag.config import (
    BM25_RECALL_K,
    CHUNK_PATH,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    DENSE_RECALL_K,
    EMBEDDING_DIM,
    EVAL_SUMMARY_PATH,
    INDEX_PATH,
    RETRIEVAL_TOP_K,
)
from campus_rag.embeddings import BGEEmbedder
from campus_rag.generator import no_retrieval_baseline, self_rag_verify
from campus_rag.memory import ConversationMemory
from campus_rag.reranker import BGEReranker
from campus_rag.retriever import (
    Bm25JiebaRetriever,
    DenseRetriever,
    HybridRRFRetriever,
    RetrievedChunk,
    TfidfRetriever,
)


TONGJI_LOGO_PATH = Path(__file__).parent / "assets" / "tongji-logo.png"

MODE_OPTIONS = [
    {
        "key": "quick",
        "label": "快速推荐",
        "short": "RRF 混合检索",
        "icon": "⚡",
        "summary": "融合 BGE 语义检索和 jieba-BM25，速度与命中率比较均衡。",
        "retriever": "rrf",
        "rerank": False,
        "verify": False,
        "baseline": False,
    },
    {
        "key": "accurate",
        "label": "高精度",
        "short": "RRF + 重排",
        "icon": "🎯",
        "summary": "先混合召回，再用 Cross-Encoder 精排，适合追求更准的办事问题。",
        "retriever": "rrf",
        "rerank": True,
        "verify": False,
        "baseline": False,
    },
    {
        "key": "dense",
        "label": "语义检索",
        "short": "BGE Dense",
        "icon": "🧠",
        "summary": "看重语义相似度，适合口语化、同义改写较多的问题。",
        "retriever": "dense",
        "rerank": False,
        "verify": False,
        "baseline": False,
    },
    {
        "key": "bm25",
        "label": "关键词检索",
        "short": "jieba-BM25",
        "icon": "🔤",
        "summary": "看重词面匹配，适合包含明确部门、材料、系统名称的问题。",
        "retriever": "bm25",
        "rerank": False,
        "verify": False,
        "baseline": False,
    },
    {
        "key": "baseline",
        "label": "无检索基线",
        "short": "LLM 直答",
        "icon": "🤖",
        "summary": "不使用知识库，只作对照实验，答案不建议作为最终依据。",
        "retriever": "none",
        "rerank": False,
        "verify": False,
        "baseline": True,
    },
    {
        "key": "self_rag",
        "label": "带校验",
        "short": "RRF + Self-RAG",
        "icon": "🔍",
        "summary": "回答后再做依据校验，适合需要确认答案是否被资料支持的场景。",
        "retriever": "rrf",
        "rerank": False,
        "verify": True,
        "baseline": False,
    },
]


def get_mode(key: str) -> dict:
    return next((mode for mode in MODE_OPTIONS if mode["key"] == key), MODE_OPTIONS[0])


def image_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_css(theme: str) -> str:
    dark = theme == "dark"
    palette = {
        "bg": "#0F172A" if dark else "#F7F9FC",
        "surface": "#111827" if dark else "#FFFFFF",
        "surface_2": "#1E293B" if dark else "#EEF3F8",
        "text": "#E5E7EB" if dark else "#172033",
        "muted": "#A7B0C0" if dark else "#64748B",
        "border": "#334155" if dark else "#D8E0EA",
        "primary": "#60A5FA" if dark else "#1D4E89",
        "primary_strong": "#93C5FD" if dark else "#0F3A68",
        "accent": "#F2C14E" if dark else "#B8842B",
        "good": "#34D399" if dark else "#059669",
        "shadow": "rgba(0,0,0,0.30)" if dark else "rgba(15,23,42,0.08)",
    }
    return f"""
    <style>
    :root {{
        --app-bg: {palette["bg"]};
        --surface: {palette["surface"]};
        --surface-2: {palette["surface_2"]};
        --text: {palette["text"]};
        --muted: {palette["muted"]};
        --border: {palette["border"]};
        --primary: {palette["primary"]};
        --primary-strong: {palette["primary_strong"]};
        --accent: {palette["accent"]};
        --good: {palette["good"]};
        --shadow: {palette["shadow"]};
    }}
    [data-testid="stAppViewContainer"] {{
        background: var(--app-bg);
        color: var(--text);
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    #MainMenu, footer {{ visibility: hidden; }}
    .block-container {{
        max-width: 1040px;
        padding-top: 1.1rem;
        padding-bottom: 2.4rem;
    }}
    [data-testid="stSidebar"] {{
        background: var(--surface);
        border-right: 1px solid var(--border);
    }}
    [data-testid="stSidebar"] * {{
        color: var(--text);
    }}
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {{
        color: var(--text);
    }}
    .hero {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.05rem 1.15rem;
        box-shadow: 0 10px 26px var(--shadow);
        margin-bottom: 0.9rem;
    }}
    .hero-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }}
    .brand {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
        min-width: 0;
    }}
    .tj-logo {{
        width: 96px;
        height: auto;
        object-fit: contain;
    }}
    .hero-kicker {{
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.1rem;
    }}
    .hero-title {{
        color: var(--text) !important;
        font-size: 1.55rem;
        line-height: 1.18;
        font-weight: 760;
        margin: 0;
    }}
    .hero-subtitle {{
        color: var(--muted) !important;
        margin: 0.28rem 0 0 0;
        font-size: 0.9rem;
        line-height: 1.55;
    }}
    .mode-pill {{
        border: 1px solid var(--border);
        background: var(--surface-2);
        color: var(--primary-strong);
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        font-size: 0.8rem;
        white-space: nowrap;
        font-weight: 700;
    }}
    .intro-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 0.8rem;
    }}
    .mode-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.82rem;
        box-shadow: 0 6px 18px var(--shadow);
        min-height: 136px;
    }}
    .mode-card.active {{
        border-color: var(--primary);
        box-shadow: inset 0 0 0 1px var(--primary), 0 8px 22px var(--shadow);
    }}
    .mode-title {{
        display: flex;
        align-items: center;
        gap: 0.4rem;
        color: var(--text);
        font-weight: 760;
        font-size: 0.95rem;
        margin-bottom: 0.35rem;
    }}
    .mode-tag {{
        display: inline-block;
        color: var(--primary-strong);
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.12rem 0.48rem;
        font-size: 0.72rem;
        margin-bottom: 0.38rem;
    }}
    .mode-desc {{
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.55;
    }}
    .sidebar-title {{
        font-size: 0.95rem;
        font-weight: 800;
        margin: 0.2rem 0 0.55rem 0;
        color: var(--text);
    }}
    .hint-box {{
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.72rem;
        margin: 0.55rem 0 0.75rem 0;
        color: var(--muted);
        font-size: 0.8rem;
        line-height: 1.55;
    }}
    .metric-row {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.45rem;
        margin: 0.5rem 0 0.75rem 0;
    }}
    .mini-metric {{
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.48rem;
    }}
    .mini-metric strong {{
        display: block;
        color: var(--text);
        font-size: 0.92rem;
    }}
    .mini-metric span {{
        color: var(--muted);
        font-size: 0.68rem;
    }}
    .retrieval-card {{
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.62rem;
        margin-bottom: 0.55rem;
    }}
    .retrieval-title {{
        display: flex;
        justify-content: space-between;
        gap: 0.5rem;
        font-size: 0.78rem;
        font-weight: 760;
        line-height: 1.4;
    }}
    .score-bar-bg {{
        height: 7px;
        background: rgba(148,163,184,0.25);
        border-radius: 999px;
        overflow: hidden;
        margin: 0.42rem 0 0.24rem 0;
    }}
    .score-bar-fill {{
        height: 7px;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        border-radius: 999px;
    }}
    .source-anchor {{
        scroll-margin-top: 88px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.72rem;
        margin-bottom: 0.58rem;
    }}
    .source-anchor:target {{
        border-color: var(--accent);
        box-shadow: inset 0 0 0 1px var(--accent), 0 8px 18px var(--shadow);
    }}
    .source-head {{
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        font-weight: 760;
        color: var(--text);
    }}
    .source-body {{
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
        margin-top: 0.35rem;
    }}
    .cite-link {{
        color: var(--primary-strong) !important;
        font-weight: 800;
        text-decoration: none;
        border-bottom: 1px solid var(--primary);
    }}
    [data-testid="stChatMessage"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        box-shadow: 0 8px 20px var(--shadow);
    }}
    [data-testid="stChatInput"] textarea {{
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        background: var(--surface) !important;
    }}
    .stButton > button,
    .stDownloadButton > button {{
        border-radius: 8px !important;
        border: 1px solid var(--border) !important;
        background: var(--surface) !important;
        color: var(--text) !important;
        font-weight: 650 !important;
    }}
    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        border-color: var(--primary) !important;
        color: var(--primary-strong) !important;
    }}
    .stRadio [role="radiogroup"] label,
    .stSegmentedControl label {{
        color: var(--text) !important;
    }}
    div[data-testid="stStatusWidget"] {{
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface);
    }}
    @media (max-width: 900px) {{
        .hero-row {{ align-items: flex-start; flex-direction: column; }}
        .intro-grid {{ grid-template-columns: 1fr; }}
        .mode-pill {{ white-space: normal; }}
        .tj-logo {{ width: 82px; }}
    }}
    </style>
    """


def render_header(mode: dict) -> None:
    logo_data_uri = image_to_data_uri(TONGJI_LOGO_PATH)
    logo_html = (
        f'<img class="tj-logo" src="{logo_data_uri}" alt="同济大学">'
        if logo_data_uri
        else ""
    )
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-row">
                <div class="brand">
                    {logo_html}
                    <div>
                        <div class="hero-kicker">Campus RAG Assistant</div>
                        <h1 class="hero-title">同小智 · 校园智能问答</h1>
                        <p class="hero-subtitle">面向同济校园办事、学习生活、奖助就业和场馆服务的检索增强问答助手。</p>
                    </div>
                </div>
                <div class="mode-pill">{mode["icon"]} 当前模式：{mode["label"]}</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_pipeline() -> dict:
    chunks = pd.read_csv(CHUNK_PATH, encoding="utf-8-sig")
    embedder = BGEEmbedder()
    dense = DenseRetriever.load(embedder, EMBEDDING_DIM)
    bm25 = Bm25JiebaRetriever.fit(chunks)
    rrf = HybridRRFRetriever(dense, bm25)
    tfidf = TfidfRetriever.load(INDEX_PATH)
    return {
        "chunks": chunks,
        "chunk_count": len(chunks),
        "dense": dense,
        "bm25": bm25,
        "rrf": rrf,
        "tfidf": tfidf,
        "reranker": BGEReranker(),
    }


def init_state() -> None:
    defaults = {
        "messages": [],
        "memory": ConversationMemory(max_turns=3),
        "mode_key": "quick",
        "theme": "light",
        "top_k": RETRIEVAL_TOP_K,
        "last_sources": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def score_to_percent(score: float, max_score: float) -> int:
    if max_score <= 0:
        return 12
    pct = int(score / max_score * 100)
    return max(8, min(100, pct))


def render_retrieval_panel(sources: list[dict], container) -> None:
    with container:
        st.markdown('<div class="sidebar-title">Top-5 检索片段</div>', unsafe_allow_html=True)
        if not sources:
            st.markdown(
                '<div class="hint-box">用户提问后，这里会显示标题、分数和相似度条，方便观察检索是否命中正确资料。</div>',
                unsafe_allow_html=True,
            )
            return

        max_score = max(float(item.get("score", 0)) for item in sources) or 1.0
        for idx, item in enumerate(sources[:5], start=1):
            pct = score_to_percent(float(item.get("score", 0)), max_score)
            title = html.escape(str(item.get("title", "")))
            score = float(item.get("score", 0))
            doc_id = html.escape(str(item.get("doc_id", "")))
            st.markdown(
                f"""
                <div class="retrieval-card">
                    <div class="retrieval-title">
                        <span>{idx}. {title}</span>
                        <span>{score:.4f}</span>
                    </div>
                    <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%;"></div></div>
                    <div style="font-size:0.68rem;color:var(--muted);">引用编号：[{idx}] · {doc_id}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_mode_intro(active_key: str) -> None:
    st.markdown("#### 先选模式，再开始提问")
    st.caption("六个模式对应不同检索/生成策略。默认推荐“快速推荐”，需要更稳时再切换到“高精度”或“带校验”。")
    cards = []
    for mode in MODE_OPTIONS:
        active = " active" if mode["key"] == active_key else ""
        icon = html.escape(mode["icon"])
        label = html.escape(mode["label"])
        short = html.escape(mode["short"])
        summary = html.escape(mode["summary"])
        cards.append(
            f'<div class="mode-card{active}">'
            f'<div class="mode-title"><span>{icon}</span><span>{label}</span></div>'
            f'<div class="mode-tag">{short}</div>'
            f'<div class="mode-desc">{summary}</div>'
            "</div>"
        )
    st.markdown(f'<div class="intro-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def markdown_conversation(messages: list[dict]) -> str:
    lines = [
        "# 同小智校园问答对话导出",
        "",
        f"- 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 对话轮数：{sum(1 for msg in messages if msg['role'] == 'user')}",
        "",
    ]
    for msg in messages:
        speaker = "用户" if msg["role"] == "user" else "同小智"
        lines.append(f"## {speaker}")
        lines.append("")
        lines.append(msg["content"])
        lines.append("")
        sources = msg.get("sources") or []
        if sources:
            lines.append("### 来源")
            for idx, source in enumerate(sources, start=1):
                lines.append(
                    f"- [{idx}] {source['title']} ({source['doc_id']}), score={source['score']:.4f}, source={source['source']}"
                )
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def get_retriever(pipeline: dict, mode: dict):
    if mode["retriever"] == "dense":
        return pipeline["dense"]
    if mode["retriever"] == "bm25":
        return pipeline["bm25"]
    return pipeline["rrf"]


def source_rows(chunks: Iterable[RetrievedChunk]) -> list[dict]:
    rows = []
    for chunk in chunks:
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "score": float(chunk.score),
                "source": chunk.source,
                "url": chunk.url,
                "content": chunk.content,
            }
        )
    return rows


def build_numbered_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for idx, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{idx}] 标题：{chunk.title}\n"
            f"来源：{chunk.source}\n"
            f"内容：{chunk.content}"
        )
    return "\n\n".join(lines)


def fallback_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "当前知识库中没有检索到足够相关的资料，建议换一个更具体的问题，或咨询对应学院/部门。"
    best = chunks[0]
    answer = [
        f"根据检索到的资料，最相关的是「{best.title}」[1]。",
        best.content,
    ]
    if len(chunks) > 1:
        refs = "、".join(f"「{chunk.title}」[{idx}]" for idx, chunk in enumerate(chunks[1:3], start=2))
        answer.append(f"也可以参考 {refs}。")
    return "\n\n".join(answer)


def generate_answer(question: str, chunks: list[RetrievedChunk], memory: ConversationMemory) -> str:
    if not DEEPSEEK_API_KEY:
        return fallback_answer(question, chunks)

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    context = build_numbered_context(chunks)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "你是同济大学校园智能问答助手“同小智”。只能依据参考资料回答。"
                "回答要简洁、分步骤，并在关键结论后标注 [1] [2] 这样的引用编号。"
                "不要使用 doc_id 作为引用编号。资料不足时明确说明建议咨询对应部门。"
            ),
        }
    ]
    messages.extend(memory.get_context())
    messages.append(
        {
            "role": "user",
            "content": f"参考资料：\n{context}\n\n问题：{question}\n请根据参考资料回答：",
        }
    )
    response = client.chat.completions.create(
        model=DEEPSEEK_CHAT_MODEL,
        messages=messages,
        temperature=0.25,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def make_citations_clickable(answer: str, sources: list[dict]) -> str:
    paragraphs = html.escape(answer).splitlines()
    text = "<br>".join(paragraphs)
    doc_to_index = {source["doc_id"]: idx for idx, source in enumerate(sources, start=1)}

    for doc_id, idx in doc_to_index.items():
        escaped_doc = re.escape(html.escape(doc_id))
        text = re.sub(
            rf"\[{escaped_doc}\]",
            f'<a class="cite-link" href="#source-{idx}">[{idx}]</a>',
            text,
        )

    def replace_number(match: re.Match) -> str:
        idx = int(match.group(1))
        if 1 <= idx <= len(sources):
            return f'<a class="cite-link" href="#source-{idx}">[{idx}]</a>'
        return match.group(0)

    return re.sub(r"\[(\d+)\]", replace_number, text)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("##### 来源")
    for idx, source in enumerate(sources, start=1):
        title = html.escape(source["title"])
        body = html.escape(source["content"][:240])
        doc_id = html.escape(source["doc_id"])
        source_name = html.escape(source["source"])
        st.markdown(
            f"""
            <div id="source-{idx}" class="source-anchor">
                <div class="source-head">
                    <span>[{idx}] {title}</span>
                    <span>{source["score"]:.4f}</span>
                </div>
                <div class="source-body">{body}...</div>
                <div style="margin-top:0.35rem;font-size:0.72rem;color:var(--muted);">{doc_id} · {source_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_message(msg: dict) -> None:
    avatar = "👤" if msg["role"] == "user" else "🏛️"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            st.markdown(
                make_citations_clickable(msg["content"], msg.get("sources", [])),
                unsafe_allow_html=True,
            )
            if msg.get("verification"):
                verdict = msg["verification"].get("verdict", "unknown")
                explanation = msg["verification"].get("explanation", "")
                st.caption(f"Self-RAG 校验：{verdict} · {explanation}")
            render_sources(msg.get("sources", []))
        else:
            st.markdown(msg["content"])


def ask(question: str, mode: dict, pipeline: dict, retrieval_container) -> dict:
    if mode["baseline"]:
        with st.status("正在处理", expanded=True) as status:
            status.write("检索中：当前模式不使用知识库，跳过检索。")
            status.write("重排中：无检索结果，跳过重排。")
            status.write("生成中：调用无检索基线回答。")
            try:
                answer = no_retrieval_baseline(
                    question,
                    backend="deepseek" if DEEPSEEK_API_KEY else "extractive",
                )
            except Exception:
                answer = no_retrieval_baseline(question, backend="extractive")
            status.write("校验中：基线模式不做依据校验。")
            status.update(label="处理完成", state="complete", expanded=False)
        render_retrieval_panel([], retrieval_container)
        return {"content": answer, "sources": [], "verification": None}

    chunks: list[RetrievedChunk] = []
    verification = None
    with st.status("正在处理", expanded=True) as status:
        status.write("检索中：召回候选知识片段。")
        retriever = get_retriever(pipeline, mode)
        chunks = retriever.retrieve(question, top_k=st.session_state.top_k)
        sources = source_rows(chunks[:5])
        st.session_state.last_sources = sources
        render_retrieval_panel(sources, retrieval_container)

        if mode["rerank"] and len(chunks) > 1:
            status.write("重排中：使用 Cross-Encoder 对候选片段精排。")
            try:
                chunks = pipeline["reranker"].rerank(question, chunks, top_k=min(5, len(chunks)))
                sources = source_rows(chunks[:5])
                st.session_state.last_sources = sources
                render_retrieval_panel(sources, retrieval_container)
            except Exception as exc:
                status.write(f"重排中：重排模型不可用，已保留原始排序。原因：{exc}")
        else:
            status.write("重排中：当前模式不启用重排，保留召回排序。")

        status.write("生成中：根据参考资料组织回答。")
        answer = generate_answer(question, chunks[:5], st.session_state.memory)

        if mode["verify"]:
            status.write("校验中：检查回答是否被检索资料支持。")
            try:
                verification = self_rag_verify(question, answer, chunks[:5])
            except Exception as exc:
                verification = {"verdict": "verify_failed", "explanation": str(exc)}
        else:
            status.write("校验中：当前模式不启用 Self-RAG 校验。")

        status.update(label="处理完成", state="complete", expanded=False)

    return {"content": answer, "sources": source_rows(chunks[:5]), "verification": verification}


st.set_page_config(
    page_title="同小智 · 校园智能问答",
    page_icon="🏛️",
    layout="wide",
    menu_items=None,
)
init_state()

mode_labels = {f"{mode['icon']} {mode['label']}": mode["key"] for mode in MODE_OPTIONS}

with st.sidebar:
    st.markdown('<div class="sidebar-title">同小智设置</div>', unsafe_allow_html=True)
    theme_choice = st.radio(
        "主题",
        ["亮色", "暗色"],
        horizontal=True,
        index=0 if st.session_state.theme == "light" else 1,
    )
    st.session_state.theme = "dark" if theme_choice == "暗色" else "light"

st.markdown(make_css(st.session_state.theme), unsafe_allow_html=True)

pipeline = load_pipeline()
mode = get_mode(st.session_state.mode_key)
render_header(mode)

with st.sidebar:
    selected_label = st.radio(
        "回答模式",
        list(mode_labels.keys()),
        index=list(mode_labels.values()).index(st.session_state.mode_key),
    )
    st.session_state.mode_key = mode_labels[selected_label]
    mode = get_mode(st.session_state.mode_key)
    st.markdown(
        f'<div class="hint-box"><b>{mode["short"]}</b><br>{mode["summary"]}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("检索参数", expanded=False):
        st.session_state.top_k = st.slider("Top-K", 1, 10, st.session_state.top_k)
        st.caption(f"Dense 召回：{DENSE_RECALL_K} · BM25 召回：{BM25_RECALL_K}")

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="mini-metric"><strong>{pipeline["chunk_count"]}</strong><span>chunks</span></div>
            <div class="mini-metric"><strong>120</strong><span>FAQ</span></div>
            <div class="mini-metric"><strong>6</strong><span>modes</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if EVAL_SUMMARY_PATH.exists():
        with st.expander("评估摘要", expanded=False):
            try:
                import json

                summary = json.loads(EVAL_SUMMARY_PATH.read_text(encoding="utf-8"))
                st.metric("Hybrid Hit@1", f"{summary.get('hit_at_1', 0):.1%}")
                st.metric("RAG 关键词召回", f"{summary.get('rag_keyword_recall', 0):.1%}")
            except Exception:
                st.caption("评估摘要暂时不可读取。")

    retrieval_slot = st.empty()
    render_retrieval_panel(st.session_state.last_sources, retrieval_slot)

    export_text = markdown_conversation(st.session_state.messages)
    st.download_button(
        "导出对话",
        data=export_text,
        file_name=f"tongxiaozi_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )

    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.session_state.last_sources = []
        st.rerun()


if not st.session_state.messages:
    render_mode_intro(st.session_state.mode_key)
    st.markdown("#### 可以这样问")
    example_cols = st.columns(3)
    examples = [
        "本科生选课申请没通过怎么办？",
        "校园卡丢失后如何挂失和补办？",
        "体质健康测试包含哪些项目？",
    ]
    for col, example in zip(example_cols, examples, strict=True):
        with col:
            if st.button(example, use_container_width=True):
                st.session_state.pending_prompt = example
                st.rerun()
else:
    for message in st.session_state.messages:
        render_message(message)


prompt = st.chat_input("输入校园问题，例如：图书馆校外怎么访问数据库？")
if st.session_state.get("pending_prompt"):
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant", avatar="🏛️"):
        result = ask(prompt, mode, pipeline, retrieval_slot)
        st.markdown(
            make_citations_clickable(result["content"], result["sources"]),
            unsafe_allow_html=True,
        )
        if result.get("verification"):
            verification = result["verification"]
            st.caption(
                f"Self-RAG 校验：{verification.get('verdict', 'unknown')} · "
                f"{verification.get('explanation', '')}"
            )
        render_sources(result["sources"])

    assistant_msg = {
        "role": "assistant",
        "content": result["content"],
        "sources": result["sources"],
        "verification": result.get("verification"),
    }
    st.session_state.messages.append(assistant_msg)
    st.session_state.memory.add(prompt, result["content"])
