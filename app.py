from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from campus_rag.config import CHUNK_PATH, EVAL_SUMMARY_PATH, INDEX_PATH
from campus_rag.generator import answer_question, no_retrieval_baseline
from campus_rag.retriever import BM25Retriever, HybridRetriever, TfidfRetriever
import pandas as pd


# ========================================
# 同济大学风格主题配置
# ========================================
TONGJI_BLUE = "#003F87"       # 同济蓝（校徽主色）
TONGJI_BLUE_LIGHT = "#0055B8" # 亮蓝
TONGJI_BLUE_DARK = "#002B5C"  # 深蓝
TONGJI_GOLD = "#D4A84B"       # 金色点缀
TONGJI_BG = "#F5F5F5"         # 背景灰
TONGJI_CARD_BG = "#FFFFFF"    # 卡片白
TEXT_DARK = "#1A1A1A"         # 深色文字
TEXT_GRAY = "#666666"         # 次要文字
TONGJI_LOGO_PATH = Path(__file__).parent / "assets" / "tongji-logo.png"


def image_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

# 自定义CSS样式
CUSTOM_CSS = f"""
/* 全局背景 */
[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at top right, rgba(212, 168, 75, 0.10), transparent 28rem),
        linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%);
}}

.block-container {{
    padding-top: 1.2rem;
    max-width: 1180px;
}}

/* 顶部标题栏 */
.main-header {{
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(90deg, rgba(0, 43, 92, 0.94) 0%, rgba(0, 63, 135, 0.94) 62%, rgba(0, 85, 184, 0.92) 100%);
    padding: 1.35rem 1.6rem 1.25rem;
    border-radius: 8px;
    margin-bottom: 1.4rem;
    box-shadow: 0 14px 34px rgba(0, 43, 92, 0.22);
    border-top: 4px solid {TONGJI_GOLD};
}}

.main-header::after {{
    content: "";
    position: absolute;
    inset: 0;
    background:
        linear-gradient(120deg, transparent 0 58%, rgba(255,255,255,0.11) 58% 59%, transparent 59%),
        linear-gradient(120deg, transparent 0 72%, rgba(255,255,255,0.08) 72% 73%, transparent 73%);
    pointer-events: none;
}}

.brand-lockup {{
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.4rem;
}}

.brand-left {{
    display: flex;
    align-items: center;
    gap: 1rem;
    min-width: 0;
}}

.tj-logo {{
    width: 164px;
    max-width: 34vw;
    height: auto;
    display: block;
    filter: drop-shadow(0 5px 12px rgba(0,0,0,0.22));
}}

.brand-divider {{
    width: 1px;
    height: 58px;
    background: rgba(255,255,255,0.35);
    flex: 0 0 auto;
}}

.main-kicker {{
    color: rgba(255,255,255,0.78);
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}}

.brand-badge {{
    border: 1px solid rgba(255,255,255,0.28);
    color: rgba(255,255,255,0.92);
    border-radius: 999px;
    padding: 0.4rem 0.75rem;
    font-size: 0.8rem;
    white-space: nowrap;
    background: rgba(255,255,255,0.08);
}}

/* 主标题 */
.main-title {{
    color: white !important;
    font-size: clamp(1.65rem, 3vw, 2.15rem) !important;
    font-weight: 700 !important;
    margin: 0 !important;
    text-shadow: 0 2px 8px rgba(0,0,0,0.18);
    letter-spacing: 0;
}}

/* 副标题 */
.main-subtitle {{
    color: rgba(255,255,255,0.86) !important;
    font-size: 0.98rem !important;
    margin: 0.45rem 0 0 !important;
    font-weight: 400;
}}

.feature-strip {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
    margin: -0.35rem 0 1.45rem;
}}

.feature-item {{
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(0, 63, 135, 0.08);
    border-top: 3px solid {TONGJI_GOLD};
    border-radius: 8px;
    padding: 0.8rem 0.95rem;
    box-shadow: 0 8px 22px rgba(0, 43, 92, 0.06);
}}

.feature-item strong {{
    display: block;
    color: {TONGJI_BLUE_DARK};
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
}}

.feature-item span {{
    color: {TEXT_GRAY};
    font-size: 0.82rem;
}}

/* 侧边栏样式 */
[data-testid="stSidebar"] {{
    background:
        linear-gradient(180deg, rgba(232, 239, 248, 0.98) 0%, rgba(220, 230, 242, 0.98) 100%);
    border-right: 1px solid rgba(0, 43, 92, 0.20);
    box-shadow: 8px 0 28px rgba(0, 43, 92, 0.08);
}}

[data-testid="stSidebar"] > div:first-child {{
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.48) 0%, rgba(255, 255, 255, 0.18) 100%);
}}

/* 侧边栏标题 */
.sidebar-header {{
    color: {TONGJI_BLUE};
    font-weight: 700;
    font-size: 1.1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(212, 168, 75, 0.75);
    margin-bottom: 1rem;
}}

/* 卡片通用样式 */
.stCard {{
    background: {TONGJI_CARD_BG};
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.04);
    transition: all 0.3s ease;
}}

.stCard:hover {{
    box-shadow: 0 8px 32px rgba(0, 63, 135, 0.12);
    transform: translateY(-2px);
}}

/* RAG回答卡片 - 蓝色边框 */
.rag-answer-card {{
    border-left: 4px solid {TONGJI_BLUE};
    background: linear-gradient(135deg, #F0F7FF 0%, #FFFFFF 72%);
    border-radius: 8px;
    box-shadow: 0 10px 24px rgba(0, 63, 135, 0.08);
}}

/* 基线回答卡片 - 金色边框 */
.baseline-answer-card {{
    border-left: 4px solid {TONGJI_GOLD};
    background: linear-gradient(135deg, #FFFBF0 0%, #FFFFFF 72%);
    border-radius: 8px;
    box-shadow: 0 10px 24px rgba(121, 87, 18, 0.07);
}}

/* 检索依据卡片 */
.citation-card {{
    border-radius: 8px;
    border: 1px solid #E8E8E8;
    margin-bottom: 0.8rem;
    overflow: hidden;
}}

/* 按钮样式 */
.stButton > button {{
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
    border: 1px solid rgba(0, 63, 135, 0.16);
    min-height: 2.6rem;
}}

.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0, 63, 135, 0.25);
}}

/* 主要按钮 */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {TONGJI_BLUE} 0%, {TONGJI_BLUE_DARK} 100%) !important;
    color: white !important;
    font-size: 1.1rem;
    padding: 0.58rem 2rem;
    border-top: 2px solid {TONGJI_GOLD};
}}

/* 示例问题按钮 */
.example-btn {{
    background: white !important;
    border: 2px solid {TONGJI_BLUE} !important;
    color: {TONGJI_BLUE} !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}}

.example-btn:hover {{
    background: {TONGJI_BLUE} !important;
    color: white !important;
}}

/* 标签样式 */
.chunk-tag {{
    background: {TONGJI_BLUE};
    color: white;
    padding: 0.2rem 0.6rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
}}

/* 分数标签 */
.score-tag {{
    background: {TONGJI_GOLD};
    color: white;
    padding: 0.2rem 0.5rem;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 600;
}}

/* Slider 样式 */
[data-testid="stSlider"] > div > div > div {{
    background: {TONGJI_BLUE};
}}

/* 展开器样式 */
.streamlit-expanderHeader {{
    border-radius: 8px;
    background: linear-gradient(90deg, #F8FAFC, white);
}}

.streamlit-expanderHeader:hover {{
    background: linear-gradient(90deg, #EEF2F7, white);
}}

/* 输入框焦点样式 */
[data-testid="stTextInput"] input:focus {{
    border-color: {TONGJI_BLUE} !important;
    box-shadow: 0 0 0 3px rgba(0, 63, 135, 0.1) !important;
}}

.section-title {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin: 1.2rem 0 0.75rem;
    color: {TEXT_DARK};
    font-size: 1.05rem;
    font-weight: 700;
}}

.section-title::before {{
    content: "";
    width: 4px;
    height: 1.15rem;
    border-radius: 4px;
    background: {TONGJI_GOLD};
}}

.sidebar-stat {{
    background: linear-gradient(135deg, {TONGJI_BLUE} 0%, {TONGJI_BLUE_DARK} 100%);
    padding: 1.05rem;
    border-radius: 8px;
    color: white;
    margin-bottom: 1.35rem;
    border-top: 3px solid {TONGJI_GOLD};
    box-shadow: 0 10px 22px rgba(0, 43, 92, 0.16);
}}

.tip-card {{
    background: rgba(255, 255, 255, 0.82);
    padding: 0.82rem;
    border-radius: 8px;
    border-left: 3px solid {TONGJI_GOLD};
    border-top: 1px solid rgba(0, 63, 135, 0.08);
    font-size: 0.82rem;
    color: {TEXT_GRAY};
}}

div[data-testid="stAlert"] {{
    border-radius: 8px;
    border: 1px solid rgba(0, 63, 135, 0.08);
}}

@media (max-width: 760px) {{
    .brand-lockup {{
        align-items: flex-start;
    }}
    .brand-left {{
        align-items: flex-start;
        flex-direction: column;
    }}
    .brand-divider {{
        display: none;
    }}
    .brand-badge {{
        display: none;
    }}
    .feature-strip {{
        grid-template-columns: 1fr;
    }}
}}

/* 隐藏元素 */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header[data-testid="stHeader"] {{
    background: transparent;
    z-index: 999999;
}}

[data-testid="stExpandSidebarButton"] {{
    display: inline-flex !important;
    visibility: visible !important;
    align-items: center;
    justify-content: center;
    width: 2.35rem;
    height: 2.35rem;
    margin-left: 0.65rem;
    margin-top: 0.35rem;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.96) !important;
    border: 1px solid rgba(0, 63, 135, 0.22) !important;
    box-shadow: 0 8px 20px rgba(0, 43, 92, 0.16);
}}

[data-testid="stExpandSidebarButton"] span {{
    color: {TONGJI_BLUE_DARK} !important;
}}

[data-testid="stExpandSidebarButton"]:hover {{
    background: #FFFFFF !important;
    box-shadow: 0 10px 24px rgba(0, 43, 92, 0.22);
}}

[data-testid="stDecoration"] {{
    display: none;
}}
"""


st.set_page_config(
    page_title="RAG 校园智能问答助手",
    page_icon="🏛️",
    layout="wide",
    menu_items=None
)

# 注入自定义CSS
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# ========================================
# 顶部标题栏
# ========================================
logo_data_uri = image_to_data_uri(TONGJI_LOGO_PATH)
logo_markup = (
    f'<img class="tj-logo" src="{logo_data_uri}" alt="同济大学校徽与校名">'
    if logo_data_uri
    else '<div class="tj-logo" style="color: white; font-weight: 700;">TONGJI UNIVERSITY</div>'
)

st.markdown(f"""
<div class="main-header">
    <div class="brand-lockup">
        <div class="brand-left">
            {logo_markup}
            <div class="brand-divider"></div>
            <div>
                <div class="main-kicker">Campus Knowledge Assistant</div>
                <h1 class="main-title">RAG 校园智能问答助手</h1>
                <p class="main-subtitle">基于校园 FAQ 知识库的检索增强问答系统 · 同济大学</p>
            </div>
        </div>
        <div class="brand-badge">TF-IDF + BM25 Hybrid Retrieval</div>
    </div>
</div>
<div class="feature-strip">
    <div class="feature-item">
        <strong>校园服务问答</strong>
        <span>聚焦图书馆、校园卡、宿舍报修等高频事项</span>
    </div>
    <div class="feature-item">
        <strong>检索增强生成</strong>
        <span>先查知识库，再生成更贴近校内语境的回答</span>
    </div>
    <div class="feature-item">
        <strong>依据可追溯</strong>
        <span>回答后展示相关片段与来源，便于核对</span>
    </div>
</div>
""", unsafe_allow_html=True)


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

# ========================================
# 左侧边栏
# ========================================
with st.sidebar:
    st.markdown('<p class="sidebar-header">系统配置</p>', unsafe_allow_html=True)
    
    # 知识库统计卡片
    st.markdown(f"""
    <div class="sidebar-stat">
        <div style="font-size: 0.85rem; opacity: 0.9;">知识库规模</div>
        <div style="font-size: 2rem; font-weight: 700;">
            {len(next(iter(retrievers.values())).chunks)}
        </div>
        <div style="font-size: 0.8rem; opacity: 0.8;">条 FAQ 文档</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 评估指标
    if EVAL_SUMMARY_PATH.exists():
        st.markdown('<p style="color: #666; font-size: 0.9rem; font-weight: 600;">评估指标</p>', unsafe_allow_html=True)
        eval_data = st.json(EVAL_SUMMARY_PATH.read_text(encoding="utf-8"), expanded=False)
    
    # 参数配置
    st.markdown('<p style="color: #666; font-size: 0.9rem; font-weight: 600; margin-top: 1.5rem;">检索参数</p>', unsafe_allow_html=True)
    top_k = st.slider("返回片段数量", min_value=1, max_value=5, value=3, 
                      help="控制每次检索返回的相关文档片段数量")
    strategy = st.selectbox("检索策略", list(retrievers.keys()),
                           help="TF-IDF: 词频分析 | BM25: 关键词匹配 | 混合检索: 两者结合")
    
    # 检索策略说明
    st.markdown(f"""
    <div class="tip-card">
        <b>混合检索</b> 结合 TF-IDF 和 BM25 优势，平衡语义理解与关键词匹配。
    </div>
    """, unsafe_allow_html=True)

examples = [
    "图书馆周末几点开门？",
    "校园卡丢了怎么办？",
    "宿舍空调坏了从哪里报修？",
    "忘记统一身份认证密码怎么办？",
]

# ========================================
# 问题输入区域
# ========================================
st.markdown('<div class="section-title">请输入您的问题</div>', unsafe_allow_html=True)

# 示例快捷问题
col1, col2, col3, col4 = st.columns(4)
example_buttons = [col1, col2, col3, col4]
example_keys = ["example_0", "example_1", "example_2", "example_3"]

for col, example, key in zip(example_buttons, examples, example_keys):
    with col:
        if st.button(f"📌 {example}", key=key, use_container_width=True):
            question = example
            st.session_state.question_input = example

# 问题输入框
if "question_input" not in st.session_state:
    st.session_state.question_input = examples[0]

question = st.text_input(
    label="",
    value=st.session_state.question_input,
    placeholder="试试输入：图书馆周末几点开门？",
    label_visibility="collapsed",
    key="question_input"
)

if st.button("🚀 生成回答", type="primary", use_container_width=True):
    with st.spinner("🔍 正在检索相关文档..."):
        result = answer_question(question, retrievers[strategy], top_k=top_k)
    
    # RAG 回答卡片
    st.markdown(f"""
    <div class="rag-answer-card" style="padding: 1.2rem 1.35rem; margin: 1.4rem 0 0.75rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
            <span style="background: {TONGJI_BLUE}; color: white; padding: 0.3rem 0.8rem; 
                         border-radius: 8px; font-weight: 600; font-size: 0.9rem;">RAG 智能回答</span>
            <span style="color: #999; font-size: 0.85rem;">基于 {top_k} 个相关文档片段</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info(result["answer"])
    
    # 基线回答卡片
    baseline_answer = no_retrieval_baseline(question)
    st.markdown(f"""
    <div class="baseline-answer-card" style="padding: 1.2rem 1.35rem; margin: 1.4rem 0 0.75rem;">
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
            <span style="background: {TONGJI_GOLD}; color: white; padding: 0.3rem 0.8rem; 
                         border-radius: 8px; font-weight: 600; font-size: 0.9rem;">基线回答</span>
            <span style="color: #999; font-size: 0.85rem;">无检索，仅依靠模型自身知识</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.warning(baseline_answer)
    
    # 检索依据区域
    st.markdown(f"""
    <div style="margin: 1.8rem 0 1rem;">
        <div class="section-title">检索依据 <span style="color: #999; font-size: 0.85rem; font-weight: 400;">共 {len(result['retrieved'])} 条相关文档</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    for i, chunk in enumerate(result["retrieved"]):
        with st.expander(
            f"**{i+1}. {chunk['title']}** `{chunk['doc_id']}` `score {chunk['score']:.4f}`",
            expanded=(i == 0)  # 默认展开第一个
        ):
            st.markdown(f"""
            <div style="padding: 1rem; background: #FAFAFA; border-radius: 8px; margin-top: 0.5rem; border-left: 3px solid {TONGJI_BLUE};">
                <p style="color: #333; line-height: 1.8; margin-bottom: 1rem;">{chunk['content']}</p>
                <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                    <span style="color: #666; font-size: 0.85rem;">{chunk['source']}</span>
                    <a href="{chunk['url']}" target="_blank" style="color: {TONGJI_BLUE}; 
                       font-size: 0.85rem;">查看原文</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ========================================
# 底部信息
# ========================================
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #999; font-size: 0.8rem; padding: 1rem;">
    <p>同济大学 · 校园智能问答助手 · 基于 RAG 技术构建</p>
    <p>Powered by TF-IDF + BM25 Hybrid Retrieval & LLM</p>
</div>
""", unsafe_allow_html=True)
