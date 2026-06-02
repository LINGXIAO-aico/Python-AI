"""Campus RAG assistant package — 七层 RAG 管线。"""

from .embeddings import BGEEmbedder
from .generator import answer_question, llm_answer, llm_answer_stream, self_rag_verify
from .memory import ConversationMemory
from .query_rewriter import hyde_rewrite, multi_query_expand, rewrite_query
from .reranker import BGEReranker
from .retriever import (
    Bm25JiebaRetriever,
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    HybridRRFRetriever,
    RetrievedChunk,
    TfidfRetriever,
)
from .vectorstore import FAISSStore

__all__ = [
    # Retrievers
    "TfidfRetriever",
    "BM25Retriever",
    "Bm25JiebaRetriever",
    "DenseRetriever",
    "HybridRetriever",
    "HybridRRFRetriever",
    "RetrievedChunk",
    # Embeddings & Vector
    "BGEEmbedder",
    "FAISSStore",
    # Reranker
    "BGEReranker",
    # Query rewriting
    "hyde_rewrite",
    "multi_query_expand",
    "rewrite_query",
    # Generation
    "answer_question",
    "llm_answer",
    "llm_answer_stream",
    "self_rag_verify",
    # Memory
    "ConversationMemory",
]
