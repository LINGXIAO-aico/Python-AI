from __future__ import annotations

import os
from typing import Sequence

from .retriever import RetrievedChunk, TfidfRetriever


def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
    lines = []
    for chunk in chunks:
        lines.append(
            f"[{chunk.doc_id}] {chunk.title}（{chunk.source}，相似度 {chunk.score}）：{chunk.content}"
        )
    return "\n".join(lines)


def call_llm(prompt: str, system_prompt: str | None = None) -> str:
    """Call an OpenAI-compatible chat API.

    For Tongyi Qianwen, set DASHSCOPE_API_KEY. The default base URL is
    DashScope's OpenAI-compatible endpoint. OPENAI_API_KEY/OPENAI_BASE_URL
    also work for other compatible providers.
    """
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No API key configured. Set DASHSCOPE_API_KEY or OPENAI_API_KEY.")

    base_url = os.getenv("OPENAI_BASE_URL")
    if not base_url and os.getenv("DASHSCOPE_API_KEY"):
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = os.getenv("OPENAI_MODEL") or os.getenv("QWEN_MODEL") or "qwen-turbo"

    from openai import OpenAI

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def extractive_answer(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return "知识库中暂未检索到足够相关的信息，建议补充更明确的问题或咨询对应部门。"

    best = chunks[0]
    answer_lines = [
        f"根据知识库中“{best.title}”的说明：{best.content}",
    ]
    if len(chunks) > 1:
        related = "；".join(f"{chunk.title}[{chunk.doc_id}]" for chunk in chunks[1:])
        answer_lines.append(f"同时可参考：{related}。")
    answer_lines.append(f"来源：{best.source}，引用编号：[{best.doc_id}]。")
    return "\n".join(answer_lines)


def llm_answer(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    context = _format_context(chunks)
    prompt = f"""请根据以下参考资料回答校园问题。
如果参考资料中没有相关信息，请明确说“暂无相关资料”，不要编造。
回答应简洁、可执行，并在关键结论后标注引用编号。

参考资料：
{context}

问题：{question}
回答："""
    return call_llm(
        prompt,
        system_prompt="你是校园智能问答助手，只能依据给定知识库上下文回答。",
    )


def answer_question(
    question: str,
    retriever: TfidfRetriever,
    top_k: int = 3,
    backend: str = "extractive",
) -> dict:
    chunks = retriever.retrieve(question, top_k=top_k)
    if backend in {"openai", "qwen", "llm"}:
        answer = llm_answer(question, chunks)
    else:
        answer = extractive_answer(question, chunks)
    return {
        "question": question,
        "answer": answer,
        "citations": [chunk.doc_id for chunk in chunks],
        "retrieved": [chunk.__dict__ for chunk in chunks],
    }


def no_retrieval_baseline(question: str, backend: str = "extractive") -> str:
    if backend in {"openai", "qwen", "llm"}:
        prompt = f"你是校园智能助手。请在不使用检索资料的情况下回答：{question}"
        return call_llm(prompt)
    return (
        "这是未接入知识库的通用回答：建议先查看学校官网、学院通知或咨询相关部门。"
        "由于没有检索校园手册，无法给出具体办理时间、材料和流程。"
    )
