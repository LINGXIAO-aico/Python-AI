from __future__ import annotations

from typing import Generator, Sequence

from openai import OpenAI

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    DEEPSEEK_REASONER_MODEL,
)
from .memory import ConversationMemory
from .retriever import RetrievedChunk


SYSTEM_PROMPT = """你是同济大学校园智能问答助手"同小智"，专门解答校园办事、教务、生活等问题。

严格遵循以下规则：
1. 只能根据【参考资料】中的内容回答，不要使用自己的知识
2. 如果参考资料中没有相关信息，明确说"抱歉，当前知识库中暂无相关资料，建议咨询对应部门"
3. 回答应简洁、清晰、分步骤，便于同学照着操作
4. 每条关键信息后标注引用编号 [doc_id]
5. 如果参考资料中包含具体的时间、地点、网址、电话，务必保留"""

VERIFY_PROMPT = """请严格审查以下AI回答是否完全被【参考资料】支持。

对于回答中的每条主张：
- 如果主张在参考资料中有明确依据 → 标记 "supported"
- 如果主张与参考资料矛盾 → 标记 "contradicted"
- 如果主张在参考资料中找不到依据 → 标记 "unsupported"

最后给出总体判断：
- "fully_supported"：所有主张都有据可循
- "partially_supported"：部分主张缺乏依据
- "unsupported"：回答基本没有依据

输出格式（JSON）：
{
  "claims": [{"text": "主张内容", "status": "supported|contradicted|unsupported"}],
  "verdict": "fully_supported|partially_supported|unsupported",
  "explanation": "一句话解释"
}

参考资料：
{context}

AI回答：
{answer}

请审查并输出JSON："""


def _get_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
    lines = []
    for chunk in chunks:
        lines.append(
            f"[{chunk.doc_id}] {chunk.title}（来源：{chunk.source}）\n{chunk.content}"
        )
    return "\n\n".join(lines)


def _format_context_compact(chunks: Sequence[RetrievedChunk]) -> str:
    return "\n".join(
        f"[{chunk.doc_id}] {chunk.content}" for chunk in chunks
    )


# ============================================================
# 流式 LLM 回答
# ============================================================

def llm_answer_stream(
    question: str,
    chunks: Sequence[RetrievedChunk],
    memory: ConversationMemory | None = None,
) -> Generator[str, None, None]:
    """DeepSeek-chat 流式生成，逐 token yield。"""
    client = _get_client()
    context = _format_context(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if memory:
        for msg in memory.get_context():
            messages.append(msg)

    messages.append({
        "role": "user",
        "content": f"参考资料：\n{context}\n\n问题：{question}\n请根据参考资料回答：",
    })

    stream = client.chat.completions.create(
        model=DEEPSEEK_CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def llm_answer(
    question: str,
    chunks: Sequence[RetrievedChunk],
    memory: ConversationMemory | None = None,
) -> str:
    """DeepSeek-chat 非流式生成（评测用）。"""
    parts = list(llm_answer_stream(question, chunks, memory))
    return "".join(parts)


# ============================================================
# Self-RAG 校验
# ============================================================

def self_rag_verify(
    question: str,
    answer: str,
    chunks: Sequence[RetrievedChunk],
) -> dict:
    """用 DeepSeek-reasoner 校验答案是否有据可循。"""
    import json

    client = _get_client()
    context = _format_context(chunks)

    prompt = VERIFY_PROMPT.format(context=context, answer=answer)

    resp = client.chat.completions.create(
        model=DEEPSEEK_REASONER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    raw = resp.choices[0].message.content or "{}"

    # 提取 JSON（reasoner 可能输出推理过程）
    try:
        # 尝试找 JSON 块
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {
            "verdict": "parse_error",
            "explanation": "校验结果解析失败",
            "raw_output": raw,
        }


# ============================================================
# 无检索基线
# ============================================================

def no_retrieval_baseline(question: str, backend: str = "extractive") -> str:
    if backend in {"openai", "qwen", "llm", "deepseek"}:
        client = _get_client()
        resp = client.chat.completions.create(
            model=DEEPSEEK_CHAT_MODEL,
            messages=[
                {"role": "system", "content": "你是校园智能助手。请在不使用检索资料的情况下回答问题。"},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return resp.choices[0].message.content or ""
    return (
        "这是未接入知识库的通用回答：建议先查看学校官网、学院通知或咨询相关部门。"
        "由于没有检索校园手册，无法给出具体办理时间、材料和流程。"
    )


# ============================================================
# 抽取式回答（离线模式保留）
# ============================================================

def extractive_answer(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return "知识库中暂未检索到足够相关的信息，建议补充更明确的问题或咨询对应部门。"
    best = chunks[0]
    lines = [f"根据知识库中「{best.title}」的说明：{best.content}"]
    if len(chunks) > 1:
        related = "；".join(f"{c.title}[{c.doc_id}]" for c in chunks[1:])
        lines.append(f"同时可参考：{related}。")
    lines.append(f"来源：{best.source}，引用编号：[{best.doc_id}]。")
    return "\n".join(lines)


# ============================================================
# 统一入口
# ============================================================

def answer_question(
    question: str,
    retriever,
    top_k: int = 5,
    backend: str = "deepseek",
    memory: ConversationMemory | None = None,
    reranker=None,
    verify: bool = False,
) -> dict:
    chunks = retriever.retrieve(question, top_k=top_k)

    if reranker and chunks:
        chunks = reranker.rerank(question, chunks, top_k=min(5, len(chunks)))

    if backend in {"openai", "qwen", "llm", "deepseek"}:
        answer = llm_answer(question, chunks, memory)
    else:
        answer = extractive_answer(question, chunks)

    result = {
        "question": question,
        "answer": answer,
        "citations": [chunk.doc_id for chunk in chunks],
        "retrieved": [chunk.to_dict() for chunk in chunks],
    }

    if verify:
        verification = self_rag_verify(question, answer, chunks)
        result["verification"] = verification

    if memory:
        memory.add(question, answer)

    return result
