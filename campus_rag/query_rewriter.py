from __future__ import annotations

from typing import Any, cast

from openai import OpenAI

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_MODEL


def _get_client() -> OpenAI:
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _call_deepseek(prompt: str, system_prompt: str = "", max_tokens: int = 512) -> str:
    client = _get_client()
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(
        model=DEEPSEEK_CHAT_MODEL,
        messages=cast(Any, messages),
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def hyde_rewrite(question: str) -> str:
    """HyDE：用 DeepSeek 生成假设答案，再用假设答案嵌入检索。"""
    system = "你是一位熟悉高校事务的助手。请根据问题，生成一段假设性的校园办事指南文档（100-200字），不要直接回答问题。"
    prompt = f"问题：{question}\n请生成一段假设性的校园办事指南文档："
    return _call_deepseek(prompt, system_prompt=system, max_tokens=300)


def multi_query_expand(question: str, n: int = 3) -> list[str]:
    """Multi-Query：生成 N 个不同视角的改写问题。"""
    system = "你是一个搜索查询改写器。请将用户问题从不同角度改写，便于检索到更全面的资料。只输出改写后的问题，每行一个，不要编号。"
    prompt = f"原问题：{question}\n请从 {n} 个不同角度改写这个问题："
    result = _call_deepseek(prompt, system_prompt=system, max_tokens=300)
    queries = [line.strip() for line in result.split("\n") if line.strip()]
    return queries[:n] if queries else [question]


def rewrite_query(question: str, method: str = "hyde") -> str | list[str]:
    """统一入口：method = 'hyde' | 'multi_query' | 'none'"""
    if method == "hyde":
        return hyde_rewrite(question)
    elif method == "multi_query":
        return multi_query_expand(question)
    elif method == "both":
        hyde_text = hyde_rewrite(question)
        mq_list = multi_query_expand(question)
        return [hyde_text] + mq_list
    else:
        return question
