#!/usr/bin/env python3
"""半自动生成评测集：基于知识库文档，用 DeepSeek 生成问题，人工抽检后使用。

输出 data/raw/eval_150.jsonl，覆盖 4 类题型：
- fact: 单跳事实题
- multi-hop: 多跳推理
- unanswerable: 知识库无答案陷阱题
- paraphrase: 改写等价问题
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from campus_rag.data import read_jsonl
from campus_rag.config import CAMPUS_KB_PATH, RAW_EVAL_PATH, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "eval_150.jsonl"
EVAL_TEMPLATE = """你是一个评测集生成器。根据以下校园知识库片段，生成评测问题。

知识库类别：{category}
标题：{title}
内容：{content}

请生成以下 3 类各 1 个问题（输出 JSON 数组）：

1. fact: 一个事实型问题，答案可以直接从这段内容中找到（如"开放时间是什么"）
2. paraphrase: 用不同措辞改写 fact 问题，问的是同一个答案
3. unanswerable: 一个与这段内容无关、但听起来像校园问题的"陷阱题"，知识库无法回答

每个问题格式：{{"question": "...", "gold_doc_id": "{doc_id}", "answer_keywords": ["关键词1", "关键词2"], "question_type": "fact|paraphrase|unanswerable"}}

注意：unanswerable 题的 gold_doc_id 设为 "NONE"，answer_keywords 为空数组。

只输出 JSON 数组，不要其他内容："""


def generate_for_doc(doc: dict) -> list[dict]:
    """对单篇文档生成 3 个评测问题。"""
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    prompt = EVAL_TEMPLATE.format(
        category=doc.get("category", ""),
        title=doc.get("title", ""),
        content=doc.get("content", ""),
        doc_id=doc.get("doc_id", ""),
    )

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content or "[]"
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"  [生成失败] {doc.get('doc_id')}: {exc}")
        return []


def main() -> None:
    if not DEEPSEEK_API_KEY:
        print("[错误] 未配置 DEEPSEEK_API_KEY。请在 deepseek.env 中设置。")
        return

    # 确定数据源 — 优先用合并KB
    from campus_rag.config import RAW_KB_PATH
    kb_path = RAW_KB_PATH
    if not kb_path.exists():
        kb_path = CAMPUS_KB_PATH if CAMPUS_KB_PATH.exists() else PROJECT_ROOT / "data" / "raw" / "campus_faq.jsonl"
    if not kb_path.exists():
        print(f"[错误] 知识库文件不存在: {kb_path}")
        return

    docs = read_jsonl(kb_path)
    print(f"知识库共有 {len(docs)} 篇文档")

    # 保留旧评测集中的人工标注题（如果存在）
    old_questions = []
    if RAW_EVAL_PATH.exists():
        old_questions = read_jsonl(RAW_EVAL_PATH)
        print(f"保留已有 {len(old_questions)} 道评测题")

    # 随机采样 50 篇文档，每篇生成问题
    sample_size = min(50, len(docs))
    sampled = random.sample(docs, sample_size)

    new_questions: list[dict] = []
    print(f"\n对 {sample_size} 篇文档生成问题...")
    for i, doc in enumerate(sampled):
        print(f"  [{i + 1}/{sample_size}] {doc.get('doc_id', '?')}: {doc.get('title', '')[:40]}")
        generated = generate_for_doc(doc)
        new_questions.extend(generated)
        print(f"    已生成 {len(generated)} 题")

    # 合并
    all_questions = old_questions + new_questions

    # 去重（按 question 文本）
    seen = set()
    unique: list[dict] = []
    for q in all_questions:
        if q["question"] not in seen:
            seen.add(q["question"])
            unique.append(q)

    random.shuffle(unique)

    # 写盘
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for q in unique:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # 统计
    type_counts = {}
    for q in unique:
        t = q.get("question_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"生成完成！共 {len(unique)} 道评测题")
    print(f"题型分布:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")
    print(f"输出: {OUTPUT_PATH}")
    print(f"\n提示: 请人工抽检约 20% 的题目，确认 gold_doc_id 和 keywords 正确。")
    print(f"将 unanswerable 设为少数（<15%），保证大部分题有标准答案。")


if __name__ == "__main__":
    main()
