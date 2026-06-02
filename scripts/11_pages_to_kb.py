#!/usr/bin/env python3
"""爬取页面清洗 → 标准化知识库格式 campus_kb.jsonl。"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CRAWLED_PATH = PROJECT_ROOT / "data" / "raw" / "crawled_pages.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "campus_kb.jsonl"

# 类目映射：根据 source_section 关键词归并
CATEGORY_MAP = {
    "学校概况": "学校概况",
    "本科招生": "本科招生",
    "教务处": "教务管理",
    "研究生院": "研究生培养",
    "图书馆": "图书资源",
    "学生事务": "学生事务",
    "体育部": "校园生活",
    "同济新闻": "同济新闻",
    "信息公开": "学校概况",
    "一卡通": "校园生活",
    "信息化": "校园生活",
    "后勤": "校园生活",
}


def normalize_category(section: str) -> str:
    for key, cat in CATEGORY_MAP.items():
        if key in section:
            return cat
    return "其他"


def clean_text(text: str) -> str:
    text = text.replace("　", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    if not CRAWLED_PATH.exists():
        print(f"[错误] 未找到爬取结果: {CRAWLED_PATH}")
        print("请先运行 python scripts/10_crawl_tongji.py")
        return

    pages = []
    with CRAWLED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pages.append(json.loads(line))

    print(f"读取到 {len(pages)} 篇爬取页面")

    records = []
    today = date.today().isoformat()

    for i, page in enumerate(pages):
        content = clean_text(page["content"])
        title = clean_text(page["title"])
        url = page["url"]
        section = page.get("source_section", "")

        # 过滤噪声：太短、疑似导航页
        if len(content) < 80:
            print(f"  [跳过-内容太短] {title[:40]} ({len(content)}字)")
            continue

        # 过滤纯英文/纯数字/导航页
        chinese_ratio = len(re.findall(r"[一-鿿]", content)) / max(len(content), 1)
        if chinese_ratio < 0.2:
            print(f"  [跳过-中文占比低] {title[:40]} ({chinese_ratio:.1%})")
            continue

        from hashlib import md5
        doc_id = f"TJ{i + 1:04d}"
        category = normalize_category(section)

        record = {
            "doc_id": doc_id,
            "category": category,
            "title": title[:80],
            "content": content[:2000] if len(content) > 2000 else content,
            "source": f"同济大学{section}",
            "url": url,
            "last_updated": today,
        }
        records.append(record)

    if not records:
        print("无有效记录。请检查爬取结果质量。")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 统计
    cats = {}
    for rec in records:
        c = rec["category"]
        cats[c] = cats.get(c, 0) + 1

    print(f"\n清洗完成！有效记录: {len(records)}")
    print(f"类目分布:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print(f"输出文件: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
