#!/usr/bin/env python3
"""爬取同济大学各二级网站公开页面，输出 crawled_pages.jsonl。

V2: 改进版 — 从列表页自动发现文章链接 + 递归抓取文章详情。
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "crawled_pages.jsonl"

REQUEST_TIMEOUT = 20.0
DELAY = 0.8
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 "
    "TongjiCampusRAG/1.0 (Academic project)"
)

# ============================================================
# 目标站点配置
# ============================================================

# 每个 site: { name, base, list_urls, article_link_pattern }
# article_link_pattern 用于从列表页筛选真正的文章链接
SITES = [
    # --- 同济新闻（文章最多）---
    {
        "section": "同济新闻",
        "base": "https://news.tongji.edu.cn/",
        "list_urls": [
            "https://news.tongji.edu.cn/tjyw1.htm",   # 同济要闻
            "https://news.tongji.edu.cn/tjkx1.htm",   # 同济快讯
            "https://news.tongji.edu.cn/mtjj1.htm",   # 媒体聚焦
            "https://news.tongji.edu.cn/rwsy1.htm",   # 人物声音
            "https://news.tongji.edu.cn/xngg.htm",    # 校内公告
            "https://news.tongji.edu.cn/jzxx1.htm",   # 讲座信息
            "https://news.tongji.edu.cn/gjsd1.htm",   # 国际视点
        ],
        "article_pattern": r"/info/\d+/\d+\.htm",
        "max_per_list": 15,
    },
    # --- 教务处（含选课/考试/学籍等）---
    {
        "section": "教务处",
        "base": "https://jwc.tongji.edu.cn/",
        "list_urls": [
            "https://jwc.tongji.edu.cn/30352/list.htm",   # 学生
            "https://jwc.tongji.edu.cn/30353/list.htm",   # 教学
            "https://jwc.tongji.edu.cn/30354/list.htm",   # 教改
            "https://jwc.tongji.edu.cn/30355/list.htm",   # 质量
            "https://jwc.tongji.edu.cn/30357/list.htm",   # 综合
            "https://jwc.tongji.edu.cn/gzdt_33828/list.htm",  # 工作动态
            "https://jwc.tongji.edu.cn/gzdt_33827/list.htm",  # 工作动态
            "https://jwc.tongji.edu.cn/kszx/list.htm",    # 考试中心
            "https://jwc.tongji.edu.cn/kczx/list.htm",    # 课程中心
        ],
        "article_pattern": r"/c\d+/[a-z0-9]+/page\.htm",
        "max_per_list": 15,
    },
    # --- 信息公开 ---
    {
        "section": "信息公开",
        "base": "https://xxgk.tongji.edu.cn/",
        "list_urls": [
            "https://xxgk.tongji.edu.cn/index.php?classid=3060",
            "https://xxgk.tongji.edu.cn/index.php?classid=4528",
            "https://xxgk.tongji.edu.cn/index.php?classid=3061",
            "https://xxgk.tongji.edu.cn/index.php?classid=3062",
            "https://xxgk.tongji.edu.cn/index.php?classid=3063",
            "https://xxgk.tongji.edu.cn/index.php?classid=3065",
            "https://xxgk.tongji.edu.cn/index.php?classid=11600",
        ],
        "article_pattern": r"(show|info|article)\.php\?",
        "max_per_list": 10,
    },
    # --- 本科招生 ---
    {
        "section": "本科招生",
        "base": "https://bkzs.tongji.edu.cn/",
        "list_urls": [
            "https://bkzs.tongji.edu.cn/welcome/generalRules",
            "https://bkzs.tongji.edu.cn/welcome/college",
            "https://bkzs.tongji.edu.cn/major/index",
            "https://bkzs.tongji.edu.cn/plan/index",
            "https://bkzs.tongji.edu.cn/luqu/admission",
            "https://bkzs.tongji.edu.cn/luqu/probability",
            "https://bkzs.tongji.edu.cn/zixun/inquiry",
            "https://bkzs.tongji.edu.cn/zixun/counselling",
            "https://bkzs.tongji.edu.cn/zixun/faq",
        ],
        "article_pattern": r"\.htm",
        "max_per_list": 8,
    },
    # --- 研究生院 ---
    {
        "section": "研究生院",
        "base": "https://gs.tongji.edu.cn/",
        "list_urls": [
            "https://gs.tongji.edu.cn/tzgg.htm",
            "https://gs.tongji.edu.cn/zsgz.htm",
            "https://gs.tongji.edu.cn/zsgz/yjszs.htm",
            "https://gs.tongji.edu.cn/jxpy.htm",
            "https://gs.tongji.edu.cn/jxpy/glgd.htm",
            "https://gs.tongji.edu.cn/jxpy/pygc.htm",
            "https://gs.tongji.edu.cn/jxpy/cxsj.htm",
            "https://gs.tongji.edu.cn/jxpy/jxgl.htm",
            "https://gs.tongji.edu.cn/xjgl.htm",
            "https://gs.tongji.edu.cn/xjgl/glgd.htm",
            "https://gs.tongji.edu.cn/xjgl/bdzc.htm",
            "https://gs.tongji.edu.cn/xjgl/xjbd.htm",
            "https://gs.tongji.edu.cn/xjgl/rcgl.htm",
            "https://gs.tongji.edu.cn/xwgz.htm",
            "https://gs.tongji.edu.cn/xwgz/xwsqd.htm",
            "https://gs.tongji.edu.cn/xwgz/xwzc.htm",
        ],
        "article_pattern": r"\.htm",
        "max_per_list": 8,
    },
    # --- 体育部 ---
    {
        "section": "体育部",
        "base": "https://sports.tongji.edu.cn/",
        "list_urls": [
            "https://sports.tongji.edu.cn/bmgk.htm",      # 部门概况
            "https://sports.tongji.edu.cn/jwjx.htm",      # 教学
            "https://sports.tongji.edu.cn/yjspy.htm",     # 研究生培养
            "https://sports.tongji.edu.cn/kxyj.htm",      # 科研
            "https://sports.tongji.edu.cn/bmgk/bmjj.htm", # 部门简介
            "https://sports.tongji.edu.cn/bmgk/szdw.htm", # 师资队伍
            "https://sports.tongji.edu.cn/jwjx/jxcg.htm", # 教学成果
            "https://sports.tongji.edu.cn/jwjx/tyktksq1.htm", # 体育课
        ],
        "article_pattern": r"\.htm",
        "max_per_list": 10,
    },
    # --- 同济主页 ---
    {
        "section": "学校概况",
        "base": "https://www.tongji.edu.cn/",
        "list_urls": [
            "https://www.tongji.edu.cn/",
        ],
        "article_pattern": r"\.htm",
        "max_per_list": 5,
    },
    # --- 学生工作部 ---
    {
        "section": "学生事务",
        "base": "https://student.tongji.edu.cn/",
        "list_urls": [
            "https://student.tongji.edu.cn/",
        ],
        "article_pattern": r"\.htm",
        "max_per_list": 5,
    },
]


def clean_html(html: str) -> str:
    """去导航/页脚/脚本，提取正文。"""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["nav", "footer", "header", "script", "style", "noscript",
                      "table", "form", "select", "button", "input"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{3,}", "  ", text)
    return text.strip()


def get_article_links(client: httpx.Client, site: dict) -> list[str]:
    """从列表页提取文章链接。"""
    all_links: list[str] = []
    base = site["base"]
    pattern = re.compile(site["article_pattern"])

    for list_url in site["list_urls"]:
        try:
            resp = client.get(list_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code >= 400:
                print(f"    [{resp.status_code}] {list_url}")
                continue
        except Exception as exc:
            print(f"    [FAIL] {list_url}: {exc}")
            continue

        soup = BeautifulSoup(resp.text, "lxml")
        count = 0
        for a in soup.select("a[href]"):
            href = a.get("href", "").strip()
            if not href:
                continue
            full = urljoin(base, href)
            # 只保留匹配文章模式的链接
            if pattern.search(full):
                text = a.get_text(strip=True)
                # 至少有中文标题
                if len(text) >= 6 and re.search(r"[一-鿿]", text):
                    all_links.append(full)
                    count += 1
                    if count >= site["max_per_list"]:
                        break
        print(f"    {list_url} -> {count} 篇")

    # 去重保序
    seen = set()
    unique = []
    for link in all_links:
        if link not in seen:
            seen.add(link)
            unique.append(link)
    return unique


def crawl_article(client: httpx.Client, url: str, section: str) -> dict | None:
    """爬取单篇文章。"""
    try:
        resp = client.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code >= 400:
            return None
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # 取标题
    title = ""
    for tag in soup.select("h1, h2, h3, title, .article-title, .content-title"):
        t = tag.get_text(strip=True)
        if len(t) > len(title):
            title = t
    if not title or len(title) < 4:
        return None
    title = title.replace("\xa0", " ").replace("　", " ").strip()

    content = clean_html(resp.text)
    if len(content) < 100:
        return None

    # 取发布日期（如果有）
    pub_date = ""
    date_patterns = [
        r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})",
    ]
    for dp in date_patterns:
        m = re.search(dp, content[:500])
        if m:
            pub_date = m.group(1)
            break

    return {
        "url": url,
        "title": title,
        "content": content[:3000] if len(content) > 3000 else content,
        "pub_date": pub_date,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "source_section": section,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = httpx.Client(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        verify=False,
        timeout=REQUEST_TIMEOUT,
    )

    all_pages: list[dict] = []
    total_articles = 0

    print("=" * 60)
    print("同济大学校园信息爬虫 V2")
    print(f"输出: {OUTPUT_PATH}")
    print("=" * 60)

    for site in SITES:
        section = site["section"]
        print(f"\n[{section}] 发现文章链接...")

        article_links = get_article_links(client, site)
        print(f"  共 {len(article_links)} 篇待抓取")

        for i, link in enumerate(article_links):
            print(f"  [{i+1}/{len(article_links)}] ", end="")
            page = crawl_article(client, link, section)
            if page:
                all_pages.append(page)
                total_articles += 1
                t = page['title'][:40].encode('gbk', errors='replace').decode('gbk')
                print(f"OK [{len(page['content'])}字] {t}")
            else:
                print("SKIP")
            time.sleep(DELAY)

    # 保存
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for page in all_pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    # 统计
    cat_counts = {}
    for p in all_pages:
        c = p["source_section"]
        cat_counts[c] = cat_counts.get(c, 0) + 1

    print(f"\n{'=' * 60}")
    print(f"爬取完成！共 {total_articles} 篇文章")
    for cat, n in sorted(cat_counts.items()):
        print(f"  {cat}: {n}")
    print(f"输出: {OUTPUT_PATH}")

    client.close()


if __name__ == "__main__":
    main()
