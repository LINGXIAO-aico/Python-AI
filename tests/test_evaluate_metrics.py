"""评测指标（纯数学部分）测试。"""

from campus_rag.evaluate import (
    keyword_recall,
    reciprocal_rank,
    ndcg_at_k,
)


class TestKeywordRecall:
    def test_all_matched(self):
        assert keyword_recall("校园卡丢失补办流程", ["校园卡", "补办"]) == 1.0

    def test_partial_match(self):
        assert keyword_recall("校园卡丢失", ["校园卡", "补办"]) == 0.5

    def test_no_match(self):
        assert keyword_recall("图书馆开放时间", ["校园卡", "补办"]) == 0.0

    def test_empty_keywords(self):
        assert keyword_recall("任何回答", []) == 0.0


class TestReciprocalRank:
    def test_rank_one(self):
        assert reciprocal_rank("A", ["A", "B", "C"]) == 1.0

    def test_rank_two(self):
        assert reciprocal_rank("A", ["B", "A", "C"]) == 0.5

    def test_not_found(self):
        assert reciprocal_rank("A", ["B", "C", "D"]) == 0.0

    def test_empty_list(self):
        assert reciprocal_rank("A", []) == 0.0


class TestNdcgAtK:
    def test_gold_at_rank_one(self):
        score = ndcg_at_k("A", ["A", "B", "C", "D", "E"], k=5)
        assert score > 0.9  # 接近 1.0

    def test_gold_not_found(self):
        assert ndcg_at_k("A", ["B", "C", "D", "E", "F"], k=5) == 0.0

    def test_gold_at_rank_three(self):
        score = ndcg_at_k("A", ["B", "C", "A", "D", "E"], k=5)
        assert 0 < score < 0.7
