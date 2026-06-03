"""多轮对话记忆模块测试。"""

from campus_rag.memory import ConversationMemory


class TestConversationMemory:
    def test_initial_empty(self):
        m = ConversationMemory(max_turns=3)
        assert len(m) == 0
        assert m.turn_count == 0
        assert m.get_context() == []

    def test_add_one_turn(self):
        m = ConversationMemory(max_turns=3)
        m.add("你好", "你好，有什么可以帮助你的？")
        assert len(m) == 2
        assert m.turn_count == 1
        ctx = m.get_context()
        assert ctx[0]["role"] == "user"
        assert ctx[1]["role"] == "assistant"

    def test_max_turns_truncation(self):
        m = ConversationMemory(max_turns=2)
        for i in range(5):
            m.add(f"问题{i}", f"回答{i}")
        # 5 turns = 10 messages, max_turns=2 → keep last 2 turns = 4 messages
        assert len(m) == 4
        assert m.turn_count == 2
        # 最旧的应该被截断
        ctx = m.get_context()
        assert ctx[0]["content"] == "问题3"

    def test_format_for_prompt(self):
        m = ConversationMemory(max_turns=3)
        m.add("你好", "你好！")
        formatted = m.format_for_prompt()
        assert "用户" in formatted
        assert "助手" in formatted
        assert "你好" in formatted

    def test_clear(self):
        m = ConversationMemory(max_turns=3)
        m.add("问题", "回答")
        m.clear()
        assert len(m) == 0
        assert m.turn_count == 0
