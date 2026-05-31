from __future__ import annotations

from .config import MAX_TURNS


class ConversationMemory:
    """多轮对话窗口管理，保留最近 N 轮。"""

    def __init__(self, max_turns: int = MAX_TURNS):
        self.max_turns = max_turns
        self._history: list[dict[str, str]] = []

    def add(self, user: str, assistant: str) -> None:
        self._history.append({"role": "user", "content": user})
        self._history.append({"role": "assistant", "content": assistant})
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-self.max_turns * 2:]

    def get_context(self) -> list[dict[str, str]]:
        return list(self._history)

    def format_for_prompt(self) -> str:
        if not self._history:
            return ""
        lines = []
        for msg in self._history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}：{msg['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._history.clear()

    @property
    def turn_count(self) -> int:
        return len(self._history) // 2

    def __len__(self) -> int:
        return len(self._history)
