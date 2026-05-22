"""
memory/agent_memory.py
JSON-backed conversation memory with sliding window.
"""

import json, os, datetime
from typing import Any

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "agent_memory.json")


class AgentMemory:
    """
    Persistent conversation memory stored as a JSON file.

    Schema:
    {
        "conversations": [
            {
                "id": "conv_<timestamp>",
                "created_at": "ISO-8601",
                "turns": [
                    {"role": "user"|"assistant", "content": "...", "ts": "ISO-8601"},
                    ...
                ]
            },
            ...
        ],
        "current_conversation_id": "conv_<timestamp>"
    }
    """

    def __init__(self, store_path: str = DEFAULT_PATH, max_history: int = 20):
        self.store_path = store_path
        self.max_history = max_history
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        self._data = self._load()
        if not self._data.get("current_conversation_id"):
            self.new_conversation()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.store_path):
            with open(self.store_path) as f:
                return json.load(f)
        return {"conversations": [], "current_conversation_id": None}

    def _save(self):
        with open(self.store_path, "w") as f:
            json.dump(self._data, f, indent=2)

    # ── Conversation management ───────────────────────────────────────────────

    def new_conversation(self) -> str:
        conv_id = f"conv_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        self._data["conversations"].append(
            {"id": conv_id, "created_at": _now(), "turns": []}
        )
        self._data["current_conversation_id"] = conv_id
        self._save()
        return conv_id

    def _current(self) -> dict:
        cid = self._data["current_conversation_id"]
        for conv in self._data["conversations"]:
            if conv["id"] == cid:
                return conv
        # Fallback – create fresh
        self.new_conversation()
        return self._current()

    # ── Turn operations ───────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str):
        conv = self._current()
        conv["turns"].append({"role": role, "content": content, "ts": _now()})
        # Trim to sliding window
        if len(conv["turns"]) > self.max_history * 2:
            conv["turns"] = conv["turns"][-(self.max_history * 2):]
        self._save()

    def get_history(self, n: int | None = None) -> list[dict]:
        """Return last n turn pairs (user+assistant) from current conversation."""
        turns = self._current()["turns"]
        if n:
            turns = turns[-(n * 2):]
        return turns

    def format_for_prompt(self, n: int = 5) -> str:
        """Return a compact text block suitable for injection into a prompt."""
        turns = self.get_history(n)
        if not turns:
            return ""
        lines = []
        for t in turns:
            prefix = "User" if t["role"] == "user" else "Assistant"
            lines.append(f"{prefix}: {t['content']}")
        return "\n".join(lines)

    # ── Inspection ────────────────────────────────────────────────────────────

    def all_conversations(self) -> list[dict]:
        return self._data["conversations"]

    def current_id(self) -> str:
        return self._data["current_conversation_id"]

    def clear_current(self):
        self._current()["turns"].clear()
        self._save()


def _now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"
