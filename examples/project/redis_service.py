from __future__ import annotations

import json
import os
from typing import Any

try:
    import redis as _redis

    _client: _redis.Redis | None = None

    def _get_client() -> _redis.Redis:
        global _client
        if _client is None:
            _client = _redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
            )
        return _client

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


_HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "10"))
_CONVERSATION_TTL = int(os.getenv("CONVERSATION_TTL", "86400"))
_MAX_MESSAGES = int(os.getenv("MAX_MESSAGES_PER_CONVERSATION", "200"))


def load_history(conversation_id: str) -> list[dict]:
    """Return the last N turns (user+assistant pairs) for a conversation."""
    if not REDIS_AVAILABLE:
        return []
    try:
        key = f"medical:{conversation_id}"
        raw = _get_client().lrange(key, 0, -1)
        messages = [json.loads(m) for m in raw]
        if _HISTORY_LIMIT > 0:
            messages = messages[-(_HISTORY_LIMIT * 2):]
        return [
            m
            for m in messages
            if m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
            and m.get("content", "").strip()
        ]
    except Exception as exc:
        print(f"[Redis] load_history error: {exc}")
        return []


def save_message(conversation_id: str, role: str, content: Any) -> None:
    """Append a message to the conversation history in Redis."""
    if not REDIS_AVAILABLE:
        return
    try:
        client = _get_client()
        key = f"medical:{conversation_id}"
        client.rpush(key, json.dumps({"role": role, "content": content}, ensure_ascii=False))
        client.ltrim(key, -_MAX_MESSAGES, -1)
        client.expire(key, _CONVERSATION_TTL)
    except Exception as exc:
        print(f"[Redis] save_message error: {exc}")


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation from Redis. Returns True if it existed."""
    if not REDIS_AVAILABLE:
        return False
    try:
        return _get_client().delete(f"medical:{conversation_id}") > 0
    except Exception as exc:
        print(f"[Redis] delete_conversation error: {exc}")
        return False
