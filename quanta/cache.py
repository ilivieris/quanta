from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any

import numpy as np
import xxhash

from quanta.config import QuantaSettings
from quanta.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingCache(ABC):
    """Abstract embedding cache interface."""

    @abstractmethod
    def get(self, key: str) -> np.ndarray | None: ...

    @abstractmethod
    def set(self, key: str, vector: np.ndarray) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class NullCache(EmbeddingCache):
    """No-op cache used when Redis is not configured."""

    def get(self, key: str) -> np.ndarray | None:
        return None

    def set(self, key: str, vector: np.ndarray) -> None:
        pass

    def delete(self, key: str) -> None:
        pass

    def close(self) -> None:
        pass


class RedisCache(EmbeddingCache):
    """Redis-backed embedding cache using np.save / np.load serialisation."""

    _client: Any
    _ttl: int

    def __init__(self, config: QuantaSettings) -> None:
        import redis as _redis  # optional dep — ImportError propagates to get_cache

        self._client = _redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            password=config.REDIS_PASSWORD,
            db=config.REDIS_DB,
            decode_responses=False,
        )
        self._ttl = config.REDIS_TTL_SECONDS

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_key(text: str) -> str:
        return f"quanta:emb:{xxhash.xxh64(text).hexdigest()}"

    @staticmethod
    def _serialise(vector: np.ndarray) -> bytes:
        buf = BytesIO()
        np.save(buf, vector, allow_pickle=False)
        return buf.getvalue()

    @staticmethod
    def _deserialise(data: bytes) -> np.ndarray:
        return np.asarray(np.load(BytesIO(data), allow_pickle=False))

    # ── EmbeddingCache interface ──────────────────────────────────────────────

    def get(self, key: str) -> np.ndarray | None:
        try:
            data = self._client.get(self._make_key(key))
            if data is None:
                return None
            return self._deserialise(data)
        except Exception:
            return None

    def set(self, key: str, vector: np.ndarray) -> None:
        try:
            self._client.setex(self._make_key(key), self._ttl, self._serialise(vector))
        except Exception as exc:
            logger.warning("RedisCache.set failed: %s", exc)

    def delete(self, key: str) -> None:
        try:
            self._client.delete(self._make_key(key))
        except Exception as exc:
            logger.warning("RedisCache.delete failed: %s", exc)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.close()


def get_cache(config: QuantaSettings) -> EmbeddingCache:
    """Return a live RedisCache or fall back to NullCache."""
    if config.REDIS_HOST is None:
        logger.info("Redis not configured — embedding cache disabled")
        return NullCache()
    try:
        cache = RedisCache(config)
        cache._client.ping()
        logger.info("Embedding cache enabled (Redis %s)", config.REDIS_HOST)
        return cache
    except Exception as exc:
        logger.warning("Redis connection failed (%s) — cache disabled", exc)
        return NullCache()
