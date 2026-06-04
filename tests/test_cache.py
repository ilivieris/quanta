"""Tests for quanta.cache — NullCache, RedisCache, and get_cache factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quanta.cache import EmbeddingCache, NullCache, RedisCache, get_cache


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_config(**kwargs: object) -> MagicMock:
    cfg = MagicMock()
    cfg.REDIS_HOST = kwargs.get("REDIS_HOST", None)
    cfg.REDIS_PORT = kwargs.get("REDIS_PORT", 6379)
    cfg.REDIS_PASSWORD = kwargs.get("REDIS_PASSWORD", None)
    cfg.REDIS_DB = kwargs.get("REDIS_DB", 0)
    cfg.REDIS_TTL_SECONDS = kwargs.get("REDIS_TTL_SECONDS", 86400)
    return cfg


def _mock_redis_module() -> tuple[MagicMock, MagicMock]:
    """Return (mock_redis_module, mock_client)."""
    mock_client = MagicMock()
    mock_mod = MagicMock()
    mock_mod.Redis.return_value = mock_client
    return mock_mod, mock_client


# ── NullCache ─────────────────────────────────────────────────────────────────

def test_null_cache_get_always_returns_none() -> None:
    cache = NullCache()
    assert cache.get("any text") is None


def test_null_cache_set_does_not_raise() -> None:
    cache = NullCache()
    cache.set("text", np.array([1.0, 2.0], dtype=np.float32))


def test_null_cache_delete_does_not_raise() -> None:
    cache = NullCache()
    cache.delete("text")


def test_null_cache_close_does_not_raise() -> None:
    cache = NullCache()
    cache.close()


def test_null_cache_is_embedding_cache() -> None:
    assert isinstance(NullCache(), EmbeddingCache)


# ── RedisCache — serialisation round-trip ─────────────────────────────────────

def test_redis_cache_round_trip() -> None:
    mock_mod, mock_client = _mock_redis_module()
    vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    stored: dict[str, bytes] = {}

    def fake_setex(key: str, ttl: int, value: bytes) -> None:
        stored[key] = value

    mock_client.setex.side_effect = fake_setex
    mock_client.get.side_effect = lambda k: stored.get(k)

    with patch.dict("sys.modules", {"redis": mock_mod}):
        cfg = _make_config(REDIS_HOST="localhost")
        cache = RedisCache(cfg)
        cache.set("hello", vector)
        result = cache.get("hello")

    assert result is not None
    np.testing.assert_array_equal(result, vector)


def test_redis_cache_get_miss_returns_none() -> None:
    mock_mod, mock_client = _mock_redis_module()
    mock_client.get.return_value = None

    with patch.dict("sys.modules", {"redis": mock_mod}):
        cfg = _make_config(REDIS_HOST="localhost")
        cache = RedisCache(cfg)
        assert cache.get("missing") is None


# ── RedisCache — exception swallowing ────────────────────────────────────────

def test_redis_get_error_returns_none() -> None:
    mock_mod, mock_client = _mock_redis_module()
    mock_client.get.side_effect = ConnectionError("connection lost")

    with patch.dict("sys.modules", {"redis": mock_mod}):
        cfg = _make_config(REDIS_HOST="localhost")
        cache = RedisCache(cfg)
        assert cache.get("any") is None


def test_redis_set_error_does_not_raise() -> None:
    mock_mod, mock_client = _mock_redis_module()
    mock_client.setex.side_effect = ConnectionError("connection lost")

    with patch.dict("sys.modules", {"redis": mock_mod}):
        cfg = _make_config(REDIS_HOST="localhost")
        cache = RedisCache(cfg)
        cache.set("any", np.array([1.0], dtype=np.float32))


def test_redis_delete_error_does_not_raise() -> None:
    mock_mod, mock_client = _mock_redis_module()
    mock_client.delete.side_effect = ConnectionError("connection lost")

    with patch.dict("sys.modules", {"redis": mock_mod}):
        cfg = _make_config(REDIS_HOST="localhost")
        cache = RedisCache(cfg)
        cache.delete("any")


# ── get_cache factory ─────────────────────────────────────────────────────────

def test_get_cache_returns_null_when_redis_host_is_none() -> None:
    cfg = _make_config(REDIS_HOST=None)
    result = get_cache(cfg)
    assert isinstance(result, NullCache)


def test_get_cache_returns_null_when_ping_fails() -> None:
    mock_mod, mock_client = _mock_redis_module()
    mock_client.ping.side_effect = ConnectionError("refused")

    with patch.dict("sys.modules", {"redis": mock_mod}):
        cfg = _make_config(REDIS_HOST="localhost")
        result = get_cache(cfg)

    assert isinstance(result, NullCache)


def test_get_cache_returns_redis_cache_on_successful_ping() -> None:
    mock_mod, mock_client = _mock_redis_module()
    mock_client.ping.return_value = True

    with patch.dict("sys.modules", {"redis": mock_mod}):
        cfg = _make_config(REDIS_HOST="localhost")
        result = get_cache(cfg)

    assert isinstance(result, RedisCache)


def test_get_cache_returns_null_when_import_fails() -> None:
    with patch.dict("sys.modules", {"redis": None}):  # type: ignore[dict-item]
        cfg = _make_config(REDIS_HOST="localhost")
        result = get_cache(cfg)

    assert isinstance(result, NullCache)
