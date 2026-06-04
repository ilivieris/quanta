"""Tests for Quanta.docstore.DocStore.

asyncpg is mocked at the pool level — no PostgreSQL instance required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from quanta.exceptions import QuantaError
from quanta.types import ChunkRecord


# ── asyncpg pool / connection mock helpers ────────────────────────────────────

def _fake_record(**kwargs):
    """Return a dict that quacks like an asyncpg Record."""
    return dict(**kwargs)


@pytest.fixture
def mock_conn():
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])
    conn.set_type_codec = AsyncMock(return_value=None)

    txn = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=None)
    txn.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=txn)
    return conn


@pytest.fixture
def mock_pool(mock_conn):
    pool = MagicMock()
    pool.close = AsyncMock()

    acq = MagicMock()
    acq.__aenter__ = AsyncMock(return_value=mock_conn)
    acq.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acq
    return pool


@pytest.fixture
def docstore(mock_pool):
    """DocStore with pool injected, bypassing real init()."""
    from quanta.config import QuantaSettings
    from quanta.docstore import DocStore

    settings = QuantaSettings(POSTGRES_USER="test", POSTGRES_PASSWORD="test")
    ds = DocStore(settings)
    ds._pool = mock_pool
    return ds


# ── init ──────────────────────────────────────────────────────────────────────

async def test_init_creates_pool_and_tables(mock_pool):
    from quanta.config import QuantaSettings
    from quanta.docstore import DocStore

    settings = QuantaSettings(POSTGRES_USER="test", POSTGRES_PASSWORD="test")
    ds = DocStore(settings)

    with patch("Quanta.docstore.asyncpg.create_pool", AsyncMock(return_value=mock_pool)):
        await ds.init()

    assert ds._pool is mock_pool


# ── add_document ──────────────────────────────────────────────────────────────

async def test_add_document_executes_upsert(docstore, mock_conn):
    await docstore.add_document("id-1", "hello world", "text", {"k": "v"})
    mock_conn.execute.assert_awaited_once()
    sql = mock_conn.execute.call_args[0][0]
    assert "ts_documents" in sql
    assert "ON CONFLICT" in sql


async def test_add_document_wraps_postgres_error(docstore, mock_conn):
    import asyncpg

    mock_conn.execute.side_effect = asyncpg.PostgresError("boom")
    with pytest.raises(QuantaError, match="add_document"):
        await docstore.add_document("id-1", "text", "text")


# ── get_document ──────────────────────────────────────────────────────────────

async def test_get_document_returns_record(docstore, mock_conn):
    now = datetime.now(tz=timezone.utc)
    mock_conn.fetchrow.return_value = _fake_record(
        id="id-1",
        content="hello",
        doc_type="text",
        metadata={"x": 1},
        created_at=now,
    )
    doc = await docstore.get_document("id-1")
    assert doc is not None
    assert doc.id == "id-1"
    assert doc.doc_type == "text"
    assert doc.metadata == {"x": 1}


async def test_get_document_missing_returns_none(docstore, mock_conn):
    mock_conn.fetchrow.return_value = None
    assert await docstore.get_document("missing") is None


# ── get_documents ─────────────────────────────────────────────────────────────

async def test_get_documents_batch_uses_any(docstore, mock_conn):
    now = datetime.now(tz=timezone.utc)
    mock_conn.fetch.return_value = [
        _fake_record(id="a", content="c1", doc_type="text", metadata={}, created_at=now),
        _fake_record(id="b", content="c2", doc_type="text", metadata={}, created_at=now),
    ]
    docs = await docstore.get_documents(["a", "b"])
    assert len(docs) == 2
    sql = mock_conn.fetch.call_args[0][0]
    assert "ANY($1)" in sql


async def test_get_documents_empty_ids_returns_empty(docstore):
    docs = await docstore.get_documents([])
    assert docs == []


# ── delete_document ───────────────────────────────────────────────────────────

async def test_delete_document_executes_delete(docstore, mock_conn):
    await docstore.delete_document("id-1")
    mock_conn.execute.assert_awaited_once()
    sql = mock_conn.execute.call_args[0][0]
    assert "DELETE" in sql
    assert "ts_documents" in sql


# ── add_chunk ─────────────────────────────────────────────────────────────────

async def test_add_chunk_executes_upsert(docstore, mock_conn):
    await docstore.add_chunk("ch-1", "doc-1", "chunk text", 0, {"tag": "a"})
    mock_conn.execute.assert_awaited_once()
    sql = mock_conn.execute.call_args[0][0]
    assert "ts_chunks" in sql


# ── add_chunks_bulk ───────────────────────────────────────────────────────────

async def test_add_chunks_bulk_single_execute(docstore, mock_conn):
    """Bulk insert must issue exactly one SQL statement regardless of batch size."""
    chunks = [
        ChunkRecord(id=f"ch-{i}", document_id="doc-1", content=f"text {i}", chunk_index=i, metadata={})
        for i in range(5)
    ]
    await docstore.add_chunks_bulk(chunks)
    mock_conn.execute.assert_awaited_once()
    sql = mock_conn.execute.call_args[0][0]
    # Single INSERT … VALUES (…), (…), …
    assert sql.count("$") >= 5 * 5  # 5 params × 5 rows


async def test_add_chunks_bulk_empty_is_noop(docstore, mock_conn):
    await docstore.add_chunks_bulk([])
    mock_conn.execute.assert_not_awaited()


# ── get_chunk ─────────────────────────────────────────────────────────────────

async def test_get_chunk_returns_record(docstore, mock_conn):
    mock_conn.fetchrow.return_value = _fake_record(
        id="ch-1", document_id="doc-1", content="hello", chunk_index=0, metadata={}
    )
    chunk = await docstore.get_chunk("ch-1")
    assert chunk is not None
    assert chunk.id == "ch-1"
    assert chunk.document_id == "doc-1"


async def test_get_chunk_missing_returns_none(docstore, mock_conn):
    mock_conn.fetchrow.return_value = None
    assert await docstore.get_chunk("x") is None


# ── get_chunks ────────────────────────────────────────────────────────────────

async def test_get_chunks_returns_list(docstore, mock_conn):
    mock_conn.fetch.return_value = [
        _fake_record(id="ch-1", document_id="doc-1", content="a", chunk_index=0, metadata={}),
    ]
    chunks = await docstore.get_chunks(["ch-1"])
    assert len(chunks) == 1
    assert chunks[0].id == "ch-1"


# ── filter_chunks ─────────────────────────────────────────────────────────────

async def test_filter_chunks_uses_jsonb_containment(docstore, mock_conn):
    mock_conn.fetch.return_value = []
    await docstore.filter_chunks({"year": 2024, "source": "arxiv"})
    sql = mock_conn.fetch.call_args[0][0]
    assert "@>" in sql


# ── close ─────────────────────────────────────────────────────────────────────

async def test_close_disposes_pool(docstore, mock_pool):
    await docstore.close()
    mock_pool.close.assert_awaited_once()
    assert docstore._pool is None
