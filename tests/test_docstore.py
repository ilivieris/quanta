"""Parametrized docstore tests covering both PostgreSQL and DuckDB backends.

DuckDB tests run in-process using tmp_path — no Docker required.
PostgreSQL tests are skipped when POSTGRES_USER is unset.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

import pytest

from quanta.config import QuantaSettings
from quanta.docstore import (
    DocStoreBackend,
    DuckDBDocStore,
    PostgresDocStore,
    get_docstore,
)
from quanta.exceptions import QuantaError
from quanta.types import ChunkRecord


# ── Parametrized fixture ───────────────────────────────────────────────────────


@pytest.fixture(params=["duckdb", "postgres"])
async def store(request: pytest.FixtureRequest, tmp_path: pytest.TempPathFactory) -> AsyncGenerator[DocStoreBackend, None]:  # type: ignore[type-arg]
    backend: str = request.param  # type: ignore[assignment]

    if backend == "postgres":
        user = os.environ.get("POSTGRES_USER")
        if not user:
            pytest.skip("POSTGRES_USER not set — skipping Postgres backend")
        password = os.environ.get("POSTGRES_PASSWORD", "")
        settings = QuantaSettings(
            POSTGRES_USER=user,
            POSTGRES_PASSWORD=password,
            POSTGRES_HOST=os.environ.get("POSTGRES_HOST", "localhost"),
            POSTGRES_PORT=int(os.environ.get("POSTGRES_PORT", "5432")),
            POSTGRES_DB=os.environ.get("POSTGRES_DB", "quanta_test"),
            DOCSTORE_BACKEND="postgres",
        )
        s = PostgresDocStore(settings)
        await s.init()
        yield s
        async with s._pool.acquire() as conn:  # type: ignore[union-attr]
            await conn.execute("TRUNCATE ts_chunks CASCADE")
            await conn.execute("TRUNCATE ts_documents CASCADE")
        await s.close()

    else:  # duckdb
        settings = QuantaSettings(
            POSTGRES_USER="unused",
            POSTGRES_PASSWORD="unused",
            DOCSTORE_BACKEND="duckdb",
            DUCKDB_PATH=str(tmp_path / "test.duckdb"),  # type: ignore[operator]
        )
        s = DuckDBDocStore(settings)
        await s.init()
        yield s
        await s.close()


# ── add → get document ────────────────────────────────────────────────────────


async def test_add_get_document(store: DocStoreBackend) -> None:
    await store.add_document("doc-1", "hello world", "text", {"k": "v"})
    doc = await store.get_document("doc-1")
    assert doc is not None
    assert doc.id == "doc-1"
    assert doc.content == "hello world"
    assert doc.doc_type == "text"
    assert doc.metadata == {"k": "v"}


async def test_add_document_upsert(store: DocStoreBackend) -> None:
    await store.add_document("doc-1", "original", "text")
    await store.add_document("doc-1", "updated", "pdf")
    doc = await store.get_document("doc-1")
    assert doc is not None
    assert doc.content == "updated"
    assert doc.doc_type == "pdf"


async def test_get_document_missing_returns_none(store: DocStoreBackend) -> None:
    assert await store.get_document("nonexistent") is None


async def test_get_documents_batch(store: DocStoreBackend) -> None:
    await store.add_document("doc-1", "c1", "text")
    await store.add_document("doc-2", "c2", "text")
    docs = await store.get_documents(["doc-1", "doc-2"])
    assert len(docs) == 2
    ids = {d.id for d in docs}
    assert ids == {"doc-1", "doc-2"}


async def test_get_documents_empty_returns_empty(store: DocStoreBackend) -> None:
    assert await store.get_documents([]) == []


# ── add → get chunk ───────────────────────────────────────────────────────────


async def test_add_get_chunk(store: DocStoreBackend) -> None:
    await store.add_document("doc-1", "content", "text")
    await store.add_chunk("ch-1", "doc-1", "chunk text", 0, {"tag": "a"})
    chunk = await store.get_chunk("ch-1")
    assert chunk is not None
    assert chunk.id == "ch-1"
    assert chunk.document_id == "doc-1"
    assert chunk.content == "chunk text"
    assert chunk.chunk_index == 0
    assert chunk.metadata == {"tag": "a"}


async def test_get_chunk_missing_returns_none(store: DocStoreBackend) -> None:
    assert await store.get_chunk("nonexistent") is None


async def test_get_chunks_batch(store: DocStoreBackend) -> None:
    await store.add_document("doc-1", "content", "text")
    await store.add_chunk("ch-1", "doc-1", "a", 0)
    await store.add_chunk("ch-2", "doc-1", "b", 1)
    chunks = await store.get_chunks(["ch-1", "ch-2"])
    assert len(chunks) == 2
    ids = {c.id for c in chunks}
    assert ids == {"ch-1", "ch-2"}


async def test_get_chunks_empty_returns_empty(store: DocStoreBackend) -> None:
    assert await store.get_chunks([]) == []


# ── add_chunks_bulk ───────────────────────────────────────────────────────────


async def test_add_chunks_bulk(store: DocStoreBackend) -> None:
    await store.add_document("doc-1", "content", "text")
    chunks = [
        ChunkRecord(id=f"ch-{i}", document_id="doc-1", content=f"text {i}", chunk_index=i, metadata={})
        for i in range(5)
    ]
    await store.add_chunks_bulk(chunks)
    fetched = await store.get_chunks([f"ch-{i}" for i in range(5)])
    assert len(fetched) == 5


async def test_add_chunks_bulk_empty_is_noop(store: DocStoreBackend) -> None:
    await store.add_chunks_bulk([])  # must not raise


# ── filter_chunks with metadata ───────────────────────────────────────────────


async def test_filter_chunks_with_metadata(store: DocStoreBackend) -> None:
    await store.add_document("doc-1", "content", "text")
    await store.add_chunk("ch-1", "doc-1", "a", 0, {"source": "arxiv", "year": "2024"})
    await store.add_chunk("ch-2", "doc-1", "b", 1, {"source": "arxiv", "year": "2023"})
    await store.add_chunk("ch-3", "doc-1", "c", 2, {"source": "other", "year": "2024"})

    results = await store.filter_chunks({"source": "arxiv"})
    assert len(results) == 2
    ids = {r.id for r in results}
    assert ids == {"ch-1", "ch-2"}


async def test_filter_chunks_multi_key(store: DocStoreBackend) -> None:
    await store.add_document("doc-1", "content", "text")
    await store.add_chunk("ch-1", "doc-1", "a", 0, {"source": "arxiv", "year": "2024"})
    await store.add_chunk("ch-2", "doc-1", "b", 1, {"source": "arxiv", "year": "2023"})

    results = await store.filter_chunks({"source": "arxiv", "year": "2024"})
    assert len(results) == 1
    assert results[0].id == "ch-1"


async def test_filter_chunks_no_match(store: DocStoreBackend) -> None:
    results = await store.filter_chunks({"source": "nonexistent"})
    assert results == []


# ── delete cascade ────────────────────────────────────────────────────────────


async def test_delete_document_cascade(store: DocStoreBackend) -> None:
    await store.add_document("doc-1", "content", "text")
    await store.add_chunk("ch-1", "doc-1", "chunk", 0)
    await store.add_chunk("ch-2", "doc-1", "chunk 2", 1)

    await store.delete_document("doc-1")

    assert await store.get_document("doc-1") is None
    assert await store.get_chunk("ch-1") is None
    assert await store.get_chunk("ch-2") is None


async def test_delete_nonexistent_is_noop(store: DocStoreBackend) -> None:
    await store.delete_document("does-not-exist")  # must not raise


# ── backend switching via config ──────────────────────────────────────────────


def test_get_docstore_factory_postgres() -> None:
    settings = QuantaSettings(
        POSTGRES_USER="u",
        POSTGRES_PASSWORD="p",
        DOCSTORE_BACKEND="postgres",
    )
    s = get_docstore(settings)
    assert isinstance(s, PostgresDocStore)


def test_get_docstore_factory_duckdb(tmp_path: pytest.TempPathFactory) -> None:  # type: ignore[type-arg]
    settings = QuantaSettings(
        POSTGRES_USER="u",
        POSTGRES_PASSWORD="p",
        DOCSTORE_BACKEND="duckdb",
        DUCKDB_PATH=str(tmp_path / "factory.duckdb"),  # type: ignore[operator]
    )
    s = get_docstore(settings)
    assert isinstance(s, DuckDBDocStore)


def test_get_docstore_factory_unknown() -> None:
    settings = QuantaSettings.model_construct(  # bypass Literal validation
        POSTGRES_USER="u",
        POSTGRES_PASSWORD="p",
        DOCSTORE_BACKEND="invalid",
        DUCKDB_PATH="./x.duckdb",
        POSTGRES_HOST="localhost",
        POSTGRES_PORT=5432,
        POSTGRES_DB="quanta",
        POSTGRES_POOL_SIZE=5,
        NEO4J_DATABASE="neo4j",
        EMBED_MODEL="test",
        EMBED_DIM=768,
        DEFAULT_BIT_WIDTH=4,
        DEFAULT_TOP_K=10,
    )
    with pytest.raises(QuantaError, match="Unknown docstore backend"):
        get_docstore(settings)  # type: ignore[arg-type]
