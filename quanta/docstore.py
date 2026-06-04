from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import asyncpg
import duckdb

from quanta.config import QuantaSettings
from quanta.exceptions import QuantaError
from quanta.types import ChunkRecord, DocumentRecord
from quanta.utils.logging import get_logger

logger = get_logger(__name__)
_T = TypeVar("_T")

# ── PostgreSQL DDL ─────────────────────────────────────────────────────────────

_PG_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS ts_documents (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ts_chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT REFERENCES ts_documents(id) ON DELETE CASCADE,
    content     TEXT,
    chunk_index INTEGER NOT NULL,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);
"""

_PG_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_ts_documents_doc_type
    ON ts_documents (doc_type);

CREATE INDEX IF NOT EXISTS idx_ts_chunks_document_id
    ON ts_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_ts_chunks_metadata_gin
    ON ts_chunks USING gin (metadata);
"""

# ── DuckDB DDL ─────────────────────────────────────────────────────────────────

_DK_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS ts_documents (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    doc_type    TEXT NOT NULL,
    metadata    JSON DEFAULT '{}',
    created_at  TIMESTAMP DEFAULT current_timestamp
);
"""

_DK_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS ts_chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT REFERENCES ts_documents(id),
    content     TEXT,
    chunk_index INTEGER NOT NULL,
    metadata    JSON DEFAULT '{}',
    created_at  TIMESTAMP DEFAULT current_timestamp
);
"""

# ── asyncpg helpers ────────────────────────────────────────────────────────────

async def _init_conn(conn: asyncpg.Connection) -> None:
    """Register JSON/JSONB codecs on every pooled connection."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    await conn.set_type_codec(
        "json",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


def _pg_row_to_document(row: asyncpg.Record) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        content=row["content"],
        doc_type=row["doc_type"],
        metadata=row["metadata"] or {},
        created_at=row["created_at"],
    )


def _pg_row_to_chunk(row: asyncpg.Record) -> ChunkRecord:
    return ChunkRecord(
        id=row["id"],
        document_id=row["document_id"],
        content=row["content"],
        chunk_index=row["chunk_index"],
        metadata=row["metadata"] or {},
    )


# ── DuckDB helpers ─────────────────────────────────────────────────────────────

_VALID_META_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _check_filter_key(key: str) -> None:
    if not _VALID_META_KEY.match(key):
        raise QuantaError(f"Invalid metadata filter key: {key!r}")


def _parse_meta(v: Any) -> dict[str, Any]:
    if v is None:
        return {}
    if isinstance(v, str):
        return json.loads(v)  # type: ignore[no-any-return]
    return dict(v)


def _dk_row_to_document(row: tuple[Any, ...]) -> DocumentRecord:
    return DocumentRecord(
        id=row[0],
        content=row[1],
        doc_type=row[2],
        metadata=_parse_meta(row[3]),
        created_at=row[4],
    )


def _dk_row_to_chunk(row: tuple[Any, ...]) -> ChunkRecord:
    return ChunkRecord(
        id=row[0],
        document_id=row[1],
        content=row[2],
        chunk_index=row[3],
        metadata=_parse_meta(row[4]),
    )


# ── Abstract base ──────────────────────────────────────────────────────────────


class DocStoreBackend(ABC):
    """Abstract interface for document and chunk persistence."""

    @abstractmethod
    async def init(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def add_document(
        self,
        id: str,
        content: str,
        doc_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    async def get_document(self, id: str) -> DocumentRecord | None: ...

    @abstractmethod
    async def get_documents(self, ids: list[str]) -> list[DocumentRecord]: ...

    @abstractmethod
    async def delete_document(self, id: str) -> None: ...

    @abstractmethod
    async def add_chunk(
        self,
        id: str,
        document_id: str,
        content: str,
        chunk_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    @abstractmethod
    async def add_chunks_bulk(self, chunks: list[ChunkRecord]) -> None: ...

    @abstractmethod
    async def get_chunk(self, id: str) -> ChunkRecord | None: ...

    @abstractmethod
    async def get_chunks(self, ids: list[str]) -> list[ChunkRecord]: ...

    @abstractmethod
    async def filter_chunks(self, filters: dict[str, Any]) -> list[ChunkRecord]: ...


# ── PostgresDocStore ───────────────────────────────────────────────────────────


class PostgresDocStore(DocStoreBackend):
    """Async PostgreSQL-backed document and chunk store using raw asyncpg."""

    def __init__(self, settings: QuantaSettings) -> None:
        self._settings = settings
        self._pool: asyncpg.Pool | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Create the connection pool and DDL tables + indexes."""
        try:
            self._pool = await asyncpg.create_pool(
                host=self._settings.POSTGRES_HOST,
                port=self._settings.POSTGRES_PORT,
                database=self._settings.POSTGRES_DB,
                user=self._settings.POSTGRES_USER,
                password=self._settings.POSTGRES_PASSWORD,
                min_size=1,
                max_size=self._settings.POSTGRES_POOL_SIZE,
                init=_init_conn,
            )
            async with self._pool.acquire() as conn, conn.transaction():
                await conn.execute(_PG_CREATE_TABLES)
                await conn.execute(_PG_CREATE_INDEXES)
        except (asyncpg.PostgresError, OSError) as exc:
            raise QuantaError(f"PostgresDocStore.init failed: {exc}") from exc
        logger.info("PostgresDocStore initialised (pool max_size=%d)", self._settings.POSTGRES_POOL_SIZE)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("PostgresDocStore connection pool closed")

    # ── Document operations ───────────────────────────────────────────────────

    async def add_document(
        self,
        id: str,
        content: str,
        doc_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                await conn.execute(
                    """
                    INSERT INTO ts_documents (id, content, doc_type, metadata)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (id) DO UPDATE
                        SET content   = EXCLUDED.content,
                            doc_type  = EXCLUDED.doc_type,
                            metadata  = EXCLUDED.metadata
                    """,
                    id, content, doc_type, metadata or {},
                )
        except asyncpg.PostgresError as exc:
            raise QuantaError(f"add_document failed for id={id!r}: {exc}") from exc

    async def get_document(self, id: str) -> DocumentRecord | None:
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                row = await conn.fetchrow(
                    "SELECT id, content, doc_type, metadata, created_at "
                    "FROM ts_documents WHERE id = $1",
                    id,
                )
        except asyncpg.PostgresError as exc:
            raise QuantaError(f"get_document failed for id={id!r}: {exc}") from exc
        return _pg_row_to_document(row) if row else None

    async def get_documents(self, ids: list[str]) -> list[DocumentRecord]:
        if not ids:
            return []
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                rows = await conn.fetch(
                    "SELECT id, content, doc_type, metadata, created_at "
                    "FROM ts_documents WHERE id = ANY($1)",
                    ids,
                )
        except asyncpg.PostgresError as exc:
            raise QuantaError(f"get_documents failed: {exc}") from exc
        return [_pg_row_to_document(r) for r in rows]

    async def delete_document(self, id: str) -> None:
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                await conn.execute("DELETE FROM ts_documents WHERE id = $1", id)
        except asyncpg.PostgresError as exc:
            raise QuantaError(f"delete_document failed for id={id!r}: {exc}") from exc

    # ── Chunk operations ──────────────────────────────────────────────────────

    async def add_chunk(
        self,
        id: str,
        document_id: str,
        content: str,
        chunk_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                await conn.execute(
                    """
                    INSERT INTO ts_chunks (id, document_id, content, chunk_index, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (id) DO UPDATE
                        SET document_id  = EXCLUDED.document_id,
                            content      = EXCLUDED.content,
                            chunk_index  = EXCLUDED.chunk_index,
                            metadata     = EXCLUDED.metadata
                    """,
                    id, document_id, content, chunk_index, metadata or {},
                )
        except asyncpg.PostgresError as exc:
            raise QuantaError(f"add_chunk failed for id={id!r}: {exc}") from exc

    async def add_chunks_bulk(self, chunks: list[ChunkRecord]) -> None:
        """Insert *chunks* in a single INSERT … VALUES statement."""
        if not chunks:
            return
        placeholders = ", ".join(
            f"(${i * 5 + 1}, ${i * 5 + 2}, ${i * 5 + 3}, ${i * 5 + 4}, ${i * 5 + 5})"
            for i in range(len(chunks))
        )
        args: list[Any] = []
        for c in chunks:
            args.extend([c.id, c.document_id, c.content, c.chunk_index, c.metadata])
        query = (
            "INSERT INTO ts_chunks (id, document_id, content, chunk_index, metadata) "
            f"VALUES {placeholders} "
            "ON CONFLICT (id) DO NOTHING"
        )
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                await conn.execute(query, *args)
        except asyncpg.PostgresError as exc:
            raise QuantaError(f"add_chunks_bulk failed ({len(chunks)} chunks): {exc}") from exc

    async def get_chunk(self, id: str) -> ChunkRecord | None:
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                row = await conn.fetchrow(
                    "SELECT id, document_id, content, chunk_index, metadata "
                    "FROM ts_chunks WHERE id = $1",
                    id,
                )
        except asyncpg.PostgresError as exc:
            raise QuantaError(f"get_chunk failed for id={id!r}: {exc}") from exc
        return _pg_row_to_chunk(row) if row else None

    async def get_chunks(self, ids: list[str]) -> list[ChunkRecord]:
        if not ids:
            return []
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                rows = await conn.fetch(
                    "SELECT id, document_id, content, chunk_index, metadata "
                    "FROM ts_chunks WHERE id = ANY($1)",
                    ids,
                )
        except asyncpg.PostgresError as exc:
            raise QuantaError(f"get_chunks failed: {exc}") from exc
        return [_pg_row_to_chunk(r) for r in rows]

    async def filter_chunks(self, filters: dict[str, Any]) -> list[ChunkRecord]:
        """Return chunks whose metadata JSONB contains all key/value pairs in *filters*.

        Uses the ``@>`` containment operator, which leverages the GIN index.
        """
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                rows = await conn.fetch(
                    "SELECT id, document_id, content, chunk_index, metadata "
                    "FROM ts_chunks WHERE metadata @> $1",
                    filters,
                )
        except asyncpg.PostgresError as exc:
            raise QuantaError(f"filter_chunks failed (filters={filters!r}): {exc}") from exc
        return [_pg_row_to_chunk(r) for r in rows]

    def _assert_ready(self) -> None:
        if self._pool is None:
            raise QuantaError("PostgresDocStore is not initialised. Call await init() first.")


# ── DuckDBDocStore ─────────────────────────────────────────────────────────────


class DuckDBDocStore(DocStoreBackend):
    """DuckDB-backed document and chunk store.

    Synchronous DuckDB calls are dispatched through a single-threaded
    ``ThreadPoolExecutor`` so they integrate cleanly with asyncio.
    """

    def __init__(self, settings: QuantaSettings) -> None:
        self._path = settings.DUCKDB_PATH
        self._conn: Any = None  # duckdb.DuckDBPyConnection
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _run(self, fn: Callable[[], _T]) -> _T:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, fn)

    def _assert_ready(self) -> Any:  # returns duckdb.DuckDBPyConnection
        if self._conn is None:
            raise QuantaError("DuckDBDocStore is not initialised. Call await init() first.")
        return self._conn

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def init(self) -> None:
        path = self._path

        def _sync() -> Any:
            conn = duckdb.connect(path)
            conn.execute(_DK_CREATE_DOCUMENTS)
            conn.execute(_DK_CREATE_CHUNKS)
            return conn

        try:
            self._conn = await self._run(_sync)
        except Exception as exc:
            raise QuantaError(f"DuckDBDocStore.init failed: {exc}") from exc
        logger.info("DuckDBDocStore initialised (path=%s)", self._path)

    async def close(self) -> None:
        if self._conn is not None:
            conn = self._conn
            self._conn = None

            def _sync() -> None:
                conn.close()

            await self._run(_sync)
            self._executor.shutdown(wait=False)
            logger.info("DuckDBDocStore closed")

    # ── Document operations ───────────────────────────────────────────────────

    async def add_document(
        self,
        id: str,
        content: str,
        doc_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn = self._assert_ready()
        meta = json.dumps(metadata or {})

        def _sync() -> None:
            conn.execute(
                "INSERT INTO ts_documents (id, content, doc_type, metadata) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (id) DO UPDATE SET "
                "content = EXCLUDED.content, doc_type = EXCLUDED.doc_type, "
                "metadata = EXCLUDED.metadata",
                [id, content, doc_type, meta],
            )

        try:
            await self._run(_sync)
        except QuantaError:
            raise
        except Exception as exc:
            raise QuantaError(f"add_document failed for id={id!r}: {exc}") from exc

    async def get_document(self, id: str) -> DocumentRecord | None:
        conn = self._assert_ready()

        def _sync() -> tuple[Any, ...] | None:
            return conn.execute(  # type: ignore[no-any-return]
                "SELECT id, content, doc_type, metadata, created_at "
                "FROM ts_documents WHERE id = ?",
                [id],
            ).fetchone()

        try:
            row = await self._run(_sync)
        except QuantaError:
            raise
        except Exception as exc:
            raise QuantaError(f"get_document failed for id={id!r}: {exc}") from exc
        return _dk_row_to_document(row) if row else None

    async def get_documents(self, ids: list[str]) -> list[DocumentRecord]:
        if not ids:
            return []
        conn = self._assert_ready()
        placeholders = ", ".join("?" * len(ids))

        def _sync() -> list[tuple[Any, ...]]:
            return conn.execute(  # type: ignore[no-any-return]
                f"SELECT id, content, doc_type, metadata, created_at "
                f"FROM ts_documents WHERE id IN ({placeholders})",
                ids,
            ).fetchall()

        try:
            rows = await self._run(_sync)
        except QuantaError:
            raise
        except Exception as exc:
            raise QuantaError(f"get_documents failed: {exc}") from exc
        return [_dk_row_to_document(r) for r in rows]

    async def delete_document(self, id: str) -> None:
        conn = self._assert_ready()

        def _sync() -> None:
            conn.execute("DELETE FROM ts_chunks WHERE document_id = ?", [id])
            conn.execute("DELETE FROM ts_documents WHERE id = ?", [id])

        try:
            await self._run(_sync)
        except QuantaError:
            raise
        except Exception as exc:
            raise QuantaError(f"delete_document failed for id={id!r}: {exc}") from exc

    # ── Chunk operations ──────────────────────────────────────────────────────

    async def add_chunk(
        self,
        id: str,
        document_id: str,
        content: str,
        chunk_index: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn = self._assert_ready()
        meta = json.dumps(metadata or {})

        def _sync() -> None:
            conn.execute(
                "INSERT INTO ts_chunks (id, document_id, content, chunk_index, metadata) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT (id) DO UPDATE SET "
                "document_id = EXCLUDED.document_id, content = EXCLUDED.content, "
                "chunk_index = EXCLUDED.chunk_index, metadata = EXCLUDED.metadata",
                [id, document_id, content, chunk_index, meta],
            )

        try:
            await self._run(_sync)
        except QuantaError:
            raise
        except Exception as exc:
            raise QuantaError(f"add_chunk failed for id={id!r}: {exc}") from exc

    async def add_chunks_bulk(self, chunks: list[ChunkRecord]) -> None:
        """Insert chunks using DuckDB executemany for performance."""
        if not chunks:
            return
        conn = self._assert_ready()
        rows = [
            (c.id, c.document_id, c.content, c.chunk_index, json.dumps(c.metadata))
            for c in chunks
        ]

        def _sync() -> None:
            conn.executemany(
                "INSERT INTO ts_chunks (id, document_id, content, chunk_index, metadata) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
                rows,
            )

        try:
            await self._run(_sync)
        except QuantaError:
            raise
        except Exception as exc:
            raise QuantaError(f"add_chunks_bulk failed ({len(chunks)} chunks): {exc}") from exc

    async def get_chunk(self, id: str) -> ChunkRecord | None:
        conn = self._assert_ready()

        def _sync() -> tuple[Any, ...] | None:
            return conn.execute(  # type: ignore[no-any-return]
                "SELECT id, document_id, content, chunk_index, metadata "
                "FROM ts_chunks WHERE id = ?",
                [id],
            ).fetchone()

        try:
            row = await self._run(_sync)
        except QuantaError:
            raise
        except Exception as exc:
            raise QuantaError(f"get_chunk failed for id={id!r}: {exc}") from exc
        return _dk_row_to_chunk(row) if row else None

    async def get_chunks(self, ids: list[str]) -> list[ChunkRecord]:
        if not ids:
            return []
        conn = self._assert_ready()
        placeholders = ", ".join("?" * len(ids))

        def _sync() -> list[tuple[Any, ...]]:
            return conn.execute(  # type: ignore[no-any-return]
                f"SELECT id, document_id, content, chunk_index, metadata "
                f"FROM ts_chunks WHERE id IN ({placeholders})",
                ids,
            ).fetchall()

        try:
            rows = await self._run(_sync)
        except QuantaError:
            raise
        except Exception as exc:
            raise QuantaError(f"get_chunks failed: {exc}") from exc
        return [_dk_row_to_chunk(r) for r in rows]

    async def filter_chunks(self, filters: dict[str, Any]) -> list[ChunkRecord]:
        """Return chunks matching all key/value pairs in *filters*.

        Uses DuckDB JSON extraction: ``json_extract_string(metadata, '$.key') = ?``
        """
        for key in filters:
            _check_filter_key(key)
        conn = self._assert_ready()
        conditions = [f"json_extract_string(metadata, '$.{key}') = ?" for key in filters]
        params = [str(v) for v in filters.values()]
        where = " AND ".join(conditions) if conditions else "1=1"

        def _sync() -> list[tuple[Any, ...]]:
            return conn.execute(  # type: ignore[no-any-return]
                f"SELECT id, document_id, content, chunk_index, metadata "
                f"FROM ts_chunks WHERE {where}",
                params,
            ).fetchall()

        try:
            rows = await self._run(_sync)
        except QuantaError:
            raise
        except Exception as exc:
            raise QuantaError(f"filter_chunks failed (filters={filters!r}): {exc}") from exc
        return [_dk_row_to_chunk(r) for r in rows]


# ── Factory ────────────────────────────────────────────────────────────────────


def get_docstore(config: QuantaSettings) -> DocStoreBackend:
    if config.DOCSTORE_BACKEND == "postgres":
        return PostgresDocStore(config)
    elif config.DOCSTORE_BACKEND == "duckdb":
        return DuckDBDocStore(config)
    else:
        raise QuantaError(f"Unknown docstore backend: {config.DOCSTORE_BACKEND}")


# ── Backward-compatibility alias ───────────────────────────────────────────────

DocStore = PostgresDocStore
