from __future__ import annotations

import json
from typing import Any

import asyncpg

from turborag.config import TurboRAGSettings
from turborag.exceptions import TurboRAGError
from turborag.types import ChunkRecord, DocumentRecord
from turborag.utils.logging import get_logger

logger = get_logger(__name__)

_CREATE_TABLES = """
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

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_ts_documents_doc_type
    ON ts_documents (doc_type);

CREATE INDEX IF NOT EXISTS idx_ts_chunks_document_id
    ON ts_chunks (document_id);

CREATE INDEX IF NOT EXISTS idx_ts_chunks_metadata_gin
    ON ts_chunks USING gin (metadata);
"""


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


def _row_to_document(row: asyncpg.Record) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        content=row["content"],
        doc_type=row["doc_type"],
        metadata=row["metadata"] or {},
        created_at=row["created_at"],
    )


def _row_to_chunk(row: asyncpg.Record) -> ChunkRecord:
    return ChunkRecord(
        id=row["id"],
        document_id=row["document_id"],
        content=row["content"],
        chunk_index=row["chunk_index"],
        metadata=row["metadata"] or {},
    )


class DocStore:
    """Async PostgreSQL-backed document and chunk store using raw asyncpg."""

    def __init__(self, settings: TurboRAGSettings) -> None:
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
                await conn.execute(_CREATE_TABLES)
                await conn.execute(_CREATE_INDEXES)
        except (asyncpg.PostgresError, OSError) as exc:
            raise TurboRAGError(f"DocStore.init failed: {exc}") from exc
        logger.info("DocStore initialised (pool max_size=%d)", self._settings.POSTGRES_POOL_SIZE)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("DocStore connection pool closed")

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
            raise TurboRAGError(f"add_document failed for id={id!r}: {exc}") from exc

    async def get_document(self, id: str) -> DocumentRecord | None:
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                row = await conn.fetchrow(
                    "SELECT id, content, doc_type, metadata, created_at "
                    "FROM ts_documents WHERE id = $1",
                    id,
                )
        except asyncpg.PostgresError as exc:
            raise TurboRAGError(f"get_document failed for id={id!r}: {exc}") from exc
        return _row_to_document(row) if row else None

    async def get_documents(self, ids: list[str]) -> list[DocumentRecord]:
        """Batch-fetch documents by ID list in a single query."""
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
            raise TurboRAGError(f"get_documents failed: {exc}") from exc
        return [_row_to_document(r) for r in rows]

    async def delete_document(self, id: str) -> None:
        """Delete a document and cascade-delete its chunks."""
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                await conn.execute("DELETE FROM ts_documents WHERE id = $1", id)
        except asyncpg.PostgresError as exc:
            raise TurboRAGError(f"delete_document failed for id={id!r}: {exc}") from exc

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
            raise TurboRAGError(f"add_chunk failed for id={id!r}: {exc}") from exc

    async def add_chunks_bulk(self, chunks: list[ChunkRecord]) -> None:
        """Insert *chunks* in a single INSERT … VALUES statement."""
        if not chunks:
            return
        # Build a single parameterised query: INSERT … VALUES ($1,…), ($6,…), …
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
            raise TurboRAGError(f"add_chunks_bulk failed ({len(chunks)} chunks): {exc}") from exc

    async def get_chunk(self, id: str) -> ChunkRecord | None:
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                row = await conn.fetchrow(
                    "SELECT id, document_id, content, chunk_index, metadata "
                    "FROM ts_chunks WHERE id = $1",
                    id,
                )
        except asyncpg.PostgresError as exc:
            raise TurboRAGError(f"get_chunk failed for id={id!r}: {exc}") from exc
        return _row_to_chunk(row) if row else None

    async def get_chunks(self, ids: list[str]) -> list[ChunkRecord]:
        """Batch-fetch chunks by ID list in a single query."""
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
            raise TurboRAGError(f"get_chunks failed: {exc}") from exc
        return [_row_to_chunk(r) for r in rows]

    async def filter_chunks(self, filters: dict[str, Any]) -> list[ChunkRecord]:
        """Return chunks whose metadata JSONB contains all key/value pairs in *filters*.

        Uses the ``@>`` containment operator, which leverages the GIN index.
        Example: ``filter_chunks({"source": "arxiv", "year": 2024})``
        """
        try:
            async with self._pool.acquire() as conn:  # type: ignore[union-attr]
                rows = await conn.fetch(
                    "SELECT id, document_id, content, chunk_index, metadata "
                    "FROM ts_chunks WHERE metadata @> $1",
                    filters,
                )
        except asyncpg.PostgresError as exc:
            raise TurboRAGError(f"filter_chunks failed (filters={filters!r}): {exc}") from exc
        return [_row_to_chunk(r) for r in rows]

    # ── Internal guard ────────────────────────────────────────────────────────

    def _assert_ready(self) -> None:
        if self._pool is None:
            raise TurboRAGError("DocStore is not initialised. Call await init() first.")
