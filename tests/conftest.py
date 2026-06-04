"""Shared pytest fixtures for the Quanta test suite.

All tests run without external services.  PostgreSQL and Neo4j are replaced
by in-process mocks; turbovec is patched via sys.modules.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from dotenv import load_dotenv
load_dotenv()

from quanta.graph import NullGraph
from quanta.types import ChunkRecord, DocumentRecord, SearchResult


# ── Basic data fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def tmp_index_dir(tmp_path):
    d = tmp_path / "indexes"
    d.mkdir()
    return str(d)


@pytest.fixture
def sample_vectors() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((10, 768)).astype(np.float32)


@pytest.fixture
def sample_ids() -> list[str]:
    return [f"doc-{i:03d}" for i in range(10)]


# ── Graph fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def null_graph() -> NullGraph:
    return NullGraph()


# ── DocStore mock ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_docstore() -> MagicMock:
    """Fully async-mocked DocStore with sensible defaults."""
    ds = MagicMock()
    ds.init = AsyncMock()
    ds.close = AsyncMock()
    ds.add_document = AsyncMock()
    ds.get_document = AsyncMock(return_value=None)
    ds.get_documents = AsyncMock(return_value=[])
    ds.delete_document = AsyncMock()
    ds.add_chunk = AsyncMock()
    ds.add_chunks_bulk = AsyncMock()
    ds.get_chunk = AsyncMock(return_value=None)
    ds.get_chunks = AsyncMock(return_value=[])
    ds.filter_chunks = AsyncMock(return_value=[])
    return ds


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(id: str, document_id: str = "parent-doc", content: str = "hello") -> ChunkRecord:
    return ChunkRecord(
        id=id,
        document_id=document_id,
        content=content,
        chunk_index=0,
        metadata={"source": "test"},
    )


def make_document(id: str, content: str = "hello doc") -> DocumentRecord:
    return DocumentRecord(
        id=id,
        content=content,
        doc_type="text",
        metadata={},
        created_at=datetime.utcnow(),
    )
