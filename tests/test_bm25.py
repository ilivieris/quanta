"""Tests for quanta.bm25 — NullBM25, TantivyBM25, and BM25 in MultiRetriever."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from quanta.bm25 import NullBM25
from quanta.exceptions import QuantaError
from quanta.types import SearchResult


# ── NullBM25 ──────────────────────────────────────────────────────────────────

def test_null_bm25_add_is_noop():
    b = NullBM25()
    b.add("doc-1", "some text")  # must not raise


def test_null_bm25_add_bulk_is_noop():
    b = NullBM25()
    b.add_bulk([("doc-1", "text a"), ("doc-2", "text b")])


def test_null_bm25_search_returns_empty():
    b = NullBM25()
    assert b.search("query") == []


def test_null_bm25_delete_is_noop():
    b = NullBM25()
    b.delete("doc-1")


def test_null_bm25_commit_is_noop():
    b = NullBM25()
    b.commit()


def test_null_bm25_close_is_noop():
    b = NullBM25()
    b.close()


# ── TantivyBM25 ───────────────────────────────────────────────────────────────

@pytest.fixture
def tantivy_bm25(tmp_path):
    pytest.importorskip("tantivy", reason="tantivy not installed — skipping TantivyBM25 tests")
    from quanta.bm25 import TantivyBM25
    from quanta.config import QuantaSettings

    cfg = QuantaSettings(
        POSTGRES_USER="u",
        POSTGRES_PASSWORD="p",
        TANTIVY_INDEX_PATH=str(tmp_path / "idx"),
    )
    b = TantivyBM25(cfg)
    yield b
    b.close()


def test_tantivy_add_commit_search(tantivy_bm25):
    tantivy_bm25.add("doc-1", "the quick brown fox jumps")
    tantivy_bm25.commit()
    results = tantivy_bm25.search("quick fox", k=5)
    ids = [r[0] for r in results]
    assert "doc-1" in ids


def test_tantivy_search_returns_scores(tantivy_bm25):
    tantivy_bm25.add("doc-1", "vector search and retrieval augmented generation")
    tantivy_bm25.commit()
    results = tantivy_bm25.search("vector retrieval", k=5)
    assert all(isinstance(score, float) for _, score in results)
    assert all(score > 0 for _, score in results)


def test_tantivy_delete_commit_search_returns_empty(tantivy_bm25):
    tantivy_bm25.add("doc-1", "hello world")
    tantivy_bm25.commit()
    tantivy_bm25.delete("doc-1")
    tantivy_bm25.commit()
    results = tantivy_bm25.search("hello", k=5)
    assert results == []


def test_tantivy_add_bulk_performance(tantivy_bm25):
    docs = [(f"doc-{i}", f"document number {i} with some content about topic {i % 10}") for i in range(1000)]
    tantivy_bm25.add_bulk(docs)
    tantivy_bm25.commit()
    results = tantivy_bm25.search("document content topic", k=10)
    assert len(results) > 0


# ── get_bm25 factory ──────────────────────────────────────────────────────────

def test_get_bm25_returns_null_when_not_configured():
    from quanta.bm25 import get_bm25
    from quanta.config import QuantaSettings

    cfg = QuantaSettings(POSTGRES_USER="u", POSTGRES_PASSWORD="p", BM25_BACKEND=None)
    backend = get_bm25(cfg)
    assert isinstance(backend, NullBM25)


# ── BM25 integration in MultiRetriever ───────────────────────────────────────

@pytest.fixture
def mock_bm25():
    """A mock BM25Backend that is NOT a NullBM25."""
    from quanta.bm25 import BM25Backend

    b = MagicMock(spec=BM25Backend)
    b.search.return_value = [("bm25-hit", 2.5)]
    return b


@pytest.fixture
def mock_docstore_for_bm25():
    from unittest.mock import AsyncMock

    ds = MagicMock()
    ds.get_chunks = AsyncMock(return_value=[])
    ds.filter_chunks = AsyncMock(return_value=[])
    return ds


async def test_bm25_leg_adds_to_score_map(mock_docstore_for_bm25, mock_bm25):
    from unittest.mock import AsyncMock

    from quanta.graph import NullGraph
    from quanta.retriever import MultiRetriever
    from quanta.types import ChunkRecord, SearchResult

    DIM = 16
    query_vec = np.zeros(DIM, dtype=np.float32)

    idx = MagicMock()
    idx.search.return_value = [SearchResult(id="dense-hit", score=0.9)]

    mock_docstore_for_bm25.get_chunks = AsyncMock(
        return_value=[
            ChunkRecord(id="dense-hit", document_id="d", content="dense", chunk_index=0, metadata={}),
            ChunkRecord(id="bm25-hit", document_id="d2", content="bm25", chunk_index=0, metadata={}),
        ]
    )

    retriever = MultiRetriever(
        indexes={"text": idx},
        docstore=mock_docstore_for_bm25,
        graph=NullGraph(),
        bm25=mock_bm25,
        bm25_weight=0.3,
        dense_weight=0.5,
        graph_weight=0.0,
    )
    results = await retriever.search(
        {"text": query_vec}, k=10, use_graph=False, query_text="quick fox"
    )

    ids = [r.id for r in results]
    assert "bm25-hit" in ids
    mock_bm25.search.assert_called_once_with("quick fox", k=10)


async def test_bm25_leg_skipped_when_query_text_is_none(mock_docstore_for_bm25, mock_bm25):
    from unittest.mock import AsyncMock

    from quanta.graph import NullGraph
    from quanta.retriever import MultiRetriever
    from quanta.types import ChunkRecord, SearchResult

    DIM = 16
    query_vec = np.zeros(DIM, dtype=np.float32)

    idx = MagicMock()
    idx.search.return_value = [SearchResult(id="dense-hit", score=0.9)]
    mock_docstore_for_bm25.get_chunks = AsyncMock(
        return_value=[
            ChunkRecord(id="dense-hit", document_id="d", content="c", chunk_index=0, metadata={})
        ]
    )

    retriever = MultiRetriever(
        indexes={"text": idx},
        docstore=mock_docstore_for_bm25,
        graph=NullGraph(),
        bm25=mock_bm25,
    )
    await retriever.search({"text": query_vec}, k=5, query_text=None)

    mock_bm25.search.assert_not_called()


async def test_bm25_scores_merge_correctly(mock_docstore_for_bm25):
    """Document appearing in both dense and BM25 gets a higher merged score."""
    from unittest.mock import AsyncMock

    from quanta.bm25 import BM25Backend
    from quanta.graph import NullGraph
    from quanta.retriever import MultiRetriever
    from quanta.types import ChunkRecord, SearchResult

    DIM = 16
    query_vec = np.zeros(DIM, dtype=np.float32)

    shared_id = "shared-doc"
    dense_only_id = "dense-only"
    bm25_only_id = "bm25-only"

    idx = MagicMock()
    idx.search.return_value = [
        SearchResult(id=shared_id, score=0.9),
        SearchResult(id=dense_only_id, score=0.5),
    ]

    bm25_mock = MagicMock(spec=BM25Backend)
    bm25_mock.search.return_value = [
        (shared_id, 3.0),
        (bm25_only_id, 1.5),
    ]

    mock_docstore_for_bm25.get_chunks = AsyncMock(
        return_value=[
            ChunkRecord(id=shared_id, document_id="d", content="c", chunk_index=0, metadata={}),
            ChunkRecord(id=dense_only_id, document_id="d", content="c", chunk_index=0, metadata={}),
            ChunkRecord(id=bm25_only_id, document_id="d", content="c", chunk_index=0, metadata={}),
        ]
    )

    retriever = MultiRetriever(
        indexes={"text": idx},
        docstore=mock_docstore_for_bm25,
        graph=NullGraph(),
        bm25=bm25_mock,
        dense_weight=0.5,
        bm25_weight=0.5,
        graph_weight=0.0,
    )
    results = await retriever.search(
        {"text": query_vec}, k=10, use_graph=False, query_text="test query"
    )

    score_map = {r.id: r.score for r in results}
    # shared_id appears in both legs → must outscore the single-leg IDs
    assert score_map[shared_id] > score_map.get(dense_only_id, 0.0)
    assert score_map[shared_id] > score_map.get(bm25_only_id, 0.0)
