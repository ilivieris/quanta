"""Tests for quanta.retriever.MultiRetriever."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from quanta.exceptions import QuantaError
from quanta.graph import NullGraph
from quanta.retriever import MultiRetriever, _normalize
from quanta.types import ChunkRecord, GraphNode, RetrievalResult, SearchResult

DIM = 64  # small dim for test speed


# ── _normalize helper ─────────────────────────────────────────────────────────

def test_normalize_empty():
    assert _normalize([]) == []


def test_normalize_single():
    assert _normalize([0.5]) == [1.0]


def test_normalize_equal_scores():
    assert _normalize([0.3, 0.3, 0.3]) == [1.0, 1.0, 1.0]


def test_normalize_range():
    result = _normalize([0.0, 0.5, 1.0])
    assert result == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_arbitrary():
    result = _normalize([2.0, 4.0, 6.0])
    assert result == pytest.approx([0.0, 0.5, 1.0])


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_index(hits: list[SearchResult]) -> MagicMock:
    idx = MagicMock()
    idx.search.return_value = hits
    return idx


def _make_chunk(id: str, doc_id: str = "doc-1") -> ChunkRecord:
    return ChunkRecord(id=id, document_id=doc_id, content=f"content:{id}", chunk_index=0, metadata={})


@pytest.fixture
def query_vec() -> np.ndarray:
    return np.random.default_rng(0).random(DIM).astype(np.float32)


@pytest.fixture
def simple_retriever(mock_docstore, null_graph):
    idx = _make_index(
        [SearchResult(id="ch-0", score=0.9), SearchResult(id="ch-1", score=0.6)]
    )
    mock_docstore.get_chunks = AsyncMock(
        return_value=[_make_chunk("ch-0"), _make_chunk("ch-1")]
    )
    return MultiRetriever(
        indexes={"text": idx},
        docstore=mock_docstore,
        graph=null_graph,
    ), idx, mock_docstore


# ── Constructor validation ────────────────────────────────────────────────────

def test_empty_indexes_raises(mock_docstore, null_graph):
    with pytest.raises(QuantaError, match="at least one"):
        MultiRetriever(indexes={}, docstore=mock_docstore, graph=null_graph)


def test_bad_dense_weight_raises(mock_docstore, null_graph):
    idx = _make_index([])
    with pytest.raises(QuantaError, match="dense_weight"):
        MultiRetriever(
            indexes={"t": idx}, docstore=mock_docstore, graph=null_graph,
            dense_weight=1.5
        )


def test_bad_graph_weight_raises(mock_docstore, null_graph):
    idx = _make_index([])
    with pytest.raises(QuantaError, match="graph_weight"):
        MultiRetriever(
            indexes={"t": idx}, docstore=mock_docstore, graph=null_graph,
            graph_weight=-0.1
        )


# ── Basic search ──────────────────────────────────────────────────────────────

async def test_search_returns_sorted_results(simple_retriever, query_vec):
    retriever, _, _ = simple_retriever
    results = await retriever.search({"text": query_vec}, k=5)

    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert all(isinstance(r, RetrievalResult) for r in results)


async def test_search_populates_content_from_docstore(simple_retriever, query_vec):
    retriever, _, _ = simple_retriever
    results = await retriever.search({"text": query_vec}, k=5)
    assert results[0].content == "content:ch-0"


async def test_search_source_is_dense_without_graph(simple_retriever, query_vec):
    retriever, _, _ = simple_retriever
    results = await retriever.search({"text": query_vec}, use_graph=False)
    assert all(r.source == "dense" for r in results)


# ── Index selection ───────────────────────────────────────────────────────────

async def test_search_unknown_index_name_raises(mock_docstore, null_graph, query_vec):
    idx = _make_index([])
    retriever = MultiRetriever(indexes={"text": idx}, docstore=mock_docstore, graph=null_graph)
    with pytest.raises(QuantaError, match="No index registered"):
        await retriever.search({"images": query_vec})


async def test_search_missing_query_vector_raises(mock_docstore, null_graph, query_vec):
    idx = _make_index([])
    retriever = MultiRetriever(
        indexes={"text": idx, "images": idx}, docstore=mock_docstore, graph=null_graph
    )
    with pytest.raises(QuantaError, match="No query vector"):
        await retriever.search({"text": query_vec}, index_names=["text", "images"])


# ── Multi-index score merging ─────────────────────────────────────────────────

async def test_merge_scores_across_two_indexes(mock_docstore, null_graph, query_vec):
    """An id that scores well in both indexes accumulates higher merged score."""
    idx_text = _make_index([
        SearchResult(id="shared", score=0.9),
        SearchResult(id="text-only", score=0.7),
    ])
    idx_img = _make_index([
        SearchResult(id="shared", score=0.8),
        SearchResult(id="img-only", score=0.6),
    ])
    mock_docstore.get_chunks = AsyncMock(return_value=[
        _make_chunk("shared"), _make_chunk("text-only"), _make_chunk("img-only"),
    ])

    retriever = MultiRetriever(
        indexes={"text": idx_text, "images": idx_img},
        docstore=mock_docstore,
        graph=null_graph,
        dense_weight=1.0,
        graph_weight=0.0,
    )
    results = await retriever.search({"text": query_vec, "images": query_vec}, k=10)

    score_map = {r.id: r.score for r in results}
    # "shared" appears in both indexes — must outscore either single-index hit
    assert score_map["shared"] > score_map["text-only"]
    assert score_map["shared"] > score_map["img-only"]


# ── Graph expansion ───────────────────────────────────────────────────────────

async def test_graph_weight_zero_equals_dense_only(mock_docstore, null_graph, query_vec):
    idx = _make_index([SearchResult(id="ch-0", score=0.9)])
    mock_docstore.get_chunks = AsyncMock(return_value=[_make_chunk("ch-0")])

    r_no_graph = MultiRetriever(
        indexes={"text": idx}, docstore=mock_docstore, graph=null_graph,
        graph_weight=0.0
    )
    r_use_graph = MultiRetriever(
        indexes={"text": idx}, docstore=mock_docstore, graph=null_graph,
        graph_weight=0.0
    )

    res1 = await r_no_graph.search({"text": query_vec}, use_graph=False)
    res2 = await r_use_graph.search({"text": query_vec}, use_graph=True)

    assert [r.id for r in res1] == [r.id for r in res2]
    assert res1[0].score == pytest.approx(res2[0].score)


async def test_graph_expanded_ids_get_graph_source(mock_docstore, query_vec):
    """IDs that come only from graph expansion have source='graph'."""
    idx = _make_index([SearchResult(id="dense-hit", score=0.9)])

    mock_graph = MagicMock(spec=["expand", "navigate", "neighbors", "close"])
    mock_graph.expand = AsyncMock(return_value=[("graph-only-id", 0.8)])

    mock_docstore.get_chunks = AsyncMock(return_value=[
        _make_chunk("dense-hit"),
        _make_chunk("graph-only-id"),
    ])

    retriever = MultiRetriever(
        indexes={"text": idx},
        docstore=mock_docstore,
        graph=mock_graph,
        graph_weight=0.3,
    )
    results = await retriever.search({"text": query_vec}, use_graph=True)

    sources = {r.id: r.source for r in results}
    assert sources["dense-hit"] == "dense"
    assert sources["graph-only-id"] == "graph"


async def test_dense_plus_graph_source_tag(mock_docstore, query_vec):
    """An id hit by both dense search AND graph expansion gets source='dense+graph'."""
    idx = _make_index([SearchResult(id="ch-0", score=0.9)])

    mock_graph = MagicMock(spec=["expand", "navigate", "neighbors", "close"])
    mock_graph.expand = AsyncMock(return_value=[("ch-0", 0.5)])  # same id

    mock_docstore.get_chunks = AsyncMock(return_value=[_make_chunk("ch-0")])

    retriever = MultiRetriever(
        indexes={"text": idx},
        docstore=mock_docstore,
        graph=mock_graph,
        graph_weight=0.3,
    )
    results = await retriever.search({"text": query_vec}, use_graph=True)

    assert results[0].source == "dense+graph"


# ── Filters ───────────────────────────────────────────────────────────────────

async def test_filters_call_filter_chunks(mock_docstore, null_graph, query_vec):
    idx = _make_index([])
    mock_docstore.filter_chunks = AsyncMock(return_value=[])

    retriever = MultiRetriever(
        indexes={"text": idx}, docstore=mock_docstore, graph=null_graph
    )
    await retriever.search({"text": query_vec}, filters={"year": 2024})

    mock_docstore.filter_chunks.assert_awaited_once_with({"year": 2024})


async def test_filters_no_chunks_returns_empty(mock_docstore, null_graph, query_vec):
    idx = _make_index([SearchResult(id="ch-0", score=0.9)])
    mock_docstore.filter_chunks = AsyncMock(return_value=[])  # nothing matches filter

    retriever = MultiRetriever(
        indexes={"text": idx}, docstore=mock_docstore, graph=null_graph
    )
    results = await retriever.search({"text": query_vec}, filters={"tag": "missing"})
    assert results == []


async def test_filters_allowed_ids_passed_to_index(mock_docstore, null_graph, query_vec):
    idx = _make_index([SearchResult(id="ch-1", score=0.9)])
    from quanta.types import ChunkRecord

    mock_docstore.filter_chunks = AsyncMock(
        return_value=[ChunkRecord(id="ch-1", document_id="d", content="c", chunk_index=0)]
    )
    mock_docstore.get_chunks = AsyncMock(return_value=[_make_chunk("ch-1")])

    retriever = MultiRetriever(
        indexes={"text": idx}, docstore=mock_docstore, graph=null_graph
    )
    await retriever.search({"text": query_vec}, filters={"tag": "x"})

    call_kwargs = idx.search.call_args
    allowed = call_kwargs.kwargs.get("allowed_ids") or call_kwargs[1].get("allowed_ids")
    assert allowed == ["ch-1"]


# ── navigate ──────────────────────────────────────────────────────────────────

async def test_navigate_delegates_to_graph(mock_docstore, null_graph):
    idx = _make_index([])
    retriever = MultiRetriever(
        indexes={"text": idx}, docstore=mock_docstore, graph=null_graph
    )
    result = await retriever.navigate("start-id", relation_type=None, hops=2)
    assert result == []  # NullGraph always returns empty


async def test_navigate_passes_relation_type(mock_docstore, query_vec):
    mock_graph = MagicMock(spec=["expand", "navigate", "neighbors", "close"])
    expected = [GraphNode(id="nb-1", title=None, relation="CITES", distance=1)]
    mock_graph.navigate = AsyncMock(return_value=expected)

    idx = _make_index([])
    retriever = MultiRetriever(
        indexes={"text": idx}, docstore=mock_docstore, graph=mock_graph
    )
    result = await retriever.navigate("doc-1", relation_type="CITES", hops=1)
    assert result == expected
    mock_graph.navigate.assert_called_once_with("doc-1", "CITES", 1)


# ── Properties ────────────────────────────────────────────────────────────────

def test_properties_accessible(mock_docstore, null_graph):
    idx = _make_index([])
    retriever = MultiRetriever(
        indexes={"text": idx}, docstore=mock_docstore, graph=null_graph
    )
    assert retriever.docstore is mock_docstore
    assert retriever.graph is null_graph
    assert "text" in retriever.indexes
