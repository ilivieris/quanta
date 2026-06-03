"""Tests for turbosearch.integrations.llama_index.TurboSearchVectorStore.

llama_index is mocked via sys.modules so the tests run without the package
being installed.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from turbosearch.exceptions import TurboSearchError
from turbosearch.graph import NullGraph
from turbosearch.retriever import HybridRetriever
from turbosearch.types import ChunkRecord, RetrievalResult, SearchResult


# ── Fake llama_index stubs ────────────────────────────────────────────────────

class FakeTextNode:
    def __init__(
        self,
        node_id: str = "node-1",
        text: str = "hello",
        embedding: list[float] | None = None,
        metadata: dict | None = None,
        ref_doc_id: str | None = None,
    ):
        self.node_id = node_id
        self.embedding = embedding
        self.metadata = metadata or {}
        self.ref_doc_id = ref_doc_id
        self._text = text

    def get_content(self) -> str:
        return self._text


class FakeNodeWithScore:
    def __init__(self, node: Any, score: float):
        self.node = node
        self.score = score


class FakeQueryResult:
    def __init__(self, nodes, similarities, ids):
        self.nodes = nodes
        self.similarities = similarities
        self.ids = ids


class FakeQuery:
    def __init__(self, embedding: list[float], top_k: int = 5):
        self.query_embedding = embedding
        self.similarity_top_k = top_k


@pytest.fixture(autouse=True)
def patch_llama_index(monkeypatch):
    """Inject fake llama_index modules so imports inside the integration succeed."""
    llama_core = MagicMock()
    llama_schema = MagicMock()
    llama_vstore_types = MagicMock()

    llama_schema.TextNode = FakeTextNode
    llama_schema.NodeWithScore = FakeNodeWithScore
    llama_vstore_types.VectorStoreQueryResult = FakeQueryResult

    monkeypatch.setitem(sys.modules, "llama_index", MagicMock())
    monkeypatch.setitem(sys.modules, "llama_index.core", llama_core)
    monkeypatch.setitem(sys.modules, "llama_index.core.schema", llama_schema)
    monkeypatch.setitem(sys.modules, "llama_index.core.vector_stores", MagicMock())
    monkeypatch.setitem(sys.modules, "llama_index.core.vector_stores.types", llama_vstore_types)


# ── Fixtures ──────────────────────────────────────────────────────────────────

DIM = 64


def _make_mock_index(hits: list[SearchResult] | None = None) -> MagicMock:
    idx = MagicMock()
    idx.search.return_value = hits or []
    idx.add = MagicMock()
    idx.remove = MagicMock(return_value=True)
    return idx


@pytest.fixture
def retriever(mock_docstore):
    mock_idx = _make_mock_index()
    r = HybridRetriever(
        indexes={"text": mock_idx},
        docstore=mock_docstore,
        graph=NullGraph(),
    )
    return r, mock_idx, mock_docstore


@pytest.fixture
def vector_store(retriever):
    from turbosearch.integrations.llama_index import TurboSearchVectorStore

    r, idx, ds = retriever
    return TurboSearchVectorStore(retriever=r, index_name="text", embed_dim=DIM)


# ── Constructor ───────────────────────────────────────────────────────────────

def test_invalid_index_name_raises(mock_docstore):
    from turbosearch.integrations.llama_index import TurboSearchVectorStore

    idx = _make_mock_index()
    r = HybridRetriever(indexes={"text": idx}, docstore=mock_docstore, graph=NullGraph())

    with pytest.raises(TurboSearchError, match="index_name"):
        TurboSearchVectorStore(retriever=r, index_name="nonexistent", embed_dim=DIM)


# ── async_add ─────────────────────────────────────────────────────────────────

async def test_async_add_persists_chunk_and_vector(vector_store, retriever):
    store = vector_store
    _, mock_idx, mock_ds = retriever

    node = FakeTextNode(
        node_id="ch-1",
        text="some text",
        embedding=[0.1] * DIM,
        metadata={"source": "pdf"},
        ref_doc_id="doc-1",
    )

    ids = await store.async_add([node])

    assert ids == ["ch-1"]
    mock_ds.add_chunk.assert_awaited_once_with(
        id="ch-1",
        document_id="doc-1",
        content="some text",
        chunk_index=0,
        metadata={"source": "pdf"},
    )
    mock_idx.add.assert_called_once()
    vec_arg = mock_idx.add.call_args[0][0]
    assert vec_arg.shape == (1, DIM)


async def test_async_add_missing_embedding_raises(vector_store):
    node = FakeTextNode(node_id="ch-1", embedding=None)
    with pytest.raises(TurboSearchError, match="embedding"):
        await vector_store.async_add([node])


async def test_async_add_missing_node_id_raises(vector_store):
    node = FakeTextNode(node_id=None, embedding=[0.1] * DIM)  # type: ignore[arg-type]
    node.node_id = None
    with pytest.raises(TurboSearchError, match="node_id"):
        await vector_store.async_add([node])


async def test_async_add_uses_chunk_index_from_metadata(vector_store, retriever):
    _, mock_idx, mock_ds = retriever
    node = FakeTextNode(
        node_id="ch-5",
        embedding=[0.2] * DIM,
        metadata={"chunk_index": 5},
        ref_doc_id="doc-1",
    )
    await vector_store.async_add([node])
    call_kwargs = mock_ds.add_chunk.call_args.kwargs
    assert call_kwargs["chunk_index"] == 5


# ── adelete ───────────────────────────────────────────────────────────────────

async def test_adelete_removes_filtered_chunks(vector_store, retriever):
    _, mock_idx, mock_ds = retriever
    mock_ds.filter_chunks = AsyncMock(return_value=[
        ChunkRecord(id="ch-1", document_id="doc-1", content="c", chunk_index=0),
        ChunkRecord(id="ch-2", document_id="doc-1", content="c", chunk_index=1),
    ])

    await vector_store.adelete("doc-1")

    assert mock_ds.delete_document.await_count == 2
    assert mock_idx.remove.call_count == 2


# ── aquery ────────────────────────────────────────────────────────────────────

async def test_aquery_delegates_to_search(vector_store, retriever):
    store = vector_store
    r, mock_idx, mock_ds = retriever

    # Make the retriever return one result
    mock_ds.get_chunks = AsyncMock(return_value=[
        ChunkRecord(id="ch-1", document_id="doc-1", content="result text", chunk_index=0),
    ])
    mock_idx.search.return_value = [SearchResult(id="ch-1", score=0.9)]

    query = FakeQuery(embedding=[0.1] * DIM, top_k=3)
    result = await store.aquery(query)

    assert len(result.nodes) == 1
    assert result.nodes[0].score == pytest.approx(0.9)
    assert result.ids == ["ch-1"]


async def test_aquery_missing_embedding_raises(vector_store):
    from turbosearch.exceptions import TurboSearchError

    query = FakeQuery(embedding=None, top_k=3)  # type: ignore[arg-type]
    query.query_embedding = None
    with pytest.raises(TurboSearchError, match="query_embedding"):
        await vector_store.aquery(query)


# ── sync stubs ────────────────────────────────────────────────────────────────

def test_sync_add_raises(vector_store):
    with pytest.raises(NotImplementedError):
        vector_store.add([])


def test_sync_query_raises(vector_store):
    with pytest.raises(NotImplementedError):
        vector_store.query(None)
