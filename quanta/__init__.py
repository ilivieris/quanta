"""Quanta — production-ready vector + graph hybrid search library."""

from quanta.config import QuantaSettings
from quanta.docstore import DocStore
from quanta.exceptions import QuantaError
from quanta.graph import GraphBackend, Neo4jGraph, NullGraph, get_graph_backend
from quanta.index import QuantaIndex
from quanta.retriever import MultiRetriever
from quanta.types import (
    ChunkRecord,
    DocumentRecord,
    GraphNode,
    RetrievalResult,
    SearchResult,
)

__version__ = "0.1.0"
__all__ = [
    "QuantaSettings",
    "DocStore",
    "GraphBackend",
    "Neo4jGraph",
    "NullGraph",
    "get_graph_backend",
    "QuantaIndex",
    "MultiRetriever",
    "QuantaError",
    "DocumentRecord",
    "ChunkRecord",
    "SearchResult",
    "GraphNode",
    "RetrievalResult",
]
