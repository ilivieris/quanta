"""TurboSearch — production-ready vector + graph hybrid search library."""

from turbosearch.config import TurboSearchSettings
from turbosearch.docstore import DocStore
from turbosearch.exceptions import TurboSearchError
from turbosearch.graph import GraphBackend, Neo4jGraph, NullGraph, get_graph_backend
from turbosearch.index import TurboIndex
from turbosearch.retriever import HybridRetriever
from turbosearch.types import (
    ChunkRecord,
    DocumentRecord,
    GraphNode,
    RetrievalResult,
    SearchResult,
)

__version__ = "0.1.0"
__all__ = [
    "TurboSearchSettings",
    "DocStore",
    "GraphBackend",
    "Neo4jGraph",
    "NullGraph",
    "get_graph_backend",
    "TurboIndex",
    "HybridRetriever",
    "TurboSearchError",
    "DocumentRecord",
    "ChunkRecord",
    "SearchResult",
    "GraphNode",
    "RetrievalResult",
]
