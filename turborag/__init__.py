"""TurboRAG — production-ready vector + graph hybrid search library."""

from turborag.config import TurboRAGSettings
from turborag.docstore import DocStore
from turborag.exceptions import TurboRAGError
from turborag.graph import GraphBackend, Neo4jGraph, NullGraph, get_graph_backend
from turborag.index import TurboIndex
from turborag.retriever import HybridRetriever
from turborag.types import (
    ChunkRecord,
    DocumentRecord,
    GraphNode,
    RetrievalResult,
    SearchResult,
)

__version__ = "0.1.0"
__all__ = [
    "TurboRAGSettings",
    "DocStore",
    "GraphBackend",
    "Neo4jGraph",
    "NullGraph",
    "get_graph_backend",
    "TurboIndex",
    "HybridRetriever",
    "TurboRAGError",
    "DocumentRecord",
    "ChunkRecord",
    "SearchResult",
    "GraphNode",
    "RetrievalResult",
]
