"""Quanta — production-ready vector + graph hybrid search library."""

from quanta.bm25 import BM25Backend, NullBM25, TantivyBM25, get_bm25
from quanta.cache import EmbeddingCache, NullCache, RedisCache, get_cache
from quanta.chunking import (
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    TextChunker,
    get_chunker,
)
from quanta.config import QuantaSettings
from quanta.docstore import (
    DocStore,
    DocStoreBackend,
    DuckDBDocStore,
    PostgresDocStore,
    get_docstore,
)
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
    "BM25Backend",
    "NullBM25",
    "TantivyBM25",
    "get_bm25",
    "EmbeddingCache",
    "NullCache",
    "RedisCache",
    "get_cache",
    "TextChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "SemanticChunker",
    "RecursiveChunker",
    "get_chunker",
    "QuantaSettings",
    "DocStoreBackend",
    "PostgresDocStore",
    "DuckDBDocStore",
    "get_docstore",
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
