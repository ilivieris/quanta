from __future__ import annotations

from abc import ABC, abstractmethod

from quanta.config import QuantaSettings
from quanta.exceptions import QuantaError
from quanta.utils.logging import get_logger

logger = get_logger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────

class BM25Backend(ABC):
    """Synchronous BM25 full-text search backend interface."""

    @abstractmethod
    def add(self, doc_id: str, text: str) -> None:
        """Stage a single document for indexing (call commit() to persist)."""

    @abstractmethod
    def add_bulk(self, docs: list[tuple[str, str]]) -> None:
        """Stage a batch of (doc_id, text) pairs (call commit() to persist)."""

    @abstractmethod
    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return up to *k* (doc_id, score) pairs ranked by BM25 relevance."""

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        """Mark *doc_id* for deletion (call commit() to persist)."""

    @abstractmethod
    def commit(self) -> None:
        """Flush staged adds/deletes to the index."""

    @abstractmethod
    def close(self) -> None:
        """Release any held resources."""


# ── NullBM25 ──────────────────────────────────────────────────────────────────

class NullBM25(BM25Backend):
    """No-op backend used when BM25_BACKEND is not configured."""

    def add(self, doc_id: str, text: str) -> None:
        pass

    def add_bulk(self, docs: list[tuple[str, str]]) -> None:
        pass

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return []

    def delete(self, doc_id: str) -> None:
        pass

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


# ── TantivyBM25 ───────────────────────────────────────────────────────────────

class TantivyBM25(BM25Backend):
    """BM25 backend powered by tantivy-py (optional extra ``[bm25]``)."""

    def __init__(self, config: QuantaSettings) -> None:
        try:
            import tantivy
        except ImportError as exc:
            raise QuantaError(
                "tantivy is required for BM25 search. "
                "Install it with: pip install quanta[bm25]"
            ) from exc

        import os

        os.makedirs(config.TANTIVY_INDEX_PATH, exist_ok=True)

        schema_builder = tantivy.SchemaBuilder()
        # raw tokenizer for id so delete_term and exact-match work correctly
        schema_builder.add_text_field("id", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field("body", stored=False)
        schema = schema_builder.build()

        self._tantivy = tantivy
        self._index = tantivy.Index(schema, path=config.TANTIVY_INDEX_PATH)
        self._writer = self._index.writer()
        logger.info("TantivyBM25 opened index at %s", config.TANTIVY_INDEX_PATH)

    def add(self, doc_id: str, text: str) -> None:
        doc = self._tantivy.Document()
        doc.add_text("id", doc_id)
        doc.add_text("body", text)
        self._writer.add_document(doc)

    def add_bulk(self, docs: list[tuple[str, str]]) -> None:
        for doc_id, text in docs:
            self.add(doc_id, text)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        self._index.reload()
        searcher = self._index.searcher()
        query_obj = self._index.parse_query(query, ["body"])
        results = searcher.search(query_obj, k)
        out: list[tuple[str, float]] = []
        for score, addr in results.hits:
            doc = searcher.doc(addr)
            doc_id = doc["id"][0]
            out.append((doc_id, float(score)))
        return out

    def delete(self, doc_id: str) -> None:
        term = self._tantivy.Term.from_field_text("id", doc_id)
        self._writer.delete_term(term)

    def commit(self) -> None:
        self._writer.commit()

    def close(self) -> None:
        pass


# ── Factory ───────────────────────────────────────────────────────────────────

def get_bm25(config: QuantaSettings) -> BM25Backend:
    """Return a :class:`TantivyBM25` when configured, otherwise :class:`NullBM25`."""
    if config.BM25_BACKEND is None:
        return NullBM25()
    if config.BM25_BACKEND == "tantivy":
        return TantivyBM25(config)
    raise QuantaError(f"Unknown BM25_BACKEND: {config.BM25_BACKEND!r}")
