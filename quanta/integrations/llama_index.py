"""LlamaIndex VectorStore integration for Quanta.

Usage::

    from Quanta import MultiRetriever, QuantaIndex, DocStore
    from quanta.integrations.llama_index import QuantaVectorStore
    from llama_index.core import VectorStoreIndex, StorageContext

    store = QuantaVectorStore(
        retriever=retriever,
        index_name="text",
        embed_dim=1536,
    )
    storage_ctx = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex(nodes, storage_context=storage_ctx)
"""

from __future__ import annotations

from typing import Any, cast

from quanta.exceptions import QuantaError
from quanta.retriever import MultiRetriever
from quanta.utils.logging import get_logger

logger = get_logger(__name__)


def _require_llama_index() -> None:
    try:
        import llama_index.core  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "llama-index-core is required. Install it with: pip install quanta[llama-index]"
        ) from exc


class QuantaVectorStore:
    """LlamaIndex-compatible ``VectorStore`` backed by a :class:`MultiRetriever`.

    Write path (``async_add`` / ``adelete``):
      Writes chunks directly to the retriever's ``docstore`` and vectors to
      ``indexes[index_name]``.

    Read path (``aquery``):
      Delegates to ``retriever.search({index_name: query_embedding}, k=...)``.
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    def __init__(
        self,
        retriever: MultiRetriever,
        index_name: str,
        embed_dim: int,
    ) -> None:
        _require_llama_index()
        if index_name not in retriever.indexes:
            raise QuantaError(
                f"index_name={index_name!r} not found in retriever. "
                f"Available: {list(retriever.indexes)}"
            )
        self._retriever = retriever
        self._index_name = index_name
        self._embed_dim = embed_dim

    # ── LlamaIndex protocol ───────────────────────────────────────────────────

    async def async_add(self, nodes: list[Any], **kwargs: Any) -> list[str]:
        """Persist *nodes* to the docstore and add their embeddings to the index."""
        import numpy as np
        from llama_index.core.schema import TextNode

        ids: list[str] = []
        idx = self._retriever.indexes[self._index_name]

        for raw_node in nodes:
            node: TextNode = cast(TextNode, raw_node)
            if node.embedding is None:
                raise QuantaError(
                    f"Node {node.node_id!r} has no embedding — run an embedder before adding."
                )
            chunk_id = node.node_id or None
            if chunk_id is None:
                raise QuantaError("LlamaIndex node is missing a node_id.")

            document_id = node.ref_doc_id or chunk_id
            chunk_index: int = node.metadata.get("chunk_index", 0)

            await self._retriever.docstore.add_document(
                id=document_id,
                content=node.get_content(),
                doc_type=node.metadata.get("doc_type", "text"),
                metadata=node.metadata,
            )
            await self._retriever.docstore.add_chunk(
                id=chunk_id,
                document_id=document_id,
                content=node.get_content(),
                chunk_index=chunk_index,
                metadata=node.metadata,
            )
            vec = np.asarray(node.embedding, dtype=np.float32).reshape(1, -1)
            idx.add(vec, [chunk_id])
            ids.append(chunk_id)

        logger.info(
            "QuantaVectorStore[%s] added %d node(s)", self._index_name, len(ids)
        )
        return ids

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete all chunks whose ``document_id`` matches *ref_doc_id*."""
        chunks = await self._retriever.docstore.filter_chunks(
            {"document_id": ref_doc_id}
        )
        idx = self._retriever.indexes[self._index_name]
        for chunk in chunks:
            await self._retriever.docstore.delete_document(chunk.id)
            idx.remove(chunk.id)
        logger.info(
            "QuantaVectorStore[%s] deleted %d chunk(s) for doc %r",
            self._index_name,
            len(chunks),
            ref_doc_id,
        )

    async def aquery(self, query: Any, **kwargs: Any) -> Any:
        """Run a vector query and return a ``VectorStoreQueryResult``."""
        from llama_index.core.schema import NodeWithScore, TextNode
        from llama_index.core.vector_stores.types import VectorStoreQueryResult

        if query.query_embedding is None:
            raise QuantaError("VectorStoreQuery must include query_embedding.")

        results = await self._retriever.search(
            query_vectors={self._index_name: query.query_embedding},
            k=query.similarity_top_k,
        )

        nodes_with_scores = [
            NodeWithScore(
                node=TextNode(
                    node_id=r.id,
                    text=r.content or "",
                    metadata=r.metadata,
                ),
                score=r.score,
            )
            for r in results
        ]

        return VectorStoreQueryResult(
            nodes=nodes_with_scores,
            similarities=[r.score for r in results],
            ids=[r.id for r in results],
        )

    # Synchronous stubs — LlamaIndex requires them even when only async is used.
    def add(self, nodes: list[Any], **kwargs: Any) -> list[str]:  # noqa: ARG002
        raise NotImplementedError("Use async_add for QuantaVectorStore.")

    def query(self, query: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        raise NotImplementedError("Use aquery for QuantaVectorStore.")
