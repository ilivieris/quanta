from __future__ import annotations

from typing import Any

import numpy as np

from quanta.config import QuantaSettings
from quanta.docstore import DocStore
from quanta.exceptions import QuantaError
from quanta.graph import GraphBackend, NullGraph
from quanta.index import QuantaIndex
from quanta.types import GraphNode, RetrievalResult
from quanta.utils.logging import get_logger

logger = get_logger(__name__)


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalise *scores* to [0, 1].  Equal scores all map to 1.0."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


class MultiRetriever:
    """Multi-index hybrid retriever combining dense ANN search with graph expansion.

    Maintains a named pool of :class:`QuantaIndex` instances so queries can be
    issued against one or more modality indexes simultaneously (e.g. ``"text"``
    and ``"images"``).  Scores from all active indexes are normalised then
    merged; an optional graph expansion step re-scores results using
    relationship distance.
    """

    def __init__(
        self,
        indexes: dict[str, QuantaIndex],
        docstore: DocStore,
        graph: GraphBackend | None = None,
        dense_weight: float = 0.7,
        graph_weight: float = 0.3,
        config: QuantaSettings | None = None,
    ) -> None:
        if not indexes:
            raise QuantaError("MultiRetriever requires at least one QuantaIndex")
        if not (0.0 <= dense_weight <= 1.0):
            raise QuantaError(f"dense_weight must be in [0, 1], got {dense_weight}")
        if not (0.0 <= graph_weight <= 1.0):
            raise QuantaError(f"graph_weight must be in [0, 1], got {graph_weight}")

        self._indexes = indexes
        self._docstore = docstore
        self._graph = graph if graph is not None else NullGraph()
        self._dense_weight = dense_weight
        self._graph_weight = graph_weight
        self._config = config

    # ── Public properties (used by integrations) ──────────────────────────────

    @property
    def indexes(self) -> dict[str, QuantaIndex]:
        return self._indexes

    @property
    def docstore(self) -> DocStore:
        return self._docstore

    @property
    def graph(self) -> GraphBackend:
        return self._graph

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(
        self,
        query_vectors: dict[str, np.ndarray],
        k: int = 10,
        use_graph: bool = True,
        graph_hops: int = 2,
        graph_seed_k: int = 5,
        filters: dict[str, Any] | None = None,
        index_names: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Search across one or more indexes and return up to *k* ranked results.

        Args:
            query_vectors:  Mapping from index name to query embedding.
            k:              Maximum number of results to return.
            use_graph:      Whether to expand results via the graph backend.
            graph_hops:     Maximum traversal depth for graph expansion.
            graph_seed_k:   Number of top dense hits to use as graph seeds.
            filters:        JSONB containment filter applied to chunk metadata
                            before vector search (e.g. ``{"year": 2024}``).
            index_names:    Restrict search to these indexes only.  Defaults to
                            all keys present in *query_vectors*.
        """
        active_names = index_names if index_names is not None else list(query_vectors.keys())
        if not active_names:
            return []

        # Validate that every requested name has a vector and a registered index
        for name in active_names:
            if name not in query_vectors:
                raise QuantaError(
                    f"No query vector provided for index {name!r}. "
                    f"Available: {list(query_vectors)}"
                )
            if name not in self._indexes:
                raise QuantaError(
                    f"No index registered with name {name!r}. "
                    f"Registered: {list(self._indexes)}"
                )

        # ── Step 1: resolve filter → allowed chunk IDs ─────────────────────
        allowed_ids: list[str] | None = None
        if filters:
            filtered_chunks = await self._docstore.filter_chunks(filters)
            if not filtered_chunks:
                logger.debug("search: filters matched 0 chunks — returning empty")
                return []
            allowed_ids = [c.id for c in filtered_chunks]

        # ── Step 2: per-index vector search + normalise ────────────────────
        # Each index contributes an equal share of dense_weight so the total
        # dense contribution is always dense_weight regardless of index count.
        per_index_weight = self._dense_weight / len(active_names)
        score_map: dict[str, float] = {}
        dense_ids: set[str] = set()

        for name in active_names:
            idx = self._indexes[name]
            query = np.asarray(query_vectors[name], dtype=np.float32)
            hits = idx.search(query, k=k, allowed_ids=allowed_ids)
            if not hits:
                continue

            norm = _normalize([h.score for h in hits])
            for hit, ns in zip(hits, norm, strict=False):
                score_map[hit.id] = score_map.get(hit.id, 0.0) + per_index_weight * ns
                dense_ids.add(hit.id)

        # ── Step 3: graph expansion ────────────────────────────────────────
        graph_ids: set[str] = set()
        if use_graph and score_map and not isinstance(self._graph, NullGraph):
            seeds = sorted(score_map, key=score_map.__getitem__, reverse=True)[:graph_seed_k]
            expanded = self._graph.expand(seed_ids=seeds, hops=graph_hops)
            for gid, g_score in expanded:
                score_map[gid] = score_map.get(gid, 0.0) + self._graph_weight * g_score
                graph_ids.add(gid)

        if not score_map:
            return []

        # ── Step 4: sort candidates, take top k, hydrate from docstore ─────
        top_ids = sorted(score_map, key=score_map.__getitem__, reverse=True)[:k]
        chunk_records = await self._docstore.get_chunks(top_ids)
        chunk_map = {c.id: c for c in chunk_records}

        results: list[RetrievalResult] = []
        for cid in top_ids:
            chunk = chunk_map.get(cid)
            in_dense = cid in dense_ids
            in_graph = cid in graph_ids

            if in_dense and in_graph:
                source = "dense+graph"
            elif in_graph:
                source = "graph"
            else:
                source = "dense"

            results.append(
                RetrievalResult(
                    id=cid,
                    score=score_map[cid],
                    source=source,
                    content=chunk.content if chunk else None,
                    metadata=chunk.metadata if chunk else {},
                    document_id=chunk.document_id if chunk else None,
                )
            )

        logger.debug(
            "search: dense_hits=%d graph_hits=%d returned=%d",
            len(dense_ids),
            len(graph_ids),
            len(results),
        )
        return results

    # ── Graph navigation ──────────────────────────────────────────────────────

    def navigate(
        self,
        start_id: str,
        relation_type: str | None = None,
        hops: int = 1,
    ) -> list[GraphNode]:
        """Passthrough to :meth:`GraphBackend.navigate`."""
        return self._graph.navigate(start_id, relation_type, hops)
