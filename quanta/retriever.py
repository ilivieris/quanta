from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from quanta.bm25 import BM25Backend, NullBM25
from quanta.cache import EmbeddingCache, NullCache
from quanta.config import QuantaSettings
from quanta.docstore import DocStoreBackend
from quanta.exceptions import QuantaError
from quanta.graph import GraphBackend, NullGraph
from quanta.index import QuantaIndex
from quanta.types import GraphNode, RetrievalResult
from quanta.utils.logging import get_logger

logger = get_logger(__name__)

#: Default rank-fusion smoothing constant (Cormack et al., SIGIR 2009).
RRF_K = 60


def _ranked_ids(scored: dict[str, float]) -> list[str]:
    """Return ids ordered by descending score (ties broken by id for determinism)."""
    return sorted(scored, key=lambda i: (-scored[i], i))


class MultiRetriever:
    """Multi-index hybrid retriever combining dense ANN search, optional BM25,
    and optional graph expansion.

    Maintains a named pool of :class:`QuantaIndex` instances so queries can be
    issued against one or more modality indexes simultaneously (e.g. ``"text"``
    and ``"images"``).

    **Fusion.**  Legs are combined with weighted Reciprocal Rank Fusion (RRF)::

        score(d) = SUM over legs of  w_leg / (rrf_k + rank_leg(d))

    Only the *rank* a document achieves within a leg matters, never its raw
    score.  This sidesteps the central problem of score-level fusion: cosine
    similarity, BM25 relevance and inverse hop-distance live on incomparable
    scales, so any min-max or z-score normalisation of them is arbitrary.

    **The graph expands, it does not score.**  Graph traversal widens the
    candidate pool; the newly admitted documents are then re-scored by the dense
    leg (a constrained search restricted to the expanded ids) and ranked on that
    content evidence like any other candidate.  With the default
    ``graph_weight=0.0`` the graph therefore contributes *no* score of its own.
    Raising ``graph_weight`` above zero adds the graph as a further fusion leg,
    ranked by hop distance -- a tie-breaker budget, and the only way for an
    expanded document that has no vector in any index to surface at all.
    """

    def __init__(
        self,
        indexes: dict[str, QuantaIndex],
        docstore: DocStoreBackend,
        graph: GraphBackend | None = None,
        dense_weight: float | None = None,
        graph_weight: float = 0.0,
        config: QuantaSettings | None = None,
        cache: EmbeddingCache | None = None,
        bm25: BM25Backend | None = None,
        bm25_weight: float = 0.3,
        rrf_k: int = RRF_K,
    ) -> None:
        if not indexes:
            raise QuantaError("MultiRetriever requires at least one QuantaIndex")
        if not (0.0 <= graph_weight <= 1.0):
            raise QuantaError(f"graph_weight must be in [0, 1], got {graph_weight}")
        if not (0.0 <= bm25_weight <= 1.0):
            raise QuantaError(f"bm25_weight must be in [0, 1], got {bm25_weight}")
        if rrf_k < 1:
            raise QuantaError(f"rrf_k must be a positive integer, got {rrf_k}")

        self._bm25 = bm25 if bm25 is not None else NullBM25()
        bm25_active = not isinstance(self._bm25, NullBM25)

        # dense_weight defaults to 0.5 when BM25 is active, 0.7 otherwise.  Under
        # RRF only the ratio between leg weights matters, not their absolute size.
        _dense = dense_weight if dense_weight is not None else (0.5 if bm25_active else 0.7)
        if not (0.0 <= _dense <= 1.0):
            raise QuantaError(f"dense_weight must be in [0, 1], got {_dense}")

        self._indexes = indexes
        self._docstore = docstore
        self._graph = graph if graph is not None else NullGraph()
        self._dense_weight = _dense
        self._graph_weight = graph_weight
        self._bm25_weight = bm25_weight
        self._rrf_k = rrf_k
        self._config = config
        self.cache: EmbeddingCache = cache if cache is not None else NullCache()

    # ── Cache helper ─────────────────────────────────────────────────────────

    def get_or_embed(self, text: str, embed_fn: Callable[[str], np.ndarray]) -> np.ndarray:
        """Return a cached embedding or call *embed_fn* and cache the result."""
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        vector = embed_fn(text)
        self.cache.set(text, vector)
        return vector

    # ── Public properties (used by integrations) ──────────────────────────────

    @property
    def indexes(self) -> dict[str, QuantaIndex]:
        return self._indexes

    @property
    def docstore(self) -> DocStoreBackend:
        return self._docstore

    @property
    def graph(self) -> GraphBackend:
        return self._graph

    @property
    def bm25(self) -> BM25Backend:
        return self._bm25

    # ── Fusion ────────────────────────────────────────────────────────────────

    def _fuse(
        self,
        dense_legs: dict[str, dict[str, float]],
        bm25_leg: dict[str, float],
        graph_leg: dict[str, float],
        dense_weight: float,
        bm25_weight: float,
    ) -> dict[str, float]:
        """Weighted Reciprocal Rank Fusion over every active leg."""
        score_map: dict[str, float] = {}
        per_index_weight = dense_weight / len(dense_legs) if dense_legs else 0.0

        def _accumulate(leg: dict[str, float], weight: float) -> None:
            if weight == 0.0 or not leg:
                return
            for rank, doc_id in enumerate(_ranked_ids(leg), start=1):
                score_map[doc_id] = score_map.get(doc_id, 0.0) + weight / (self._rrf_k + rank)

        for leg in dense_legs.values():
            _accumulate(leg, per_index_weight)
        _accumulate(bm25_leg, bm25_weight)
        _accumulate(graph_leg, self._graph_weight)
        return score_map

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
        query_text: str | None = None,
    ) -> list[RetrievalResult]:
        """Search across one or more indexes and return up to *k* ranked results.

        Args:
            query_vectors:  Mapping from index name to query embedding.
            k:              Maximum number of results to return.
            use_graph:      Whether to expand the candidate pool via the graph.
            graph_hops:     Maximum traversal depth for graph expansion.
            graph_seed_k:   Number of top fused hits to use as graph seeds.
            filters:        JSONB containment filter applied to chunk metadata.
            index_names:    Restrict search to these indexes only.
            query_text:     Raw query string used for the optional BM25 leg.
        """
        active_names = index_names if index_names is not None else list(query_vectors.keys())
        if not active_names:
            return []

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

        # ── Step 2: determine active legs ──────────────────────────────────
        bm25_active = not isinstance(self._bm25, NullBM25) and query_text is not None
        eff_bm25 = self._bm25_weight if bm25_active else 0.0

        graph_active = use_graph and not isinstance(self._graph, NullGraph)
        # Fetch enough dense hits to supply high-quality graph seeds.
        fetch_k = max(k, graph_seed_k) if graph_active else k

        # ── Step 3: per-index dense search (raw scores → per-leg ranking) ──
        dense_legs: dict[str, dict[str, float]] = {}
        for name in active_names:
            idx = self._indexes[name]
            query = np.asarray(query_vectors[name], dtype=np.float32)
            leg: dict[str, float] = {}
            for hit in idx.search(query, k=fetch_k, allowed_ids=allowed_ids):
                # keep the best score per id in case a backend returns duplicates
                if hit.id not in leg or hit.score > leg[hit.id]:
                    leg[hit.id] = float(hit.score)
            dense_legs[name] = leg

        dense_ids: set[str] = {doc_id for leg in dense_legs.values() for doc_id in leg}

        # ── Step 4: BM25 leg ───────────────────────────────────────────────
        bm25_leg: dict[str, float] = {}
        if bm25_active:
            for bid, bscore in self._bm25.search(query_text, k=k):  # type: ignore[arg-type]
                if bid not in bm25_leg or bscore > bm25_leg[bid]:
                    bm25_leg[bid] = float(bscore)
        bm25_ids = set(bm25_leg)

        # ── Step 5: graph expansion (candidate enlargement, not scoring) ───
        # Seeds come from a preliminary fusion of the content legs.  Newly
        # admitted ids are re-scored by every dense index via a constrained
        # search, so they compete on content evidence like any other candidate.
        graph_leg: dict[str, float] = {}
        graph_ids: set[str] = set()
        if graph_active:
            preliminary = self._fuse(dense_legs, bm25_leg, {}, self._dense_weight, eff_bm25)
            if preliminary:
                seeds = _ranked_ids(preliminary)[:graph_seed_k]
                expanded = await self._graph.expand(seed_ids=seeds, hops=graph_hops)
                for gid, g_score in expanded:
                    graph_leg[gid] = float(g_score)  # g_score = 1.0 / hop_distance
                graph_ids = set(graph_leg)

                new_ids = [gid for gid in graph_leg if gid not in dense_ids]
                if new_ids:
                    allowset = set(new_ids)
                    for name in active_names:
                        idx = self._indexes[name]
                        query = np.asarray(query_vectors[name], dtype=np.float32)
                        for hit in idx.search(query, k=len(new_ids), allowed_ids=new_ids):
                            if hit.id not in allowset:
                                continue  # backend ignored the allowlist — skip
                            leg = dense_legs[name]
                            if hit.id not in leg or hit.score > leg[hit.id]:
                                leg[hit.id] = float(hit.score)
                            dense_ids.add(hit.id)

        # ── Step 6: final fusion ───────────────────────────────────────────
        score_map = self._fuse(dense_legs, bm25_leg, graph_leg, self._dense_weight, eff_bm25)
        if not score_map:
            return []

        # ── Step 7: take top k and hydrate from the docstore ───────────────
        top_ids = _ranked_ids(score_map)[:k]
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
            "search: dense_hits=%d bm25_hits=%d graph_hits=%d returned=%d",
            len(dense_ids),
            len(bm25_ids),
            len(graph_ids),
            len(results),
        )
        return results

    # ── Graph navigation ──────────────────────────────────────────────────────

    async def navigate(
        self,
        start_id: str,
        relation_type: str | None = None,
        hops: int = 1,
    ) -> list[GraphNode]:
        """Passthrough to :meth:`GraphBackend.navigate`."""
        return await self._graph.navigate(start_id, relation_type, hops)
