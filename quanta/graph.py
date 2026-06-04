from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from quanta.config import QuantaSettings
from quanta.exceptions import QuantaError
from quanta.types import GraphNode
from quanta.utils.logging import get_logger

logger = get_logger(__name__)

# Relationship type names are interpolated into Cypher (parameters can't be
# used for type names), so we enforce a strict identifier format.
_SAFE_REL_TYPE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _validate_rel_type(rel_type: str) -> str:
    if not _SAFE_REL_TYPE.match(rel_type):
        raise QuantaError(
            f"rel_type must be an upper-case Cypher identifier (e.g. RELATED_TO), "
            f"got: {rel_type!r}"
        )
    return rel_type


def _validate_hops(hops: int) -> int:
    if not isinstance(hops, int) or hops < 1:
        raise QuantaError(f"hops must be a positive integer, got {hops!r}")
    return hops


# ── Abstract base ─────────────────────────────────────────────────────────────

class GraphBackend(ABC):
    """Read-oriented graph backend interface."""

    @abstractmethod
    def expand(
        self,
        seed_ids: list[str],
        hops: int = 2,
        limit: int = 50,
    ) -> list[tuple[str, float]]:
        """Return (doc_id, score) pairs reachable from *seed_ids* within *hops*.

        Score is inversely proportional to shortest-path distance.
        """

    @abstractmethod
    def navigate(
        self,
        start_id: str,
        relation_type: str | None,
        hops: int,
    ) -> list[GraphNode]:
        """Traverse from *start_id* up to *hops* steps.

        Optionally constrain to a specific relationship type at the final hop.
        """

    @abstractmethod
    def neighbors(
        self,
        node_id: str,
        relation_type: str | None,
    ) -> list[GraphNode]:
        """Return direct (1-hop) neighbours of *node_id*."""

    @abstractmethod
    def close(self) -> None:
        """Release any held resources."""


# ── NullGraph ─────────────────────────────────────────────────────────────────

class NullGraph(GraphBackend):
    """No-op backend used when Neo4j is not configured."""

    def expand(self, seed_ids: list[str], hops: int = 2, limit: int = 50) -> list[tuple[str, float]]:
        return []

    def navigate(self, start_id: str, relation_type: str | None, hops: int) -> list[GraphNode]:
        return []

    def neighbors(self, node_id: str, relation_type: str | None) -> list[GraphNode]:
        return []

    def close(self) -> None:
        pass


# ── Neo4jGraph ────────────────────────────────────────────────────────────────

class Neo4jGraph(GraphBackend):
    """Sync Neo4j backend (neo4j-python-driver v5).

    The sync driver is used deliberately — simpler than async and sufficient
    for graph expansion which happens per-query, not per-request at scale.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore[attr-defined]
        except ImportError as exc:
            raise QuantaError(
                "neo4j driver is required. Install it with: pip install quanta[neo4j]"
            ) from exc

        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Neo4jGraph driver created for %s (database=%s)", uri, database)

    # ── Internal helper ───────────────────────────────────────────────────────

    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        try:
            with self._driver.session(database=self._database) as session:
                return session.run(query, **params).data()  # type: ignore[no-any-return]
        except Exception as exc:
            raise QuantaError(f"Neo4j query failed: {exc}") from exc

    # ── GraphBackend interface ────────────────────────────────────────────────

    def expand(
        self,
        seed_ids: list[str],
        hops: int = 2,
        limit: int = 50,
    ) -> list[tuple[str, float]]:
        """Expand a set of seed document IDs via graph traversal.

        Returns (doc_id, score) pairs sorted by descending score where
        score = 1.0 / shortest_path_distance_from_any_seed.
        """
        if not seed_ids:
            return []
        _validate_hops(hops)
        query = f"""
            UNWIND $seeds AS seed_id
            MATCH (start:Document {{id: seed_id}})
            MATCH path = (start)-[*1..{hops}]-(neighbor:Document)
            WHERE NOT neighbor.id IN $seeds
            WITH neighbor.id AS doc_id, min(length(path)) AS dist
            RETURN doc_id, 1.0 / dist AS score
            ORDER BY score DESC LIMIT $limit
        """
        rows = self._run(query, seeds=seed_ids, limit=limit)
        return [(r["doc_id"], float(r["score"])) for r in rows]

    def navigate(
        self,
        start_id: str,
        relation_type: str | None,
        hops: int,
    ) -> list[GraphNode]:
        """Traverse from *start_id* returning all reachable Document nodes.

        When *relation_type* is given, only paths whose final relationship
        matches that type are returned.
        """
        _validate_hops(hops)
        # Bind a path variable so length() and last(relationships()) work correctly.
        # r*1..n binds r as a list of relationships — length(r) would fail there.
        query = f"""
            MATCH (start:Document {{id: $id}})
            MATCH path = (start)-[*1..{hops}]-(neighbor:Document)
            WHERE ($rel_type IS NULL OR type(last(relationships(path))) = $rel_type)
            WITH neighbor.id AS id, neighbor.title AS title,
                 type(last(relationships(path))) AS relation,
                 min(length(path)) AS distance
            RETURN id, title, relation, distance
            ORDER BY distance
        """
        rows = self._run(query, id=start_id, rel_type=relation_type)
        return [
            GraphNode(
                id=r["id"],
                title=r.get("title"),
                relation=r.get("relation"),
                distance=int(r["distance"]),
            )
            for r in rows
        ]

    def neighbors(
        self,
        node_id: str,
        relation_type: str | None,
    ) -> list[GraphNode]:
        """Return direct (1-hop) neighbours, optionally filtered by relationship type."""
        query = """
            MATCH (start:Document {id: $id})-[r]-(neighbor:Document)
            WHERE $rel_type IS NULL OR type(r) = $rel_type
            RETURN neighbor.id AS id, neighbor.title AS title,
                   type(r) AS relation, 1 AS distance
        """
        rows = self._run(query, id=node_id, rel_type=relation_type)
        return [
            GraphNode(
                id=r["id"],
                title=r.get("title"),
                relation=r.get("relation"),
                distance=1,
            )
            for r in rows
        ]

    # ── Write helpers (not in ABC — used by indexing pipelines) ──────────────

    def upsert_node(self, doc_id: str, properties: dict[str, Any]) -> None:
        """Create or update a Document node."""
        self._run(
            "MERGE (d:Document {id: $id}) SET d += $props",
            id=doc_id,
            props=properties,
        )

    def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a directed relationship between two Document nodes."""
        safe_type = _validate_rel_type(rel_type.upper())
        self._run(
            f"MATCH (a:Document {{id: $source}}), (b:Document {{id: $target}}) "
            f"MERGE (a)-[r:{safe_type}]->(b) SET r += $props",
            source=source_id,
            target=target_id,
            props=properties or {},
        )

    def delete_node(self, doc_id: str) -> None:
        """Delete a Document node and all its relationships."""
        self._run(
            "MATCH (d:Document {id: $id}) DETACH DELETE d",
            id=doc_id,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._driver.close()
        logger.info("Neo4jGraph driver closed")


# ── Factory ───────────────────────────────────────────────────────────────────

def get_graph_backend(config: QuantaSettings) -> GraphBackend:
    """Return a configured :class:`Neo4jGraph` when Neo4j is available, otherwise
    a :class:`NullGraph` that silently no-ops every call."""
    if config.graph_configured:
        backend: GraphBackend = Neo4jGraph(
            uri=config.NEO4J_URI,       # type: ignore[arg-type]
            user=config.NEO4J_USER,     # type: ignore[arg-type]
            password=config.NEO4J_PASSWORD,  # type: ignore[arg-type]
            database=config.NEO4J_DATABASE,
        )
        logger.info("Graph backend: Neo4jGraph (%s)", config.NEO4J_URI)
    else:
        backend = NullGraph()
        logger.info("Graph backend: NullGraph (NEO4J_URI not configured)")
    return backend
