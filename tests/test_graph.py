"""Tests for turborag.graph — NullGraph and Neo4jGraph."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from turborag.exceptions import TurboRAGError
from turborag.graph import NullGraph, get_graph_backend
from turborag.types import GraphNode


# ── NullGraph ─────────────────────────────────────────────────────────────────

def test_null_graph_expand_returns_empty():
    g = NullGraph()
    assert g.expand(["id-1", "id-2"], hops=2, limit=10) == []


def test_null_graph_navigate_returns_empty():
    g = NullGraph()
    assert g.navigate("id-1", relation_type="REL", hops=2) == []


def test_null_graph_neighbors_returns_empty():
    g = NullGraph()
    assert g.neighbors("id-1", relation_type=None) == []


def test_null_graph_close_is_noop():
    g = NullGraph()
    g.close()  # must not raise


def test_null_graph_expand_empty_seeds():
    g = NullGraph()
    assert g.expand([], hops=3) == []


# ── Neo4jGraph ────────────────────────────────────────────────────────────────

@pytest.fixture
def neo4j_mock():
    """Patch neo4j.GraphDatabase so no real driver is created."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.data.return_value = []
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_driver.session.return_value = mock_session

    mock_gdb = MagicMock()
    mock_gdb.driver.return_value = mock_driver

    with patch.dict(sys.modules, {"neo4j": MagicMock(GraphDatabase=mock_gdb)}):
        from turborag.graph import Neo4jGraph

        graph = Neo4jGraph(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="secret",
            database="neo4j",
        )
        yield graph, mock_driver, mock_session, mock_result


def test_neo4j_graph_expand_calls_session(neo4j_mock):
    graph, driver, session, result = neo4j_mock
    result.data.return_value = [
        {"doc_id": "neighbor-1", "score": 0.5},
    ]
    hits = graph.expand(["seed-1"], hops=2, limit=20)
    assert hits == [("neighbor-1", 0.5)]
    session.run.assert_called_once()
    cypher = session.run.call_args[0][0]
    assert "UNWIND" in cypher
    assert "LIMIT" in cypher


def test_neo4j_graph_expand_empty_seeds_returns_empty(neo4j_mock):
    graph, *_ = neo4j_mock
    assert graph.expand([]) == []


def test_neo4j_graph_navigate_calls_session(neo4j_mock):
    graph, driver, session, result = neo4j_mock
    result.data.return_value = [
        {"id": "n-1", "title": "Title A", "relation": "RELATED_TO", "distance": 1},
    ]
    nodes = graph.navigate("start-id", relation_type=None, hops=2)
    assert len(nodes) == 1
    assert isinstance(nodes[0], GraphNode)
    assert nodes[0].id == "n-1"
    assert nodes[0].distance == 1
    cypher = session.run.call_args[0][0]
    assert "MATCH" in cypher


def test_neo4j_graph_neighbors_1hop(neo4j_mock):
    graph, driver, session, result = neo4j_mock
    result.data.return_value = [
        {"id": "nb-1", "title": None, "relation": "CITES", "distance": 1},
    ]
    nodes = graph.neighbors("doc-1", relation_type="CITES")
    assert len(nodes) == 1
    assert nodes[0].relation == "CITES"
    cypher = session.run.call_args[0][0]
    # Single-hop — no variable-length pattern
    assert "*" not in cypher


def test_neo4j_graph_upsert_node_calls_merge(neo4j_mock):
    graph, driver, session, result = neo4j_mock
    graph.upsert_node("doc-1", {"title": "Hello"})
    session.run.assert_called_once()
    cypher = session.run.call_args[0][0]
    assert "MERGE" in cypher


def test_neo4j_graph_upsert_edge_valid(neo4j_mock):
    graph, driver, session, result = neo4j_mock
    graph.upsert_edge("a", "b", "RELATED_TO")
    cypher = session.run.call_args[0][0]
    assert "RELATED_TO" in cypher


def test_neo4j_graph_upsert_edge_invalid_rel_type_raises(neo4j_mock):
    graph, *_ = neo4j_mock
    with pytest.raises(TurboRAGError, match="identifier"):
        graph.upsert_edge("a", "b", "bad rel type!")


def test_neo4j_graph_delete_node(neo4j_mock):
    graph, driver, session, result = neo4j_mock
    graph.delete_node("doc-1")
    cypher = session.run.call_args[0][0]
    assert "DETACH DELETE" in cypher


def test_neo4j_graph_close_calls_driver_close(neo4j_mock):
    graph, driver, *_ = neo4j_mock
    graph.close()
    driver.close.assert_called_once()


def test_neo4j_hops_validation_raises(neo4j_mock):
    graph, *_ = neo4j_mock
    with pytest.raises(TurboRAGError, match="hops"):
        graph.expand(["id-1"], hops=0)


# ── Factory ───────────────────────────────────────────────────────────────────

def test_get_graph_backend_returns_null_when_unconfigured():
    from turborag.config import TurboRAGSettings

    settings = TurboRAGSettings(POSTGRES_USER="u", POSTGRES_PASSWORD="p")
    backend = get_graph_backend(settings)
    assert isinstance(backend, NullGraph)


def test_get_graph_backend_returns_neo4j_when_configured(monkeypatch):
    from turborag.config import TurboRAGSettings

    mock_gdb = MagicMock()
    mock_gdb.driver.return_value = MagicMock()

    with patch.dict(sys.modules, {"neo4j": MagicMock(GraphDatabase=mock_gdb)}):
        from turborag.graph import Neo4jGraph

        settings = TurboRAGSettings(
            POSTGRES_USER="u",
            POSTGRES_PASSWORD="p",
            NEO4J_URI="bolt://localhost:7687",
            NEO4J_USER="neo4j",
            NEO4J_PASSWORD="secret",
        )
        backend = get_graph_backend(settings)
    assert isinstance(backend, Neo4jGraph)
