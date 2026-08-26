#!/usr/bin/env python3
"""Triple-signal retrieval over the medical knowledge graph via ``MultiRetriever``.

The agent in ``agent.py`` calls ``QuantaIndex`` directly through its tool layer.
This script instead drives the full Quanta pipeline — weighted reciprocal rank
fusion over a dense leg, a BM25 leg and a graph leg, with graph expansion and
allowlist re-scoring — against the same corpus, so the fusion rule itself can be
observed.

It shows three configurations of the same query:

    1. dense only                       (use_graph=False, no query_text)
    2. dense + BM25                     (query_text supplied)
    3. dense + BM25 + graph expansion   (use_graph=True)

Prerequisites: a populated Neo4j graph (``populate_medical_graph.py``) and the
persisted indexes under ``./medical_indexes`` (created on the first run of
``agent.py``).

Usage:
    python -m examples.project.multiretriever_demo
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import tempfile
import time
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from examples.project.indexer import build_indexer
from quanta import (
    DuckDBDocStore,
    GraphNode,
    MultiRetriever,
    QuantaSettings,
    TantivyBM25,
)
from quanta.graph import GraphBackend

# ── The clinical-knowledge subgraph ───────────────────────────────────────────
# Entities carry different primary-key properties, so every lookup coalesces
# them into the single string id that the vector indexes are keyed by.
_ID = "coalesce(n.icdCode, n.procCode, n.patientId, n.doctorId)"

# Relationship types that carry clinical meaning for retrieval. Restricting the
# traversal keeps expansion interpretable and bounds its fan-out.
_REL_TYPES = "COMORBID_WITH|HAS_DIAGNOSIS|UNDERWENT|PERFORMED_BY|TREATED_BY"


class MedicalGraphBackend(GraphBackend):
    """A :class:`GraphBackend` over the hospital ontology.

    Quanta's built-in :class:`~quanta.graph.Neo4jGraph` assumes a homogeneous
    ``(:Document {id})`` model. This corpus is heterogeneous — ``Patient``,
    ``Doctor``, ``Diagnosis``, ``Procedure`` — with a different key property per
    label, so the backend interface is implemented directly against the real
    schema instead. No projection or duplication of the graph is required.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    async def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, **params)
            return await result.data()

    async def expand(
        self, seed_ids: list[str], hops: int = 2, limit: int = 50
    ) -> list[tuple[str, float]]:
        if not seed_ids:
            return []
        query = f"""
            UNWIND $seeds AS seed_id
            MATCH (s)
            WHERE coalesce(s.icdCode, s.procCode, s.patientId, s.doctorId) = seed_id
            MATCH path = (s)-[:{_REL_TYPES}*1..{int(hops)}]-(n)
            WITH {_ID} AS doc_id, min(length(path)) AS dist
            WHERE doc_id IS NOT NULL AND NOT doc_id IN $seeds
            RETURN doc_id, 1.0 / dist AS score
            ORDER BY score DESC, doc_id
            LIMIT $limit
        """
        rows = await self._run(query, seeds=seed_ids, limit=limit)
        return [(r["doc_id"], float(r["score"])) for r in rows]

    async def navigate(
        self, start_id: str, relation_type: str | None, hops: int
    ) -> list[GraphNode]:
        query = f"""
            MATCH (s)
            WHERE coalesce(s.icdCode, s.procCode, s.patientId, s.doctorId) = $id
            MATCH path = (s)-[:{_REL_TYPES}*1..{int(hops)}]-(n)
            WHERE $rel_type IS NULL OR type(last(relationships(path))) = $rel_type
            WITH {_ID} AS id, coalesce(n.name, n.icdCode) AS title,
                 type(last(relationships(path))) AS relation,
                 min(length(path)) AS distance
            WHERE id IS NOT NULL
            RETURN id, title, relation, distance ORDER BY distance, id
        """
        rows = await self._run(query, id=start_id, rel_type=relation_type)
        return [
            GraphNode(
                id=r["id"], title=r.get("title"),
                relation=r.get("relation"), distance=int(r["distance"]),
            )
            for r in rows
        ]

    async def neighbors(self, node_id: str, relation_type: str | None) -> list[GraphNode]:
        return await self.navigate(node_id, relation_type, hops=1)

    async def close(self) -> None:
        await self._driver.close()


# ── Corpus loading ────────────────────────────────────────────────────────────

_CORPUS_CYPHER = """
MATCH (d:Diagnosis)
RETURN d.icdCode AS id, 'diagnosis' AS node_type, d.name AS name,
       d.clinicalDescription AS text
UNION ALL
MATCH (p:Procedure)
RETURN p.procCode AS id, 'procedure' AS node_type, p.name AS name,
       p.procedureDescription AS text
UNION ALL
MATCH (pt:Patient)
RETURN pt.patientId AS id, 'patient' AS node_type, pt.name AS name,
       pt.summary AS text
"""


async def _load_corpus(docstore: DuckDBDocStore, graph_conn) -> dict[str, dict]:
    """Mirror the narrative fields into the docstore, keyed by the index ids."""
    rows = graph_conn.query(_CORPUS_CYPHER) or []
    corpus: dict[str, dict] = {}
    for row in rows:
        if not row["id"] or not row["text"]:
            continue
        rec = dict(row)
        corpus[rec["id"]] = rec
        await docstore.add_document(rec["id"], rec["text"], rec["node_type"])
        await docstore.add_chunk(
            rec["id"], rec["id"], rec["text"], 0,
            metadata={"node_type": rec["node_type"], "name": rec["name"]},
        )
    return corpus


def _show(title: str, results, corpus: dict[str, dict]) -> None:
    print(f"\n  {title}")
    if not results:
        print("    (no results)")
        return
    print(f"    {'rank':<5}{'id':<10}{'score':>10}  {'source':<12}name")
    for i, r in enumerate(results, 1):
        name = (corpus.get(r.id) or {}).get("name", "?")
        print(f"    {i:<5}{r.id:<10}{r.score:>10.6f}  {r.source:<12}{name}")


async def _trace_expansion(
    retriever: MultiRetriever,
    indexer,
    qv: dict[str, np.ndarray],
    corpus: dict[str, dict],
    args: argparse.Namespace,
) -> None:
    """Report the candidate pool before and after expansion.

    Reproduces the two passes of ``MultiRetriever.search()`` explicitly so the
    enlargement of the candidate set — the whole point of the graph leg — is
    visible as a number rather than inferred from the ranking.
    """
    fetch_k = max(args.k, args.seed_k)
    pool: set[str] = set()
    for name, idx in retriever.indexes.items():
        pool.update(h.id for h in idx.search(qv[name], k=fetch_k))

    seeds = [r.id for r in await retriever.search(
        qv, k=args.seed_k, use_graph=False)][: args.seed_k]
    expanded = await retriever.graph.expand(seed_ids=seeds, hops=args.hops)
    expanded_ids = {i for i, _ in expanded}
    new_ids = sorted(expanded_ids - pool)

    rescored = 0
    for name, idx in retriever.indexes.items():
        if new_ids:
            rescored += len(idx.search(qv[name], k=len(new_ids), allowed_ids=new_ids))

    print(
        f"\n    pool before expansion: {len(pool)}"
        f" | seeds: {seeds}"
        f" | reached by BFS: {len(expanded_ids)}"
        f" | new to pool: {len(new_ids)}"
        f" | re-scored by a dense index: {rescored}"
        f" | pool after: {len(pool | set(new_ids))}"
    )


async def main(args: argparse.Namespace) -> None:
    indexer = build_indexer(uri=args.uri, user=args.user, password=args.password)
    graph_conn = indexer._graph

    tmp_dir = tempfile.mkdtemp(prefix="quanta_demo_")
    settings = QuantaSettings(
        DOCSTORE_BACKEND="duckdb",
        DUCKDB_PATH=os.path.join(tmp_dir, "demo.duckdb"),
        BM25_BACKEND="tantivy",
        TANTIVY_INDEX_PATH=os.path.join(tmp_dir, "tantivy"),
    )

    docstore = DuckDBDocStore(settings)
    await docstore.init()
    corpus = await _load_corpus(docstore, graph_conn)
    print(f"[corpus] {len(corpus)} entities mirrored into the docstore")

    bm25 = TantivyBM25(settings)
    bm25.add_bulk([(cid, rec["text"]) for cid, rec in corpus.items()])
    bm25.commit()

    graph = MedicalGraphBackend(uri=args.uri, user=args.user, password=args.password)

    retriever = MultiRetriever(
        indexes={
            "diagnoses": indexer._indexes["diagnoses"],
            "procedures": indexer._indexes["procedures"],
            "patients": indexer._indexes["patients"],
        },
        docstore=docstore,
        graph=graph,
        bm25=bm25,
        dense_weight=0.5,
        bm25_weight=0.3,
        graph_weight=0.0,   # the graph expands the pool; it does not score
    )

    queries = [
        "Ασθενής με ινσουλινοαντίσταση και πολυουρία",   # type 2 diabetes, in Greek
        "chronic kidney disease requiring dialysis",       # cross-lingual
        "καρδιακή ανεπάρκεια και δύσπνοια",                # heart failure
    ]

    for q in queries:
        print("\n" + "=" * 78)
        print(f"QUERY: {q}")
        vec = indexer.embed(q)
        qv = {"diagnoses": vec, "procedures": vec, "patients": vec}

        t0 = time.perf_counter()
        dense = await retriever.search(qv, k=args.k, use_graph=False)
        t_dense = (time.perf_counter() - t0) * 1000
        _show(f"1. dense only  [{t_dense:.0f} ms]", dense, corpus)

        t0 = time.perf_counter()
        hybrid = await retriever.search(qv, k=args.k, use_graph=False, query_text=q)
        t_hybrid = (time.perf_counter() - t0) * 1000
        _show(f"2. dense + BM25  [{t_hybrid:.0f} ms]", hybrid, corpus)

        t0 = time.perf_counter()
        full = await retriever.search(
            qv, k=args.k, use_graph=True, query_text=q,
            graph_hops=args.hops, graph_seed_k=args.seed_k,
        )
        t_full = (time.perf_counter() - t0) * 1000
        _show(f"3. dense + BM25 + graph expansion  [{t_full:.0f} ms]", full, corpus)

        await _trace_expansion(retriever, indexer, qv, corpus, args)

        before = {r.id: i for i, r in enumerate(hybrid, 1)}
        moved = [
            (r.id, before[r.id], i)
            for i, r in enumerate(full, 1)
            if r.id in before and before[r.id] != i
        ]
        admitted = [r.id for r in full if r.id not in before]
        if admitted:
            print("    entered top-k after expansion: " + ", ".join(
                f"{i} ({(corpus.get(i) or {}).get('name', '?')})" for i in admitted))
        if moved:
            print("    rank changes from re-scoring: " + ", ".join(
                f"{i}: {a}->{b}" for i, a, b in moved))
        if not admitted and not moved:
            print("    expansion changed neither membership nor order of the top-k")

    await docstore.close()
    await graph.close()
    graph_conn.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="quantapass")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--seed-k", type=int, default=2, dest="seed_k")
    asyncio.run(main(parser.parse_args()))
