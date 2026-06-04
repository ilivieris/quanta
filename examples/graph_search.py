"""
graph_search.py — Graph-augmented retrieval with Quanta.

Demonstrates how combining dense vector search with a Neo4j knowledge graph
surfaces documents that pure ANN search would rank low or miss entirely.

Use case: a small Greek legal document corpus covering GDPR / data protection.
Five documents are connected by legal citation, interpretation, and amendment
relationships.  A query about "controller obligations" naturally scores the
core GDPR statutes highly.  Graph traversal then promotes related court
decisions and circulars that dense search alone would rank poorly.

Run:
    python examples/graph_search.py

Required .env variables:
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=your_password
    DOCSTORE_BACKEND=duckdb
    DUCKDB_PATH=./examples/graph_search_example.duckdb

    # QuantaSettings validates POSTGRES_* even for DuckDB backend;
    # set these to any placeholder — they are NOT used here.
    POSTGRES_USER=_unused_
    POSTGRES_PASSWORD=_unused_

Optional:
    RUN_CLEANUP=false   # set to true to delete test data on next run
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import numpy as np
from dotenv import load_dotenv

from quanta import (
    MultiRetriever,
    Neo4jGraph,
    QuantaIndex,
    QuantaSettings,
    RetrievalResult,
    get_docstore,
)
from quanta.docstore import DocStoreBackend

# ── Document catalogue ─────────────────────────────────────────────────────────

DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "doc_001",
        "title": "Ν. 4624/2019 — Προστασία δεδομένων προσωπικού χαρακτήρα",
        "doc_type": "law",
        "content": (
            "Ο νόμος 4624/2019 ενσωματώνει τον Κανονισμό (ΕΕ) 2016/679 στο "
            "ελληνικό δίκαιο. Ορίζει τις υποχρεώσεις του υπευθύνου επεξεργασίας "
            "και τα δικαιώματα των υποκειμένων των δεδομένων."
        ),
    },
    {
        "id": "doc_002",
        "title": "ΣτΕ 1234/2021 — Απόφαση για παραβίαση GDPR",
        "doc_type": "decision",
        "content": (
            "Το Συμβούλιο της Επικρατείας έκρινε παράνομη την επεξεργασία "
            "προσωπικών δεδομένων χωρίς νόμιμη βάση. Επικαλείται τον Ν. 4624/2019 "
            "και την Οδηγία ΕΕ 2016/679 ως νομικές βάσεις κρίσης."
        ),
    },
    {
        "id": "doc_003",
        "title": "Εγκύκλιος ΑΠΔΠΧ 2022 — Ερμηνεία ΣτΕ 1234/2021",
        "doc_type": "circular",
        "content": (
            "Η ΑΠΔΠΧ εκδίδει ερμηνευτική εγκύκλιο για την εφαρμογή της απόφασης "
            "ΣτΕ 1234/2021. Παρέχει κατευθύνσεις για τη νόμιμη επεξεργασία "
            "ειδικών κατηγοριών δεδομένων προσωπικού χαρακτήρα."
        ),
    },
    {
        "id": "doc_004",
        "title": "Ν. 3471/2006 — Προστασία δεδομένων στις τηλεπικοινωνίες",
        "doc_type": "law",
        "content": (
            "Ο νόμος 3471/2006 ρυθμίζει την προστασία δεδομένων προσωπικού "
            "χαρακτήρα στις ηλεκτρονικές επικοινωνίες. Τροποποιήθηκε ώστε να "
            "εναρμονιστεί με τον Ν. 4624/2019."
        ),
    },
    {
        "id": "doc_005",
        "title": "Οδηγία ΕΕ 2016/679 — GDPR",
        "doc_type": "directive",
        "content": (
            "Ο Γενικός Κανονισμός Προστασίας Δεδομένων (GDPR) θεσπίζει κανόνες "
            "για την επεξεργασία δεδομένων προσωπικού χαρακτήρα. Ορίζει τις αρχές "
            "νομιμότητας και τα δικαιώματα των υποκειμένων."
        ),
    },
]

# Directed relationships: (source_id, target_id, RELATION_TYPE)
EDGES: list[tuple[str, str, str]] = [
    ("doc_002", "doc_001", "CITES"),
    ("doc_002", "doc_005", "CITES"),
    ("doc_003", "doc_002", "INTERPRETS"),
    ("doc_001", "doc_004", "AMENDS"),
]

# ── Embeddings ─────────────────────────────────────────────────────────────────
# Deterministic 8-dimensional fake embeddings reflecting semantic similarity
# to the query "υποχρεώσεις υπευθύνου επεξεργασίας" (GDPR controller obligations).
#
# In production, replace with your embedding model output.
#
# Designed approximate cosine similarities vs. query:
#   doc_001, doc_005  →  ~0.98  (high — core GDPR instruments)
#   doc_004           →  ~0.73  (medium — related but narrower scope)
#   doc_002, doc_003  →  ~0.37  (low — secondary / procedural documents)

_DIM = 8


def _unit(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    return arr / float(np.linalg.norm(arr))


# Query vector: close to doc_001 and doc_005.
QUERY_VEC: np.ndarray = _unit([1.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])

EMBEDDINGS: dict[str, np.ndarray] = {
    "doc_001": _unit([1.0, 0.4, 0.2, 0.1, 0.2, 0.0, 0.0, 0.0]),  # cosine ≈ 0.98
    "doc_005": _unit([0.9, 0.5, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0]),  # cosine ≈ 0.97
    "doc_004": _unit([0.5, 0.3, 0.6, 0.4, 0.0, 0.0, 0.3, 0.0]),  # cosine ≈ 0.73
    "doc_002": _unit([0.2, 0.1, 0.8, 0.6, 0.4, 0.0, 0.0, 0.0]),  # cosine ≈ 0.37
    "doc_003": _unit([0.1, 0.2, 0.6, 0.8, 0.3, 0.0, 0.0, 0.1]),  # cosine ≈ 0.33
}

# ── Graph path explanations (static — mirrors the EDGES above) ─────────────────

_GRAPH_PATHS: dict[str, str] = {
    "doc_002": "hop=1 from doc_001 via CITES",
    "doc_003": "hop=2 from doc_001 via doc_002 → INTERPRETS",
    "doc_004": "hop=1 from doc_001 via AMENDS",
}


# ── Setup helpers ──────────────────────────────────────────────────────────────

async def _build_graph(graph: Neo4jGraph) -> None:
    """Upsert Document nodes and edges in Neo4j (MERGE — safe to re-run)."""
    print("\n[GRAPH] Upserting document nodes ...")
    for doc in DOCUMENTS:
        await graph.upsert_node(
            doc["id"],
            {"title": doc["title"], "doc_type": doc["doc_type"]},
        )
        print(f"  MERGE :Document {{id: {doc['id']!r}}}")

    print("[GRAPH] Upserting edges ...")
    for src, tgt, rel in EDGES:
        await graph.upsert_edge(src, tgt, rel)
        print(f"  MERGE ({src})-[:{rel}]->({tgt})")


async def _populate_docstore(docstore: DocStoreBackend) -> None:
    """Add one document + one chunk per document (upserts on re-run)."""
    print("\n[DOCSTORE] Adding documents and chunks ...")
    for doc in DOCUMENTS:
        await docstore.add_document(
            id=doc["id"],
            content=doc["content"],
            doc_type=doc["doc_type"],
            metadata={"title": doc["title"]},
        )
        # Chunk id equals document id — simplifies result display for this demo.
        await docstore.add_chunk(
            id=doc["id"],
            document_id=doc["id"],
            content=doc["content"],
            chunk_index=0,
            metadata={"title": doc["title"], "doc_type": doc["doc_type"]},
        )
        print(f"  + {doc['id']}: {doc['title'][:55]}")


def _build_index() -> QuantaIndex:
    """Create a QuantaIndex with the deterministic 8-dim embeddings."""
    print("\n[INDEX] Building vector index (dim=8, bit_width=4) ...")
    idx = QuantaIndex(name="legal_text", dim=_DIM, bit_width=4)
    doc_ids = [d["id"] for d in DOCUMENTS]
    vectors = np.stack([EMBEDDINGS[did] for did in doc_ids]).astype(np.float32)
    idx.add(vectors, doc_ids)
    print(f"  Indexed {len(idx)} vectors")
    return idx


# ── Output helpers ─────────────────────────────────────────────────────────────

def _sep(text: str = "") -> None:
    width = max(len(text) + 2, 52)
    print("─" * width)
    if text:
        print(f" {text}")
        print("─" * width)


def _print_results(label: str, results: list[RetrievalResult]) -> None:
    print(f"\n[{label}]")
    for rank, r in enumerate(results, 1):
        title = r.metadata.get("title", r.id)
        source_col = f"source={r.source}"
        print(f"  {rank}. {r.id:<10}  score={r.score:.3f}  {source_col:<22}  {title}")


def _print_comparison(
    results_a: list[RetrievalResult],
    results_b: list[RetrievalResult],
) -> None:
    ids_a = {r.id for r in results_a}
    ids_b = {r.id for r in results_b}

    newly_found = ids_b - ids_a
    graph_confirmed = {r.id for r in results_b if "graph" in r.source}
    highlighted = newly_found | graph_confirmed

    print("\nGraph expansion surfaced:")
    if not highlighted:
        print("  (no additional documents reached via graph traversal)")
        return

    # Show newly found first, then confirmed
    for doc_id in sorted(highlighted, key=lambda x: (x not in newly_found, x)):
        title = next((d["title"] for d in DOCUMENTS if d["id"] == doc_id), doc_id)
        path = _GRAPH_PATHS.get(doc_id, "reached via graph traversal")
        tag = " [NEW]" if doc_id in newly_found else ""
        print(f"  • {doc_id} ({title}){tag}: {path}")


# ── Cleanup ────────────────────────────────────────────────────────────────────

async def _cleanup(graph: Neo4jGraph, docstore: DocStoreBackend) -> None:
    print("\n[CLEANUP] Removing test data ...")
    for doc in DOCUMENTS:
        await graph.delete_node(doc["id"])
        await docstore.delete_document(doc["id"])
    print("  Done — Neo4j nodes and docstore records removed.")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    # ── 1. SETUP ──────────────────────────────────────────────────────────────
    load_dotenv()

    # QuantaSettings validates POSTGRES_* even when DOCSTORE_BACKEND=duckdb.
    # setdefault avoids overriding real credentials that may already be set.
    os.environ.setdefault("POSTGRES_USER", "_unused_")
    os.environ.setdefault("POSTGRES_PASSWORD", "_unused_")

    cfg = QuantaSettings()
    run_cleanup = os.environ.get("RUN_CLEANUP", "false").lower() == "true"

    if not cfg.graph_configured:
        print(
            "ERROR: Neo4j is not configured.\n"
            "  Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD in your .env file."
        )
        sys.exit(1)

    try:
        graph = Neo4jGraph(
            uri=cfg.NEO4J_URI,           # type: ignore[arg-type]
            user=cfg.NEO4J_USER,         # type: ignore[arg-type]
            password=cfg.NEO4J_PASSWORD, # type: ignore[arg-type]
            database=cfg.NEO4J_DATABASE,
        )
    except Exception as exc:
        msg = str(exc)
        if "neo4j driver" in msg or "pip install" in msg:
            print(f"Missing dependency: {exc}\n  Run: pip install quanta[neo4j]")
        else:
            print(f"Could not connect to Neo4j. Is it running? Check NEO4J_URI\n  Detail: {exc}")
        sys.exit(1)

    try:
        await graph._driver.verify_connectivity()
        print(f"[SETUP] Connected to Neo4j at {cfg.NEO4J_URI}")
    except Exception as exc:
        print(f"Could not connect to Neo4j. Is it running? Check NEO4J_URI\n  Detail: {exc}")
        await graph.close()
        sys.exit(1)

    docstore = get_docstore(cfg)
    await docstore.init()
    print(f"[SETUP] Connected to docstore ({cfg.DOCSTORE_BACKEND}: {cfg.DUCKDB_PATH})")

    idx: QuantaIndex | None = None

    try:
        # ── 2. BUILD THE GRAPH ─────────────────────────────────────────────────
        await _build_graph(graph)

        # ── 3. ADD DOCUMENTS TO DOCSTORE ──────────────────────────────────────
        await _populate_docstore(docstore)

        # ── 4. CREATE VECTOR INDEX ─────────────────────────────────────────────
        idx = _build_index()

        # ── 5 & 6. QUERIES AND RESULTS ─────────────────────────────────────────
        retriever = MultiRetriever(
            indexes={"legal_text": idx},
            docstore=docstore,
            graph=graph,
        )

        query_text = "υποχρεώσεις υπευθύνου επεξεργασίας"

        print()
        _sep(f'QUERY: "{query_text}"')

        # Query A — dense vector search only
        results_a = await retriever.search(
            query_vectors={"legal_text": QUERY_VEC},
            k=5,
            use_graph=False,
        )
        _print_results("A] Dense-only results", results_a)

        # Query B — dense search + 2-hop graph expansion from top-3 seeds
        results_b = await retriever.search(
            query_vectors={"legal_text": QUERY_VEC},
            k=5,
            use_graph=True,
            graph_hops=2,
            graph_seed_k=3,
        )
        _print_results("B] Dense + Graph results", results_b)

        _print_comparison(results_a, results_b)

    finally:
        # ── 7. CLEANUP (optional) ──────────────────────────────────────────────
        if run_cleanup and idx is not None:
            await _cleanup(graph, docstore)
        elif not run_cleanup:
            print(
                "\n[INFO] Set RUN_CLEANUP=true in .env to remove test data on the next run."
            )

        await graph.close()
        await docstore.close()


if __name__ == "__main__":
    asyncio.run(main())
