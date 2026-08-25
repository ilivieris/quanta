# Quanta — Technical Overview

## The Problem

Standard RAG pipelines break in predictable ways as corpora grow:

**Float32 vectors eat RAM.** A corpus of 1 million 768-dimensional embeddings
requires ~2.86 GB in float32. At 10 million documents it becomes 28.6 GB —
exceeding the RAM of most application servers before the rest of the stack gets
any memory.

**Dense-only retrieval misses keyword matches.** Embedding models compress
meaning into a fixed-dimensional space. Rare terms, product codes, legal article
numbers, and proper nouns that appear infrequently in training data are poorly
represented. A query for "Regulation 2016/679 Article 5" may not surface the
exact document if its embedding is similar to many other regulatory texts.

**No way to exploit known document relationships.** A corpus of legal documents
has explicit structure: laws cite other laws, court decisions cite statutes,
circulars interpret decisions. Dense search has no knowledge of these
relationships. A document written in procedural language may rank low for a
substantive query even though it is the most relevant document in the corpus
given its structural position.

**Post-filter recall loss.** Many vector databases filter after ANN search:
they retrieve the top-K vectors, then discard those that don't match the
metadata predicate. With strict filters and large K, a significant fraction of
candidates are discarded. The effective K shrinks and recall drops.

---

## Design Decisions

### 1. TurboQuant Compression

Quanta stores vectors using turbovec's `IdMapIndex`, which quantises float32
vectors to integers of configurable bit-width (1, 2, 4, or 8 bits per
dimension). At 4-bit quantisation, the memory formula is:

```
RAM (bytes) = n_vectors × dim × 4 / 8 = n_vectors × dim / 2
```

For 1 million 768-dimensional vectors:
- float32 baseline: `1,000,000 × 768 × 4 = 2,861 MB (~2.86 GB)`
- 4-bit quantised: `1,000,000 × 768 / 2 = 368 MB (~0.36 GB)`

This is an **8× memory reduction** without a separate ANN server. The index
lives in-process; there is no network round-trip.

The trade-off: 4-bit quantisation introduces recall loss. This loss is small in
practice for high-dimensional vectors (≥256 dimensions) but is corpus- and
query-dependent. Use `bit_width=8` when recall is critical and memory allows.

Quanta does not use HNSW or IVF approximate indexing. turbovec's `IdMapIndex`
performs exact search over quantised vectors. This means search time scales
linearly with corpus size. For corpora up to a few hundred thousand vectors,
this is fast enough for online retrieval. For corpora in the tens of millions,
consider a pre-filtering step with `allowed_ids` to constrain the search space.

### 2. Pre-filter Constrained Query

When `filters` are passed to `MultiRetriever.search()`, Quanta resolves them
to a set of allowed chunk IDs before touching the vector index:

```
filter_chunks({"topic": "AI", "year": 2024})
    → [chunk IDs that match]
    → QuantaIndex.search(query, k=k, allowed_ids=[...])
```

This is pre-filter search, not post-filter. The ANN search only considers
vectors in the allowed set. PostgreSQL uses a GIN index on the `metadata JSONB`
column; DuckDB uses `json_extract_string`. Both resolve the filter before the
vector search.

The recall impact: pre-filtering a corpus of 100,000 chunks to a subset of
10,000 means the ANN search operates over 10,000 candidates instead of 100,000.
If the filter is highly selective, this can reduce recall. The mitigation is to
increase `k` relative to the expected filter selectivity.

### 3. Rank Fusion, Not Score Fusion

The three legs produce scores on incomparable scales. Cosine similarity is
bounded in `[-1, 1]` and, for a well-trained embedding model over a topical
corpus, is compressed into a narrow band near the top. BM25 is unbounded and
grows with query length and corpus statistics. Inverse hop distance takes values
in `{1, 1/2, 1/3, ...}` and is a property of the graph, not of the query.

Normalising these onto a common range does not make them comparable — it only
hides the incompatibility behind an arbitrary affine map. Min-max normalisation
is worse than arbitrary: it is *query-dependent*. The same document with the same
cosine similarity receives a different normalised score depending on how good
the rest of that query's result list happened to be, and a leg whose top hits
are all nearly tied has its tiny score differences stretched to fill `[0, 1]`.

Quanta therefore fuses on **rank**, not score, using weighted Reciprocal Rank
Fusion (Cormack et al., SIGIR 2009):

```
score(d) = SUM over legs of  w_leg / (rrf_k + rank_leg(d))
```

where `rank_leg(d)` is the 1-based position of `d` in that leg's ranked list and
`rrf_k` (default 60) damps the influence of the very top ranks. Documents absent
from a leg simply contribute nothing from it. The only thing a leg has to agree
on is an ordering, which every retrieval signal can produce by construction.

Leg weights are `dense_weight / n_active_indexes` per dense index,
`bm25_weight` for the BM25 leg, and `graph_weight` for the graph leg. Under RRF
only the *ratios* between them matter; scaling all weights by a constant leaves
the ranking unchanged, so no renormalisation step is needed.

### 4. Graph as Candidate Expander

The graph's role is frequently misunderstood. It is not a relevance scorer.

Retrieval in Quanta happens in two passes. The first pass ranks the corpus by
content: dense search over each index, plus BM25 when a query string is given.
The top `graph_seed_k` documents of the preliminary fusion become seeds, and
`Neo4jGraph.expand()` runs a bounded BFS from them, returning document ids
within `graph_hops` steps.

Those ids are then **re-scored, not injected**. For every dense index, Quanta
issues a second search constrained to the expanded ids via the `allowed_ids`
allowlist, and merges the resulting hits into that index's ranked list before
the final fusion. An expanded document therefore competes on its own cosine
similarity to the query, exactly like a document the first pass had returned
directly. What the graph changed is *which documents were considered*, not
*how well they rank*:

```
graph.expand(seeds)          →  candidate ids
index.search(q, allowed_ids) →  real cosine scores for those ids
weighted RRF                 →  final ranking
```

This is why the default is `graph_weight=0.0`. At that setting the graph
contributes no term to the score at all, and the entire benefit is recall: the
pool now contains documents that ranked below the first pass's `k` cutoff but
are structurally adjacent to a strong hit.

Raising `graph_weight` above zero adds the graph as a further fusion leg, ranked
by hop distance. This is a tie-breaker budget, and it is the *only* way for an
expanded document that has no vector in any index to appear in the results at
all. It should be spent carefully: because RRF compresses each leg into
`w / (60 + rank)`, a graph weight comparable to `dense_weight` will routinely
promote 1-hop neighbours over content-relevant documents.

The graph surfaces documents you know are structurally related because you built
the edges. Dense search surfaces documents that are semantically similar. When a
court decision uses different terminology than the statute it cites, dense
search will miss the citation relationship. The graph closes that gap — not by
scoring, but by inclusion.

Pipeline (expanded):

```
# Pass 1: content legs
for each active index:
    leg[name] = {id: raw_score for hits of index.search(q, k=fetch_k)}
bm25_leg     = {id: raw_score for hits of bm25.search(query_text, k)}

# Preliminary fusion picks the seeds (graph leg empty)
seeds = top graph_seed_k of RRF(dense legs, bm25_leg)

# Pass 2: candidate enlargement + re-scoring
graph_leg = {id: 1/hop_distance for id in graph.expand(seeds, hops)}
new_ids   = graph_leg keys not already found by any dense leg
for each active index:
    merge index.search(q, allowed_ids=new_ids) into leg[name]

# Final fusion (graph leg participates only when graph_weight > 0)
score(d) = SUM over legs of  w_leg / (rrf_k + rank_leg(d))
```

### 5. Dual Docstore

Quanta supports two docstore backends with identical async interfaces:

`PostgresDocStore` — production-grade, concurrent, indexed. Uses raw `asyncpg`
with a configurable connection pool. The `metadata JSONB` column has a GIN
index that makes `filter_chunks()` fast even at large scale. Tables are created
idempotently at `init()` time. Suitable for production APIs with concurrent
readers and writers.

`DuckDBDocStore` — zero-setup, single-process, embedded. DuckDB is bundled with
Quanta's core dependencies. Synchronous DuckDB calls are dispatched through a
single-threaded `ThreadPoolExecutor` so they integrate cleanly with asyncio.
Suitable for local development, CI, batch indexing pipelines, and single-process
applications. Does not support concurrent writes from multiple processes.

The backend is selected explicitly via `DOCSTORE_BACKEND` (or by instantiating
the class directly), not auto-detected from available services. Explicit
configuration is production-safe: the application fails loudly at startup if the
configured backend is unreachable, rather than silently falling back to a
different backend.

### 6. Optional Everything — Null Object Pattern

Every pluggable component has a null implementation:

| Component | Active class    | Null class   | Activation condition         |
|-----------|-----------------|--------------|------------------------------|
| Graph     | `Neo4jGraph`    | `NullGraph`  | `NEO4J_URI` configured       |
| BM25      | `TantivyBM25`   | `NullBM25`   | `BM25_BACKEND=tantivy`       |
| Cache     | `RedisCache`    | `NullCache`  | `REDIS_HOST` configured      |

Null implementations are not stubs; they are production no-ops. `NullGraph`
returns an empty list from every method. `NullBM25` silently drops all adds and
returns nothing from search. `NullCache` always misses. Their presence means:

- `MultiRetriever` has no conditional branches for optional components. It
  always calls `self._graph.expand()`, `self._bm25.search()`, and
  `self.cache.get()`. The null objects handle the "not configured" case.
- You can start with zero external services and add components incrementally
  without changing application code — only configuration changes.
- Tests run without external services. Every component can be replaced with
  its null object or a mock.

---

## Component Map

| Component       | Technology        | Optional | Fallback       |
|-----------------|-------------------|----------|----------------|
| Vector index    | turbovec          | No       | —              |
| Docstore        | asyncpg / DuckDB  | No       | —              |
| Graph           | Neo4j (Bolt v5)   | Yes      | `NullGraph`    |
| BM25            | tantivy-py        | Yes      | `NullBM25`     |
| Embedding cache | Redis             | Yes      | `NullCache`    |
| RAG framework   | LlamaIndex        | Yes      | Direct API     |

---

## Retrieval Pipeline (Step by Step)

**Full pipeline with all components active:**

1. **Metadata pre-filter** — if `filters` are supplied, `filter_chunks()` is
   called on the docstore. The result is a set of `allowed_ids`. If the filter
   matches zero chunks, `search()` returns immediately with an empty list.

2. **Determine active legs** — BM25 is active only when the backend is not a
   `NullBM25` *and* `query_text` was supplied. The graph is active only when the
   backend is not a `NullGraph` *and* `use_graph=True`. When the graph is
   active, the dense legs fetch `max(k, graph_seed_k)` hits so there are enough
   high-quality seeds.

3. **Per-index vector search** — for each active `QuantaIndex`, `search()` is
   called with the query vector and `allowed_ids` (if any). Each index keeps its
   own ranked list of `(id, raw_score)`; raw scores are never mixed across legs.

4. **BM25 search** — if active, `TantivyBM25.search(query_text, k)` produces a
   second ranked list.

5. **Graph expansion** — a preliminary weighted RRF over the content legs picks
   the top `graph_seed_k` seeds. `Neo4jGraph.expand(seeds, hops=graph_hops)`
   runs a Cypher BFS returning `(doc_id, 1/dist)` pairs for non-seed neighbours.
   Expanded IDs that no dense leg had found are then **re-scored**: each dense
   index runs a second `search(query, allowed_ids=new_ids)` and the resulting
   hits are merged into that index's ranked list.

6. **Weighted RRF** — every leg contributes `w_leg / (rrf_k + rank_leg(d))`.
   Dense legs carry `dense_weight / n_active_indexes` each, BM25 carries
   `bm25_weight`, and the graph leg (ranked by hop distance) carries
   `graph_weight` — zero by default, so the graph normally contributes no score.

7. **Sort and hydrate** — candidates are sorted by descending fused score; the
   top `k` IDs are kept. `docstore.get_chunks(top_ids)` retrieves content and
   metadata in a single query. Results are assembled as `RetrievalResult`
   dataclasses with `source` tags.

**Unconstrained path (no filters):** steps 2–7 only.
**Dense-only path (`use_graph=False`, no BM25):** steps 2, 3, 6, 7.

---

## Benchmarks

turbovec recall and QPS benchmarks are published in the turbovec repository.
Quanta-level end-to-end benchmarks (including docstore hydration and graph
expansion latency) have **not yet been benchmarked**.

Memory reduction from 4-bit quantisation is computed analytically:
`n × dim × 4 / 8` bytes vs `n × dim × 4` bytes for float32, yielding an 8×
reduction. This is confirmed by the demo at
[examples/turbovec_4bit_quantization_demo.py](examples/turbovec_4bit_quantization_demo.py).

---

## Limitations

**Single-process writes (DuckDB).** The DuckDB docstore uses a single file
connection and a single-threaded executor. Concurrent writes from multiple
processes are not supported. Use PostgreSQL for multi-process or multi-worker
deployments.

**No distributed index.** A `QuantaIndex` lives in one process's memory. There
is no built-in sharding, replication, or distributed search. For corpora
requiring multiple machines, you would need to implement query fan-out and
result merging yourself.

**Tantivy sweet spot is under ~50K documents.** The tantivy-py bindings are
well-suited for small to medium corpora. For larger BM25 indexes, a dedicated
search engine (Elasticsearch, Typesense) is more appropriate.

**Graph must be user-provided.** Quanta does not extract relationships from
document content. You must supply nodes and edges via `upsert_node()` and
`upsert_edge()`. This is a feature — it means your graph encodes domain
knowledge rather than noisy auto-extracted relationships — but it requires
upfront work.

**No built-in embedding model.** You must embed your documents and queries
before passing vectors to Quanta. This is intentional: embedding model choice
is application-specific and changes frequently. Quanta is model-agnostic.

**graph_seed_k sensitivity.** Too few seeds and the graph expansion misses
valuable paths. Too many seeds and distant, loosely-related documents enter the
pool. The optimal value depends on graph density and corpus size. There is no
automatic tuning.

**Linear search time.** `QuantaIndex` performs exact search over quantised
vectors. Search time is `O(n × dim)`. For corpora above ~500K vectors, use
`allowed_ids` filtering or consider an approximate index.

---

## Comparison

Comparison against common alternatives as of the knowledge cutoff. Capabilities
may have changed.

| Capability                | Quanta   | FAISS    | Qdrant   | Chroma   | Milvus   |
|---------------------------|----------|----------|----------|----------|----------|
| Compressed vectors        | ✅ 4-bit  | ✅ PQ/SQ  | ✅        | ❌        | ✅        |
| Metadata pre-filter ANN   | ✅        | ❌        | ✅        | ⚠️ partial| ✅        |
| Graph retrieval           | ✅        | ❌        | ❌        | ❌        | ❌        |
| BM25 hybrid search        | ✅        | ❌        | ⚠️ sparse | ❌        | ⚠️ sparse |
| Embedded (no server)      | ✅        | ✅        | ❌        | ✅        | ❌        |
| LlamaIndex native         | ✅        | ✅        | ✅        | ✅        | ✅        |

**Notes:**
- FAISS has no metadata storage or filtering; those must be implemented outside.
- Qdrant and Milvus support sparse vector hybrid search, which approximates
  BM25 but is not identical.
- Chroma supports metadata filtering but applies it as post-filter by default
  in some configurations.
- "Embedded" means the service runs in-process without a separate server.
  Quanta (DuckDB backend), FAISS, and Chroma fit this profile.

Quanta's differentiator is the combination of graph-augmented retrieval with
an embedded, compressed vector index. No other library in this table supports
graph expansion as a first-class retrieval component.
