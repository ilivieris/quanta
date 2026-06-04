# Quanta

Production-ready **hybrid search** library for Python — combines quantised
vector ANN search (turbovec) with optional Neo4j graph expansion and an async
PostgreSQL document/chunk store.

---

## Features

- **Multi-index ANN search** — query `n` named `QuantaIndex` instances in one
  call (e.g. `"text"` + `"images"`), scores normalised and merged
- **Graph expansion** — Neo4j traversal widens recall beyond pure ANN; a
  `NullGraph` fallback keeps the interface identical when Neo4j is absent
- **Chunk-level retrieval** — PostgreSQL stores documents and chunks separately;
  metadata GIN index enables pre-filtered vector search
- **Quantised vectors** — turbovec `IdMapIndex` with configurable bit-width
  (1 / 2 / 4 / 8) and xxhash-64 ID mapping
- **LlamaIndex integration** — drop-in `VectorStore` for `VectorStoreIndex`
- **Zero-config fallbacks** — `NullGraph`, `NullGraph`, missing `.env` → safe
  defaults everywhere
- **Structured logging** — ISO-8601 timestamps, stdlib only

---

## Architecture

```
 Query: {"text": vec, "images": vec}
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│                     MultiRetriever                         │
│                                                             │
│  ┌──────────────────┐     ┌──────────────────┐             │
│  │   QuantaIndex     │     │   QuantaIndex     │             │
│  │    "text"        │     │   "images"       │             │
│  │  turbovec ANN    │     │  turbovec ANN    │             │
│  │  xxhash-64 IDs   │     │  xxhash-64 IDs   │             │
│  └────────┬─────────┘     └────────┬─────────┘             │
│           │  min-max norm           │  min-max norm         │
│           └────────────┬────────────┘                       │
│                        │  × (dense_weight / n_indexes)      │
│                        │  Σ per-index contributions         │
│                        │                                    │
│             top graph_seed_k IDs                            │
│                        │                                    │
│                ┌───────▼────────┐                           │
│                │  GraphBackend  │                           │
│                │  Neo4jGraph /  │  + graph_weight × g_score │
│                │  NullGraph     │  (BFS up to hops=N)       │
│                └───────┬────────┘                           │
│                        │  final top-k IDs                   │
│                ┌───────▼────────┐                           │
│                │    DocStore    │  hydrate chunks            │
│                │  PostgreSQL    │  (content, metadata,       │
│                │  asyncpg       │   document_id)             │
│                └────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
 list[RetrievalResult(id, score, source, content, metadata)]
```

---

## Installation

```bash
# Core — vector search + document store
pip install Quanta

# With Neo4j graph support
pip install "quanta[neo4j]"

# With LlamaIndex integration
pip install "quanta[llama-index]"

# Everything
pip install "Quanta[all]"

# Development
pip install "Quanta[dev]"
```

---

## Docker quickstart

```bash
cp .env.example .env
# Edit .env — set POSTGRES_USER and POSTGRES_PASSWORD at minimum

# PostgreSQL only
docker compose up -d postgres

# PostgreSQL + Neo4j (graph-augmented retrieval)
docker compose --profile graph up -d
```

Services are healthy when their `healthcheck` passes:

```bash
docker compose ps          # all services → healthy
```

---

## Environment setup

Copy `.env.example` to `.env` and fill in your values:

```bash
# PostgreSQL (required)
POSTGRES_USER=tsuser
POSTGRES_PASSWORD=changeme

# Neo4j (optional — only needed for graph expansion)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=changeme_neo4j
```

All remaining variables have sensible defaults — see the
[Configuration reference](#configuration-reference) below.

---

## Usage examples

### a) Basic: create a QuantaIndex, add vectors, search

```python
import numpy as np
from Quanta import QuantaIndex

# Create a 768-dimensional index (e.g. for sentence-transformers)
idx = QuantaIndex(name="articles", dim=768, bit_width=4, index_dir="./indexes")

# Add vectors
vectors = np.random.rand(100, 768).astype(np.float32)
ids = [f"chunk-{i}" for i in range(100)]
idx.add(vectors, ids)

# Search
query = np.random.rand(768).astype(np.float32)
results = idx.search(query, k=5)

for r in results:
    print(r.id, r.score)

# Persist to disk
idx.save()

# Reload later
idx2 = QuantaIndex.load("articles", index_dir="./indexes")
```

### b) Full pipeline: DocStore + MultiRetriever

```python
import asyncio
import numpy as np
from Quanta import QuantaSettings, DocStore, QuantaIndex, MultiRetriever, NullGraph

async def main():
    settings = QuantaSettings()  # reads from .env

    # Initialise the document store
    docstore = DocStore(settings)
    await docstore.init()

    # Create a vector index
    idx = QuantaIndex(name="text", dim=768, index_dir="./indexes")
    idx.initialize(dimension=768)   # or just construct with dim=

    # Build the retriever (NullGraph = no Neo4j)
    retriever = MultiRetriever(
        indexes={"text": idx},
        docstore=docstore,
        graph=NullGraph(),
        dense_weight=1.0,
    )

    # Add chunks directly via docstore + index
    await docstore.add_document("doc-1", "Original document text", "text")
    await docstore.add_chunk(
        id="chunk-1",
        document_id="doc-1",
        content="First chunk of the document.",
        chunk_index=0,
        metadata={"year": 2024, "source": "arxiv"},
    )
    idx.add(np.random.rand(1, 768).astype(np.float32), ["chunk-1"])

    # Search
    results = await retriever.search(
        query_vectors={"text": np.random.rand(768).astype(np.float32)},
        k=5,
    )
    for r in results:
        print(f"[{r.score:.3f}] [{r.source}] {r.content}")

    await docstore.close()

asyncio.run(main())
```

### c) With Neo4j graph expansion

```python
import asyncio
import numpy as np
from Quanta import (
    QuantaSettings, DocStore, QuantaIndex,
    MultiRetriever, get_graph_backend, Neo4jGraph,
)

async def main():
    settings = QuantaSettings()   # NEO4J_URI must be set in .env
    docstore = DocStore(settings)
    await docstore.init()

    idx = QuantaIndex(name="text", dim=768, index_dir="./indexes")

    # Factory returns Neo4jGraph if NEO4J_URI is set, else NullGraph
    graph = get_graph_backend(settings)

    retriever = MultiRetriever(
        indexes={"text": idx},
        docstore=docstore,
        graph=graph,
        dense_weight=0.7,
        graph_weight=0.3,
    )

    # Build the graph (outside the retriever — one-time indexing step)
    if isinstance(graph, Neo4jGraph):
        graph.upsert_node("doc-1", {"title": "Paper on RAG"})
        graph.upsert_node("doc-2", {"title": "Survey on embeddings"})
        graph.upsert_edge("doc-1", "doc-2", "CITES")

    # Hybrid search: ANN + graph expansion up to 2 hops
    results = await retriever.search(
        query_vectors={"text": np.random.rand(768).astype(np.float32)},
        k=10,
        use_graph=True,
        graph_hops=2,
        graph_seed_k=5,
    )
    for r in results:
        print(f"[{r.source}] [{r.score:.3f}] {r.id}")

    # Direct graph navigation
    neighbors = retriever.navigate("doc-1", relation_type="CITES", hops=1)
    for n in neighbors:
        print(n.id, n.relation, n.distance)

    await docstore.close()
    graph.close()

asyncio.run(main())
```

### d) LlamaIndex integration

```python
import asyncio
from Quanta import QuantaSettings, DocStore, QuantaIndex, MultiRetriever, NullGraph
from quanta.integrations.llama_index import QuantaVectorStore

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

async def main():
    settings = QuantaSettings()
    docstore = DocStore(settings)
    await docstore.init()

    idx = QuantaIndex(name="llamaidx", dim=1536, index_dir="./indexes")

    retriever = MultiRetriever(
        indexes={"llamaidx": idx},
        docstore=docstore,
        graph=NullGraph(),
    )

    # Wrap as a LlamaIndex VectorStore
    store = QuantaVectorStore(
        retriever=retriever,
        index_name="llamaidx",
        embed_dim=1536,
    )

    # Use as a standard LlamaIndex storage context
    storage_ctx = StorageContext.from_defaults(vector_store=store)
    docs = [Document(text="Quanta makes hybrid search easy.")]
    li_index = VectorStoreIndex.from_documents(docs, storage_context=storage_ctx)

    # Query
    engine = li_index.as_query_engine()
    response = await engine.aquery("What does Quanta do?")
    print(response)

    await docstore.close()

asyncio.run(main())
```

---

## Configuration reference

All settings are loaded from environment variables (or a `.env` file in the
working directory). No prefix is required.

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | PostgreSQL server host |
| `POSTGRES_PORT` | `5432` | PostgreSQL server port |
| `POSTGRES_DB` | `Quanta` | Database name |
| `POSTGRES_USER` | *(required)* | Database username |
| `POSTGRES_PASSWORD` | *(required)* | Database password |
| `POSTGRES_POOL_SIZE` | `5` | Max async connection pool size |
| `NEO4J_URI` | `None` | Neo4j Bolt URI — omit to use `NullGraph` |
| `NEO4J_USER` | `None` | Neo4j username |
| `NEO4J_PASSWORD` | `None` | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j target database |
| `DEFAULT_BIT_WIDTH` | `4` | turbovec quantisation bit-width (1/2/4/8) |
| `DEFAULT_TOP_K` | `10` | Default `k` for vector searches |

---

## Database schema

Two tables are created automatically on `DocStore.init()`:

```sql
-- Parent documents
ts_documents (id TEXT PK, content TEXT, doc_type TEXT,
              metadata JSONB, created_at TIMESTAMPTZ)

-- Chunks — searchable units; cascade-delete with parent
ts_chunks    (id TEXT PK, document_id TEXT FK → ts_documents,
              content TEXT, chunk_index INT,
              metadata JSONB, created_at TIMESTAMPTZ)
```

Indexes created automatically:

| Index | Type | Purpose |
|---|---|---|
| `idx_ts_documents_doc_type` | B-tree | Filter by document type |
| `idx_ts_chunks_document_id` | B-tree | FK join / parent lookup |
| `idx_ts_chunks_metadata_gin` | GIN | `metadata @> $filter` containment |

---

## Development

```bash
# Install in editable mode with dev extras
pip install -e ".[dev]"

# Run all tests (no external services needed)
pytest

# Run a single file
pytest tests/test_retriever.py -v

# Lint
ruff check Quanta/

# Type-check
mypy Quanta/
```

---

## License

MIT
