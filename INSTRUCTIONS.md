# turboRAG — Οδηγίες Εκκίνησης

## Προαπαιτούμενα

Βεβαιώσου ότι έχεις εγκατεστημένα:

- Python 3.10+
- Docker & Docker Compose
- Git

---

## Βήμα 1 — Κλωνοποίηση repo

```bash
git clone https://github.com/<username>/turboRAG.git
cd turboRAG
```

---

## Βήμα 2 — Αντιγραφή και συμπλήρωση .env

```bash
cp .env.example .env
```

Άνοιξε το `.env` και συμπλήρωσε τις τιμές:

```env
# ── PostgreSQL ──────────────────────────────────────────
TURBORAG_POSTGRES_HOST=localhost
TURBORAG_POSTGRES_PORT=5432
TURBORAG_POSTGRES_DB=turborag
TURBORAG_POSTGRES_USER=turborag
TURBORAG_POSTGRES_PASSWORD=ΒΑΛΕ_ΔΙΚΟ_ΣΟΥ_ΚΩΔΙΚΟ

# ── Neo4j (προαιρετικό — άφησε κενό αν δεν χρησιμοποιείς γράφο) ──
TURBORAG_NEO4J_URI=bolt://localhost:7687
TURBORAG_NEO4J_USER=neo4j
TURBORAG_NEO4J_PASSWORD=ΒΑΛΕ_ΔΙΚΟ_ΣΟΥ_ΚΩΔΙΚΟ
TURBORAG_NEO4J_DATABASE=neo4j

# ── Index defaults ───────────────────────────────────────
TURBORAG_DEFAULT_BIT_WIDTH=4
TURBORAG_DEFAULT_TOP_K=10
```

> ⚠️ Μην βάλεις ποτέ το `.env` στο git. Είναι ήδη στο `.gitignore`.

---

## Βήμα 3 — Εκκίνηση services με Docker

### Μόνο PostgreSQL (χωρίς γράφο)

```bash
docker compose up -d postgres
```

### PostgreSQL + Neo4j (με γράφο)

```bash
docker compose --profile graph up -d
```

Επαλήθευση ότι τρέχουν:

```bash
docker compose ps
```

Πρέπει να δεις `healthy` για postgres (και neo4j αν το εκκίνησες).

---

## Βήμα 4 — Δημιουργία virtual environment

```bash
python -m venv .venv
```

Ενεργοποίηση:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

Επαλήθευση ότι χρησιμοποιείς τον σωστό Python:

```bash
which python   # macOS/Linux → πρέπει να δείχνει στο .venv
python --version
```

> ℹ️ Το `.venv/` είναι ήδη στο `.gitignore` — δεν ανεβαίνει στο repo.

---

## Βήμα 5 — Εγκατάσταση Python library

```bash
# Βασική εγκατάσταση
pip install -e .

# Με LlamaIndex integration
pip install -e ".[llama-index]"

# Με Neo4j support
pip install -e ".[neo4j]"

# Όλα μαζί
pip install -e ".[llama-index,neo4j]"

# Για development (tests, linting)
pip install -e ".[llama-index,neo4j,dev]"
```

---

## Βήμα 6 — Επαλήθευση εγκατάστασης

```bash
python -c "import turborag; print('OK')"
```

---

## Βήμα 7 — Τρέξε τα tests

```bash
pytest tests/ -v
```

Όλα τα tests τρέχουν χωρίς external services (χρησιμοποιούν mocks).

---

## Βήμα 8 — Πρώτη χρήση

```python
import asyncio
import numpy as np
from turborag import TurboIndex
from turborag.docstore import DocStore
from turborag.config import get_settings
from turborag.retriever import HybridRetriever
from turborag.graph import get_graph_backend

async def main():
    settings = get_settings()  # φορτώνει από .env

    # DocStore (PostgreSQL)
    docstore = DocStore(settings)
    await docstore.init()  # δημιουργεί tables αν δεν υπάρχουν

    # Vector indexes
    text_index  = TurboIndex(name="text",   dim=768)
    image_index = TurboIndex(name="images", dim=800)

    # Graph backend (NullGraph αν δεν έχεις Neo4j)
    graph = get_graph_backend(settings)

    # Retriever
    retriever = HybridRetriever(
        indexes={"text": text_index, "images": image_index},
        docstore=docstore,
        graph=graph,
    )

    # Προσθήκη vectors (παράδειγμα με τυχαία)
    vectors = np.random.randn(10, 768).astype(np.float32)
    ids     = [f"chunk_{i}" for i in range(10)]
    text_index.add(vectors, ids)

    # Αναζήτηση
    query = np.random.randn(768).astype(np.float32)
    results = await retriever.search(
        query_vectors={"text": query},
        k=5,
    )
    for r in results:
        print(r.id, r.score, r.source)

    await docstore.close()

asyncio.run(main())
```

---

## Βήμα 9 — LlamaIndex integration (προαιρετικό)

```python
from llama_index.core import VectorStoreIndex, StorageContext
from turborag.integrations.llama_index import TurboRAGVectorStore

vector_store = TurboRAGVectorStore(
    index=text_index,
    docstore=docstore,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes=[], storage_context=storage_context)

# Πρόσθεσε documents
index.insert_nodes(my_nodes)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("Ποιες είναι οι υποχρεώσεις του υπευθύνου επεξεργασίας;")
print(response)
```

---

## Βήμα 10 — Neo4j Browser (προαιρετικό)

Αν εκκίνησες Neo4j, άνοιξε στον browser:

```
http://localhost:7474
```

Login με `neo4j` / τον κωδικό που έβαλες στο `.env`.

---

## Χρήσιμες εντολές Docker

```bash
# Σταμάτημα όλων
docker compose down

# Σταμάτημα + διαγραφή volumes (ΠΡΟΣΟΧΗ: χάνεις τα δεδομένα)
docker compose down -v

# Logs postgres
docker compose logs -f postgres

# Logs neo4j
docker compose logs -f neo4j

# Restart μόνο postgres
docker compose restart postgres
```

---

## Linting & Type checking

```bash
# Ruff
ruff check turborag/

# Mypy
mypy turborag/

# Και τα δύο μαζί
ruff check turborag/ && mypy turborag/
```

---

## Συχνά προβλήματα

**`Connection refused` στο PostgreSQL**
→ Βεβαιώσου ότι τρέχει: `docker compose ps`
→ Περίμενε 5-10 δευτερόλεπτα μετά το `docker compose up`

**`ModuleNotFoundError: turborag`**
→ Βεβαιώσου ότι το venv είναι ενεργό: `source .venv/bin/activate`
→ Μετά τρέξε `pip install -e .` μέσα στον φάκελο του repo

**`TurboRAGError: Neo4j not configured`**
→ Φυσιολογικό αν δεν έχεις ορίσει `TURBORAG_NEO4J_URI` — το σύστημα πέφτει σε NullGraph αυτόματα

**turbovec compilation error κατά την εγκατάσταση**
→ Χρειάζεσαι Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`