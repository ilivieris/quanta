"""
LlamaIndex integration demo for Quanta.

Requires:
    pip install llama-index-core sentence-transformers

EMBED_MODEL and EMBED_DIM are read from .env (or use the defaults in QuantaSettings).
"""
import asyncio
import logging
import time

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStoreQuery
from sentence_transformers import SentenceTransformer

from Quanta import QuantaIndex
from quanta.config import get_settings
from quanta.docstore import DocStore
from quanta.integrations.llama_index import QuantaVectorStore
from quanta.retriever import MultiRetriever

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s.%(msecs)03d  [%(levelname)-4s]  [%(name)-27s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("LlamaIndex integration demo")
log.setLevel(logging.INFO)

# ── Sample corpus ─────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    # GDPR
    (
        "doc_gdpr_1",
        "Ο υπεύθυνος επεξεργασίας υποχρεούται να εφαρμόζει κατάλληλα τεχνικά και "
        "οργανωτικά μέτρα για να διασφαλίζει και να μπορεί να αποδείξει ότι η "
        "επεξεργασία διενεργείται σύμφωνα με τον παρόντα κανονισμό.",
    ),
    (
        "doc_gdpr_2",
        "Ο υπεύθυνος επεξεργασίας τηρεί αρχεία δραστηριοτήτων επεξεργασίας υπό "
        "την ευθύνη του. Τα αρχεία αυτά περιέχουν όλες τις ακόλουθες πληροφορίες: "
        "το όνομα και τα στοιχεία επικοινωνίας του υπευθύνου επεξεργασίας.",
    ),
    (
        "doc_gdpr_3",
        "Ο υπεύθυνος επεξεργασίας γνωστοποιεί παραβίαση δεδομένων προσωπικού "
        "χαρακτήρα στην αρμόδια εποπτική αρχή το αργότερο εντός 72 ωρών αφότου "
        "αποκτήσει γνώση της παραβίασης.",
    ),
    (
        "doc_gdpr_4",
        "Το υποκείμενο των δεδομένων έχει το δικαίωμα να λαμβάνει από τον "
        "υπεύθυνο επεξεργασίας επιβεβαίωση για το αν τα δεδομένα προσωπικού "
        "χαρακτήρα που το αφορούν υφίστανται ή όχι επεξεργασία.",
    ),
    (
        "doc_gdpr_5",
        "Ο υπεύθυνος επεξεργασίας ορίζει Υπεύθυνο Προστασίας Δεδομένων όταν η "
        "βασική δραστηριότητα συνίσταται σε πράξεις επεξεργασίας που απαιτούν "
        "τακτική και συστηματική παρακολούθηση των υποκειμένων σε μεγάλη κλίμακα.",
    ),
    # Αστρονομία
    (
        "doc_astro_1",
        "Η Γη περιστρέφεται γύρω από τον Ήλιο σε μια ελλειπτική τροχιά με μέση "
        "απόσταση περίπου 150 εκατομμύρια χιλιόμετρα, ολοκληρώνοντας μία πλήρη "
        "περιφορά σε 365,25 ημέρες.",
    ),
    (
        "doc_astro_2",
        "Ο Ήλιος είναι ένας αστέρας τύπου G που βρίσκεται στον βραχίονα Ωρίωνα "
        "του γαλαξία μας. Η μάζα του αντιστοιχεί στο 99,86% της συνολικής μάζας "
        "του Ηλιακού Συστήματος.",
    ),
    (
        "doc_astro_3",
        "Η Σελήνη απέχει κατά μέσο όρο 384.400 χιλιόμετρα από τη Γη και "
        "ολοκληρώνει μία πλήρη περιφορά γύρω από αυτήν σε 27,3 ημέρες. "
        "Η βαρυτική της επίδραση προκαλεί τα φαινόμενα της παλίρροιας.",
    ),
    (
        "doc_astro_4",
        "Οι μαύρες τρύπες είναι περιοχές του χωρόχρονου όπου η βαρύτητα είναι "
        "τόσο ισχυρή ώστε τίποτα, ούτε καν το φως, δεν μπορεί να διαφύγει πέρα "
        "από τον ορίζοντα γεγονότων.",
    ),
    (
        "doc_astro_5",
        "Ο γαλαξίας μας, ο Γαλαξίας, περιέχει εκτιμώμενα 200-400 δισεκατομμύρια "
        "αστέρια και έχει διάμετρο περίπου 100.000 ετών φωτός. Στο κέντρο του "
        "βρίσκεται μια υπερμαζική μαύρη τρύπα γνωστή ως Sagittarius A*.",
    ),
]


def _build_embed_model(model_name: str) -> BaseEmbedding:
    """Wrap SentenceTransformer so LlamaIndex uses it globally (no OpenAI)."""
    log.info("Building SentenceTransformer wrapper: %s", model_name)
    t0 = time.perf_counter()
    st = SentenceTransformer(model_name)
    log.info("SentenceTransformer loaded in %.2f s", time.perf_counter() - t0)

    class _STEmbed(BaseEmbedding):
        def _get_text_embedding(self, text: str) -> list[float]:
            vec = st.encode(text, normalize_embeddings=True).tolist()
            return vec

        def _get_query_embedding(self, query: str) -> list[float]:
            vec = st.encode(query, normalize_embeddings=True).tolist()
            return vec

        async def _aget_query_embedding(self, query: str) -> list[float]:
            vec = st.encode(query, normalize_embeddings=True).tolist()
            return vec

    log.info("_STEmbed wrapper ready  embed_batch_size=32")
    return _STEmbed(embed_batch_size=32)


async def main() -> None:
    t_start = time.perf_counter()
    log.info("=== LlamaIndex + Quanta demo starting ===")

    cfg = get_settings()
    log.info("Settings loaded  EMBED_MODEL=%s  EMBED_DIM=%d", cfg.EMBED_MODEL, cfg.EMBED_DIM)

    print(f"Embedding model : {cfg.EMBED_MODEL}")
    print(f"Embedding dim   : {cfg.EMBED_DIM}")

    # ── Embedding model ───────────────────────────────────────────────────────
    log.info("Building embed model …")
    embed_model = _build_embed_model(cfg.EMBED_MODEL)
    Settings.embed_model = embed_model
    log.info("LlamaIndex Settings.embed_model set to _STEmbed")

    # ── Quanta infrastructure ───────────────────────────────────────────────
    log.info("Initialising DocStore (PostgreSQL) …")
    t0 = time.perf_counter()
    docstore = DocStore(cfg)
    await docstore.init()
    log.info("DocStore ready in %.2f s", time.perf_counter() - t0)

    log.info("Creating QuantaIndex  name=text  dim=%d  bit_width=4 (default)", cfg.EMBED_DIM)
    text_index = QuantaIndex(name="text", dim=cfg.EMBED_DIM)

    log.info("Building MultiRetriever (no graph — NullGraph)")
    retriever = MultiRetriever(
        indexes={"text": text_index},
        docstore=docstore,
    )

    # ── LlamaIndex vector store ───────────────────────────────────────────────
    log.info("Creating QuantaVectorStore  index_name=text  embed_dim=%d", cfg.EMBED_DIM)
    vector_store = QuantaVectorStore(
        retriever=retriever,
        index_name="text",
        embed_dim=cfg.EMBED_DIM,
    )

    log.info("Creating StorageContext and VectorStoreIndex (empty) …")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=[], storage_context=storage_context)
    log.info("VectorStoreIndex ready  (0 nodes at init)")

    # ── Embed & ingest ────────────────────────────────────────────────────────
    log.info("Embedding %d documents …", len(SAMPLE_TEXTS))
    print(f"\nInserting {len(SAMPLE_TEXTS)} nodes ...")

    nodes: list[TextNode] = []
    for doc_id, text in SAMPLE_TEXTS:
        t0 = time.perf_counter()
        embedding = embed_model.get_text_embedding(text)
        elapsed = time.perf_counter() - t0
        log.info("Embedded %-14s  dim=%d  norm=%.4f  elapsed=%.3f s",
                 doc_id, len(embedding),
                 sum(x * x for x in embedding) ** 0.5,
                 elapsed)
        nodes.append(TextNode(
            node_id=doc_id,
            text=text,
            embedding=embedding,
            metadata={"source": "GDPR", "chunk_index": 0},
        ))

    log.info("All %d nodes embedded — calling async_add …", len(nodes))
    t0 = time.perf_counter()
    added_ids = await index._vector_store.async_add(nodes)
    elapsed = time.perf_counter() - t0
    log.info("async_add done in %.3f s  added_ids=%s", elapsed, added_ids)

    # ── Query ─────────────────────────────────────────────────────────────────
    question = "Ποιες είναι οι υποχρεώσεις του υπευθύνου επεξεργασίας;"
    log.info("Building query embedding for: %r", question)
    print(f"\nQuery: {question}\n")

    t0 = time.perf_counter()
    query_embedding = embed_model.get_query_embedding(question)
    log.info("Query embedding ready  dim=%d  norm=%.4f  elapsed=%.3f s",
             len(query_embedding),
             sum(x * x for x in query_embedding) ** 0.5,
             time.perf_counter() - t0)

    log.info("Executing VectorStoreQuery  similarity_top_k=3")
    vs_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=3,
    )

    t0 = time.perf_counter()
    result = await vector_store.aquery(vs_query)
    elapsed = time.perf_counter() - t0
    log.info("aquery done in %.3f s  nodes_returned=%d  similarities=%s",
             elapsed,
             len(result.nodes),
             [f"{s:.4f}" for s in (result.similarities or [])])

    for rank, nws in enumerate(result.nodes, 1):
        node_id   = nws.node.node_id
        score     = nws.score or 0.0
        content   = nws.node.get_content()
        metadata  = nws.node.metadata
        log.info("Result rank=%d  id=%-14s  score=%.4f  content_len=%d  metadata=%s",
                 rank, node_id, score, len(content), metadata)
        print(f"[{score:.4f}] {node_id}")
        print(f"  {content[:120]}...")
        print()

    # ── Teardown ──────────────────────────────────────────────────────────────
    total = time.perf_counter() - t_start
    log.info("Closing DocStore …")
    await docstore.close()
    log.info("DocStore closed")
    log.info("=== done  total=%.2f s ===", total)


asyncio.run(main())
