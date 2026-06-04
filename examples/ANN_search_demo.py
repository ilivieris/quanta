"""
turbovec · Pure ANN Search Demo
================================

Indexes 10 GDPR articles with a 4-bit quantised turbovec IdMapIndex,
then retrieves the top-4 nearest neighbours for a Greek-language query
using MultiRetriever with graph expansion explicitly disabled.

Demonstrates: IdMapIndex.search() — dense vector similarity, no graph.

Prerequisites
─────────────
  .env with POSTGRES vars + EMBED_MODEL
  pip install Quanta sentence-transformers
"""

import asyncio
import logging
import textwrap
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from Quanta import QuantaIndex
from quanta.config import get_settings
from quanta.docstore import DocStore
from quanta.graph import get_graph_backend
from quanta.retriever import MultiRetriever

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s.%(msecs)03d  [%(levelname)-4s]  [%(name)-27s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("turbovec · ANN search")
log.setLevel(logging.INFO)

# ── GDPR articles ──────────────────────────────────────────────────────────────

ARTICLES: list[tuple[str, str, str]] = [
    (
        "art_5", "Αρχές επεξεργασίας",
        "Τα δεδομένα προσωπικού χαρακτήρα υποβάλλονται σε επεξεργασία νόμιμα, "
        "δίκαια και με διαφανή τρόπο. Συλλέγονται για καθορισμένους, ρητούς και "
        "νόμιμους σκοπούς (περιορισμός σκοπού) και δεν υποβάλλονται σε περαιτέρω "
        "επεξεργασία ασυμβίβαστη με αυτούς. Εφαρμόζεται αρχή ελαχιστοποίησης.",
    ),
    (
        "art_6", "Νομιμότητα επεξεργασίας",
        "Η επεξεργασία είναι σύννομη εφόσον ισχύει τουλάχιστον μία από τις "
        "ακόλουθες προϋποθέσεις: το υποκείμενο έχει δώσει τη συγκατάθεσή του, "
        "η επεξεργασία είναι αναγκαία για εκτέλεση σύμβασης ή για συμμόρφωση "
        "με έννομη υποχρέωση του υπευθύνου επεξεργασίας.",
    ),
    (
        "art_7", "Όροι συγκατάθεσης",
        "Όταν η επεξεργασία βασίζεται στη συγκατάθεση, ο υπεύθυνος μπορεί να "
        "αποδείξει ότι το υποκείμενο των δεδομένων συγκατατέθηκε. Το υποκείμενο "
        "έχει δικαίωμα ανάκλησης ανά πάσα στιγμή χωρίς να επηρεαστεί η νομιμότητα "
        "προηγούμενης επεξεργασίας.",
    ),
    (
        "art_13", "Ενημέρωση κατά τη συλλογή",
        "Κατά τη συλλογή δεδομένων από το υποκείμενο, ο υπεύθυνος παρέχει "
        "πληροφορίες για: ταυτότητα και στοιχεία επικοινωνίας, σκοπούς και νομική "
        "βάση επεξεργασίας, αποδέκτες, μεταφορές σε τρίτες χώρες και δικαιώματα "
        "πρόσβασης, διόρθωσης, διαγραφής και εναντίωσης.",
    ),
    (
        "art_17", "Δικαίωμα διαγραφής",
        "Το υποκείμενο των δεδομένων έχει δικαίωμα να ζητήσει διαγραφή δεδομένων "
        "χωρίς αδικαιολόγητη καθυστέρηση εφόσον τα δεδομένα δεν είναι πλέον "
        "αναγκαία ή η συγκατάθεση ανακλήθηκε ή τα δεδομένα επεξεργάστηκαν παρανόμως.",
    ),
    (
        "art_25", "Προστασία από σχεδιασμού",
        "Ο υπεύθυνος εφαρμόζει κατάλληλα τεχνικά και οργανωτικά μέτρα κατά τον "
        "σχεδιασμό και εξ ορισμού ώστε να ενσωματώνονται οι αρχές προστασίας "
        "δεδομένων και να τηρείται η αρχή ελαχιστοποίησης. Ψευδωνυμοποίηση και "
        "privacy-by-design αποτελούν βασικά εργαλεία.",
    ),
    (
        "art_30", "Αρχεία δραστηριοτήτων",
        "Κάθε υπεύθυνος επεξεργασίας τηρεί αρχείο δραστηριοτήτων που περιλαμβάνει: "
        "όνομα και στοιχεία επικοινωνίας, σκοπούς επεξεργασίας, κατηγορίες "
        "υποκειμένων και δεδομένων, αποδέκτες, μεταφορές σε τρίτες χώρες και "
        "προθεσμίες διαγραφής. Τηρείται γραπτώς ή ηλεκτρονικά.",
    ),
    (
        "art_32", "Ασφάλεια επεξεργασίας",
        "Ο υπεύθυνος και ο εκτελών εφαρμόζουν τεχνικά και οργανωτικά μέτρα "
        "ανάλογα του κινδύνου: ψευδωνυμοποίηση, κρυπτογράφηση, εμπιστευτικότητα, "
        "ακεραιότητα, διαθεσιμότητα, ανθεκτικότητα και αποκατάσταση μετά από "
        "περιστατικό. Αξιολόγηση κινδύνου διεξάγεται τακτικά.",
    ),
    (
        "art_33", "Γνωστοποίηση παραβίασης",
        "Σε περίπτωση παραβίασης δεδομένων ο υπεύθυνος γνωστοποιεί στην εποπτική "
        "αρχή εντός 72 ωρών από τότε που αποκτά γνώση. Η γνωστοποίηση περιλαμβάνει "
        "φύση παραβίασης, κατηγορίες και αριθμό υποκειμένων, πιθανές συνέπειες "
        "και μέτρα αντιμετώπισης.",
    ),
    (
        "art_37", "Υπεύθυνος Προστασίας Δεδομένων",
        "Ορισμός ΥΠΔ είναι υποχρεωτικός όταν η βασική δραστηριότητα συνίσταται σε "
        "τακτική και συστηματική παρακολούθηση υποκειμένων σε μεγάλη κλίμακα ή σε "
        "επεξεργασία ειδικών κατηγοριών. Ο ΥΠΔ έχει εξειδικευμένες γνώσεις δικαίου "
        "και πρακτικής προστασίας δεδομένων.",
    ),
]

TITLES: dict[str, str] = {aid: title for aid, title, _ in ARTICLES}

# ── Helpers ────────────────────────────────────────────────────────────────────

LINE = "─" * 72


def section(title: str) -> None:
    pad = (70 - len(title)) // 2
    print(f"\n{'─' * pad}  {title}  {'─' * pad}")


def wrap(text: str, indent: str = "    ") -> str:
    return textwrap.fill(text, width=70, initial_indent=indent, subsequent_indent=indent)


def _vec_stats(v: np.ndarray) -> str:
    return (
        f"shape={v.shape}  dtype={v.dtype}  "
        f"norm={float(np.linalg.norm(v)):.4f}  "
        f"min={float(v.min()):.4f}  max={float(v.max()):.4f}  "
        f"mean={float(v.mean()):.4f}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    t_start = time.perf_counter()

    cfg = get_settings()
    log.info("Settings loaded")

    print(LINE)
    print("  turbovec · Pure ANN Search Demo")
    print(LINE)
    print(f"  Embed   : {cfg.EMBED_MODEL}")
    print(f"  Dim     : {cfg.EMBED_DIM}  |  bit_width: 4")
    print(LINE)

    # ── Embedding model ───────────────────────────────────────────────────────
    log.info("Loading SentenceTransformer model: %s", cfg.EMBED_MODEL)
    t0 = time.perf_counter()
    print("\n[1/3]  Loading embedding model …")
    st = SentenceTransformer(cfg.EMBED_MODEL)
    log.info("Model loaded in %.2f s", time.perf_counter() - t0)

    def embed(texts: list[str]) -> np.ndarray:
        vecs = st.encode(texts, normalize_embeddings=True).astype(np.float32)
        log.debug("embed() %d texts → shape=%s", len(texts), vecs.shape)
        return vecs

    # ── Infrastructure ────────────────────────────────────────────────────────
    log.info("Initialising DocStore + QuantaIndex + MultiRetriever …")
    print("[2/3]  Initialising DocStore + QuantaIndex + MultiRetriever …")
    t0 = time.perf_counter()
    docstore = DocStore(cfg)
    await docstore.init()
    log.info("DocStore ready in %.2f s", time.perf_counter() - t0)

    text_index = QuantaIndex(name="gdpr", dim=cfg.EMBED_DIM, bit_width=4)
    graph = get_graph_backend(cfg)
    retriever = MultiRetriever(
        indexes={"text": text_index},
        docstore=docstore,
        graph=graph,
    )

    # ── Ingest ────────────────────────────────────────────────────────────────
    log.info("Ingesting %d GDPR articles …", len(ARTICLES))
    print("[3/3]  Embedding + indexing GDPR articles …\n")

    t0 = time.perf_counter()
    texts = [text for _, _, text in ARTICLES]
    embeddings = embed(texts)
    log.info("Batch embed done: shape=%s  elapsed=%.2f s",
             embeddings.shape, time.perf_counter() - t0)

    for idx_i, (art_id, title, text) in enumerate(ARTICLES):
        try:
            await docstore.add_document(
                id=art_id, content=text, doc_type="gdpr_article",
                metadata={"title": title, "article": art_id},
            )
            await docstore.add_chunk(
                id=art_id, document_id=art_id, content=text, chunk_index=0,
                metadata={"title": title, "article": art_id},
            )
            log.info("DocStore ✓  %s  '%s'", art_id, title)
        except Exception as exc:
            log.warning("DocStore skip (already exists): %s — %s", art_id, exc)
        print(f"    ✓  {art_id:8s}  {title}")

    ids = [art_id for art_id, _, _ in ARTICLES]
    text_index.add(embeddings, ids)
    log.info("QuantaIndex: %d vectors added", len(text_index))
    print(f"\n    turbovec IdMapIndex: {len(text_index)} vectors  (dim={cfg.EMBED_DIM}, 4-bit)")

    # ── Q1: Pure dense ANN — no graph ─────────────────────────────────────────
    section("Q1 · Pure ANN search  [turbovec]")
    query = "Ποιες είναι οι υποχρεώσεις του υπευθύνου επεξεργασίας;"
    log.info("Query: %r", query)
    print(f"  Query: {query}\n")

    t0 = time.perf_counter()
    q_vec = embed([query])[0]
    log.debug("Query vector: %s", _vec_stats(q_vec))

    results = await retriever.search(
        query_vectors={"text": q_vec}, k=4, use_graph=False,
    )
    elapsed = time.perf_counter() - t0
    log.info("ANN search done in %.3f s  results=%d", elapsed, len(results))

    for r in results:
        log.debug("  id=%s  score=%.6f  source=%s", r.id, r.score, r.source)
        print(f"  [{r.score:.4f}]  {r.id:8s}  {TITLES[r.id]}")
        print(wrap((r.content or "")[:110] + "…"))

    # ── Teardown ──────────────────────────────────────────────────────────────
    total = time.perf_counter() - t_start
    print(f"\n{LINE}")
    await docstore.close()
    log.info("Total time: %.2f s", total)
    print("  Done.")
    print(LINE)


asyncio.run(main())
