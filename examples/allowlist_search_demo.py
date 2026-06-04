"""
turbovec · Allowlist-Filtered Search Demo
==========================================

Indexes 10 GDPR articles with turbovec, then runs the same query twice:
once against the full index and once restricted to a caller-supplied
allowlist of document IDs — demonstrating that irrelevant articles are
excluded regardless of their cosine similarity to the query.

Demonstrates: IdMapIndex.search(allowed_ids=[...])

Prerequisites
─────────────
  .env with EMBED_MODEL
  pip install Quanta sentence-transformers
"""

import logging
import textwrap
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from quanta import QuantaIndex
from quanta.config import get_settings

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s.%(msecs)03d  [%(levelname)-4s]  [%(name)-27s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("turbovec · allowlist search")
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


def _vec_stats(v: np.ndarray) -> str:
    return (
        f"shape={v.shape}  dtype={v.dtype}  "
        f"norm={float(np.linalg.norm(v)):.4f}  "
        f"min={float(v.min()):.4f}  max={float(v.max()):.4f}  "
        f"mean={float(v.mean()):.4f}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.perf_counter()

    cfg = get_settings()
    log.info("Settings loaded")

    print(LINE)
    print("  turbovec · Allowlist-Filtered Search Demo")
    print(LINE)
    print(f"  Embed : {cfg.EMBED_MODEL}")
    print(f"  Dim   : {cfg.EMBED_DIM}  |  bit_width: 4")
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

    # ── Build QuantaIndex ──────────────────────────────────────────────────────
    log.info("Creating QuantaIndex  name=gdpr  dim=%d  bit_width=4", cfg.EMBED_DIM)
    print("[2/3]  Embedding articles + building QuantaIndex …\n")

    t0 = time.perf_counter()
    texts = [text for _, _, text in ARTICLES]
    embeddings = embed(texts)
    log.info("Batch embed done: shape=%s  elapsed=%.2f s",
             embeddings.shape, time.perf_counter() - t0)

    text_index = QuantaIndex(name="gdpr", dim=cfg.EMBED_DIM, bit_width=4)
    ids = [art_id for art_id, _, _ in ARTICLES]
    text_index.add(embeddings, ids)

    for art_id, title, _ in ARTICLES:
        print(f"    ✓  {art_id:8s}  {title}")

    log.info("QuantaIndex: %d vectors", len(text_index))
    print(f"\n    turbovec IdMapIndex: {len(text_index)} vectors  (dim={cfg.EMBED_DIM}, 4-bit)")

    # ── Q3: Allowlist-filtered ANN search ─────────────────────────────────────
    section("Q3 · Allowlist-filtered ANN  [turbovec]")
    allowed = ["art_7", "art_13", "art_17"]
    query = "Δικαιώματα υποκειμένου δεδομένων"
    log.info("Query: %r  allowed=%s", query, allowed)
    print(f"  Query   : {query}")
    print(f"  Allowed : {allowed}\n")

    t0 = time.perf_counter()
    q_vec = embed([query])[0]
    log.debug("Query vector: %s", _vec_stats(q_vec))

    print("  Without allowlist (all 10 articles):")
    t1 = time.perf_counter()
    full_results = text_index.search(q_vec, k=3)
    log.info("Full search: %.3f s  results=%d", time.perf_counter() - t1, len(full_results))
    for r in full_results:
        log.debug("  id=%s  score=%.6f", r.id, r.score)
        print(f"    [{r.score:.4f}]  {r.id:8s}  {TITLES[r.id]}")

    print("\n  With allowlist (only art_7, art_13, art_17):")
    t1 = time.perf_counter()
    filtered_results = text_index.search(q_vec, k=3, allowed_ids=allowed)
    log.info("Filtered search: %.3f s  results=%d  allowlist_size=%d",
             time.perf_counter() - t1, len(filtered_results), len(allowed))
    for r in filtered_results:
        log.debug("  id=%s  score=%.6f", r.id, r.score)
        print(f"    [{r.score:.4f}]  {r.id:8s}  {TITLES[r.id]}")

    log.info("Total time: %.2f s", time.perf_counter() - t_start)
    print(f"\n{LINE}")
    print("  Done.")
    print(LINE)


main()
