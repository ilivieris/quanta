"""
turbovec · Disk Persistence Demo
==================================

Indexes 10 GDPR articles, saves the turbovec IdMapIndex to disk, reloads
it into a fresh object, and verifies that a probe query returns the same
top-1 result from both the original and the reloaded index.

Demonstrates: QuantaIndex.save() / QuantaIndex.load() round-trip.

Files written
─────────────
  ./indexes/gdpr.tvim        — quantised vector data
  ./indexes/gdpr.ids.json    — id → position mapping

Prerequisites
─────────────
  .env with EMBED_MODEL
  pip install Quanta sentence-transformers
"""

import logging
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
log = logging.getLogger("turbovec · disk persistence")
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
    print("  turbovec · Disk Persistence Demo")
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

    # ── Q7: Save / Load round-trip ────────────────────────────────────────────
    section("Q7 · turbovec save() / load()  [disk persistence]")
    print("[3/3]  Save → Load → verify …\n")

    log.info("Saving index to ./indexes/gdpr.tvim …")
    t0 = time.perf_counter()
    text_index.save()
    log.info("QuantaIndex.save() done in %.3f s", time.perf_counter() - t0)

    log.info("Loading index from ./indexes/ …")
    t0 = time.perf_counter()
    text_index_loaded = QuantaIndex.load("gdpr", index_dir="./indexes")
    log.info("QuantaIndex.load() done in %.3f s  size=%d",
             time.perf_counter() - t0, len(text_index_loaded))

    probe = "κρυπτογράφηση ασφάλεια ψευδωνυμοποίηση"
    probe_vec = embed([probe])[0]
    log.debug("Probe vector: %s", _vec_stats(probe_vec))

    r_orig   = text_index.search(probe_vec, k=1)[0]
    r_loaded = text_index_loaded.search(probe_vec, k=1)[0]
    log.info("Original → id=%s  score=%.6f", r_orig.id, r_orig.score)
    log.info("Reloaded → id=%s  score=%.6f", r_loaded.id, r_loaded.score)

    match = r_orig.id == r_loaded.id
    log.info("Round-trip: %s", "PASS" if match else "FAIL")

    print("  Saved to   : ./indexes/gdpr.tvim  +  ./indexes/gdpr.ids.json")
    print(f"  Original   → top-1: {r_orig.id:8s}  score={r_orig.score:.6f}")
    print(f"  Reloaded   → top-1: {r_loaded.id:8s}  score={r_loaded.score:.6f}")
    print(f"  Round-trip : {'✓ identical' if match else '✗ differ'}")

    log.info("Total time: %.2f s", time.perf_counter() - t_start)
    print(f"\n{LINE}")
    print("  Done.")
    print(LINE)


main()
