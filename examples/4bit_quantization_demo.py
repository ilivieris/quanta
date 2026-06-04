"""
turbovec · 4-bit Quantisation Memory Demo
==========================================

Indexes 10 GDPR articles with a 4-bit quantised turbovec IdMapIndex,
then measures and compares the memory footprint of the same vectors
stored as float32 versus 4-bit product-quantised form.

Demonstrates: memory vs float32 comparison — 4-bit IdMapIndex.

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
log = logging.getLogger("turbovec · 4-bit quantisation")
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

# ── Helpers ────────────────────────────────────────────────────────────────────

LINE = "─" * 72


def section(title: str) -> None:
    pad = (70 - len(title)) // 2
    print(f"\n{'─' * pad}  {title}  {'─' * pad}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.perf_counter()

    cfg = get_settings()
    log.info("Settings loaded")

    print(LINE)
    print("  turbovec · 4-bit Quantisation Memory Demo")
    print(LINE)
    print(f"  Embed : {cfg.EMBED_MODEL}")
    print(f"  Dim   : {cfg.EMBED_DIM}  |  bit_width: 4  (4-bit product quantisation)")
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

    # ── Q8: Memory footprint — 4-bit vs float32 ───────────────────────────────
    section("Q8 · turbovec memory: float32 vs 4-bit quantised")
    print("[3/3]  Computing memory footprint …\n")

    n = len(text_index)
    d = cfg.EMBED_DIM
    f32_bytes = n * d * 4
    q4_bytes  = n * d // 2
    ratio     = f32_bytes / q4_bytes

    log.info("n=%d  dim=%d  float32=%d B  4-bit=%d B  ratio=%.1f×",
             n, d, f32_bytes, q4_bytes, ratio)

    m_f32_1m = (1_000_000 * d * 4) / 1024**3
    m_q4_1m  = (1_000_000 * d // 2) / 1024**3
    log.debug("1M-vector projection: float32=%.2f GB  4-bit=%.3f GB", m_f32_1m, m_q4_1m)

    print(f"  Vectors     : {n}")
    print(f"  Dimension   : {d}")
    print(f"  float32     : {f32_bytes:>8,} bytes  ({f32_bytes / 1024:.1f} KB)")
    print(f"  4-bit quant : {q4_bytes:>8,} bytes  ({q4_bytes  / 1024:.1f} KB)")
    print(f"  Reduction   : {ratio:.0f}×  less memory")
    print(f"\n  At 1 M vectors: float32 ≈ {m_f32_1m:.1f} GB  →  4-bit ≈ {m_q4_1m:.2f} GB")

    log.info("Total time: %.2f s", time.perf_counter() - t_start)
    print(f"\n{LINE}")
    print("  Done.")
    print(LINE)


main()
