"""
Medical Graph Population Script
================================
Connects to Neo4j using the Neo4jConnection utility and populates
a rich medical graph with randomised sample data.

Changes vs original
-------------------
- Patient nodes gain a `summary` field (free text, embedding-ready):
  a 2-3 sentence clinical summary aggregating diagnoses, medications,
  procedures and admissions.
- Doctor nodes gain an `expertise` field: a short narrative of their
  specialisation and procedural focus.
- Diagnosis nodes gain a `clinicalDescription` field: a paragraph
  describing the condition, typical presentation and treatment approach.
  `severity` and `chronic` have been moved to the HAS_DIAGNOSIS edge so
  each patient can have their own per-diagnosis severity/chronicity.
- Procedure nodes gain a `procedureDescription` field: a short
  narrative of the procedure, indication and expected outcome.
  These four fields are the targets for sentence-transformer embeddings.
- `frequency` has been moved from the Medication node to the PRESCRIBED
  edge so each patient can have a personalised dosing schedule.
- ADMITTED_TO edges are now sequenced to prevent overlapping admissions;
  `dischargeDate` is NULL when the patient is still admitted.

Usage
-----
    python populate_medical_graph.py \
        --uri bolt://localhost:7687 \
        --user neo4j \
        --password secret \
        [--clean]

Dependencies
------------
    pip install neo4j faker
"""

import argparse
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from examples.project.neo4j_connection import Neo4jConnection, neo4j_connection

try:
    from faker import Faker
except ImportError:
    print("[ERROR] 'faker' is not installed.  Run: pip install faker")
    sys.exit(1)

fake = Faker("el_GR")
random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# Reference data
# ══════════════════════════════════════════════════════════════════════════════

SPECIALTIES = [
    "Cardiology", "Neurology", "Oncology", "Orthopedics",
    "Gastroenterology", "Endocrinology", "Pulmonology", "Nephrology",
    "Dermatology", "Psychiatry",
]

# (icd, name, clinical_description)
DIAGNOSES = [
    (
        "I10", "Essential hypertension",
        "Η ουσιώδης υπέρταση είναι χρόνια ανύψωση της αρτηριακής πίεσης χωρίς "
        "αναγνωρίσιμη δευτερογενή αιτία. Παρουσιάζεται συχνά ασυμπτωματικά και "
        "ανακαλύπτεται τυχαία κατά τη ρουτίνα εξέταση. Η αγωγή περιλαμβάνει "
        "τροποποίηση τρόπου ζωής και αντιυπερτασικά φάρμακα όπως αναστολείς ΜΕΑ, "
        "αποκλειστές ασβεστίου και διουρητικά. Μακροχρόνια αρρύθμιστη υπέρταση "
        "αυξάνει τον κίνδυνο αγγειακού εγκεφαλικού επεισοδίου και καρδιακής ανεπάρκειας.",
    ),
    (
        "E11", "Type 2 diabetes mellitus",
        "Ο σακχαρώδης διαβήτης τύπου 2 χαρακτηρίζεται από ινσουλινοαντίσταση και "
        "προοδευτική δυσλειτουργία των β-κυττάρων του παγκρέατος. Τυπική παρουσίαση "
        "περιλαμβάνει πολυουρία, πολυδιψία και κόπωση. Η θεραπεία ξεκινά με "
        "μετφορμίνη και τροποποίηση διατροφής, με δυνατότητα κλιμάκωσης σε ινσουλίνη. "
        "Στενή παρακολούθηση HbA1c και νεφρικής λειτουργίας είναι απαραίτητη.",
    ),
    (
        "J45", "Asthma",
        "Το άσθμα είναι χρόνια φλεγμονώδης νόσος των αεραγωγών με επεισοδιακή "
        "βρογχόσπαση, συριγμό και δύσπνοια. Ο ασθενής έχει υπεραντιδραστικούς "
        "αεραγωγούς σε εκλυτικούς παράγοντες όπως αλλεργιογόνα και άσκηση. "
        "Η θεραπεία βασίζεται σε εισπνεόμενα κορτικοστεροειδή για συντήρηση "
        "και βραχείας δράσης β2-αγωνιστές για ανακούφιση.",
    ),
    (
        "I21", "Acute myocardial infarction",
        "Το οξύ έμφραγμα του μυοκαρδίου είναι ισχαιμική νέκρωση καρδιακού ιστού "
        "λόγω ξαφνικής απόφραξης στεφανιαίας αρτηρίας, συνήθως από θρόμβο. "
        "Παρουσιάζεται με έντονο θωρακικό άλγος, εφίδρωση και δύσπνοια. "
        "Άμεση επαναιμάτωση με πρωτογενή αγγειοπλαστική ή θρομβόλυση είναι κρίσιμη "
        "για τη διάσωση μυοκαρδίου. Μακροχρόνια αγωγή με αντιαιμοπεταλιακά και στατίνες.",
    ),
    (
        "K21", "Gastroesophageal reflux",
        "Η γαστροοισοφαγική παλινδρόμηση αναφέρεται στην παλινδρόμηση γαστρικού "
        "περιεχομένου στον οισοφάγο. Τυπικά συμπτώματα είναι η καούρα και η "
        "αναγωγή, συχνά επιδεινούμενα μετά από γεύμα ή σε ύπτια θέση. "
        "Αντιμετωπίζεται με αναστολείς αντλίας πρωτονίων, διατροφικές αλλαγές "
        "και ανύψωση κεφαλής κλίνης.",
    ),
    (
        "G40", "Epilepsy",
        "Η επιληψία χαρακτηρίζεται από επαναλαμβανόμενες μη προκλητές επιληπτικές "
        "κρίσεις λόγω παθολογικής εκκένωσης νευρώνων. Η κλινική εικόνα ποικίλει "
        "από εστιακές κρίσεις με διαταραχή συνείδησης έως γενικευμένες τονικοκλονικές. "
        "Ο ηλεκτροεγκεφαλογράφος και η απεικόνιση εγκεφάλου αποτελούν βασικά "
        "διαγνωστικά εργαλεία. Αντιεπιληπτικά φάρμακα επιτυγχάνουν έλεγχο στην "
        "πλειονότητα των ασθενών.",
    ),
    (
        "C34", "Malignant neoplasm of bronchus",
        "Ο κακοήθης όγκος βρόγχου και πνεύμονα είναι από τους συχνότερους "
        "καρκίνους παγκοσμίως, συνδέεται στενά με το κάπνισμα. Διακρίνεται σε "
        "μικροκυτταρικό και μη-μικροκυτταρικό. Συμπτώματα περιλαμβάνουν χρόνιο "
        "βήχα, αιμόπτυση και απώλεια βάρους. Η θεραπεία εξαρτάται από στάδιο "
        "και ιστολογία: χειρουργείο, χημειοθεραπεία, ακτινοθεραπεία ή ανοσοθεραπεία.",
    ),
    (
        "N18", "Chronic kidney disease",
        "Η χρόνια νεφρική νόσος είναι προοδευτική και μη αναστρέψιμη μείωση της "
        "νεφρικής λειτουργίας. Αιτίες περιλαμβάνουν διαβήτη, υπέρταση και "
        "σπειραματονεφρίτιδες. Παρακολουθείται με GFR και αλβουμινουρία. "
        "Σε τελικό στάδιο απαιτείται αιμοκάθαρση ή μεταμόσχευση νεφρού. "
        "Ο έλεγχος αιτιολογίας επιβραδύνει την εξέλιξη.",
    ),
    (
        "F32", "Major depressive disorder",
        "Η μείζων καταθλιπτική διαταραχή χαρακτηρίζεται από επίμονη καταθλιπτική "
        "διάθεση, απώλεια ενδιαφέροντος, διαταραχές ύπνου και όρεξης, κόπωση "
        "και σκέψεις αξιοθρηνήτου ή θανάτου. Η θεραπεία συνδυάζει SSRIs και "
        "ψυχοθεραπεία γνωστικής-συμπεριφορικής κατεύθυνσης. Η παρακολούθηση "
        "υποτροπής είναι κρίσιμη μακροπρόθεσμα.",
    ),
    (
        "M54", "Dorsalgia / back pain",
        "Το άλγος ράχης είναι ένα από τα συχνότερα μυοσκελετικά αιτήματα. "
        "Μπορεί να είναι οξύ ή χρόνιο, μηχανικό ή ριζιτικό. Συνηθέστερες αιτίες "
        "είναι οσφυαλγία από μυϊκή καταπόνηση, εκφυλισμός δίσκου και σπονδυλαρθρίτιδα. "
        "Αντιμετώπιση περιλαμβάνει φυσικοθεραπεία, ΜΣΑΦ και στις επίμονες περιπτώσεις "
        "παρεμβατικές τεχνικές.",
    ),
    (
        "I50", "Heart failure",
        "Η καρδιακή ανεπάρκεια είναι κλινικό σύνδρομο στο οποίο η καρδιά αδυνατεί "
        "να καλύψει τις μεταβολικές ανάγκες του οργανισμού. Παρουσιάζεται με δύσπνοια "
        "κόπωση και οιδήματα κάτω άκρων. Ο υπερηχοκαρδιογράφημα εκτιμά το κλάσμα "
        "εξώθησης. Θεραπεία βασίζεται σε αναστολείς ΜΕΑ, β-αποκλειστές και "
        "διουρητικά, με πιθανή ανάγκη συσκευής ή μεταμόσχευσης σε σοβαρές περιπτώσεις.",
    ),
    (
        "E78", "Hyperlipidaemia",
        "Η υπερλιπιδαιμία αναφέρεται σε αυξημένα επίπεδα χοληστερόλης ή/και "
        "τριγλυκεριδίων στο αίμα, αποτελώντας σημαντικό παράγοντα κινδύνου "
        "αθηροσκλήρωσης και καρδιαγγειακής νόσου. Η θεραπεία ξεκινά με διατροφικές "
        "παρεμβάσεις και αν δεν επαρκεί, χορηγούνται στατίνες. Τακτική παρακολούθηση "
        "λιπιδαιμικού προφίλ είναι απαραίτητη.",
    ),
]

MEDICATIONS = [
    ("MED001", "Metformin",     500),
    ("MED002", "Lisinopril",     10),
    ("MED003", "Atorvastatin",   20),
    ("MED004", "Aspirin",       100),
    ("MED005", "Omeprazole",     20),
    ("MED006", "Amlodipine",      5),
    ("MED007", "Metoprolol",     50),
    ("MED008", "Salbutamol",    100),
    ("MED009", "Fluoxetine",     20),
    ("MED010", "Levetiracetam", 500),
    ("MED011", "Furosemide",     40),
    ("MED012", "Warfarin",        5),
]

SEVERITIES            = ["mild", "moderate", "severe"]
PRESCRIPTION_FREQS    = ["once daily", "twice daily", "three times daily",
                         "once daily at night", "as needed", "every 8 hours",
                         "every 12 hours", "weekly"]

# (code, name, duration, anaesthesia, procedure_description)
PROCEDURES = [
    (
        "PROC001", "Coronary angiography", 60, True,
        "Η στεφανιογραφία είναι επεμβατική απεικονιστική μέθοδος για την αξιολόγηση "
        "των στεφανιαίων αρτηριών. Εισάγεται καθετήρας από τη μηριαία ή κερκιδική "
        "αρτηρία και χορηγείται σκιαγραφικό. Ενδείκνυται σε υποψία στεφανιαίας νόσου "
        "ή πριν επέμβαση επαναιμάτωσης. Επιτρέπει άμεση αγγειοπλαστική αν χρειαστεί.",
    ),
    (
        "PROC002", "Upper GI endoscopy", 30, True,
        "Η γαστροσκόπηση είναι ενδοσκοπική εξέταση οισοφάγου, στομάχου και "
        "δωδεκαδακτύλου. Ενδείκνυται για διερεύνηση δυσπεψίας, αιμορραγίας ή "
        "κατάποσης ξένου σώματος. Επιτρέπει βιοψία και θεραπευτικές παρεμβάσεις "
        "όπως αιμόσταση ή αφαίρεση πολυπόδων.",
    ),
    (
        "PROC003", "Bronchoscopy", 45, True,
        "Η βρογχοσκόπηση είναι ενδοσκοπική εξέταση της τραχείας και βρόγχων. "
        "Χρησιμοποιείται για διερεύνηση αιμόπτυσης, ύποπτων βλαβών ή ξένων σωμάτων "
        "και για λήψη βιοψιών. Επιτρέπει επίσης βρογχοκυψελιδικό έκπλυμα για "
        "μικροβιολογική ανάλυση σε λοιμώξεις.",
    ),
    (
        "PROC004", "Echocardiography", 30, False,
        "Το υπερηχοκαρδιογράφημα είναι μη επεμβατική απεικόνιση της καρδιάς με "
        "υπέρηχους. Αξιολογεί δομή, λειτουργία, βαλβίδες και κλάσμα εξώθησης. "
        "Αποτελεί βασικό εργαλείο στη διάγνωση καρδιακής ανεπάρκειας, βαλβιδοπαθειών "
        "και περικαρδιακής συλλογής.",
    ),
    (
        "PROC005", "Lumbar puncture", 20, True,
        "Η οσφυονωτιαία παρακέντηση είναι επεμβατική τεχνική λήψης εγκεφαλονωτιαίου "
        "υγρού. Ενδείκνυται για διάγνωση μηνιγγίτιδας, υπαραχνοειδούς αιμορραγίας "
        "και νευρολογικών νοσημάτων. Το δείγμα αναλύεται για κύτταρα, πρωτεΐνη, "
        "γλυκόζη και μικρόβια.",
    ),
    (
        "PROC006", "Bone marrow biopsy", 30, True,
        "Η βιοψία μυελού των οστών λαμβάνεται συνήθως από την οπίσθια λαγόνια "
        "άκανθα υπό τοπική αναισθησία. Ενδείκνυται στη διερεύνηση αιματολογικών "
        "κακοηθειών, αναιμιών και σταδιοποίηση λεμφωμάτων. Η ιστολογική εξέταση "
        "αξιολογεί κυτταρικότητα και παθολογικές διηθήσεις.",
    ),
    (
        "PROC007", "Renal biopsy", 45, True,
        "Η νεφρική βιοψία είναι η επεμβατική λήψη ιστού νεφρού υπό υπερηχογραφική "
        "καθοδήγηση. Ενδείκνυται για τη διάγνωση σπειραματονεφρίτιδας, νεφρωτικού "
        "συνδρόμου και αιτίας νεφρικής ανεπάρκειας. Η ιστολογική εξέταση καθοδηγεί "
        "θεραπευτικές αποφάσεις.",
    ),
    (
        "PROC008", "Colonoscopy", 45, True,
        "Η κολονοσκόπηση είναι η ενδοσκοπική εξέταση του παχέος εντέρου. "
        "Ενδείκνυται για διαλογή καρκίνου παχέος εντέρου, διερεύνηση αιμορραγίας "
        "ή αλλαγών συνηθειών εντέρου. Επιτρέπει αφαίρεση πολυπόδων και βιοψία. "
        "Προετοιμασία με εντερική κάθαρση είναι απαραίτητη.",
    ),
    (
        "PROC009", "MRI brain", 60, False,
        "Η μαγνητική τομογραφία εγκεφάλου παρέχει λεπτομερή απεικόνιση εγκεφαλικού "
        "παρεγχύματος χωρίς ιοντίζουσα ακτινοβολία. Ενδείκνυται σε εγκεφαλικά "
        "αγγειακά επεισόδια, όγκους, φλεγμονές και επιληψία. Με σκιαγραφικό "
        "γαδολίνιο ανιχνεύει διαταραχές αιματοεγκεφαλικού φραγμού.",
    ),
    (
        "PROC010", "CT thorax", 20, False,
        "Η αξονική τομογραφία θώρακα παρέχει λεπτομερείς εικόνες πνευμόνων, "
        "μεσοθωρακίου και αγγείων. Ενδείκνυται για διερεύνηση πνευμονικής εμβολής, "
        "διάμεσης πνευμονοπάθειας και νεοπλασμάτων. Η χαμηλής δόσης CT αποτελεί "
        "εργαλείο διαλογής για καρκίνο πνεύμονα σε υψηλού κινδύνου άτομα.",
    ),
    (
        "PROC011", "Electroencephalogram (EEG)", 60, False,
        "Το ηλεκτροεγκεφαλογράφημα καταγράφει ηλεκτρική δραστηριότητα εγκεφάλου "
        "μέσω ηλεκτροδίων τοποθετημένων στο τριχωτό κεφαλής. Βασικό εργαλείο "
        "διάγνωσης επιληψίας, εγκεφαλοπαθειών και διαταραχών ύπνου. "
        "Επιτρεπτό επί ηρεμίας, ύπνου ή με φωτοδιέγερση.",
    ),
    (
        "PROC012", "Haemodialysis session", 240, False,
        "Η συνεδρία αιμοκάθαρσης αντικαθιστά τη νεφρική λειτουργία σε ασθενείς "
        "με τελικού σταδίου νεφρική νόσο. Αίμα διέρχεται από εξωσωματικό κύκλωμα "
        "μέσω ημιδιαπερατής μεμβράνης που αφαιρεί τοξίνες και περίσσεια υγρών. "
        "Συνήθως διενεργείται τρεις φορές εβδομαδιαίως για 4 ώρες.",
    ),
]

HOSPITALS = [
    ("HOSP001", "General Hospital of Athens",    1200, "Athens"),
    ("HOSP002", "University Hospital of Patras",  800, "Patras"),
    ("HOSP003", "Hippocration Hospital",           900, "Thessaloniki"),
    ("HOSP004", "General Hospital of Kalamata",   350, "Kalamata"),
    ("HOSP005", "Evangelismos Hospital",          1100, "Athens"),
]

BLOOD_TYPES   = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
WARDS         = ["Cardiology", "Neurology", "Oncology", "General Medicine", "ICU", "Surgery"]
OUTCOMES      = ["successful", "complicated", "routine", "excellent", "satisfactory"]

# ── specialty → expertise narrative templates ──────────────────────────────────
_EXPERTISE_TEMPLATES = {
    "Cardiology": (
        "Ειδικός καρδιολόγος με εστίαση στη στεφανιαία νόσο, καρδιακή ανεπάρκεια "
        "και επεμβατική καρδιολογία. Εκτεταμένη εμπειρία σε στεφανιογραφίες και "
        "αγγειοπλαστικές. Παρακολουθεί ασθενείς μετά από οξύ έμφραγμα και "
        "εμφύτευση βηματοδότη."
    ),
    "Neurology": (
        "Νευρολόγος με εξειδίκευση στην επιληψία, εγκεφαλικά αγγειακά επεισόδια "
        "και νευροεκφυλιστικές παθήσεις. Διενεργεί ηλεκτροεγκεφαλογραφήματα και "
        "συμμετέχει σε διεπιστημονικές ομάδες για τη διαχείριση πολλαπλής σκλήρυνσης."
    ),
    "Oncology": (
        "Ογκολόγος με εμπειρία στη χημειοθεραπεία, ανοσοθεραπεία και στοχευμένες "
        "θεραπείες. Παρακολουθεί ασθενείς με καρκίνο πνεύμονα, μαστού και "
        "γαστρεντερικές κακοήθειες. Συνεργάζεται στενά με ακτινοθεραπευτές "
        "και χειρουργούς ογκολόγους."
    ),
    "Orthopedics": (
        "Ορθοπαιδικός χειρουργός εξειδικευμένος σε αρθροπλαστικές γόνατος και "
        "ισχίου, αθλητικές κακώσεις και παθήσεις σπονδυλικής στήλης. "
        "Χρησιμοποιεί ελάχιστα επεμβατικές τεχνικές για ταχύτερη αποκατάσταση."
    ),
    "Gastroenterology": (
        "Γαστρεντερολόγος με εξειδίκευση στην ενδοσκόπηση, φλεγμονώδη νόσο εντέρου "
        "και ηπατολογία. Εκτελεί γαστροσκοπήσεις, κολονοσκοπήσεις και ERCP. "
        "Παρακολουθεί ασθενείς με κίρρωση και χρόνια ηπατίτιδα."
    ),
    "Endocrinology": (
        "Ενδοκρινολόγος με εστίαση στο σακχαρώδη διαβήτη τύπου 1 και 2, "
        "θυρεοειδοπάθειες και παθήσεις επινεφριδίων. Παρέχει εκπαίδευση χρήσης "
        "αντλίας ινσουλίνης και ερμηνεία συνεχούς καταγραφής γλυκόζης."
    ),
    "Pulmonology": (
        "Πνευμονολόγος με εμπειρία στη ΧΑΠ, άσθμα, διάμεσες πνευμονοπάθειες "
        "και πνευμονική υπέρταση. Διενεργεί βρογχοσκοπήσεις και σπιρομετρήσεις. "
        "Εξειδικευμένος στη διαχείριση ασθενών με αναπνευστική ανεπάρκεια."
    ),
    "Nephrology": (
        "Νεφρολόγος με εξειδίκευση στη χρόνια νεφρική νόσο, αιμοκάθαρση και "
        "περιτοναϊκή κάθαρση. Παρακολουθεί ασθενείς πρε- και μετα-μεταμόσχευση "
        "νεφρού. Εμπειρία σε νεφρικές βιοψίες και αντιμετώπιση σπειραματονεφρίτιδων."
    ),
    "Dermatology": (
        "Δερματολόγος με εστίαση στη δερματοογκολογία, ψωρίαση και αλλεργικές "
        "δερματοπάθειες. Εκτελεί δερμοσκοπήσεις, βιοψίες και εκτομές δερματικών "
        "βλαβών. Εμπειρία στη χρήση βιολογικών παραγόντων για φλεγμονώδεις παθήσεις."
    ),
    "Psychiatry": (
        "Ψυχίατρος με εξειδίκευση στη μείζονα καταθλιπτική διαταραχή, αγχώδεις "
        "διαταραχές και ψύχωση. Συνδυάζει ψυχοφαρμακολογία με ψυχοθεραπευτικές "
        "τεχνικές. Εμπειρία σε κρίσεις και ακούσια νοσηλεία."
    ),
}

# ══════════════════════════════════════════════════════════════════════════════
# Patient summary builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_patient_summary(
    name: str,
    dob: str,
    blood: str,
    diag_names: list[str],
    med_names: list[str],
    proc_names: list[str],
    hosp_names: list[str],
) -> str:
    """
    Build a free-text clinical summary for a patient.
    This is the field that will be embedded for semantic search.
    """
    age = date.today().year - int(dob[:4])
    parts = [
        f"Ασθενής {name}, ηλικίας {age} ετών, ομάδα αίματος {blood}."
    ]
    if diag_names:
        parts.append(
            "Ενεργές διαγνώσεις: " + ", ".join(diag_names) + "."
        )
    if med_names:
        parts.append(
            "Τρέχουσα αγωγή: " + ", ".join(med_names) + "."
        )
    if proc_names:
        parts.append(
            "Διενεργηθείσες διαγνωστικές/θεραπευτικές πράξεις: "
            + ", ".join(proc_names) + "."
        )
    if hosp_names:
        parts.append(
            "Νοσηλεύτηκε σε: " + ", ".join(set(hosp_names)) + "."
        )
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Schema constraints & indexes
# ══════════════════════════════════════════════════════════════════════════════

CONSTRAINTS = [
    "CREATE CONSTRAINT patient_id   IF NOT EXISTS FOR (n:Patient)   REQUIRE n.patientId   IS UNIQUE",
    "CREATE CONSTRAINT doctor_id    IF NOT EXISTS FOR (n:Doctor)    REQUIRE n.doctorId    IS UNIQUE",
    "CREATE CONSTRAINT diag_code    IF NOT EXISTS FOR (n:Diagnosis) REQUIRE n.icdCode     IS UNIQUE",
    "CREATE CONSTRAINT med_id       IF NOT EXISTS FOR (n:Medication) REQUIRE n.drugId     IS UNIQUE",
    "CREATE CONSTRAINT proc_code    IF NOT EXISTS FOR (n:Procedure) REQUIRE n.procCode    IS UNIQUE",
    "CREATE CONSTRAINT hospital_id  IF NOT EXISTS FOR (n:Hospital)  REQUIRE n.hospitalId IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX patient_name  IF NOT EXISTS FOR (n:Patient)   ON (n.name)",
    "CREATE INDEX doctor_spec   IF NOT EXISTS FOR (n:Doctor)    ON (n.specialty)",
    "CREATE INDEX dx_edge_severity IF NOT EXISTS FOR ()-[r:HAS_DIAGNOSIS]-() ON (r.severity)",
    "CREATE INDEX med_name      IF NOT EXISTS FOR (m:Medication) ON (m.name)",
    "CREATE FULLTEXT INDEX doctorNames    IF NOT EXISTS FOR (n:Doctor)    ON EACH [n.name]",
    "CREATE FULLTEXT INDEX patientNames   IF NOT EXISTS FOR (n:Patient)   ON EACH [n.name]",
    "CREATE FULLTEXT INDEX diagnosisNames IF NOT EXISTS FOR (n:Diagnosis) ON EACH [n.name]",
]


# ══════════════════════════════════════════════════════════════════════════════
# Node creation
# ══════════════════════════════════════════════════════════════════════════════

def create_hospitals(graph: Neo4jConnection) -> None:
    print("[INFO] Creating Hospital nodes …")
    for hid, name, beds, city in HOSPITALS:
        graph.query(
            """
            MERGE (h:Hospital {hospitalId: $hid})
            SET h.name      = $name,
                h.bedCount  = $beds,
                h.city      = $city,
                h.address   = $address
            """,
            parameters={
                "hid":     hid,
                "name":    name,
                "beds":    beds,
                "city":    city,
                "address": fake.address().replace("\n", ", "),
            },
        )


def create_diagnoses(graph: Neo4jConnection) -> None:
    print("[INFO] Creating Diagnosis nodes …")
    for icd, name, description in DIAGNOSES:
        graph.query(
            """
            MERGE (d:Diagnosis {icdCode: $icd})
            SET d.name                = $name,
                d.clinicalDescription = $description
            """,
            parameters={
                "icd":         icd,
                "name":        name,
                "description": description,
            },
        )


def create_medications(graph: Neo4jConnection) -> None:
    print("[INFO] Creating Medication nodes …")
    for did, name, dosage in MEDICATIONS:
        graph.query(
            """
            MERGE (m:Medication {drugId: $did})
            SET m.name     = $name,
                m.dosageMg = $dosage
            """,
            parameters={"did": did, "name": name, "dosage": dosage},
        )


def create_procedures(graph: Neo4jConnection) -> None:
    print("[INFO] Creating Procedure nodes …")
    for pid, name, duration, anaesthesia, description in PROCEDURES:
        graph.query(
            """
            MERGE (p:Procedure {procCode: $pid})
            SET p.name                   = $name,
                p.durationMin            = $duration,
                p.anaesthesia            = $anaesthesia,
                p.procedureDescription   = $description
            """,
            parameters={
                "pid":         pid,
                "name":        name,
                "duration":    duration,
                "anaesthesia": anaesthesia,
                "description": description,
            },
        )


def create_doctors(
    graph: Neo4jConnection, n: int = 30
) -> tuple[list[str], dict[str, str]]:
    print(f"[INFO] Creating {n} Doctor nodes …")
    hospital_ids = [h[0] for h in HOSPITALS]
    doctor_ids: list[str] = []
    doctor_hospital: dict[str, str] = {}   # did → hid
    for i in range(1, n + 1):
        did        = f"DOC{i:04d}"
        name       = fake.name()
        specialty  = random.choice(SPECIALTIES)
        expertise  = _EXPERTISE_TEMPLATES[specialty]
        license_no = f"GR-{random.randint(10000, 99999)}"
        years_exp  = random.randint(2, 35)
        hid        = random.choice(hospital_ids)
        doctor_hospital[did] = hid
        graph.query(
            """
            MERGE (d:Doctor {doctorId: $did})
            SET d.name       = $name,
                d.specialty  = $specialty,
                d.expertise  = $expertise,
                d.licenseNo  = $licenseNo,
                d.yearsExp   = $yearsExp
            """,
            parameters={
                "did":       did,
                "name":      name,
                "specialty": specialty,
                "expertise": expertise,
                "licenseNo": license_no,
                "yearsExp":  years_exp,
                "hid":       hid,
            },
        )
        doctor_ids.append(did)
    return doctor_ids, doctor_hospital


def create_patients(graph: Neo4jConnection, n: int = 100) -> list[str]:
    """
    Creates Patient nodes WITHOUT summary — summaries are added later
    in enrich_patient_summaries() after relationships are built.
    """
    print(f"[INFO] Creating {n} Patient nodes …")
    patient_ids = []
    for i in range(1, n + 1):
        pid = f"PAT{i:05d}"
        dob = str(
            date(random.randint(1940, 2005), random.randint(1, 12), random.randint(1, 28))
        )
        graph.query(
            """
            MERGE (p:Patient {patientId: $pid})
            SET p.name        = $name,
                p.dateOfBirth = $dob,
                p.bloodType   = $blood,
                p.phone       = $phone
            """,
            parameters={
                "pid":   pid,
                "name":  fake.name(),
                "dob":   dob,
                "blood": random.choice(BLOOD_TYPES),
                "phone": fake.phone_number(),
            },
        )
        patient_ids.append(pid)
    return patient_ids


# ══════════════════════════════════════════════════════════════════════════════
# Relationship creation  (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════

def rand_date(start_year: int = 2020, end_year: int = 2024) -> str:
    start = date(start_year, 1, 1)
    end   = date(end_year, 12, 31)
    return str(start + timedelta(days=random.randint(0, (end - start).days)))


def rand_end_date(start: str, max_days: int = 365) -> str:
    d = date.fromisoformat(start)
    return str(d + timedelta(days=random.randint(30, max_days)))


def link_patients_to_doctors(
    graph: Neo4jConnection,
    patient_ids: list[str],
    doctor_ids: list[str],
    patient_hospitals: dict[str, list[str]],
    doctor_hospital: dict[str, str],
) -> dict[str, list[str]]:
    print("[INFO] Linking Patients → Doctors (TREATED_BY) …")
    patient_doctors: dict[str, list[str]] = {}
    for pid in patient_ids:
        hosps = set(patient_hospitals.get(pid, []))
        # Only doctors at the patient's hospital(s); fall back to all if none match
        eligible = [d for d in doctor_ids if doctor_hospital.get(d) in hosps] or doctor_ids
        k = min(random.randint(1, 3), len(eligible))
        assigned = random.sample(eligible, k=k)
        patient_doctors[pid] = assigned
        for did in assigned:
            graph.query(
                """
                MATCH (p:Patient {patientId: $pid})
                MATCH (d:Doctor  {doctorId:  $did})
                MERGE (p)-[r:TREATED_BY]->(d)
                SET r.since            = $since,
                    r.primaryPhysician = $primary
                """,
                parameters={
                    "pid":     pid, "did": did,
                    "since":   rand_date(2018, 2024),
                    "primary": random.random() < 0.3,
                },
            )
    return patient_doctors


def link_patients_to_diagnoses(
    graph: Neo4jConnection,
    patient_ids: list[str],
    patient_doctors: dict[str, list[str]],
) -> dict[str, list[str]]:
    print("[INFO] Linking Patients → Diagnoses (HAS_DIAGNOSIS) …")
    diag_codes = [d[0] for d in DIAGNOSES]
    patient_diagnoses: dict[str, list[str]] = {}
    for pid in patient_ids:
        assigned = random.sample(diag_codes, k=random.randint(1, 4))
        patient_diagnoses[pid] = assigned
        doctors = patient_doctors.get(pid, [])
        for icd in assigned:
            dx_date = rand_date(2015, 2024)
            graph.query(
                """
                MATCH (p:Patient   {patientId: $pid})
                MATCH (d:Diagnosis {icdCode:   $icd})
                MERGE (p)-[r:HAS_DIAGNOSIS]->(d)
                SET r.date        = $date,
                    r.confirmed   = $confirmed,
                    r.notes       = $notes,
                    r.severity    = $severity,
                    r.chronic     = $chronic,
                    r.diagnosedBy = $diagnosedBy
                """,
                parameters={
                    "pid":         pid, "icd": icd,
                    "date":        dx_date,
                    "confirmed":   random.random() < 0.85,
                    "notes":       random.choice([
                        "Routine screening", "Emergency presentation",
                        "Follow-up consultation", "Referral from GP", "",
                    ]),
                    "severity":    random.choice(SEVERITIES),
                    "chronic":     random.random() < 0.5,
                    "diagnosedBy": random.choice(doctors) if doctors else None,
                },
            )
    return patient_diagnoses


def link_patients_to_medications(
    graph: Neo4jConnection,
    patient_ids: list[str],
    patient_diagnoses: dict[str, list[str]],
) -> None:
    print("[INFO] Linking Patients → Medications (PRESCRIBED) …")
    med_ids = [m[0] for m in MEDICATIONS]
    for pid in patient_ids:
        diags = patient_diagnoses.get(pid, [])
        for mid in random.sample(med_ids, k=random.randint(1, 5)):
            start = rand_date(2019, 2024)
            graph.query(
                """
                MATCH (p:Patient    {patientId: $pid})
                MATCH (m:Medication {drugId:    $mid})
                MERGE (p)-[r:PRESCRIBED]->(m)
                SET r.startDate  = $start,
                    r.endDate    = $end,
                    r.active     = $active,
                    r.frequency  = $frequency,
                    r.icdCode    = $icdCode
                """,
                parameters={
                    "pid":       pid, "mid": mid,
                    "start":     start,
                    "end":       rand_end_date(start) if random.random() < 0.4 else None,
                    "active":    random.random() < 0.7,
                    "frequency": random.choice(PRESCRIPTION_FREQS),
                    "icdCode":   random.choice(diags) if diags else None,
                },
            )


def link_patients_to_procedures(
    graph: Neo4jConnection, patient_ids: list[str], doctor_ids: list[str]
) -> None:
    print("[INFO] Linking Patients → Procedures (UNDERWENT) …")
    proc_codes = [p[0] for p in PROCEDURES]
    for pid in patient_ids:
        num = random.randint(0, 3)
        if num == 0:
            continue
        for pcode in random.sample(proc_codes, k=num):
            did   = random.choice(doctor_ids)
            pdate = rand_date(2019, 2024)
            graph.query(
                """
                MATCH (pat:Patient   {patientId: $pid})
                MATCH (pr:Procedure  {procCode:  $pcode})
                MATCH (doc:Doctor    {doctorId:  $did})
                MERGE (pat)-[r:UNDERWENT]->(pr)
                SET r.date        = $date,
                    r.outcome     = $outcome,
                    r.performedBy = $did
                MERGE (pr)-[s:PERFORMED_BY]->(doc)
                SET s.role = $role
                """,
                parameters={
                    "pid":     pid, "pcode": pcode, "did": did,
                    "date":    pdate,
                    "outcome": random.choice(OUTCOMES),
                    "role":    random.choice(["lead surgeon", "assistant", "supervising", "consultant"]),
                },
            )


def link_patients_to_hospitals(
    graph: Neo4jConnection, patient_ids: list[str]
) -> dict[str, list[str]]:
    print("[INFO] Linking Patients → Hospitals (ADMITTED_TO) …")
    hospital_ids = [h[0] for h in HOSPITALS]
    patient_hospitals: dict[str, list[str]] = {}   # pid → [hid, ...]
    for pid in patient_ids:
        if random.random() < 0.6:
            num_admissions = random.randint(1, 2)
            selected = random.sample(hospital_ids, k=num_admissions)
            patient_hospitals[pid] = selected

            # Start from a date early enough to allow sequencing within the range
            adm_date = date(random.randint(2019, 2022), random.randint(1, 12), random.randint(1, 28))

            for idx, hid in enumerate(selected):
                is_last = (idx == num_admissions - 1)
                # Only the last admission may still be ongoing (15 % chance)
                still_admitted = is_last and random.random() < 0.15

                los = random.randint(1, 21)
                dis_date = adm_date + timedelta(days=los)

                graph.query(
                    """
                    MATCH (p:Patient  {patientId:  $pid})
                    MATCH (h:Hospital {hospitalId: $hid})
                    MERGE (p)-[r:ADMITTED_TO]->(h)
                    SET r.admissionDate = $adm,
                        r.dischargeDate = $dis,
                        r.ward          = $ward,
                        r.lengthOfStay  = $los
                    """,
                    parameters={
                        "pid":  pid, "hid": hid,
                        "adm":  str(adm_date),
                        "dis":  None if still_admitted else str(dis_date),
                        "ward": random.choice(WARDS),
                        "los":  None if still_admitted else los,
                    },
                )

                # Next admission starts 30–180 days after discharge
                adm_date = dis_date + timedelta(days=random.randint(30, 180))

    return patient_hospitals


def link_comorbid_diagnoses(graph: Neo4jConnection) -> None:
    print("[INFO] Adding Diagnosis comorbidity edges (COMORBID_WITH) …")
    pairs = [
        ("I10", "E11"), ("I10", "E78"), ("I10", "I50"),
        ("E11", "E78"), ("E11", "N18"), ("I50", "N18"),
        ("I21", "I50"), ("J45", "G40"), ("F32", "M54"),
    ]
    for a, b in pairs:
        graph.query(
            """
            MATCH (da:Diagnosis {icdCode: $a})
            MATCH (db:Diagnosis {icdCode: $b})
            MERGE (da)-[r:COMORBID_WITH]->(db)
            SET r.evidenceLevel = $ev,
                r.studyCount    = $sc
            """,
            parameters={
                "a": a, "b": b,
                "ev": random.choice(["strong", "moderate", "emerging"]),
                "sc": random.randint(5, 200),
            },
        )


def link_drug_interactions(graph: Neo4jConnection) -> None:
    print("[INFO] Adding Medication interaction edges (INTERACTS_WITH) ...")
    interactions = [
        ("Warfarin",      "Aspirin",       "high",   "Αυξημένος κίνδυνος αιμορραγίας"),
        ("Warfarin",      "Metformin",     "low",    "Ήπια μεταβολή αντιπηκτικής δράσης"),
        ("Lisinopril",    "Furosemide",    "medium", "Κίνδυνος υπότασης"),
        ("Metoprolol",    "Amlodipine",    "low",    "Προσεκτική παρακολούθηση ΑΠ"),
        ("Fluoxetine",    "Metoprolol",    "medium", "Αυξημένα επίπεδα μετοπρολόλης"),
        ("Levetiracetam", "Furosemide",    "low",    "Πιθανή μεταβολή επιπέδων φαρμάκου"),
        ("Warfarin",      "Atorvastatin",  "medium", "Αύξηση INR"),
        ("Aspirin",       "Furosemide",    "medium", "Μείωση διουρητικής δράσης"),
        ("Omeprazole",    "Metformin",     "low",    "Ήπια αλληλεπίδραση απορρόφησης"),
        ("Fluoxetine",    "Aspirin",       "medium", "Αυξημένος κίνδυνος αιμορραγίας"),
    ]
    for drug_a, drug_b, severity, effect in interactions:
        graph.query(
            """
            MATCH (a:Medication {name: $drug_a})
            MATCH (b:Medication {name: $drug_b})
            MERGE (a)-[r:INTERACTS_WITH]->(b)
            SET r.severity = $severity,
                r.effect   = $effect
            MERGE (b)-[s:INTERACTS_WITH]->(a)
            SET s.severity = $severity,
                s.effect   = $effect
            """,
            parameters={
                "drug_a":   drug_a,
                "drug_b":   drug_b,
                "severity": severity,
                "effect":   effect,
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# Patient summary enrichment  (NEW)
# ══════════════════════════════════════════════════════════════════════════════

def enrich_patient_summaries(graph: Neo4jConnection, patient_ids: list[str]) -> None:
    """
    For each patient, traverse the graph to collect clinical context and
    write a free-text `summary` field suitable for embedding.
    """
    print("[INFO] Enriching Patient nodes with clinical summaries …")

    # Fetch everything in one query per patient to minimise round-trips.
    for pid in patient_ids:
        result = graph.query(
            """
            MATCH (p:Patient {patientId: $pid})
            OPTIONAL MATCH (p)-[:HAS_DIAGNOSIS]->(dx:Diagnosis)
            OPTIONAL MATCH (p)-[rx:PRESCRIBED]->(med:Medication) WHERE rx.active = true
            OPTIONAL MATCH (p)-[:UNDERWENT]->(proc:Procedure)
            OPTIONAL MATCH (p)-[:ADMITTED_TO]->(h:Hospital)
            RETURN p.name        AS name,
                   p.dateOfBirth AS dob,
                   p.bloodType   AS blood,
                   collect(DISTINCT dx.name)   AS diagnoses,
                   collect(DISTINCT med.name)  AS medications,
                   collect(DISTINCT proc.name) AS procedures,
                   collect(DISTINCT h.name)    AS hospitals
            """,
            parameters={"pid": pid},
        )
        if not result:
            continue
        row = result[0]
        summary = _build_patient_summary(
            name       = row["name"],
            dob        = row["dob"],
            blood      = row["blood"],
            diag_names = row["diagnoses"],
            med_names  = row["medications"],
            proc_names = row["procedures"],
            hosp_names = row["hospitals"],
        )
        graph.query(
            "MATCH (p:Patient {patientId: $pid}) SET p.summary = $summary",
            parameters={"pid": pid, "summary": summary},
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Populate a medical Neo4j graph with random data")
    parser.add_argument("--uri",      default="bolt://localhost:7687")
    parser.add_argument("--user",     default="neo4j")
    parser.add_argument("--password", default="quantapass")
    parser.add_argument("--patients", type=int, default=100)
    parser.add_argument("--doctors",  type=int, default=30)
    parser.add_argument("--clean",    action="store_true")
    parser.add_argument("--demo",     action="store_true")
    args = parser.parse_args()

    graph = neo4j_connection(
        neo4j_settings={
            "connection_url": args.uri,
            "username":       args.user,
            "password":       args.password,
        },
        clean_graph=args.clean,
    )

    print("[INFO] Applying constraints and indexes …")
    for stmt in CONSTRAINTS + INDEXES:
        graph.query(stmt)

    create_hospitals(graph)
    create_diagnoses(graph)
    create_medications(graph)
    create_procedures(graph)

    doctor_ids, doctor_hospital = create_doctors(graph, n=args.doctors)
    patient_ids = create_patients(graph, n=args.patients)

    # Hospitals first — needed to constrain which doctors can treat each patient
    patient_hospitals = link_patients_to_hospitals(graph, patient_ids)
    patient_doctors   = link_patients_to_doctors(graph, patient_ids, doctor_ids,
                                                  patient_hospitals, doctor_hospital)
    patient_diagnoses = link_patients_to_diagnoses(graph, patient_ids, patient_doctors)
    link_patients_to_medications(graph, patient_ids, patient_diagnoses)
    link_patients_to_procedures(graph, patient_ids, doctor_ids)
    link_comorbid_diagnoses(graph)
    link_drug_interactions(graph)

    # ── NEW: enrich summaries after all relationships are in place ─────────
    enrich_patient_summaries(graph, patient_ids)

    # schema = graph.get_schema()
    # print("\n[INFO] Graph schema:")
    # print(schema)

    graph.close()
    print("\n[INFO] Done ✓")
    print("\n[INFO] Embedding-ready text fields:")
    print("  Patient   → p.summary")
    print("  Doctor    → d.expertise")
    print("  Diagnosis → d.clinicalDescription")
    print("  Procedure → p.procedureDescription")


if __name__ == "__main__":
    main()