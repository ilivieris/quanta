"""
Medical Graph Population Script — English edition
================================================
English-language twin of ``populate_medical_graph.py``. It builds the exact
same Neo4j ontology (same node labels, relationship types, constraints and
fulltext indexes) but every embedding-target text field is written in English
and patient / doctor names are generated with the ``en_US`` Faker locale.

Use this when you want to drive the agent with an English embedding model
(e.g. ``sentence-transformers/all-mpnet-base-v2``) or with the on-device
``needle`` router, which is English-centric.

Embedding-ready text fields (identical to the Greek edition):
  Patient      -> p.summary
  Doctor       -> d.expertise
  Diagnosis    -> d.clinicalDescription
  Procedure    -> p.procedureDescription
  MedicalImage -> BiomedCLIP image embeddings (image_indexer.build())

Usage
-----
    python populate_medical_graph_en.py \
        --uri bolt://localhost:7687 \
        --user neo4j \
        --password secret \
        [--clean] [--no-images]

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

fake = Faker("en_US")
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
        "Essential hypertension is a chronic elevation of arterial blood pressure "
        "with no identifiable secondary cause. It is frequently asymptomatic and "
        "discovered incidentally on routine examination. Management combines "
        "lifestyle modification with antihypertensive agents such as ACE "
        "inhibitors, calcium-channel blockers and diuretics. Long-standing "
        "uncontrolled hypertension raises the risk of stroke and heart failure.",
    ),
    (
        "E11", "Type 2 diabetes mellitus",
        "Type 2 diabetes mellitus is characterised by insulin resistance and "
        "progressive dysfunction of the pancreatic beta cells. Typical "
        "presentation includes polyuria, polydipsia and fatigue. First-line "
        "treatment is metformin together with dietary change, with escalation to "
        "insulin when required. Close monitoring of HbA1c and renal function is "
        "essential.",
    ),
    (
        "J45", "Asthma",
        "Asthma is a chronic inflammatory airway disease with episodic "
        "bronchospasm, wheeze and dyspnoea. Patients have hyper-responsive "
        "airways that react to triggers such as allergens and exercise. Treatment "
        "relies on inhaled corticosteroids for maintenance and short-acting "
        "beta-2 agonists for symptom relief.",
    ),
    (
        "I21", "Acute myocardial infarction",
        "Acute myocardial infarction is ischaemic necrosis of myocardium caused "
        "by sudden occlusion of a coronary artery, usually by thrombus. It "
        "presents with severe chest pain, sweating and dyspnoea. Immediate "
        "reperfusion by primary angioplasty or thrombolysis is critical to "
        "salvage myocardium. Long-term therapy includes antiplatelet agents and "
        "statins.",
    ),
    (
        "K21", "Gastroesophageal reflux",
        "Gastro-oesophageal reflux disease refers to reflux of gastric contents "
        "into the oesophagus. Typical symptoms are heartburn and regurgitation, "
        "often worse after meals or when lying supine. It is managed with "
        "proton-pump inhibitors, dietary change and elevation of the head of the "
        "bed.",
    ),
    (
        "G40", "Epilepsy",
        "Epilepsy is characterised by recurrent unprovoked seizures caused by "
        "abnormal neuronal discharge. The clinical picture ranges from focal "
        "seizures with impaired awareness to generalised tonic-clonic seizures. "
        "EEG and brain imaging are the core diagnostic tools. Antiepileptic drugs "
        "achieve seizure control in the majority of patients.",
    ),
    (
        "C34", "Malignant neoplasm of bronchus",
        "Malignant neoplasm of the bronchus and lung is one of the most common "
        "cancers worldwide and is strongly linked to smoking. It is divided into "
        "small-cell and non-small-cell types. Symptoms include chronic cough, "
        "haemoptysis and weight loss. Treatment depends on stage and histology: "
        "surgery, chemotherapy, radiotherapy or immunotherapy.",
    ),
    (
        "N18", "Chronic kidney disease",
        "Chronic kidney disease is a progressive and irreversible decline in "
        "renal function. Common causes include diabetes, hypertension and "
        "glomerulonephritis. It is monitored with GFR and albuminuria. End-stage "
        "disease requires dialysis or kidney transplantation. Treating the "
        "underlying cause slows progression.",
    ),
    (
        "F32", "Major depressive disorder",
        "Major depressive disorder is characterised by persistent low mood, loss "
        "of interest, disturbed sleep and appetite, fatigue and thoughts of "
        "worthlessness or death. Treatment combines SSRIs with cognitive "
        "behavioural psychotherapy. Long-term monitoring for relapse is "
        "important.",
    ),
    (
        "M54", "Dorsalgia / back pain",
        "Dorsalgia, or back pain, is one of the most common musculoskeletal "
        "complaints. It may be acute or chronic, mechanical or radicular. The "
        "commonest causes are muscular strain, disc degeneration and "
        "spondyloarthritis. Management includes physiotherapy, NSAIDs and, in "
        "persistent cases, interventional techniques.",
    ),
    (
        "I50", "Heart failure",
        "Heart failure is a clinical syndrome in which the heart cannot meet the "
        "metabolic demands of the body. It presents with dyspnoea, fatigue and "
        "lower-limb oedema. Echocardiography assesses the ejection fraction. "
        "Treatment is based on ACE inhibitors, beta-blockers and diuretics, with "
        "device therapy or transplantation in severe cases.",
    ),
    (
        "E78", "Hyperlipidaemia",
        "Hyperlipidaemia refers to raised blood cholesterol and/or triglycerides "
        "and is a major risk factor for atherosclerosis and cardiovascular "
        "disease. Treatment starts with dietary intervention and, when "
        "insufficient, statins are added. Regular monitoring of the lipid profile "
        "is required.",
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
        "Coronary angiography is an invasive imaging technique for assessing the "
        "coronary arteries. A catheter is introduced from the femoral or radial "
        "artery and contrast is injected. It is indicated for suspected coronary "
        "artery disease or before a revascularisation procedure. It allows "
        "immediate angioplasty when required.",
    ),
    (
        "PROC002", "Upper GI endoscopy", 30, True,
        "Upper GI endoscopy is an endoscopic examination of the oesophagus, "
        "stomach and duodenum. It is indicated for investigation of dyspepsia, "
        "bleeding or foreign-body ingestion. It permits biopsy and therapeutic "
        "interventions such as haemostasis or polyp removal.",
    ),
    (
        "PROC003", "Bronchoscopy", 45, True,
        "Bronchoscopy is an endoscopic examination of the trachea and bronchi. It "
        "is used to investigate haemoptysis, suspicious lesions or foreign bodies "
        "and to take biopsies. It also allows bronchoalveolar lavage for "
        "microbiological analysis in infection.",
    ),
    (
        "PROC004", "Echocardiography", 30, False,
        "Echocardiography is non-invasive ultrasound imaging of the heart. It "
        "assesses structure, function, valves and ejection fraction. It is a core "
        "tool in the diagnosis of heart failure, valvular disease and pericardial "
        "effusion.",
    ),
    (
        "PROC005", "Lumbar puncture", 20, True,
        "Lumbar puncture is an invasive technique for sampling cerebrospinal "
        "fluid. It is indicated for the diagnosis of meningitis, subarachnoid "
        "haemorrhage and neurological disease. The sample is analysed for cells, "
        "protein, glucose and microorganisms.",
    ),
    (
        "PROC006", "Bone marrow biopsy", 30, True,
        "Bone marrow biopsy is usually taken from the posterior iliac crest under "
        "local anaesthesia. It is indicated in the investigation of "
        "haematological malignancy and anaemia and for lymphoma staging. "
        "Histology assesses cellularity and pathological infiltration.",
    ),
    (
        "PROC007", "Renal biopsy", 45, True,
        "Renal biopsy is the invasive sampling of kidney tissue under ultrasound "
        "guidance. It is indicated for the diagnosis of glomerulonephritis, "
        "nephrotic syndrome and the cause of renal failure. Histology guides "
        "treatment decisions.",
    ),
    (
        "PROC008", "Colonoscopy", 45, True,
        "Colonoscopy is the endoscopic examination of the large bowel. It is "
        "indicated for colorectal cancer screening and investigation of bleeding "
        "or altered bowel habit. It allows polyp removal and biopsy. Bowel "
        "preparation with a cleansing regimen is required.",
    ),
    (
        "PROC009", "MRI brain", 60, False,
        "Brain MRI provides detailed imaging of the cerebral parenchyma without "
        "ionising radiation. It is indicated in stroke, tumours, inflammation and "
        "epilepsy. With gadolinium contrast it detects disruption of the "
        "blood-brain barrier.",
    ),
    (
        "PROC010", "CT thorax", 20, False,
        "CT of the thorax provides detailed images of the lungs, mediastinum and "
        "vessels. It is indicated for the investigation of pulmonary embolism, "
        "interstitial lung disease and neoplasia. Low-dose CT is a screening tool "
        "for lung cancer in high-risk individuals.",
    ),
    (
        "PROC011", "Electroencephalogram (EEG)", 60, False,
        "The electroencephalogram records the brain's electrical activity through "
        "electrodes placed on the scalp. It is a core tool for diagnosing "
        "epilepsy, encephalopathy and sleep disorders. It can be performed at "
        "rest, during sleep or with photic stimulation.",
    ),
    (
        "PROC012", "Haemodialysis session", 240, False,
        "A haemodialysis session replaces renal function in patients with "
        "end-stage kidney disease. Blood passes through an extracorporeal circuit "
        "across a semipermeable membrane that removes toxins and excess fluid. It "
        "is usually performed three times a week for four hours.",
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
        "Cardiologist focused on coronary artery disease, heart failure and "
        "interventional cardiology. Extensive experience with coronary "
        "angiography and angioplasty. Follows patients after acute myocardial "
        "infarction and pacemaker implantation."
    ),
    "Neurology": (
        "Neurologist specialising in epilepsy, stroke and neurodegenerative "
        "disease. Performs electroencephalograms and takes part in "
        "multidisciplinary teams managing multiple sclerosis."
    ),
    "Oncology": (
        "Oncologist experienced in chemotherapy, immunotherapy and targeted "
        "therapy. Follows patients with lung, breast and gastrointestinal "
        "cancer. Works closely with radiation oncologists and surgical "
        "oncologists."
    ),
    "Orthopedics": (
        "Orthopaedic surgeon specialising in knee and hip arthroplasty, sports "
        "injuries and spinal disease. Uses minimally invasive techniques for "
        "faster recovery."
    ),
    "Gastroenterology": (
        "Gastroenterologist specialising in endoscopy, inflammatory bowel "
        "disease and hepatology. Performs gastroscopy, colonoscopy and ERCP. "
        "Follows patients with cirrhosis and chronic hepatitis."
    ),
    "Endocrinology": (
        "Endocrinologist focused on type 1 and type 2 diabetes, thyroid disease "
        "and adrenal disorders. Provides insulin-pump training and "
        "interpretation of continuous glucose monitoring."
    ),
    "Pulmonology": (
        "Pulmonologist experienced in COPD, asthma, interstitial lung disease "
        "and pulmonary hypertension. Performs bronchoscopy and spirometry. "
        "Specialises in managing patients with respiratory failure."
    ),
    "Nephrology": (
        "Nephrologist specialising in chronic kidney disease, haemodialysis and "
        "peritoneal dialysis. Follows patients before and after kidney "
        "transplantation. Experienced in renal biopsy and management of "
        "glomerulonephritis."
    ),
    "Dermatology": (
        "Dermatologist focused on dermato-oncology, psoriasis and allergic skin "
        "disease. Performs dermoscopy, biopsy and excision of skin lesions. "
        "Experienced with biologic agents for inflammatory disease."
    ),
    "Psychiatry": (
        "Psychiatrist specialising in major depressive disorder, anxiety "
        "disorders and psychosis. Combines psychopharmacology with "
        "psychotherapeutic techniques. Experienced in crisis care and "
        "involuntary admission."
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
    parts = [f"Patient {name}, age {age}, blood type {blood}."]
    if diag_names:
        parts.append("Active diagnoses: " + ", ".join(diag_names) + ".")
    if med_names:
        parts.append("Current medications: " + ", ".join(med_names) + ".")
    if proc_names:
        parts.append("Procedures performed: " + ", ".join(proc_names) + ".")
    if hosp_names:
        parts.append("Admitted at: " + ", ".join(sorted(set(hosp_names))) + ".")
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Schema constraints & indexes
# ══════════════════════════════════════════════════════════════════════════════

# ── Image type configuration ──────────────────────────────────────────────────

# medmnist dataset → (description, modality)
_IMAGE_TYPE_META: dict[str, dict] = {
    "chest_xray": {"description": "Chest X-ray",         "modality": "X-Ray",      "dataset": "ChestMNIST"},
    "ct_scan":    {"description": "CT scan",              "modality": "CT",         "dataset": "OrganAMNIST"},
    "pathology":  {"description": "Histopathology slide", "modality": "Microscopy", "dataset": "PathMNIST"},
}

# ICD code → preferred image type for random assignment
_DIAG_TO_IMAGE_TYPE: dict[str, str] = {
    "I10": "chest_xray",  # hypertension
    "E11": "ct_scan",     # diabetes
    "J45": "chest_xray",  # asthma
    "I21": "chest_xray",  # MI
    "K21": "ct_scan",     # GERD
    "G40": "chest_xray",  # epilepsy (fallback — no brain MRI in medmnist)
    "C34": "pathology",   # lung cancer
    "N18": "ct_scan",     # CKD
    "F32": "chest_xray",  # depression (fallback)
    "M54": "ct_scan",     # back pain
    "I50": "chest_xray",  # heart failure
    "E78": "chest_xray",  # hyperlipidemia
}

CONSTRAINTS = [
    "CREATE CONSTRAINT patient_id   IF NOT EXISTS FOR (n:Patient)      REQUIRE n.patientId   IS UNIQUE",
    "CREATE CONSTRAINT doctor_id    IF NOT EXISTS FOR (n:Doctor)        REQUIRE n.doctorId    IS UNIQUE",
    "CREATE CONSTRAINT diag_code    IF NOT EXISTS FOR (n:Diagnosis)     REQUIRE n.icdCode     IS UNIQUE",
    "CREATE CONSTRAINT med_id       IF NOT EXISTS FOR (n:Medication)    REQUIRE n.drugId      IS UNIQUE",
    "CREATE CONSTRAINT proc_code    IF NOT EXISTS FOR (n:Procedure)     REQUIRE n.procCode    IS UNIQUE",
    "CREATE CONSTRAINT hospital_id  IF NOT EXISTS FOR (n:Hospital)      REQUIRE n.hospitalId  IS UNIQUE",
    "CREATE CONSTRAINT image_id     IF NOT EXISTS FOR (n:MedicalImage)  REQUIRE n.imageId     IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX patient_name     IF NOT EXISTS FOR (n:Patient)       ON (n.name)",
    "CREATE INDEX doctor_spec      IF NOT EXISTS FOR (n:Doctor)        ON (n.specialty)",
    "CREATE INDEX dx_edge_severity IF NOT EXISTS FOR ()-[r:HAS_DIAGNOSIS]-() ON (r.severity)",
    "CREATE INDEX med_name         IF NOT EXISTS FOR (m:Medication)    ON (m.name)",
    "CREATE INDEX image_type_idx   IF NOT EXISTS FOR (n:MedicalImage)  ON (n.imageType)",
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
# Relationship creation
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
        ("Warfarin",      "Aspirin",       "high",   "Increased bleeding risk"),
        ("Warfarin",      "Metformin",     "low",    "Minor change in anticoagulant effect"),
        ("Lisinopril",    "Furosemide",    "medium", "Risk of hypotension"),
        ("Metoprolol",    "Amlodipine",    "low",    "Monitor blood pressure closely"),
        ("Fluoxetine",    "Metoprolol",    "medium", "Elevated metoprolol levels"),
        ("Levetiracetam", "Furosemide",    "low",    "Possible change in drug levels"),
        ("Warfarin",      "Atorvastatin",  "medium", "INR elevation"),
        ("Aspirin",       "Furosemide",    "medium", "Reduced diuretic effect"),
        ("Omeprazole",    "Metformin",     "low",    "Minor absorption interaction"),
        ("Fluoxetine",    "Aspirin",       "medium", "Increased bleeding risk"),
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
# Patient summary enrichment
# ══════════════════════════════════════════════════════════════════════════════

def enrich_patient_summaries(graph: Neo4jConnection, patient_ids: list[str]) -> None:
    """
    For each patient, traverse the graph to collect clinical context and
    write a free-text `summary` field suitable for embedding.
    """
    print("[INFO] Enriching Patient nodes with clinical summaries …")

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
# Medical image download & creation
# ══════════════════════════════════════════════════════════════════════════════

def _download_medmnist_images(image_dir: str, n_per_type: int = 60) -> dict[str, list[str]]:
    """Download medmnist sample images to disk. Returns {image_type: [filepath, ...]}."""
    try:
        import medmnist
    except ImportError:
        print("[ERROR] medmnist not installed.  Run: pip install medmnist")
        return {}

    try:
        from PIL import Image as PILImage
        import numpy as np
    except ImportError:
        print("[ERROR] Pillow/numpy not installed.  Run: pip install Pillow numpy")
        return {}

    Path(image_dir).mkdir(parents=True, exist_ok=True)
    type_to_files: dict[str, list[str]] = {}

    for img_type, meta in _IMAGE_TYPE_META.items():
        ds_name = meta["dataset"]
        DataClass = getattr(medmnist, ds_name, None)
        if DataClass is None:
            print(f"[WARN] medmnist class {ds_name!r} not found — skipping {img_type}")
            continue

        dataset = DataClass(split="test", download=True)

        files: list[str] = []
        for idx in range(min(n_per_type, len(dataset))):
            img_data, _ = dataset[idx]

            if hasattr(img_data, "convert"):
                img = img_data.convert("RGB").resize((224, 224), PILImage.BILINEAR)
            else:
                arr = np.array(img_data)
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=-1)
                elif arr.shape[-1] == 1:
                    arr = np.repeat(arr, 3, axis=-1)
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                img = PILImage.fromarray(arr.astype(np.uint8)).resize((224, 224), PILImage.BILINEAR)

            filepath = str(Path(image_dir) / f"{img_type}_{idx:04d}.png")
            img.save(filepath)
            files.append(filepath)

        type_to_files[img_type] = files
        print(f"[Images] Saved {len(files)} {img_type} images → {image_dir}/")

    return type_to_files


def create_and_link_medical_images(
    graph: Neo4jConnection,
    image_dir: str,
    patient_ids: list[str],
    patient_diagnoses: dict[str, list[str]],
    images_per_patient: tuple[int, int] = (1, 3),
) -> None:
    """Create MedicalImage nodes and link them to patients based on their diagnoses."""
    print("[INFO] Downloading medical images (medmnist) …")
    type_to_files = _download_medmnist_images(image_dir)
    if not type_to_files:
        print("[WARN] No images available — skipping MedicalImage creation")
        return

    print("[INFO] Creating MedicalImage nodes and linking to patients …")
    img_counter = 0
    for pid in patient_ids:
        diags = patient_diagnoses.get(pid, [])
        n_images = random.randint(*images_per_patient)

        preferred_types = list({_DIAG_TO_IMAGE_TYPE.get(d, "chest_xray") for d in diags})

        for _ in range(n_images):
            img_type = random.choice(preferred_types) if preferred_types else "chest_xray"
            pool = type_to_files.get(img_type) or type_to_files.get("chest_xray", [])
            if not pool:
                continue

            filepath = random.choice(pool)
            img_id   = f"IMG{img_counter:05d}"
            img_counter += 1
            meta     = _IMAGE_TYPE_META[img_type]

            graph.query(
                """
                MERGE (img:MedicalImage {imageId: $img_id})
                SET img.filePath     = $filepath,
                    img.imageType    = $img_type,
                    img.modality     = $modality,
                    img.capturedDate = $date,
                    img.description  = $description,
                    img.patientId    = $pid
                WITH img
                MATCH (p:Patient {patientId: $pid})
                MERGE (p)-[:HAS_IMAGE]->(img)
                """,
                parameters={
                    "img_id":      img_id,
                    "filepath":    filepath,
                    "img_type":    img_type,
                    "modality":    meta["modality"],
                    "date":        rand_date(2020, 2024),
                    "description": meta["description"],
                    "pid":         pid,
                },
            )

    print(f"[INFO] Created {img_counter} MedicalImage nodes")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Populate an English medical Neo4j graph with random data")
    parser.add_argument("--uri",        default="bolt://localhost:7687")
    parser.add_argument("--user",       default="neo4j")
    parser.add_argument("--password",   default="quantapass")
    parser.add_argument("--patients",   type=int, default=100)
    parser.add_argument("--doctors",    type=int, default=30)
    parser.add_argument("--clean",      action="store_true")
    parser.add_argument("--demo",       action="store_true")
    parser.add_argument("--no-images",  action="store_true",
                        help="Skip medical image download and MedicalImage node creation")
    parser.add_argument("--image-dir",  default="./medical_images",
                        help="Directory to store downloaded medical images (default: ./medical_images)")
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

    patient_hospitals = link_patients_to_hospitals(graph, patient_ids)
    patient_doctors   = link_patients_to_doctors(graph, patient_ids, doctor_ids,
                                                  patient_hospitals, doctor_hospital)
    patient_diagnoses = link_patients_to_diagnoses(graph, patient_ids, patient_doctors)
    link_patients_to_medications(graph, patient_ids, patient_diagnoses)
    link_patients_to_procedures(graph, patient_ids, doctor_ids)
    link_comorbid_diagnoses(graph)
    link_drug_interactions(graph)

    enrich_patient_summaries(graph, patient_ids)

    if not args.no_images:
        create_and_link_medical_images(
            graph, args.image_dir, patient_ids, patient_diagnoses
        )

    graph.close()
    print("\n[INFO] Done ✓")
    print("\n[INFO] Embedding-ready text fields (all English):")
    print("  Patient       → p.summary")
    print("  Doctor        → d.expertise")
    print("  Diagnosis     → d.clinicalDescription")
    print("  Procedure     → p.procedureDescription")
    print("  MedicalImage  → image embeddings via BiomedCLIP (image_indexer.build())")


if __name__ == "__main__":
    main()
