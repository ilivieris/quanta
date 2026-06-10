# Medical Search — Hybrid AI Agent for Hospital Knowledge Graphs

A self-contained demonstration of **semantic search + graph traversal + LLM-driven tool use**, applied to a realistic synthetic hospital dataset stored in Neo4j. The agent answers natural-language questions from medical staff by autonomously selecting and chaining 10 specialised tools that combine dense vector similarity, fulltext fuzzy matching, structured Cypher queries, and multi-hop graph traversal.

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Architecture Overview](#2-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [Knowledge Graph Ontology](#4-knowledge-graph-ontology)
   - [Node Types](#41-node-types)
   - [Relationship Types](#42-relationship-types)
   - [Comorbidity Network](#43-comorbidity-network)
   - [Drug Interaction Network](#44-drug-interaction-network)
   - [Indexes & Constraints](#45-indexes--constraints)
5. [Semantic Embedding Strategy](#5-semantic-embedding-strategy)
6. [Agent Tools Reference](#6-agent-tools-reference)
   - [Search & Lookup](#61-search--lookup)
   - [Graph Traversal](#62-graph-traversal)
   - [Patient Intelligence](#63-patient-intelligence)
   - [Doctor Matching](#64-doctor-matching)
7. [Agent Decision Strategy](#7-agent-decision-strategy)
8. [Vector Index — QuantaIndex](#8-vector-index--quantaindex)
9. [Conversation Memory — Redis](#9-conversation-memory--redis)
10. [Project Structure](#10-project-structure)
11. [Quick Start](#11-quick-start)
12. [Configuration Reference](#12-configuration-reference)
13. [Example Queries](#13-example-queries)

---

## 1. Project Purpose

This project serves as a **proof-of-concept** for a hybrid medical information retrieval system that blends three complementary paradigms:

| Paradigm | When it shines | Implemented via |
|---|---|---|
| **Semantic Search** | Fuzzy concepts, symptoms, descriptions without exact IDs | `sentence-transformers` + `QuantaIndex` |
| **Graph Search** | Relationships, traversals, comorbidities, drug interactions | Neo4j + Cypher |
| **Structured Query** | Exact filters, aggregations, date/age ranges, counts | Read-only Cypher via the agent |

A GPT-4.1-mini agent acts as the reasoning layer, receiving user questions in free natural language and autonomously deciding which combination of tools to call, in what order, and how to synthesise the results into a coherent clinical answer.

The dataset is fully synthetic (generated with `Faker` using a Greek locale) and represents a realistic graph of patients, doctors, diagnoses, medications, procedures, and hospitals for a network of Greek hospitals.

---

## 2. Architecture Overview

```
User (natural language)
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  MedicalSearchAgent  (agent.py)                                  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  System Prompt (ontology rules + Cypher rules + schema)   │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  GPT-4.1-mini ──► tool_calls ──► dispatch_tool()                 │
│       ▲                                   │                      │
│       └──── tool results (JSON) ◄─────────┘                      │
└──────────────────────────────────────────────────────────────────┘
           │                          │
           ▼                          ▼
  ┌─────────────────┐       ┌──────────────────────┐
  │  MedicalIndexer │       │   Neo4j Graph DB     │
  │  (indexer.py)   │       │  (neo4j_connection)  │
  │                 │       │                      │
  │  QuantaIndex ×4 │       │  6 node types        │
  │  patients       │       │  8 relationship types│
  │  doctors        │       │  Fulltext indexes    │
  │  diagnoses      │       │  Comorbidity edges   │
  │  procedures     │       │  Drug interactions   │
  └─────────────────┘       └──────────────────────┘
           │
           ▼
  ./medical_indexes/
  (persisted 4-bit vectors)

  ┌──────────────────┐
  │  Redis (optional)│  ← conversation history per session UUID
  └──────────────────┘
```

**Query lifecycle:**

1. User types a question at the REPL.
2. The agent loads conversation history from Redis (if available).
3. GPT-4.1-mini receives the system prompt (which includes the live graph schema), the history, and the question.
4. The model emits one or more `tool_calls`.
5. `dispatch_tool()` routes each call to the correct Python function, which queries Neo4j or the vector indexes.
6. Tool results are appended to the message list and the model generates its next turn.
7. Steps 4–6 repeat until the model produces a plain text answer (no tool calls).
8. The answer is saved to Redis and printed to the user.

---

## 3. Technology Stack

| Component | Library / Service | Role |
|---|---|---|
| Graph database | **Neo4j** ≥ 5 (bolt protocol) | Stores the medical knowledge graph |
| Vector index | **QuantaIndex** (from this repo) | 4-bit quantised dense vector search |
| Embedding model | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768-dim multilingual embeddings |
| LLM | **OpenAI GPT-4.1-mini-2025-04-14** | Reasoning layer, tool selection |
| Conversation store | **Redis** (optional) | Persists multi-turn chat history |
| Data generation | **Faker** (`el_GR` locale) | Synthetic Greek patient/doctor names |
| Settings | **pydantic-settings** + `.env` | Centralised configuration |

---

## 4. Knowledge Graph Ontology

### 4.1 Node Types

#### `Patient` (default: 100 nodes)

| Property | Type | Description |
|---|---|---|
| `patientId` | String (unique) | Primary key, e.g. `PAT00001` |
| `name` | String | Full name (Greek, generated by Faker) |
| `dateOfBirth` | Date string | Used to compute age dynamically in Cypher |
| `bloodType` | String | One of `A+`, `A-`, `B+`, `B-`, `AB+`, `AB-`, `O+`, `O-` |
| `phone` | String | Contact number |
| `summary` | String | **Embedding target** — free-text clinical summary aggregating diagnoses, medications, procedures, and admissions |

The `summary` field is constructed after all relationships are built and reads: *"Patient X, age Y, blood type Z. Active diagnoses: ... Current medications: ... Procedures performed: ... Hospitalised at: ..."*

#### `Doctor` (default: 30 nodes)

| Property | Type | Description |
|---|---|---|
| `doctorId` | String (unique) | Primary key, e.g. `DOC0001` |
| `name` | String | Full name |
| `specialty` | String | One of 10 medical specialties |
| `expertise` | String | **Embedding target** — narrative description of procedural focus and subspecialisation |
| `licenseNo` | String | Medical license number, e.g. `GR-12345` |
| `yearsExp` | Integer | Years of experience (2–35) |
| `hospitalId` | String | Foreign key to their affiliated Hospital (stored as a property, **no WORKS_AT relationship**) |

The 10 available specialties are: **Cardiology, Neurology, Oncology, Orthopedics, Gastroenterology, Endocrinology, Pulmonology, Nephrology, Dermatology, Psychiatry**.

#### `Diagnosis` (13 nodes)

| Property | Type | Description |
|---|---|---|
| `icdCode` | String (unique) | ICD-10 code, e.g. `I10` |
| `name` | String | Human-readable name |
| `clinicalDescription` | String | **Embedding target** — clinical paragraph (in Greek) covering presentation, aetiology, and treatment approach |

| ICD Code | Diagnosis |
|---|---|
| `I10` | Essential hypertension |
| `E11` | Type 2 diabetes mellitus |
| `J45` | Asthma |
| `I21` | Acute myocardial infarction |
| `K21` | Gastroesophageal reflux |
| `G40` | Epilepsy |
| `C34` | Malignant neoplasm of bronchus |
| `N18` | Chronic kidney disease |
| `F32` | Major depressive disorder |
| `M54` | Dorsalgia / back pain |
| `I50` | Heart failure |
| `E78` | Hyperlipidaemia |

#### `Procedure` (12 nodes)

| Property | Type | Description |
|---|---|---|
| `procCode` | String (unique) | Primary key, e.g. `PROC001` |
| `name` | String | Procedure name |
| `durationMin` | Integer | Typical duration in minutes |
| `anaesthesia` | Boolean | Whether anaesthesia is required |
| `procedureDescription` | String | **Embedding target** — narrative covering indication, technique, and expected outcome |

Procedures: Coronary angiography, Upper GI endoscopy, Bronchoscopy, Echocardiography, Lumbar puncture, Bone marrow biopsy, Renal biopsy, Colonoscopy, MRI brain, CT thorax, EEG, Haemodialysis session.

#### `Medication` (12 nodes)

| Property | Type | Description |
|---|---|---|
| `drugId` | String (unique) | Primary key, e.g. `MED001` |
| `name` | String | Drug name |
| `dosageMg` | Integer | Standard dosage in mg |

Medications: Metformin, Lisinopril, Atorvastatin, Aspirin, Omeprazole, Amlodipine, Metoprolol, Salbutamol, Fluoxetine, Levetiracetam, Furosemide, Warfarin.

#### `Hospital` (5 nodes)

| Property | Type | Description |
|---|---|---|
| `hospitalId` | String (unique) | Primary key, e.g. `HOSP001` |
| `name` | String | Hospital name |
| `bedCount` | Integer | Total bed capacity |
| `city` | String | City of operation |
| `address` | String | Full address |

Hospitals: General Hospital of Athens (1200 beds), University Hospital of Patras (800), Hippocration Hospital — Thessaloniki (900), General Hospital of Kalamata (350), Evangelismos Hospital — Athens (1100).

---

### 4.2 Relationship Types

```
(Patient)-[:TREATED_BY]->(Doctor)
(Patient)-[:HAS_DIAGNOSIS]->(Diagnosis)
(Patient)-[:PRESCRIBED]->(Medication)
(Patient)-[:UNDERWENT]->(Procedure)
(Patient)-[:ADMITTED_TO]->(Hospital)
(Procedure)-[:PERFORMED_BY]->(Doctor)
(Diagnosis)-[:COMORBID_WITH]-(Diagnosis)
(Medication)-[:INTERACTS_WITH]-(Medication)
```

#### `TREATED_BY`
| Property | Description |
|---|---|
| `since` | Date when the care relationship started |
| `primaryPhysician` | Boolean — whether this doctor is the primary physician |

Doctors are matched to patients by hospital affiliation. Each patient has 1–3 treating doctors.

#### `HAS_DIAGNOSIS`

> **Note:** `severity` and `chronic` live on the **edge**, not the `Diagnosis` node. This allows different patients to have different severity levels for the same condition.

| Property | Description |
|---|---|
| `date` | Date of diagnosis |
| `confirmed` | Boolean (85% true probability) |
| `notes` | Free-text note (e.g. "Emergency presentation") |
| `severity` | `mild` / `moderate` / `severe` |
| `chronic` | Boolean |
| `diagnosedBy` | `doctorId` of the diagnosing doctor |

#### `PRESCRIBED`

> **Note:** `frequency` lives on the edge (per-patient dosing schedule), not the `Medication` node.

| Property | Description |
|---|---|
| `startDate` | Prescription start date |
| `endDate` | End date (`null` for open prescriptions) |
| `active` | Boolean — whether the prescription is currently active |
| `frequency` | Dosing schedule (e.g. "twice daily", "as needed") |
| `icdCode` | The diagnosis for which the drug was prescribed |

#### `UNDERWENT`
| Property | Description |
|---|---|
| `date` | Date the procedure was performed |
| `outcome` | `successful`, `complicated`, `routine`, `excellent`, `satisfactory` |
| `performedBy` | `doctorId` of the performing doctor |

#### `ADMITTED_TO`

> A `dischargeDate = null` indicates the patient is **currently admitted**.

| Property | Description |
|---|---|
| `admissionDate` | Date of admission |
| `dischargeDate` | Date of discharge (`null` if still admitted) |
| `ward` | Ward name (Cardiology, ICU, Surgery, etc.) |
| `lengthOfStay` | Length in days (`null` if still admitted) |

Admissions are sequenced — the next admission always starts after the previous discharge. 15% of last admissions are ongoing.

#### `PERFORMED_BY`
| Property | Description |
|---|---|
| `role` | `lead surgeon`, `assistant`, `supervising`, `consultant` |

#### `COMORBID_WITH` (bidirectional)
| Property | Description |
|---|---|
| `evidenceLevel` | `strong`, `moderate`, or `emerging` |
| `studyCount` | Number of supporting studies (5–200) |

#### `INTERACTS_WITH` (bidirectional)
| Property | Description |
|---|---|
| `severity` | `high`, `medium`, or `low` |
| `effect` | Clinical description of the interaction (in Greek) |

---

### 4.3 Comorbidity Network

Nine clinically-validated comorbidity pairs are encoded in the graph:

| Pair | Condition A | Condition B |
|---|---|---|
| I10 ↔ E11 | Hypertension | Type 2 diabetes |
| I10 ↔ E78 | Hypertension | Hyperlipidaemia |
| I10 ↔ I50 | Hypertension | Heart failure |
| E11 ↔ E78 | Type 2 diabetes | Hyperlipidaemia |
| E11 ↔ N18 | Type 2 diabetes | Chronic kidney disease |
| I50 ↔ N18 | Heart failure | Chronic kidney disease |
| I21 ↔ I50 | Myocardial infarction | Heart failure |
| J45 ↔ G40 | Asthma | Epilepsy |
| F32 ↔ M54 | Major depression | Back pain |

---

### 4.4 Drug Interaction Network

Ten bidirectional drug interactions, modelled by severity:

| Drug A | Drug B | Severity | Clinical Effect |
|---|---|---|---|
| Warfarin | Aspirin | **High** | Increased bleeding risk |
| Lisinopril | Furosemide | Medium | Hypotension risk |
| Fluoxetine | Metoprolol | Medium | Elevated metoprolol levels |
| Warfarin | Atorvastatin | Medium | INR elevation |
| Aspirin | Furosemide | Medium | Reduced diuretic effect |
| Fluoxetine | Aspirin | Medium | Increased bleeding risk |
| Warfarin | Metformin | Low | Mild anticoagulant effect change |
| Metoprolol | Amlodipine | Low | Monitor blood pressure |
| Levetiracetam | Furosemide | Low | Possible drug level change |
| Omeprazole | Metformin | Low | Mild absorption interaction |

---

### 4.5 Indexes & Constraints

**Uniqueness constraints:**
- `Patient.patientId`, `Doctor.doctorId`, `Diagnosis.icdCode`, `Medication.drugId`, `Procedure.procCode`, `Hospital.hospitalId`

**Range indexes:**
- `Patient(name)`, `Doctor(specialty)`, `Medication(name)`
- `HAS_DIAGNOSIS[severity]` — relationship property index for severity-based filtering

**Fulltext (Lucene) indexes** — used by `fuzzy_name_search`:
- `doctorNames` on `Doctor.name`
- `patientNames` on `Patient.name`
- `diagnosisNames` on `Diagnosis.name`

---

## 5. Semantic Embedding Strategy

The project uses `paraphrase-multilingual-mpnet-base-v2` (768 dimensions, supports 50+ languages including Greek) to embed four rich narrative fields:

| Index | Source Field | Content |
|---|---|---|
| `patients` | `Patient.summary` | Auto-generated clinical summary: age, blood type, active diagnoses, current medications, procedures, hospitals |
| `doctors` | `Doctor.expertise` | Specialty narrative covering subspecialisation, procedures performed, and patient population |
| `diagnoses` | `Diagnosis.clinicalDescription` | Clinical paragraph covering aetiology, presentation, and treatment (in Greek) |
| `procedures` | `Procedure.procedureDescription` | Procedural narrative covering indication, technique, and expected outcome (in Greek) |

Vectors are stored in **4-bit quantised** QuantaIndex files under `./medical_indexes/`:

```
medical_indexes/
├── patients.tvim          # quantised vectors
├── patients.ids.json      # id → position mapping
├── doctors.tvim
├── doctors.ids.json
├── diagnoses.tvim
├── diagnoses.ids.json
├── procedures.tvim
├── procedures.ids.json
└── metadata.json          # id → {node_type, display_name, raw_text}
```

On startup, if the index files already exist they are loaded from disk (fast path). Otherwise, the indexer queries Neo4j, embeds all texts, and saves the indexes (slow path, runs once).

---

## 6. Agent Tools Reference

All 10 tools are defined as OpenAI function-call schemas in `tools.py` and dispatched by `dispatch_tool()`.

### 6.1 Search & Lookup

#### `semantic_search`

Dense vector similarity search over one or more node types.

| Parameter | Type | Description |
|---|---|---|
| `query` | string | Natural-language query |
| `node_types` | array of enum | One or more of `patient`, `doctor`, `diagnosis`, `procedure` |
| `k` | integer | Max results (default 5) |
| `allowed_ids` | array of string | Optional ID allowlist to restrict the search space |

The query is embedded and cosine-compared against the selected QuantaIndex slices. Results from multiple node types are merged and re-ranked by score before the top-k are returned.

**Best for:** Symptom descriptions, condition concepts, doctor specialisations, procedure types — anything without an exact identifier.

---

#### `fuzzy_name_search`

Fulltext fuzzy search using Neo4j's Lucene `~` (edit-distance) operator.

| Parameter | Type | Description |
|---|---|---|
| `name` | string | Name or partial name to search |
| `index_name` | enum | `doctorNames`, `patientNames`, or `diagnosisNames` |
| `threshold` | float | Minimum Lucene score (default 0.7) |
| `limit` | integer | Max results (default 10) |

Each token of the input is suffixed with `~` to enable fuzzy matching (Levenshtein distance ≤ 2 by default). This handles typos, transliterations, and partial spellings.

**Best for:** Name resolution when the user provides a name that may be misspelt. The agent uses this tool **before any other tool** when a name is mentioned.

---

#### `cypher_query`

Executes a read-only Cypher query against the Neo4j graph and returns rows as plain dictionaries.

| Parameter | Type | Description |
|---|---|---|
| `cypher` | string | A MATCH / RETURN / WITH / UNWIND … query |

Write operations (`CREATE`, `MERGE`, `DELETE`, `SET`, `REMOVE`) are blocked by a regex guard at the Python layer.

**Best for:** Exact ID lookups, age/date filters, aggregations, counts, complex multi-hop patterns.

**Critical Cypher rules the agent follows:**
- Age is never a property; always computed: `date().year - date(p.dateOfBirth).year`
- Active medications: `[rx:PRESCRIBED] WHERE rx.active = true`
- Diagnosis severity: `[r:HAS_DIAGNOSIS] r.severity`
- Doctor's hospital: `d.hospitalId` (no `WORKS_AT` relationship)
- Currently admitted patients: `[r:ADMITTED_TO] WHERE r.dischargeDate IS NULL`

---

### 6.2 Graph Traversal

#### `graph_expand`

Fetches the immediate neighborhood of a known node up to a configurable number of hops.

| Parameter | Type | Description |
|---|---|---|
| `node_id` | string | The node's identifier (`patientId`, `doctorId`, `icdCode`, or `procCode`) |
| `node_type` | enum | `patient`, `doctor`, `diagnosis`, or `procedure` |
| `hops` | integer | Traversal depth: 1 (default), 2, or 3 |

Returns the **center node** (with all properties) and up to **50 neighbours**, each annotated with relationship type, direction (`incoming`/`outgoing`), and node type.

**Best for:** Exploring all connections around a specific entity — "what else is connected to this patient?", "which procedures has this doctor performed?"

---

#### `comorbidity_search`

Traverses `COMORBID_WITH` edges from a source diagnosis, filtered by minimum evidence level and ranked by study count.

| Parameter | Type | Description |
|---|---|---|
| `icd_code` | string | ICD-10 code of the source diagnosis |
| `min_evidence` | enum | `strong`, `moderate` (default), or `emerging` |
| `limit` | integer | Max results (default 10) |

Evidence level hierarchy: `strong` > `moderate` > `emerging`. Setting `min_evidence="moderate"` returns both `strong` and `moderate` evidence comorbidities, excluding emerging.

**Best for:** Clinical co-occurrence analysis, differential diagnosis support, population risk stratification.

---

### 6.3 Patient Intelligence

#### `get_patient_timeline`

Returns all dated clinical events for a patient in strict chronological order.

| Parameter | Type | Description |
|---|---|---|
| `patient_id` | string | The patient's `patientId` |

Collects four event types from separate queries and merges them into a unified sorted timeline:

| Event Type | Source | Key Fields |
|---|---|---|
| `diagnosis` | `HAS_DIAGNOSIS` edge | date, description, severity, chronic, diagnosedBy |
| `admission` | `ADMITTED_TO` edge | admissionDate, hospital, ward, lengthOfStay, dischargeDate |
| `procedure` | `UNDERWENT` edge | date, procedure name, outcome |
| `prescription` | `PRESCRIBED` edge | startDate, medication, active, frequency, linked ICD |

Events with null dates are omitted. Results are sorted by the `date` field.

**Best for:** Patient history review, chronological progression analysis, pre-admission workup.

---

#### `find_similar_patients`

Finds patients with a clinically similar profile to a reference patient using vector similarity on `Patient.summary`.

| Parameter | Type | Description |
|---|---|---|
| `patient_id` | string | Reference patient's `patientId` |
| `k` | integer | Number of similar patients (default 5) |
| `age_band` | integer | ±years around reference age (default ±10, set 0 to disable) |

**Algorithm:**
1. Fetch the reference patient's `summary` and `dateOfBirth` from Neo4j.
2. (Optional) Build an `allowed_ids` list of patients within the age band.
3. Embed the reference patient's summary and run vector search.
4. Remove the reference patient from results if it appears.
5. Enrich hits with `name` and `dateOfBirth` from Neo4j.

**Best for:** Cohort analysis, case matching, research queries like "find patients with a profile similar to this one".

---

#### `drug_interaction_check`

Checks all active medications of a patient for known pairwise drug interactions.

| Parameter | Type | Description |
|---|---|---|
| `patient_id` | string | The patient's `patientId` |

**Algorithm:**
1. Fetch all active medications via `PRESCRIBED` edges where `rx.active = true`.
2. For every pair of active medications, look for an `INTERACTS_WITH` edge.
3. Sort interactions by severity: `high` → `medium` → `low`.

Returns: list of active medications, all detected interactions (with severity and clinical effect), and a `high_severity_count` summary.

**Best for:** Polypharmacy safety checks, pre-prescription review, medication reconciliation on admission.

---

#### `high_risk_patients`

Identifies the highest-risk patients across the hospital network using a composite clinical score.

| Parameter | Type | Description |
|---|---|---|
| `min_age` | integer | Minimum age threshold (default 65) |
| `min_severe_diagnoses` | integer | Minimum severe diagnosis count (default 1) |
| `min_admissions` | integer | Minimum hospital admission count (default 2) |
| `limit` | integer | Max patients to return (default 20) |

**Risk score formula:**

```
risk_score = (severe_diagnosis_count × 3)
           + (admission_count × 2)
           + active_medication_count
           + (2 if age ≥ 75 else 0)
```

Severe diagnoses are weighted highest because they indicate the most acute clinical burden. The age bonus of +2 for patients ≥75 reflects elevated baseline fragility.

**Best for:** Proactive population health management, discharge planning prioritisation, resource allocation.

---

### 6.4 Doctor Matching

#### `doctor_expertise_match`

Finds the most suitable doctors for a clinical condition using semantic similarity on `Doctor.expertise` narratives.

| Parameter | Type | Description |
|---|---|---|
| `condition_description` | string | Free-text description of the clinical condition or care need |
| `k` | integer | Number of doctors to return (default 5) |

Embeds the condition description and runs vector search against the `doctors` QuantaIndex. Each result is enriched with `name`, `specialty`, `yearsExp`, and `hospitalId` from Neo4j.

**Best for:** Referral matching, finding the right specialist for a complex case, interdisciplinary consultation planning.

---

## 7. Agent Decision Strategy

The system prompt encodes 12 prioritised decision rules that guide the agent's tool selection:

| Priority | Trigger | Tool(s) |
|---|---|---|
| 1 | Any name mentioned (patient, doctor, diagnosis) | `fuzzy_name_search` first, then chain |
| 2 | Patient history / chronological progression | `get_patient_timeline` |
| 3 | Drug safety, interactions, polypharmacy | `drug_interaction_check` |
| 4 | Co-occurring diseases, related conditions | `comorbidity_search` (chain with `semantic_search` to get ICD code) |
| 5 | "Find patients like X" / cohort analysis | `find_similar_patients` |
| 6 | Population-wide high-risk identification | `high_risk_patients` |
| 7 | All connections / full context around a node | `graph_expand` (hops=1 or 2) |
| 8 | Doctor-to-condition matching / referrals | `doctor_expertise_match` |
| 9 | Exact matches, date/age filters, aggregations | `cypher_query` |
| 10 | Symptoms or concepts without exact IDs | `semantic_search` |
| 11 | Multi-step pipelines | Chain: `fuzzy_name_search` → `graph_expand`, or `semantic_search` → `comorbidity_search` |
| 12 | Language | Reply in the same language the user writes in |

The agent never exposes raw Cypher, internal node IDs, or tool call details in its final answer.

---

## 8. Vector Index — QuantaIndex

`QuantaIndex` is the vector store from the parent `quanta` library. It uses **4-bit scalar quantisation** (configurable via `DEFAULT_BIT_WIDTH`) to compress float32 embeddings by 8×, enabling efficient in-memory search with minimal RAM overhead.

```python
# Each index: one per node type
QuantaIndex(name="patients", dim=768, bit_width=4, index_dir="./medical_indexes")

# Build: embed all texts → add vectors + IDs
index.add(vectors, ids)
index.save()

# Search: embed query → find top-k nearest neighbours
index.search(query_vec, k=5, allowed_ids=["PAT00001", ...])  # optional filter
```

The `allowed_ids` parameter enables pre-filtering before the ANN search — used by `find_similar_patients` to restrict the search to the age band.

---

## 9. Conversation Memory — Redis

When Redis is available, the agent maintains **per-session conversation history** across multiple turns.

```
Key pattern:  medical:{conversation_id}
Type:         Redis list (append-only)
TTL:          86400 seconds (24 hours, configurable via CONVERSATION_TTL)
Max messages: 200 per conversation (configurable via MAX_MESSAGES_PER_CONVERSATION)
History window: last 10 turns × 2 messages = 20 messages sent to the LLM
```

Session IDs are UUIDs persisted to `.conversation_id` in the module directory between runs. REPL commands:
- `new` — start a fresh conversation (new UUID)
- `clear` — delete the current conversation's history from Redis
- `quit` — exit

If Redis is unavailable (not installed or connection fails), the agent runs in **stateless mode** — each query is answered independently without memory of previous turns.

Environment variables: `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`.

---

## 10. Project Structure

```
medical_search/
├── populate_medical_graph.py   # Step 1: create the Neo4j graph with synthetic data
├── agent.py                    # Step 2: LLM agent REPL
├── indexer.py                  # MedicalIndexer: build/load/search vector indexes
├── tools.py                    # 10 tool implementations + OpenAI schema definitions
├── redis_service.py            # Redis-backed conversation history
├── neo4j_connection.py         # Neo4jConnection utility (also shared with examples/)
├── __init__.py
└── README.md

medical_indexes/                # Auto-created on first run of agent.py
├── patients.tvim
├── patients.ids.json
├── doctors.tvim
├── doctors.ids.json
├── diagnoses.tvim
├── diagnoses.ids.json
├── procedures.tvim
├── procedures.ids.json
└── metadata.json
```

---

## 11. Quick Start

### Prerequisites

- Neo4j ≥ 5 running locally (default: `bolt://localhost:7687`, user `neo4j`, password `quantapass`)
- Python ≥ 3.11
- The `quanta` package installed (from the parent repo)

```bash
pip install neo4j faker sentence-transformers openai redis python-dotenv pydantic-settings
```

### Step 1 — Populate the graph

```bash
cd medical_search
python populate_medical_graph.py
```

This creates 100 patients, 30 doctors, 13 diagnoses, 12 procedures, 12 medications, and 5 hospitals in Neo4j, builds all relationships, enriches patient summaries, and adds comorbidity and drug interaction edges.

Optional flags:
```bash
python populate_medical_graph.py --patients 200 --doctors 50  # larger dataset
python populate_medical_graph.py --clean                       # wipe and repopulate
python populate_medical_graph.py --uri bolt://myhost:7687 --user neo4j --password secret
```

### Step 2 — Run the agent

```bash
python agent.py
```

On first run, the indexer queries Neo4j and builds the vector indexes (takes ~30–60 seconds). Subsequent runs load from disk instantly.

```bash
python agent.py --conversation-id <uuid>   # resume a specific conversation
python agent.py --uri bolt://myhost:7687 --user neo4j --password secret
```

---

## 12. Configuration Reference

Copy `.env.example` to `.env` and set the following variables:

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `quantapass` | Neo4j password |
| `EMBED_MODEL` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | HuggingFace model identifier |
| `EMBED_DIM` | `768` | Embedding dimensionality |
| `DEFAULT_BIT_WIDTH` | `4` | Quantisation bit-width for QuantaIndex |
| `REDIS_HOST` | *(empty)* | Redis hostname; leave empty to disable |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | *(empty)* | Redis password |
| `REDIS_DB` | `0` | Redis database index |
| `HISTORY_LIMIT` | `10` | Number of conversation turns kept in context |
| `CONVERSATION_TTL` | `86400` | Redis key TTL in seconds (24 h) |

---

## 13. Example Queries

Below are representative natural-language queries that exercise different tool combinations:

```
> Ποιοι γιατροί ειδικεύονται στη νεφρολογία και χρόνια νεφρική νόσο;
```
*Agent uses: `doctor_expertise_match` with "nephrology chronic kidney disease"*

```
> Δείξε μου το ιατρικό ιστορικό του ασθενή Γεώργιου Παπαδόπουλου.
```
*Agent chains: `fuzzy_name_search(patientNames)` → `get_patient_timeline`*

```
> Ελέγξε για αλληλεπιδράσεις φαρμάκων για τον ασθενή PAT00042.
```
*Agent uses: `drug_interaction_check`*

```
> Ποιες παθήσεις συνυπάρχουν συχνά με διαβήτη τύπου 2;
```
*Agent chains: `semantic_search(diagnosis, "type 2 diabetes")` → `comorbidity_search(E11)`*

```
> Βρες μου ασθενείς με παρόμοιο κλινικό προφίλ με τον PAT00015, ηλικίας ±10 ετών.
```
*Agent uses: `find_similar_patients(age_band=10)`*

```
> Ποιοι ασθενείς άνω των 70 ετών έχουν σοβαρές διαγνώσεις και πολλαπλές νοσηλείες;
```
*Agent uses: `high_risk_patients(min_age=70)`*

```
> Δείξε όλες τις συνδέσεις (διαγνώσεις, φάρμακα, διαδικασίες) γύρω από τον ασθενή PAT00008.
```
*Agent uses: `graph_expand(node_type=patient, hops=1)`*

```
> Πόσοι ασθενείς με καρδιακή ανεπάρκεια νοσηλεύονται αυτή τη στιγμή;
```
*Agent uses: `cypher_query` with MATCH filtering on `HAS_DIAGNOSIS` and `ADMITTED_TO` where `dischargeDate IS NULL`*
