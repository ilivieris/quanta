from __future__ import annotations

import json
import re
import sys
import os

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from examples.project.neo4j_connection import Neo4jConnection

from examples.project.indexer import MedicalIndexer

# ── Read-only write-keyword guard ──────────────────────────────────────────────
_WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE)\b",
    re.IGNORECASE,
)

# Mapping from tool-facing node type name → indexer method
_SEARCH_DISPATCH: dict[str, str] = {
    "patient":   "search_patients",
    "doctor":    "search_doctors",
    "diagnosis": "search_diagnoses",
    "procedure": "search_procedures",
}

# Mapping from tool-facing node type → (Neo4j label, id property)
_NODE_TYPE_MAP: dict[str, tuple[str, str]] = {
    "patient":   ("Patient",   "patientId"),
    "doctor":    ("Doctor",    "doctorId"),
    "diagnosis": ("Diagnosis", "icdCode"),
    "procedure": ("Procedure", "procCode"),
}

# Evidence hierarchy strongest-first; used to build allowed-level slices
_EVIDENCE_LEVELS: list[str] = ["strong", "moderate", "emerging"]

_SEVERITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}

# ── Tool implementations ───────────────────────────────────────────────────────

def run_semantic_search(
    indexer: MedicalIndexer,
    query: str,
    node_types: list[str],
    k: int = 5,
    allowed_ids: list[str] | None = None,
) -> list[dict]:
    """Embed query, search across requested node types, re-rank, return top-k."""
    query_vec = indexer.embed(query)

    merged: list[dict] = []
    for nt in node_types:
        method_name = _SEARCH_DISPATCH.get(nt)
        if method_name is None:
            raise ValueError(
                f"Unknown node_type {nt!r}. Valid: {list(_SEARCH_DISPATCH)}"
            )
        method = getattr(indexer, method_name)
        merged.extend(method(query_vec, k=k, allowed_ids=allowed_ids))

    merged.sort(key=lambda r: r["score"], reverse=True)
    return merged[:k]


def run_cypher_query(
    graph: Neo4jConnection,
    cypher: str,
) -> list[dict]:
    """Execute a read-only Cypher query and return rows as plain dicts."""
    if _WRITE_PATTERN.search(cypher):
        raise ValueError(
            "Write operations (CREATE / MERGE / DELETE / SET / REMOVE) are not "
            "permitted. Supply a read-only Cypher query."
        )
    rows = graph.query(cypher) or []
    return [dict(row) for row in rows]


def run_fuzzy_name_search(
    graph: Neo4jConnection,
    name: str,
    index_name: str,
    threshold: float = 0.7,
    limit: int = 10,
) -> list[dict]:
    """Fulltext fuzzy search using Neo4j's Lucene ~ operator."""
    # Build a Lucene fuzzy query: each token gets a ~ suffix for edit-distance matching
    fuzzy_query = " ".join(token + "~" for token in name.split())

    cypher = (
        "CALL db.index.fulltext.queryNodes($index_name, $query) "
        "YIELD node, score "
        "WHERE score >= $threshold "
        "RETURN node, score "
        "LIMIT $limit"
    )
    rows = graph.query(
        cypher,
        parameters={
            "index_name": index_name,
            "query":      fuzzy_query,
            "threshold":  threshold,
            "limit":      limit,
        },
    ) or []

    results = []
    for row in rows:
        node = row["node"]
        node_props = dict(node)
        labels = list(node.labels) if hasattr(node, "labels") else []
        node_type = labels[0] if labels else "Unknown"

        # Best-effort extraction of a display id and name from known schemas
        node_id = (
            node_props.get("patientId")
            or node_props.get("doctorId")
            or node_props.get("icdCode")
            or node_props.get("procCode")
            or node_props.get("id")
            or ""
        )
        display_name = (
            node_props.get("name")
            or node_props.get("summary")
            or node_props.get("expertise")
            or node_props.get("clinicalDescription")
            or node_props.get("procedureDescription")
            or node_id
        )

        results.append({
            "id":         str(node_id),
            "name":       str(display_name),
            "score":      float(row["score"]),
            "node_type":  node_type,
        })

    return results


def run_drug_interaction_check(
    graph: Neo4jConnection,
    patient_id: str,
) -> dict:
    """Checks all active medications of a patient for known drug interactions."""
    rows = graph.query(
        """
        MATCH (p:Patient {patientId: $patient_id})-[r:PRESCRIBED]->(m:Medication)
        WHERE r.active = true
        RETURN m.name AS name
        """,
        parameters={"patient_id": patient_id},
    ) or []

    active_meds = [row["name"] for row in rows if row.get("name")]

    interactions: list[dict] = []
    for i in range(len(active_meds)):
        for j in range(i + 1, len(active_meds)):
            hit = graph.query(
                """
                MATCH (m1:Medication {name: $drug_a})-[r:INTERACTS_WITH]->(m2:Medication {name: $drug_b})
                RETURN r.severity AS severity, r.effect AS effect
                """,
                parameters={"drug_a": active_meds[i], "drug_b": active_meds[j]},
            ) or []
            if hit:
                interactions.append({
                    "drug_a":   active_meds[i],
                    "drug_b":   active_meds[j],
                    "severity": hit[0]["severity"],
                    "effect":   hit[0]["effect"],
                })

    interactions.sort(key=lambda x: _SEVERITY_ORDER.get(x["severity"], 99))

    return {
        "patient_id":          patient_id,
        "active_medications":  active_meds,
        "interactions":        interactions,
        "high_severity_count": sum(1 for i in interactions if i["severity"] == "high"),
    }


def run_graph_expand(
    graph: Neo4jConnection,
    node_id: str,
    node_type: str,
    hops: int = 1,
) -> dict:
    """Fetches the immediate neighborhood of a node up to `hops` depth."""
    mapping = _NODE_TYPE_MAP.get(node_type)
    if mapping is None:
        raise ValueError(f"Unknown node_type {node_type!r}. Valid: {list(_NODE_TYPE_MAP)}")
    label, id_field = mapping

    center_rows = graph.query(
        f"MATCH (c:{label} {{{id_field}: $node_id}}) RETURN c",
        parameters={"node_id": node_id},
    ) or []

    if not center_rows:
        return {"center": {"id": node_id, "type": label, "properties": {}}, "neighbors": []}

    c = center_rows[0]["c"]
    c_props = dict(c)
    c_labels = list(c.labels) if hasattr(c, "labels") else [label]
    center = {
        "id": node_id,
        "type": c_labels[0] if c_labels else label,
        "properties": c_props,
    }

    neighbor_rows = graph.query(
        f"MATCH (c:{label} {{{id_field}: $node_id}})-[r*1..{hops}]-(n) "
        "RETURN r, n LIMIT 50",
        parameters={"node_id": node_id},
    ) or []

    neighbors: list[dict] = []
    for row in neighbor_rows:
        rel_path = row["r"]
        n = row["n"]
        if rel_path is None or n is None:
            continue

        rel_list = rel_path if isinstance(rel_path, list) else [rel_path]
        first_rel = rel_list[0]
        last_rel = rel_list[-1]
        rel_type = last_rel.type if hasattr(last_rel, "type") else type(last_rel).__name__

        try:
            direction = "outgoing" if first_rel.start_node == c.element_id else "incoming"
        except Exception:
            direction = "unknown"

        n_props = dict(n)
        n_labels = list(n.labels) if hasattr(n, "labels") else []
        n_type = n_labels[0] if n_labels else "Unknown"
        n_id = (
            n_props.get("patientId")
            or n_props.get("doctorId")
            or n_props.get("icdCode")
            or n_props.get("procCode")
            or n_props.get("medicationId")
            or n_props.get("hospitalId")
            or n_props.get("id")
            or ""
        )
        neighbors.append({
            "id": str(n_id),
            "type": n_type,
            "relationship": rel_type,
            "direction": direction,
            "properties": n_props,
        })

    return {"center": center, "neighbors": neighbors}


def run_comorbidity_search(
    graph: Neo4jConnection,
    icd_code: str,
    min_evidence: str = "moderate",
    limit: int = 10,
) -> list[dict]:
    """Traverses COMORBID_WITH edges from the given diagnosis."""
    if min_evidence not in _EVIDENCE_LEVELS:
        raise ValueError(f"min_evidence must be one of {_EVIDENCE_LEVELS}")
    allowed_levels = _EVIDENCE_LEVELS[: _EVIDENCE_LEVELS.index(min_evidence) + 1]

    rows = graph.query(
        """
        MATCH (d:Diagnosis {icdCode: $icd_code})-[r:COMORBID_WITH]-(co:Diagnosis)
        WHERE r.evidenceLevel IN $allowed_levels
        RETURN co.icdCode AS icd_code, co.name AS name,
               r.evidenceLevel AS evidence_level, r.studyCount AS study_count
        ORDER BY r.studyCount DESC
        LIMIT $limit
        """,
        parameters={
            "icd_code": icd_code,
            "allowed_levels": allowed_levels,
            "limit": limit,
        },
    ) or []

    return [
        {
            "icd_code":       row["icd_code"],
            "name":           row["name"],
            "evidence_level": row["evidence_level"],
            "study_count":    row["study_count"],
        }
        for row in rows
    ]


def run_get_patient_timeline(
    graph: Neo4jConnection,
    patient_id: str,
) -> list[dict]:
    """Returns all dated clinical events for a patient sorted chronologically."""
    events: list[dict] = []

    rows = graph.query(
        """
        MATCH (p:Patient {patientId: $pid})-[r:HAS_DIAGNOSIS]->(d:Diagnosis)
        WHERE r.date IS NOT NULL
        RETURN r.date AS date, d.name AS description,
               r.diagnosedBy AS diagnosedBy, r.severity AS severity, r.chronic AS chronic
        """,
        parameters={"pid": patient_id},
    ) or []
    for row in rows:
        events.append({"date": row["date"], "event_type": "diagnosis",
                        "description": row["description"],
                        "extra": {"diagnosedBy": row["diagnosedBy"],
                                  "severity": row["severity"],
                                  "chronic": row["chronic"]}})

    rows = graph.query(
        """
        MATCH (p:Patient {patientId: $pid})-[r:ADMITTED_TO]->(h:Hospital)
        WHERE r.admissionDate IS NOT NULL
        RETURN r.admissionDate AS date, h.name AS description,
               r.ward AS ward, r.lengthOfStay AS lengthOfStay,
               r.dischargeDate AS dischargeDate
        """,
        parameters={"pid": patient_id},
    ) or []
    for row in rows:
        events.append({"date": row["date"], "event_type": "admission",
                        "description": row["description"],
                        "extra": {"ward": row["ward"],
                                  "lengthOfStay": row["lengthOfStay"],
                                  "dischargeDate": row["dischargeDate"]}})

    rows = graph.query(
        """
        MATCH (p:Patient {patientId: $pid})-[r:UNDERWENT]->(proc:Procedure)
        WHERE r.date IS NOT NULL
        RETURN r.date AS date, proc.name AS description, r.outcome AS outcome
        """,
        parameters={"pid": patient_id},
    ) or []
    for row in rows:
        events.append({"date": row["date"], "event_type": "procedure",
                        "description": row["description"],
                        "extra": {"outcome": row["outcome"]}})

    rows = graph.query(
        """
        MATCH (p:Patient {patientId: $pid})-[r:PRESCRIBED]->(m:Medication)
        WHERE r.startDate IS NOT NULL
        RETURN r.startDate AS date, m.name AS description,
               r.active AS active, r.endDate AS endDate,
               r.frequency AS frequency, r.icdCode AS icdCode
        """,
        parameters={"pid": patient_id},
    ) or []
    for row in rows:
        events.append({"date": row["date"], "event_type": "prescription",
                        "description": row["description"],
                        "extra": {"active": row["active"], "endDate": row["endDate"],
                                  "frequency": row["frequency"],
                                  "icdCode": row["icdCode"]}})

    events.sort(key=lambda e: e["date"] or "")
    return events


def run_high_risk_patients(
    graph: Neo4jConnection,
    min_age: int = 65,
    min_severe_diagnoses: int = 1,
    min_admissions: int = 2,
    limit: int = 20,
) -> list[dict]:
    """Identifies high-risk patients based on composite clinical criteria."""
    rows = graph.query(
        """
        MATCH (p:Patient)
        WHERE date().year - date(p.dateOfBirth).year >= $min_age
        OPTIONAL MATCH (p)-[rd:HAS_DIAGNOSIS {severity: "severe"}]->(d:Diagnosis)
        OPTIONAL MATCH (p)-[:ADMITTED_TO]->(h:Hospital)
        OPTIONAL MATCH (p)-[rx:PRESCRIBED]->(m:Medication) WHERE rx.active = true
        WITH p,
             date().year - date(p.dateOfBirth).year AS age,
             count(DISTINCT d) AS severe_dx,
             count(DISTINCT h) AS admissions,
             count(DISTINCT m) AS active_meds
        WHERE severe_dx >= $min_severe_diagnoses
          AND admissions >= $min_admissions
        RETURN p.patientId AS patient_id, p.name AS name, age,
               severe_dx, admissions, active_meds
        ORDER BY (severe_dx*3 + admissions*2 + active_meds) DESC
        LIMIT $limit
        """,
        parameters={
            "min_age":              min_age,
            "min_severe_diagnoses": min_severe_diagnoses,
            "min_admissions":       min_admissions,
            "limit":                limit,
        },
    ) or []

    results = []
    for row in rows:
        age        = row["age"]
        severe_dx  = row["severe_dx"]
        admissions = row["admissions"]
        active_meds = row["active_meds"]
        risk_score = (
            severe_dx * 3
            + admissions * 2
            + active_meds
            + (2 if age >= 75 else 0)
        )
        results.append({
            "patient_id":             row["patient_id"],
            "name":                   row["name"],
            "age":                    age,
            "severe_diagnosis_count": severe_dx,
            "admission_count":        admissions,
            "active_medication_count": active_meds,
            "risk_score":             risk_score,
        })

    results.sort(key=lambda r: r["risk_score"], reverse=True)
    return results


def run_find_similar_patients(
    indexer: MedicalIndexer,
    graph: Neo4jConnection,
    patient_id: str,
    k: int = 5,
    age_band: int = 10,
) -> list[dict]:
    """Finds patients with a similar clinical profile using vector similarity."""
    # Step 1: fetch target patient's summary and date of birth
    patient_rows = graph.query(
        "MATCH (p:Patient {patientId: $patient_id}) "
        "RETURN p.summary AS summary, p.dateOfBirth AS dob",
        parameters={"patient_id": patient_id},
    ) or []
    if not patient_rows:
        return []

    summary = patient_rows[0]["summary"] or ""
    dob = patient_rows[0]["dob"]

    # Step 2: optional age-band filter → builds allowed_ids
    allowed_ids: list[str] | None = None
    if age_band > 0 and dob:
        age_rows = graph.query(
            """
            MATCH (p:Patient)
            WHERE abs(date().year - date(p.dateOfBirth).year
                  - (date().year - date($dob).year)) <= $age_band
              AND p.patientId <> $patient_id
            RETURN p.patientId AS pid
            """,
            parameters={"dob": str(dob), "patient_id": patient_id, "age_band": age_band},
        ) or []
        allowed_ids = [row["pid"] for row in age_rows]
        if not allowed_ids:
            return []

    # Steps 3 & 4: embed summary and search
    query_vec = indexer.embed(summary)
    hits = indexer.search_patients(query_vec, k=k + 1, allowed_ids=allowed_ids)

    # Drop the target patient if it appears in its own results
    hits = [h for h in hits if h["id"] != patient_id][:k]
    if not hits:
        return []

    # Batch-enrich with name and date_of_birth from Neo4j
    hit_ids = [h["id"] for h in hits]
    enrich_rows = graph.query(
        "MATCH (p:Patient) WHERE p.patientId IN $ids "
        "RETURN p.patientId AS pid, p.name AS name, p.dateOfBirth AS dob",
        parameters={"ids": hit_ids},
    ) or []
    enrich_map = {row["pid"]: row for row in enrich_rows}

    results = []
    for hit in hits:
        extra = enrich_map.get(hit["id"], {})
        results.append({
            "patient_id":      hit["id"],
            "name":            extra.get("name") or hit["display_name"],
            "score":           hit["score"],
            "date_of_birth":   str(extra.get("dob") or ""),
            "summary_snippet": (hit["raw_text"] or "")[:120],
        })
    return results


def run_doctor_expertise_match(
    indexer: MedicalIndexer,
    graph: Neo4jConnection,
    condition_description: str,
    k: int = 5,
) -> list[dict]:
    """Finds the most suitable doctors for a condition using semantic similarity."""
    query_vec = indexer.embed(condition_description)
    hits = indexer.search_doctors(query_vec, k=k)
    if not hits:
        return []

    results = []
    for hit in hits:
        enrich = graph.query(
            """
            MATCH (d:Doctor {doctorId: $did})
            RETURN d.name AS name, d.specialty AS specialty,
                   d.yearsExp AS years_exp, d.hospitalId AS hospital_id
            """,
            parameters={"did": hit["id"]},
        ) or []
        info = enrich[0] if enrich else {}
        results.append({
            "doctor_id":   hit["id"],
            "name":        info.get("name") or hit["display_name"],
            "specialty":   info.get("specialty"),
            "years_exp":   info.get("years_exp"),
            "hospital_id": info.get("hospital_id"),
            "score":       hit["score"],
        })
    return results


# ── OpenAI tool definitions ────────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": (
                "Performs dense vector similarity search over one or more medical "
                "node types (patients, doctors, diagnoses, procedures). Optionally "
                "restricts results to an allowlist of IDs. Returns ranked results "
                "with similarity scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "node_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["patient", "doctor", "diagnosis", "procedure"],
                        },
                        "description": (
                            "Node types to search. Provide one or more from the "
                            "allowed enum values."
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5).",
                        "default": 5,
                    },
                    "allowed_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of node IDs to restrict search to. "
                            "Results outside this set are excluded."
                        ),
                    },
                },
                "required": ["query", "node_types"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cypher_query",
            "description": (
                "Executes a read-only Cypher query against the Neo4j medical graph "
                "and returns matching rows. Use for exact matches, filters by age / "
                "date, aggregations, or relationship traversals. "
                "Write operations (CREATE, MERGE, DELETE, SET, REMOVE) are rejected."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cypher": {
                        "type": "string",
                        "description": "A read-only Cypher query (MATCH / RETURN / WITH / UNWIND …).",
                    },
                },
                "required": ["cypher"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fuzzy_name_search",
            "description": (
                "Uses Neo4j fulltext indexes to find nodes by approximate name, "
                "handling typos and partial matches via the Lucene ~ fuzzy operator. "
                "Available indexes: doctorNames, patientNames, diagnosisNames."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name (or partial name) to search for.",
                    },
                    "index_name": {
                        "type": "string",
                        "enum": ["doctorNames", "patientNames", "diagnosisNames"],
                        "description": "Which fulltext index to query.",
                    },
                    "threshold": {
                        "type": "number",
                        "description": (
                            "Minimum Lucene relevance score to include a result "
                            "(default 0.7). Lower values return more, fuzzier matches."
                        ),
                        "default": 0.7,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 10).",
                        "default": 10,
                    },
                },
                "required": ["name", "index_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_expand",
            "description": (
                "Fetches the immediate neighborhood of a node in the medical graph "
                "up to a given number of hops. Returns the center node and up to 50 "
                "reachable neighbors with relationship type and direction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": (
                            "Identifier of the node to expand: patientId for patient, "
                            "doctorId for doctor, icdCode for diagnosis, procCode for procedure."
                        ),
                    },
                    "node_type": {
                        "type": "string",
                        "enum": ["patient", "doctor", "diagnosis", "procedure"],
                        "description": "The type of the node to expand.",
                    },
                    "hops": {
                        "type": "integer",
                        "description": "Traversal depth (default 1, min 1, max 3).",
                        "default": 1,
                        "minimum": 1,
                        "maximum": 3,
                    },
                },
                "required": ["node_id", "node_type"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "comorbidity_search",
            "description": (
                "Finds diagnoses that co-occur with a given ICD code by traversing "
                "COMORBID_WITH edges. Filters by minimum evidence level and returns "
                "results sorted by study count descending."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "icd_code": {
                        "type": "string",
                        "description": "ICD code of the source diagnosis.",
                    },
                    "min_evidence": {
                        "type": "string",
                        "enum": ["strong", "moderate", "emerging"],
                        "description": (
                            "Minimum evidence level to include (default 'moderate'). "
                            "Hierarchy: strong > moderate > emerging. "
                            "'moderate' includes strong and moderate but excludes emerging."
                        ),
                        "default": "moderate",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 10).",
                        "default": 10,
                    },
                },
                "required": ["icd_code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_patient_timeline",
            "description": (
                "Returns all dated clinical events for a patient in chronological "
                "order: diagnoses, hospital admissions, procedures, and prescriptions. "
                "Events with null dates are omitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "The patient's unique identifier (patientId).",
                    },
                },
                "required": ["patient_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_patients",
            "description": (
                "Finds patients with a clinically similar profile to a given patient "
                "using vector similarity on patient summaries. Optionally restricts "
                "the search to patients within a similar age band."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "The patientId of the reference patient.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of similar patients to return (default 5).",
                        "default": 5,
                    },
                    "age_band": {
                        "type": "integer",
                        "description": (
                            "±years around the reference patient's age to restrict search. "
                            "Set to 0 to disable age filtering (default 10)."
                        ),
                        "default": 10,
                    },
                },
                "required": ["patient_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "drug_interaction_check",
            "description": (
                "Checks all active medications of a patient for known drug interactions. "
                "Returns the list of active medications, every detected interaction with "
                "severity and clinical effect, and a count of high-severity interactions. "
                "Interactions are ordered high → medium → low severity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {
                        "type": "string",
                        "description": "The patient's unique identifier (patientId).",
                    },
                },
                "required": ["patient_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "high_risk_patients",
            "description": (
                "Identifies high-risk patients using a composite score based on age, "
                "severe diagnoses, hospital admissions, and active medications. "
                "Returns patients ranked by risk_score descending."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_age": {
                        "type": "integer",
                        "description": "Minimum patient age to consider (default 65).",
                        "default": 65,
                    },
                    "min_severe_diagnoses": {
                        "type": "integer",
                        "description": "Minimum severe-diagnosis count required (default 1).",
                        "default": 1,
                    },
                    "min_admissions": {
                        "type": "integer",
                        "description": "Minimum hospital-admission count required (default 2).",
                        "default": 2,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of patients to return (default 20).",
                        "default": 20,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "doctor_expertise_match",
            "description": (
                "Finds the most suitable doctors for a clinical condition using "
                "semantic similarity on Doctor.expertise. Each result includes "
                "hospital_id — the hospital where the doctor works (there is no "
                "WORKS_AT relationship; hospital affiliation is a Doctor property)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition_description": {
                        "type": "string",
                        "description": "Free-text description of the clinical condition or requirement.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of doctors to return (default 5).",
                        "default": 5,
                    },
                },
                "required": ["condition_description"],
                "additionalProperties": False,
            },
        },
    },
]


# ── Dispatch ───────────────────────────────────────────────────────────────────

def dispatch_tool(
    tool_name: str,
    tool_args: dict,
    indexer: MedicalIndexer,
    graph: Neo4jConnection,
) -> str:
    """Route a tool call to the correct function and return a JSON string."""
    if tool_name == "semantic_search":
        print(f"[INFO] Running semantic_search | query={tool_args['query']!r} node_types={tool_args['node_types']} k={tool_args.get('k', 5)}")
        result = run_semantic_search(
            indexer=indexer,
            query=tool_args["query"],
            node_types=tool_args["node_types"],
            k=tool_args.get("k", 5),
            allowed_ids=tool_args.get("allowed_ids"),
        )
        print(f"[INFO] semantic_search → {len(result)} results | top: {[r.get('display_name') or r.get('id') for r in result[:3]]}")

    elif tool_name == "cypher_query":
        print(f"[INFO] Running cypher_query | cypher={tool_args['cypher']!r}")
        result = run_cypher_query(
            graph=graph,
            cypher=tool_args["cypher"],
        )
        print(f"[INFO] cypher_query → {len(result)} rows")

    elif tool_name == "fuzzy_name_search":
        print(f"[INFO] Running fuzzy_name_search | name={tool_args['name']!r} index={tool_args['index_name']!r}")
        result = run_fuzzy_name_search(
            graph=graph,
            name=tool_args["name"],
            index_name=tool_args["index_name"],
            threshold=tool_args.get("threshold", 0.7),
            limit=tool_args.get("limit", 10),
        )
        print(f"[INFO] fuzzy_name_search → {len(result)} matches | top: {[r['name'] for r in result[:3]]}")

    elif tool_name == "graph_expand":
        print(f"[INFO] Running graph_expand | node_id={tool_args['node_id']!r} node_type={tool_args['node_type']!r} hops={tool_args.get('hops', 1)}")
        result = run_graph_expand(
            graph=graph,
            node_id=tool_args["node_id"],
            node_type=tool_args["node_type"],
            hops=tool_args.get("hops", 1),
        )
        neighbor_types = {}
        for n in result.get("neighbors", []):
            neighbor_types[n["type"]] = neighbor_types.get(n["type"], 0) + 1
        print(f"[INFO] graph_expand → {len(result.get('neighbors', []))} neighbors | by type: {neighbor_types}")

    elif tool_name == "comorbidity_search":
        print(f"[INFO] Running comorbidity_search | icd_code={tool_args['icd_code']!r} min_evidence={tool_args.get('min_evidence', 'moderate')!r}")
        result = run_comorbidity_search(
            graph=graph,
            icd_code=tool_args["icd_code"],
            min_evidence=tool_args.get("min_evidence", "moderate"),
            limit=tool_args.get("limit", 10),
        )
        print(f"[INFO] comorbidity_search → {len(result)} comorbidities | {[r['name'] for r in result[:3]]}")

    elif tool_name == "get_patient_timeline":
        print(f"[INFO] Running get_patient_timeline | patient_id={tool_args['patient_id']!r}")
        result = run_get_patient_timeline(
            graph=graph,
            patient_id=tool_args["patient_id"],
        )
        event_types = {}
        for e in result:
            event_types[e["event_type"]] = event_types.get(e["event_type"], 0) + 1
        print(f"[INFO] get_patient_timeline → {len(result)} events | by type: {event_types}")

    elif tool_name == "find_similar_patients":
        print(f"[INFO] Running find_similar_patients | patient_id={tool_args['patient_id']!r} k={tool_args.get('k', 5)} age_band=±{tool_args.get('age_band', 10)}yr")
        result = run_find_similar_patients(
            indexer=indexer,
            graph=graph,
            patient_id=tool_args["patient_id"],
            k=tool_args.get("k", 5),
            age_band=tool_args.get("age_band", 10),
        )
        print(f"[INFO] find_similar_patients → {len(result)} patients | top scores: {[round(r['score'], 3) for r in result[:3]]}")

    elif tool_name == "doctor_expertise_match":
        print(f"[INFO] Running doctor_expertise_match | condition={tool_args['condition_description']!r} k={tool_args.get('k', 5)}")
        result = run_doctor_expertise_match(
            indexer=indexer,
            graph=graph,
            condition_description=tool_args["condition_description"],
            k=tool_args.get("k", 5),
        )
        print(f"[INFO] doctor_expertise_match → {len(result)} doctors | {[r['name'] for r in result[:3]]}")

    elif tool_name == "drug_interaction_check":
        print(f"[INFO] Running drug_interaction_check | patient_id={tool_args['patient_id']!r}")
        result = run_drug_interaction_check(
            graph=graph,
            patient_id=tool_args["patient_id"],
        )
        print(f"[INFO] drug_interaction_check → {len(result['active_medications'])} active meds | {len(result['interactions'])} interactions | high-severity: {result['high_severity_count']}")

    elif tool_name == "high_risk_patients":
        print(f"[INFO] Running high_risk_patients | min_age={tool_args.get('min_age', 65)} min_severe_dx={tool_args.get('min_severe_diagnoses', 1)} min_admissions={tool_args.get('min_admissions', 2)}")
        result = run_high_risk_patients(
            graph=graph,
            min_age=tool_args.get("min_age", 65),
            min_severe_diagnoses=tool_args.get("min_severe_diagnoses", 1),
            min_admissions=tool_args.get("min_admissions", 2),
            limit=tool_args.get("limit", 20),
        )
        print(f"[INFO] high_risk_patients → {len(result)} patients | top risk scores: {[r['risk_score'] for r in result[:3]]}")

    else:
        raise ValueError(
            f"Unknown tool {tool_name!r}. "
            f"Valid tools: semantic_search, cypher_query, fuzzy_name_search, "
            f"graph_expand, comorbidity_search, get_patient_timeline, "
            f"find_similar_patients, doctor_expertise_match, "
            f"drug_interaction_check, high_risk_patients"
        )

    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))
