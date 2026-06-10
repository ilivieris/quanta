"""Unit tests for medical_search.tools."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from examples.project.tools import (
    TOOLS,
    dispatch_tool,
    run_doctor_expertise_match,
    run_drug_interaction_check,
    run_find_similar_patients,
    run_high_risk_patients,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_VEC = np.zeros(384, dtype=np.float32)


def _patient_hit(pid: str, score: float, raw_text: str = "Patient summary text") -> dict:
    return {"id": pid, "score": score, "node_type": "Patient",
            "display_name": pid, "raw_text": raw_text}


def _doctor_hit(did: str, score: float) -> dict:
    return {"id": did, "score": score, "node_type": "Doctor",
            "display_name": did, "raw_text": "Cardiology expert"}


def _make_indexer(hits: list[dict], kind: str = "patients") -> MagicMock:
    indexer = MagicMock()
    indexer.embed.return_value = _VEC
    if kind == "patients":
        indexer.search_patients.return_value = hits
    else:
        indexer.search_doctors.return_value = hits
    return indexer


# ── find_similar_patients ─────────────────────────────────────────────────────

class TestFindSimilarPatients:

    def test_returns_empty_when_patient_not_found(self):
        graph = MagicMock()
        graph.query.return_value = []
        indexer = _make_indexer([])

        result = run_find_similar_patients(indexer=indexer, graph=graph, patient_id="P999")

        assert result == []
        indexer.embed.assert_not_called()

    def test_no_age_filter_returns_similar_patients(self):
        graph = MagicMock()
        graph.query.side_effect = [
            [{"summary": "Diabetic patient with hypertension", "dob": "1970-01-01"}],
            [{"pid": "P002", "name": "John Doe", "dob": "1971-05-15"}],
        ]
        indexer = _make_indexer([_patient_hit("P002", 0.95, "Diabetic, obese, hypertensive")])

        result = run_find_similar_patients(
            indexer=indexer, graph=graph, patient_id="P001", k=3, age_band=0
        )

        assert len(result) == 1
        r = result[0]
        assert r["patient_id"] == "P002"
        assert r["name"] == "John Doe"
        assert r["score"] == pytest.approx(0.95)
        assert r["date_of_birth"] == "1971-05-15"
        assert len(r["summary_snippet"]) <= 120

    def test_age_filter_passed_as_allowed_ids(self):
        graph = MagicMock()
        graph.query.side_effect = [
            [{"summary": "Summary", "dob": "1970-01-01"}],
            [{"pid": "P002"}, {"pid": "P003"}],
            [
                {"pid": "P002", "name": "Jane Smith", "dob": "1972-03-10"},
                {"pid": "P003", "name": "Bob Jones", "dob": "1968-11-20"},
            ],
        ]
        indexer = _make_indexer([
            _patient_hit("P002", 0.92),
            _patient_hit("P003", 0.88),
        ])

        result = run_find_similar_patients(
            indexer=indexer, graph=graph, patient_id="P001", k=5, age_band=10
        )

        _, kwargs = indexer.search_patients.call_args
        assert set(kwargs["allowed_ids"]) == {"P002", "P003"}
        assert len(result) == 2
        assert result[0]["patient_id"] == "P002"
        assert result[1]["patient_id"] == "P003"

    def test_target_patient_excluded_from_results(self):
        graph = MagicMock()
        graph.query.side_effect = [
            [{"summary": "Summary", "dob": "1970-01-01"}],
            [{"pid": "P002", "name": "Other", "dob": "1971-01-01"}],
        ]
        # Search returns self first, then another patient
        indexer = _make_indexer([
            _patient_hit("P001", 1.0),
            _patient_hit("P002", 0.91),
        ])

        result = run_find_similar_patients(
            indexer=indexer, graph=graph, patient_id="P001", k=5, age_band=0
        )

        assert all(r["patient_id"] != "P001" for r in result)
        assert len(result) == 1
        assert result[0]["patient_id"] == "P002"

    def test_empty_age_band_returns_early(self):
        graph = MagicMock()
        graph.query.side_effect = [
            [{"summary": "Summary", "dob": "1970-01-01"}],
            [],  # No patients in age band
        ]
        indexer = _make_indexer([])

        result = run_find_similar_patients(
            indexer=indexer, graph=graph, patient_id="P001", k=5, age_band=5
        )

        assert result == []
        indexer.embed.assert_not_called()

    def test_summary_snippet_truncated_to_120(self):
        long_summary = "x" * 300
        graph = MagicMock()
        graph.query.side_effect = [
            [{"summary": long_summary, "dob": "1970-01-01"}],
            [{"pid": "P002", "name": "Pat", "dob": "1971-01-01"}],
        ]
        indexer = _make_indexer([_patient_hit("P002", 0.80, long_summary)])

        result = run_find_similar_patients(
            indexer=indexer, graph=graph, patient_id="P001", k=1, age_band=0
        )

        assert result[0]["summary_snippet"] == "x" * 120

    def test_missing_dob_skips_age_filter(self):
        """Patient with no dateOfBirth should skip age filtering and still search."""
        graph = MagicMock()
        graph.query.side_effect = [
            [{"summary": "Summary", "dob": None}],
            [{"pid": "P002", "name": "Pat", "dob": None}],
        ]
        indexer = _make_indexer([_patient_hit("P002", 0.85)])

        result = run_find_similar_patients(
            indexer=indexer, graph=graph, patient_id="P001", k=1, age_band=10
        )

        # Only two graph.query calls: fetch patient + enrichment (no age filter)
        assert graph.query.call_count == 2
        assert len(result) == 1

    def test_k_respected(self):
        graph = MagicMock()
        graph.query.side_effect = [
            [{"summary": "Summary", "dob": "1970-01-01"}],
            [{"pid": f"P{i:03d}", "name": f"Pat{i}", "dob": "1972-01-01"} for i in range(2, 4)],
        ]
        hits = [_patient_hit(f"P{i:03d}", 1.0 - i * 0.1) for i in range(2, 6)]
        indexer = _make_indexer(hits)

        result = run_find_similar_patients(
            indexer=indexer, graph=graph, patient_id="P001", k=2, age_band=0
        )

        args, kwargs = indexer.search_patients.call_args
        assert kwargs["k"] == 3  # k+1 passed to search
        assert len(result) <= 2


# ── doctor_expertise_match ────────────────────────────────────────────────────

class TestDoctorExpertiseMatch:

    def test_no_hospital_filter_searches_all_doctors(self):
        graph = MagicMock()
        graph.query.return_value = [
            {"name": "Dr. Smith", "specialty": "Cardiology",
             "years_exp": 15, "hospitals": ["General Hospital"]}
        ]
        indexer = _make_indexer([_doctor_hit("D001", 0.92)], kind="doctors")

        result = run_doctor_expertise_match(
            indexer=indexer, graph=graph, condition_description="cardiac arrhythmia"
        )

        _, kwargs = indexer.search_doctors.call_args
        assert kwargs.get("allowed_ids") is None
        assert len(result) == 1
        r = result[0]
        assert r["doctor_id"] == "D001"
        assert r["name"] == "Dr. Smith"
        assert r["specialty"] == "Cardiology"
        assert r["years_exp"] == 15
        assert r["hospitals"] == ["General Hospital"]
        assert r["score"] == pytest.approx(0.92)

    def test_hospital_filter_restricts_search(self):
        graph = MagicMock()
        graph.query.side_effect = [
            [{"did": "D001"}, {"did": "D002"}],  # WORKS_AT filter
            [{"name": "Dr. A", "specialty": "Neurology", "years_exp": 10,
              "hospitals": ["Hosp A"]}],
            [{"name": "Dr. B", "specialty": "Neurology", "years_exp": 8,
              "hospitals": ["Hosp A"]}],
        ]
        indexer = _make_indexer(
            [_doctor_hit("D001", 0.90), _doctor_hit("D002", 0.85)], kind="doctors"
        )

        result = run_doctor_expertise_match(
            indexer=indexer, graph=graph,
            condition_description="epilepsy", hospital_id="H001"
        )

        _, kwargs = indexer.search_doctors.call_args
        assert set(kwargs["allowed_ids"]) == {"D001", "D002"}
        assert len(result) == 2
        assert result[0]["doctor_id"] == "D001"
        assert result[1]["doctor_id"] == "D002"

    def test_no_doctors_at_hospital_returns_empty(self):
        graph = MagicMock()
        graph.query.return_value = []
        indexer = _make_indexer([], kind="doctors")

        result = run_doctor_expertise_match(
            indexer=indexer, graph=graph,
            condition_description="diabetes", hospital_id="H999"
        )

        assert result == []
        indexer.embed.assert_not_called()

    def test_no_hits_from_search_returns_empty(self):
        graph = MagicMock()
        indexer = _make_indexer([], kind="doctors")

        result = run_doctor_expertise_match(
            indexer=indexer, graph=graph, condition_description="rare disease"
        )

        assert result == []

    def test_missing_graph_info_falls_back_to_display_name(self):
        graph = MagicMock()
        graph.query.return_value = []  # enrichment returns nothing
        indexer = _make_indexer([_doctor_hit("D001", 0.70)], kind="doctors")

        result = run_doctor_expertise_match(
            indexer=indexer, graph=graph, condition_description="oncology"
        )

        assert result[0]["name"] == "D001"  # falls back to display_name

    def test_per_doctor_enrichment_query_called(self):
        graph = MagicMock()
        graph.query.side_effect = [
            [{"name": "Dr. X", "specialty": "Ortho", "years_exp": 5, "hospitals": []}],
            [{"name": "Dr. Y", "specialty": "Ortho", "years_exp": 7, "hospitals": []}],
        ]
        indexer = _make_indexer(
            [_doctor_hit("D001", 0.90), _doctor_hit("D002", 0.85)], kind="doctors"
        )

        run_doctor_expertise_match(
            indexer=indexer, graph=graph, condition_description="fracture"
        )

        # One enrichment query per hit (2 hits → 2 calls; no hospital filter)
        assert graph.query.call_count == 2


# ── dispatch routing ──────────────────────────────────────────────────────────

class TestDispatch:

    def test_dispatch_find_similar_patients(self):
        with patch("medical_search.tools.run_find_similar_patients") as mock_fn:
            mock_fn.return_value = [{"patient_id": "P001", "score": 0.9,
                                     "name": "X", "date_of_birth": "", "summary_snippet": ""}]
            indexer, graph = MagicMock(), MagicMock()

            raw = dispatch_tool("find_similar_patients",
                                {"patient_id": "P001", "k": 3}, indexer, graph)

            mock_fn.assert_called_once_with(
                indexer=indexer, graph=graph, patient_id="P001", k=3, age_band=10
            )
            parsed = json.loads(raw)
            assert parsed[0]["patient_id"] == "P001"

    def test_dispatch_doctor_expertise_match(self):
        with patch("medical_search.tools.run_doctor_expertise_match") as mock_fn:
            mock_fn.return_value = [{"doctor_id": "D001", "score": 0.9,
                                     "name": "Dr. Z", "specialty": None,
                                     "years_exp": None, "hospitals": []}]
            indexer, graph = MagicMock(), MagicMock()

            raw = dispatch_tool("doctor_expertise_match",
                                {"condition_description": "heart disease",
                                 "hospital_id": "H001"}, indexer, graph)

            mock_fn.assert_called_once_with(
                indexer=indexer, graph=graph,
                condition_description="heart disease", k=5, hospital_id="H001"
            )
            parsed = json.loads(raw)
            assert parsed[0]["doctor_id"] == "D001"

    def test_dispatch_unknown_tool_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch_tool("nonexistent_tool", {}, MagicMock(), MagicMock())


# ── TOOLS list sanity ─────────────────────────────────────────────────────────

def test_tools_list_contains_new_entries():
    names = {t["function"]["name"] for t in TOOLS}
    assert "find_similar_patients" in names
    assert "doctor_expertise_match" in names


def test_find_similar_patients_schema():
    defn = next(t for t in TOOLS if t["function"]["name"] == "find_similar_patients")
    params = defn["function"]["parameters"]
    assert "patient_id" in params["required"]
    assert "k" not in params["required"]
    assert "age_band" not in params["required"]


def test_doctor_expertise_match_schema():
    defn = next(t for t in TOOLS if t["function"]["name"] == "doctor_expertise_match")
    params = defn["function"]["parameters"]
    assert "condition_description" in params["required"]
    assert "k" not in params["required"]
    assert "hospital_id" not in params["required"]


# ── drug_interaction_check ────────────────────────────────────────────────────

# Interaction responses keyed by (drug_a, drug_b) tuple (order matches the
# inner loop: active_meds[i], active_meds[j] with i < j).
_INTERACTION_DB: dict[tuple, dict] = {
    ("Warfarin",  "Aspirin"):      {"severity": "high",   "effect": "Αυξημένος κίνδυνος αιμορραγίας"},
    ("Warfarin",  "Atorvastatin"): {"severity": "medium", "effect": "Αύξηση INR"},
    ("Warfarin",  "Metformin"):    {"severity": "low",    "effect": "Ήπια μεταβολή αντιπηκτικής δράσης"},
}


def _graph_with_meds_and_interactions(
    med_names: list[str],
    known: dict[tuple, dict] | None = None,
) -> MagicMock:
    """
    Returns a mock graph where:
      - the first query (fetch active meds) yields med_names
      - each subsequent query (one per pair) returns the matching interaction
        row from *known*, or [] if the pair is absent.
    The pair order matches the nested loop: (med_names[i], med_names[j]) i<j.
    """
    if known is None:
        known = _INTERACTION_DB

    pairs = [
        (med_names[i], med_names[j])
        for i in range(len(med_names))
        for j in range(i + 1, len(med_names))
    ]

    responses: list = [[{"name": n} for n in med_names]]
    for pair in pairs:
        info = known.get(pair)
        responses.append([info] if info else [])

    graph = MagicMock()
    graph.query.side_effect = responses
    return graph


class TestDrugInteractionCheck:

    def test_known_high_severity_interaction(self):
        graph = _graph_with_meds_and_interactions(["Warfarin", "Aspirin"])

        result = run_drug_interaction_check(graph=graph, patient_id="P001")

        assert result["patient_id"] == "P001"
        assert set(result["active_medications"]) == {"Warfarin", "Aspirin"}
        assert len(result["interactions"]) == 1
        assert result["interactions"][0]["severity"] == "high"
        assert result["high_severity_count"] == 1

    def test_no_interactions_returns_empty_list(self):
        graph = _graph_with_meds_and_interactions(["Metformin", "Lisinopril"])

        result = run_drug_interaction_check(graph=graph, patient_id="P002")

        assert result["interactions"] == []
        assert result["high_severity_count"] == 0

    def test_interactions_sorted_high_medium_low(self):
        # Pairs in loop order: W+A(high), W+At(medium), W+M(low), A+At([]), A+M([]), At+M([])
        graph = _graph_with_meds_and_interactions(
            ["Warfarin", "Aspirin", "Atorvastatin", "Metformin"]
        )

        result = run_drug_interaction_check(graph=graph, patient_id="P003")

        severities = [i["severity"] for i in result["interactions"]]
        order = {"high": 0, "medium": 1, "low": 2}
        assert severities == sorted(severities, key=lambda s: order[s])
        assert len(result["interactions"]) == 3

    def test_high_severity_count_accurate(self):
        # Warfarin+Aspirin=high, Warfarin+Atorvastatin=medium → high_severity_count=1
        graph = _graph_with_meds_and_interactions(["Warfarin", "Aspirin", "Atorvastatin"])

        result = run_drug_interaction_check(graph=graph, patient_id="P004")

        assert result["high_severity_count"] == 1

    def test_no_active_medications(self):
        graph = MagicMock()
        graph.query.return_value = []

        result = run_drug_interaction_check(graph=graph, patient_id="P005")

        assert result["active_medications"] == []
        assert result["interactions"] == []
        assert result["high_severity_count"] == 0
        assert graph.query.call_count == 1  # only the meds fetch; no pair queries

    def test_single_medication_no_pairs(self):
        graph = _graph_with_meds_and_interactions(["Warfarin"])

        result = run_drug_interaction_check(graph=graph, patient_id="P006")

        assert result["interactions"] == []
        assert graph.query.call_count == 1  # only the meds fetch

    def test_graph_queried_once_per_pair(self):
        # 4 meds → C(4,2)=6 pairs → 1 (meds) + 6 (pairs) = 7 queries total
        graph = _graph_with_meds_and_interactions(
            ["Warfarin", "Aspirin", "Atorvastatin", "Metformin"]
        )

        run_drug_interaction_check(graph=graph, patient_id="P007")

        assert graph.query.call_count == 7

    def test_bidirectional_edge_found_via_graph(self):
        # The graph stores INTERACTS_WITH in both directions; even if our loop
        # queries (Aspirin, Warfarin), the edge exists and should be found.
        # We mock that specific direction returning a result.
        custom = {("Aspirin", "Warfarin"): {"severity": "high", "effect": "bleeding"}}
        graph = _graph_with_meds_and_interactions(["Aspirin", "Warfarin"], known=custom)

        result = run_drug_interaction_check(graph=graph, patient_id="P008")

        assert len(result["interactions"]) == 1
        assert result["interactions"][0]["severity"] == "high"


# ── high_risk_patients ────────────────────────────────────────────────────────

class TestHighRiskPatients:

    def _row(self, pid: str, name: str, age: int,
             severe_dx: int, admissions: int, active_meds: int) -> dict:
        return {
            "patient_id": pid, "name": name, "age": age,
            "severe_dx": severe_dx, "admissions": admissions,
            "active_meds": active_meds,
        }

    def test_basic_risk_score_formula(self):
        graph = MagicMock()
        graph.query.return_value = [
            self._row("P001", "Alice", 70, 2, 3, 4),
        ]

        result = run_high_risk_patients(graph=graph)

        # score = 2*3 + 3*2 + 4 + 0 (age<75) = 6+6+4 = 16
        assert result[0]["risk_score"] == 16
        assert result[0]["patient_id"] == "P001"

    def test_age_75_adds_two_to_risk_score(self):
        graph = MagicMock()
        graph.query.return_value = [
            self._row("P001", "Bob", 75, 1, 2, 1),
        ]

        result = run_high_risk_patients(graph=graph)

        # score = 1*3 + 2*2 + 1 + 2 = 3+4+1+2 = 10
        assert result[0]["risk_score"] == 10

    def test_age_74_does_not_add_bonus(self):
        graph = MagicMock()
        graph.query.return_value = [
            self._row("P001", "Carol", 74, 1, 2, 1),
        ]

        result = run_high_risk_patients(graph=graph)

        # score = 1*3 + 2*2 + 1 + 0 = 8
        assert result[0]["risk_score"] == 8

    def test_results_sorted_by_risk_score_descending(self):
        graph = MagicMock()
        graph.query.return_value = [
            self._row("P001", "Low",  68, 1, 2, 0),   # score = 3+4+0 = 7
            self._row("P002", "High", 76, 3, 4, 5),   # score = 9+8+5+2 = 24
            self._row("P003", "Mid",  70, 2, 2, 2),   # score = 6+4+2 = 12
        ]

        result = run_high_risk_patients(graph=graph)

        scores = [r["risk_score"] for r in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0]["patient_id"] == "P002"

    def test_empty_result(self):
        graph = MagicMock()
        graph.query.return_value = []

        result = run_high_risk_patients(graph=graph)

        assert result == []

    def test_cypher_receives_correct_parameters(self):
        graph = MagicMock()
        graph.query.return_value = []

        run_high_risk_patients(
            graph=graph, min_age=70, min_severe_diagnoses=2,
            min_admissions=3, limit=10
        )

        _, kwargs = graph.query.call_args
        params = kwargs["parameters"]
        assert params["min_age"] == 70
        assert params["min_severe_diagnoses"] == 2
        assert params["min_admissions"] == 3
        assert params["limit"] == 10

    def test_result_fields_complete(self):
        graph = MagicMock()
        graph.query.return_value = [self._row("P001", "Dan", 80, 2, 3, 5)]

        result = run_high_risk_patients(graph=graph)

        r = result[0]
        assert set(r) == {
            "patient_id", "name", "age",
            "severe_diagnosis_count", "admission_count",
            "active_medication_count", "risk_score",
        }


# ── dispatch for new tools ────────────────────────────────────────────────────

class TestDispatchSafetyTools:

    def test_dispatch_drug_interaction_check(self):
        with patch("medical_search.tools.run_drug_interaction_check") as mock_fn:
            mock_fn.return_value = {
                "patient_id": "P001", "active_medications": [],
                "interactions": [], "high_severity_count": 0,
            }
            indexer, graph = MagicMock(), MagicMock()

            raw = dispatch_tool("drug_interaction_check",
                                {"patient_id": "P001"}, indexer, graph)

            mock_fn.assert_called_once_with(graph=graph, patient_id="P001")
            assert json.loads(raw)["patient_id"] == "P001"

    def test_dispatch_high_risk_patients_defaults(self):
        with patch("medical_search.tools.run_high_risk_patients") as mock_fn:
            mock_fn.return_value = []
            indexer, graph = MagicMock(), MagicMock()

            dispatch_tool("high_risk_patients", {}, indexer, graph)

            mock_fn.assert_called_once_with(
                graph=graph, min_age=65, min_severe_diagnoses=1,
                min_admissions=2, limit=20
            )

    def test_dispatch_high_risk_patients_custom(self):
        with patch("medical_search.tools.run_high_risk_patients") as mock_fn:
            mock_fn.return_value = []
            indexer, graph = MagicMock(), MagicMock()

            dispatch_tool("high_risk_patients",
                          {"min_age": 70, "limit": 5}, indexer, graph)

            mock_fn.assert_called_once_with(
                graph=graph, min_age=70, min_severe_diagnoses=1,
                min_admissions=2, limit=5
            )


# ── TOOLS list sanity for safety tools ───────────────────────────────────────

def test_tools_list_contains_safety_tools():
    names = {t["function"]["name"] for t in TOOLS}
    assert "drug_interaction_check" in names
    assert "high_risk_patients" in names


def test_drug_interaction_check_schema():
    defn = next(t for t in TOOLS if t["function"]["name"] == "drug_interaction_check")
    params = defn["function"]["parameters"]
    assert params["required"] == ["patient_id"]


def test_high_risk_patients_schema_all_optional():
    defn = next(t for t in TOOLS if t["function"]["name"] == "high_risk_patients")
    params = defn["function"]["parameters"]
    assert params["required"] == []
    props = params["properties"]
    assert set(props) == {"min_age", "min_severe_diagnoses", "min_admissions", "limit"}


