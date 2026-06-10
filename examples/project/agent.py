from __future__ import annotations

import argparse
import json
import os
import time
import uuid

import openai
from dotenv import load_dotenv

load_dotenv()

from examples.project.indexer import MedicalIndexer, build_indexer
from examples.project.redis_service import REDIS_AVAILABLE, delete_conversation, load_history, save_message
from examples.project.tools import TOOLS, dispatch_tool

_SYSTEM_PROMPT_BASE = """\
You are a medical information assistant with access to a hospital knowledge
graph. You help medical staff retrieve patient information, doctor details,
diagnoses, procedures, medications, and hospital admissions.

You have the following tools:

SEARCH & LOOKUP
- semantic_search: dense vector search over patients, doctors, diagnoses, or
  procedures using natural language. Use when the user describes a concept,
  symptom, or condition without exact identifiers.
- fuzzy_name_search: fulltext fuzzy search for a name that may have typos or
  alternative spellings. Indexes available: doctorNames, patientNames,
  diagnosisNames.
- cypher_query: structured read-only Cypher query for exact identifiers,
  age/date filters, aggregations, counts, or relationship traversals.

GRAPH TRAVERSAL
- graph_expand: fetches the immediate neighborhood of a known node (patient,
  doctor, diagnosis, or procedure) up to a given number of hops (1–3).
  Use to explore all connections of a specific entity.
- comorbidity_search: traverses COMORBID_WITH edges from a diagnosis ICD code
  to find clinically co-occurring conditions ranked by study evidence.

PATIENT INTELLIGENCE
- get_patient_timeline: returns all dated clinical events for a patient
  (diagnoses, admissions, procedures, prescriptions) sorted chronologically.
- find_similar_patients: finds patients with a similar clinical profile using
  vector similarity on patient summaries, with optional age-band filtering.
- drug_interaction_check: checks all active medications of a patient for
  known drug-drug interactions, ranked by severity (high → medium → low).
- high_risk_patients: identifies high-risk patients across the population
  using a composite score of age, severe diagnoses, admissions, and active
  medications.

DOCTOR MATCHING
- doctor_expertise_match: finds the most suitable doctors for a clinical
  condition using semantic similarity on doctor expertise narratives.

Decision strategy:
1. NAME RESOLUTION — If the user mentions any name (patient, doctor, or
   diagnosis), always start with fuzzy_name_search to resolve the correct ID
   before running any other tool. Never guess IDs.
2. TIMELINE — For questions about a patient's history, events, or
   chronological progression, use get_patient_timeline.
3. MEDICATION SAFETY — For any question about drug safety, interactions, or
   polypharmacy risk, use drug_interaction_check.
4. COMORBIDITIES — For questions about diseases that co-occur or are
   clinically related to a known diagnosis, use comorbidity_search with the
   ICD code. Chain with semantic_search if you need to find the ICD code first.
5. SIMILAR PATIENTS — For questions like "find patients like X" or cohort
   analysis, use find_similar_patients.
6. POPULATION RISK — For questions about which patients are most at-risk
   across the hospital system, use high_risk_patients.
7. NETWORK EXPLORATION — When the user asks for all connections, relationships,
   or the full context around a specific node, use graph_expand (hops=1 for
   immediate neighbors, hops=2 for broader network).
8. DOCTOR MATCHING — When matching a doctor to a condition or specialty, use
   doctor_expertise_match.
9. STRUCTURED FILTERS — For exact matches, date/age filters, aggregations,
   or counts, use cypher_query. Compute age from dateOfBirth, never from an
   age property.
10. OPEN-ENDED MEDICAL CONCEPTS — For symptoms, treatments, or descriptions
    without exact identifiers, use semantic_search on the relevant node type.
11. CHAIN TOOLS when needed: fuzzy_name_search → graph_expand, or
    semantic_search → comorbidity_search, or fuzzy_name_search → cypher_query.
12. Always present results in Greek if the user writes in Greek.
13. Never expose raw Cypher, internal IDs, or tool call details in your
    final answer.

Cypher rules — follow these exactly:
- Node labels are CASE-SENSITIVE: Patient, Doctor, Diagnosis, Procedure,
  Medication, Hospital.
- Patients do NOT have an `age` property. Compute age from `dateOfBirth`:
    WHERE date().year - date(p.dateOfBirth).year >= 65
- Hospital names are stored in `h.name`. To filter by hospital name use:
    WHERE toLower(h.name) CONTAINS toLower('Ευαγγελισμός')
- Medication active status is on the relationship: [rx:PRESCRIBED] WHERE rx.active = true
- Medication frequency (dosing schedule) is on the relationship: [rx:PRESCRIBED] rx.frequency
- Medication prescribed-for diagnosis is on the relationship: [rx:PRESCRIBED] rx.icdCode
- Diagnosis severity and chronic flag are on the relationship: [r:HAS_DIAGNOSIS] r.severity, r.chronic
- The doctor who made a diagnosis is on the relationship: [r:HAS_DIAGNOSIS] r.diagnosedBy (stores doctorId)
- Patients still admitted to a hospital have dischargeDate = null on the ADMITTED_TO relationship
- There is NO WORKS_AT relationship. A doctor's hospital is stored as a property: d.hospitalId
- Always RETURN meaningful fields (name, id, etc.) not whole nodes.\
"""


def _build_system_prompt(graph) -> str:
    """Append the live graph schema so the LLM generates correct Cypher."""
    try:
        schema = graph.get_schema()
    except Exception:
        schema = "(schema unavailable)"
    return _SYSTEM_PROMPT_BASE + "\n\nGraph schema:\n" + schema


class MedicalSearchAgent:
    def __init__(self, indexer: MedicalIndexer, graph) -> None:
        self.indexer = indexer
        self.graph = graph
        self.client = openai.OpenAI(
            api_key="..."
        )
        self.model = "gpt-4.1-mini-2025-04-14"
        self.system_prompt = _build_system_prompt(graph)

    def query(self, user_message: str, conversation_id: str | None = None) -> str:
        """Run a full tool-calling loop and return the final text response."""
        history = load_history(conversation_id) if conversation_id else []

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]

        answer = ""
        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                max_tokens=32768,
                tool_choice="auto",
            )

            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls

            if not tool_calls:
                answer = assistant_message.content or ""
                break

            # Append the assistant turn (with tool_calls) to history
            messages.append(assistant_message.model_dump(exclude_unset=True))

            # Execute each tool call and append its result
            for tool_call in tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                    result_str = dispatch_tool(
                        tool_name=tool_call.function.name,
                        tool_args=args,
                        indexer=self.indexer,
                        graph=self.graph,
                    )
                except Exception as exc:
                    result_str = json.dumps({"error": str(exc)}, ensure_ascii=False)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_str,
                    }
                )

        if conversation_id:
            save_message(conversation_id, "user", user_message)
            save_message(conversation_id, "assistant", answer)

        return answer


# ── REPL ──────────────────────────────────────────────────────────────────────


_SESSION_FILE = os.path.join(os.path.dirname(__file__), ".conversation_id")


def _load_session_id() -> str:
    try:
        with open(_SESSION_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return str(uuid.uuid4())


def _save_session_id(conversation_id: str) -> None:
    try:
        with open(_SESSION_FILE, "w") as f:
            f.write(conversation_id)
    except OSError:
        pass


def run_repl(agent: MedicalSearchAgent, conversation_id: str | None = None) -> None:
    if not REDIS_AVAILABLE:
        print("Medical Search Agent — type 'quit' to exit.")
        print("[Warning] Redis not available — conversation history disabled.\n")
        conversation_id = None
    else:
        if conversation_id is None:
            conversation_id = _load_session_id()
        _save_session_id(conversation_id)
        print(f"Medical Search Agent — conversation: {conversation_id}")
        print("Commands: 'quit' to exit, 'new' to start a new conversation, 'clear' to clear history.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye.")
            break
        if user_input.lower() == "new":
            conversation_id = str(uuid.uuid4())
            _save_session_id(conversation_id)
            print(f"[New conversation: {conversation_id}]\n")
            continue
        if user_input.lower() == "clear":
            delete_conversation(conversation_id)
            print("[History cleared]\n")
            continue

        t0 = time.perf_counter()
        try:
            answer = agent.query(user_input, conversation_id=conversation_id)
        except Exception as exc:
            answer = f"[Error] {exc}"
        elapsed = time.perf_counter() - t0

        print(answer)
        print(f"  ({elapsed:.2f}s)\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Search Agent REPL")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j bolt URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="quantapass", help="Neo4j password")
    parser.add_argument("--conversation-id", default=None, help="Resume an existing conversation by ID")
    args = parser.parse_args()

    print(f"[INIT] Connecting to Neo4j at {args.uri} and building indexes …")
    indexer = build_indexer(uri=args.uri, user=args.user, password=args.password)
    graph = indexer._graph

    agent = MedicalSearchAgent(indexer=indexer, graph=graph)
    run_repl(agent, conversation_id=args.conversation_id)
