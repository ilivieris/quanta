from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import sys
import time
import uuid

import openai
from dotenv import load_dotenv

load_dotenv()

from examples.project.indexer import ImageIndexer, MedicalIndexer, build_image_indexer, build_indexer
from examples.project.redis_service import REDIS_AVAILABLE, delete_conversation, load_history, save_message
from examples.project.tools import TOOLS, dispatch_tool


# ── Local LLM config ──────────────────────────────────────────────────────────

_LOCAL_LLM_URL   = os.environ.get("LOCAL_LLM_URL",   "https://llm-serving.eu-dev.novelcore.org").rstrip("/")
_LOCAL_LLM_MODEL = os.environ.get("LOCAL_LLM_MODEL", "Qwen/Qwen3.5-9B")
_MAX_TOKENS            = int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "32768"))
_MAX_TOOL_RESULT_CHARS = 4000


# ── Inline tool-call parsing ──────────────────────────────────────────────────
# The vLLM deployment accepts `tools` but does not parse the model's tool
# invocations: `tool_calls` comes back null and the call arrives as raw text
# in `content`, in Qwen's XML format:
#   <tool_call>
#   <function=fuzzy_name_search>
#   <parameter=name>
#   Σάρρα Μπαλαμπάνη
#   </parameter>
#   </function>
#   </tool_call>
# (Older Qwen builds emit a JSON object inside <tool_call> instead.)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE  = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
_THINK_RE     = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_inline_tool_calls(content: str) -> list[tuple[str, dict]]:
    """Extract (tool_name, args) pairs from raw <tool_call> text."""
    calls: list[tuple[str, dict]] = []
    for block in _TOOL_CALL_RE.findall(content):
        if block.lstrip().startswith("{"):
            try:
                payload = json.loads(block)
                calls.append((payload["name"], payload.get("arguments") or {}))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
            continue
        for name, body in _FUNCTION_RE.findall(block):
            args: dict = {}
            for key, value in _PARAMETER_RE.findall(body):
                value = value.strip()
                try:
                    args[key] = json.loads(value)
                except json.JSONDecodeError:
                    args[key] = value
            calls.append((name, args))
    return calls


def _strip_markup(content: str) -> str:
    """Remove <think> and stray <tool_call> blocks from a final answer."""
    content = _THINK_RE.sub("", content)
    content = _TOOL_CALL_RE.sub("", content)
    return content.strip()


class _StreamPrinter:
    """Echo streamed deltas live, while assembling the full content.

    Text is held back by a few characters so that a <tool_call> or <think>
    marker split across SSE chunks is never partially printed; once a marker
    appears, everything from it onward is swallowed (the tool loop handles it).
    """

    _MARKERS = ("<tool_call>", "<think>")
    _HOLDBACK = max(len(marker) for marker in _MARKERS)

    def __init__(self, echo: bool) -> None:
        self.echo = echo
        self.buffer = ""
        self.printed = 0
        self.suppressed = False

    def feed(self, delta: str) -> None:
        self.buffer += delta
        self._flush(final=False)

    def finish(self) -> str:
        """Flush any held-back tail and return the full raw content."""
        self._flush(final=True)
        return self.buffer

    def _flush(self, final: bool) -> None:
        if not self.echo or self.suppressed:
            return
        positions = [i for m in self._MARKERS if (i := self.buffer.find(m)) != -1]
        if positions:
            limit = min(positions)
            self.suppressed = True
        else:
            limit = len(self.buffer) if final else len(self.buffer) - self._HOLDBACK
        if limit > self.printed:
            print(self.buffer[self.printed:limit], end="", flush=True)
            self.printed = limit


# ── System prompt (identical to agent.py) ────────────────────────────────────

_SYSTEM_PROMPT_BASE = """\
You are a medical information assistant with access to a hospital knowledge
graph. You help medical staff retrieve patient information, doctor details,
diagnoses, procedures, medications, and hospital admissions.

===========================================================================
MANDATORY FIRST STEP — APPLIES TO EVERY SINGLE REQUEST, NO EXCEPTIONS:
===========================================================================
Whenever the user's message contains ANY name — a doctor name, patient name,
hospital name, diagnosis name, or any other named entity — you MUST call
fuzzy_name_search FIRST before calling any other tool.

DO NOT pass names to cypher_query. Names are NOT stored as lookup keys in
the graph. A Cypher query with {name: "..."} or WHERE x.name = "..." will
ALWAYS return 0 results. The graph is indexed by IDs (DOC0013, PAT0042, etc).

The ONLY valid workflow when a name is mentioned:
  STEP 1 → fuzzy_name_search(name="<the name>", index_name="<doctorNames|patientNames|diagnosisNames>")
  STEP 2 → use the returned ID in all subsequent tool calls

NEVER skip Step 1. NEVER guess an ID. NEVER filter by name in Cypher.
===========================================================================

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

IMAGE SEARCH
- image_search: searches for similar medical images using either a text
  description (cross-modal text→image) or an image file path
  (image→image similarity). Supports filtering by image_type.
- get_patient_images: retrieves all medical images for a specific patient
  with metadata (modality, capture date, description).

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
    final answer. Never describe the tool calls you made. Never say
    "I called fuzzy_name_search" or "the function chain is". Just present
    the data directly as a human-readable answer.
14. IMAGE QUERIES — When the user message contains "[Image attached: PATH]",
    call image_search with query_type="image" and query=PATH. For text
    questions about images (e.g., "βρες chest X-rays"), use query_type="text".
15. PATIENT IMAGES — To list all images of a specific patient, use
    get_patient_images after resolving the patient ID.
16. IMAGE DISPLAY — When presenting image results (from image_search or
    get_patient_images), always include the file_path field as a clickable
    reference so the user knows exactly which file to open. Format it as:
    "Αρχείο: <file_path>".

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
- TREATED_BY direction: Patient points TO Doctor: (p:Patient)-[:TREATED_BY]->(d:Doctor)
  NEVER write (d:Doctor)-[:TREATED_BY]->(p:Patient) — wrong direction always returns 0 rows.
- Patient ID property is `patientId`, NOT `id`. Always use p.patientId, never p.id.
- Always RETURN meaningful fields (name, id, etc.) not whole nodes.
- If the user sends a greeting, small talk, or a message unrelated to medical
  data (e.g. "Καλημέρα", "Hello", "Thanks"), respond conversationally WITHOUT
  calling any tool.

EXAMPLE TOOL CHAINS — follow these patterns exactly:

PATTERN 1 — "φέρε τους ασθενείς που παρακολουθεί ο/η γιατρός X":
  Step 1: fuzzy_name_search(name="X", index_name="doctorNames")
          → returns [{id: "DOC0010", name: "Σάρρα Μπαλαμπάνη", ...}]
  Step 2: cypher_query(cypher="MATCH (p:Patient)-[:TREATED_BY]->(d:Doctor {doctorId: 'DOC0010'}) RETURN p.name AS patientName")

PATTERN 2 — "ποιος γιατρός παρακολουθεί τον/την ασθενή X":
  Step 1: fuzzy_name_search(name="X", index_name="patientNames")
          → returns [{id: "PAT0042", name: "...", ...}]
  Step 2: cypher_query(cypher="MATCH (p:Patient {patientId: 'PAT0042'})-[:TREATED_BY]->(d:Doctor) RETURN d.name AS doctorName")

PATTERN 3 — "φέρε πληροφορίες / ιστορικό για τον/την ασθενή X":
  Step 1: fuzzy_name_search(name="X", index_name="patientNames")
          → returns [{id: "PAT0042", name: "Υπαπαντή Κουθούρη", ...}]
  Step 2: get_patient_timeline(patient_id="PAT0042")
  *** WRONG: get_patient_timeline(patient_id="Υπαπαντή Κουθούρη") — NEVER use the name ***

PATTERN 4 — "ποιες παθήσεις / φάρμακα έχει ο ασθενής X":
  Step 1: fuzzy_name_search(name="X", index_name="patientNames")
          → returns [{id: "PAT0042", ...}]
  Step 2: cypher_query(cypher="MATCH (p:Patient {patientId: 'PAT0042'})-[r:HAS_DIAGNOSIS]->(d:Diagnosis) RETURN d.name AS diagnosis, r.severity AS severity")

PATTERN 5 — "έλεγξε αλληλεπιδράσεις φαρμάκων για τον ασθενή X":
  Step 1: fuzzy_name_search(name="X", index_name="patientNames")
          → returns [{id: "PAT0042", ...}]
  Step 2: drug_interaction_check(patient_id="PAT0042")
  *** WRONG: drug_interaction_check(patient_id="Υπαπαντή Κουθούρη") — NEVER use the name ***

PATTERN 6 — "δείξε μου όλες τις συνδέσεις του ασθενή X" (network/graph exploration):
  Step 1: fuzzy_name_search(name="X", index_name="patientNames")
          → returns [{id: "PAT0042", ...}]
  Step 2: graph_expand(node_id="PAT0042", node_type="patient", hops=1)

PATTERN 7 — "πόσοι ασθενείς / γιατροί υπάρχουν στη βάση" (count):
  Step 1: cypher_query(cypher="MATCH (p:Patient) RETURN count(p) AS total")

PATTERN 8 — "ποιοι ασθενείς παίρνουν / έχουν συνταγογραφηθεί φάρμακο X":
  Medication nodes are filtered by name directly — NO fuzzy_name_search needed.
  Always add WHERE rx.active = true to return only current prescriptions.
  Step 1: cypher_query(cypher="MATCH (p:Patient)-[rx:PRESCRIBED]->(m:Medication {name: 'Lisinopril'}) WHERE rx.active = true RETURN p.name AS patientName")
  *** WRONG: semantic_search for a medication name — medications are not in the semantic index ***
  *** WRONG: omitting WHERE rx.active = true — returns historical prescriptions too ***

PATTERN 9 — "ποιους ασθενείς παρακολουθούν οι γιατροί X και Y" (multiple doctors):
  Step 1: fuzzy_name_search(name="X", index_name="doctorNames") → returns [{id: "DOC0001", ...}]
  Step 2: fuzzy_name_search(name="Y", index_name="doctorNames") → returns [{id: "DOC0002", ...}]
  Step 3: cypher_query(cypher="MATCH (p:Patient)-[:TREATED_BY]->(d:Doctor) WHERE d.doctorId IN ['DOC0001', 'DOC0002'] RETURN p.name AS patientName, d.name AS doctorName ORDER BY d.name")
  *** WRONG: MATCH (d:Doctor)-[:TREATED_BY]->(p:Patient) — ALWAYS wrong direction, returns 0 rows ***
  *** WRONG: using UNION with reversed direction in either branch ***
  Use WHERE d.doctorId IN [...] — never use UNION for multiple doctors.\
"""


def _build_system_prompt(graph) -> str:
    try:
        schema = graph.get_schema()
    except Exception:
        schema = "(schema unavailable)"
    return _SYSTEM_PROMPT_BASE + "\n\nGraph schema:\n" + schema


# ── Agent ─────────────────────────────────────────────────────────────────────

class MedicalSearchAgent:
    def __init__(
        self,
        indexer: MedicalIndexer,
        graph,
        image_indexer: ImageIndexer | None = None,
    ) -> None:
        self.indexer = indexer
        self.graph = graph
        self.image_indexer = image_indexer
        self.client = openai.OpenAI(
            base_url=f"{_LOCAL_LLM_URL}/v1",
            api_key="not-needed",
        )
        self.model = _LOCAL_LLM_MODEL
        self.system_prompt = _build_system_prompt(graph)

    def query(self, user_message: str, conversation_id: str | None = None, stream: bool = True) -> str:
        history = load_history(conversation_id) if conversation_id else []
        history = []
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]

        answer = ""
        iteration = 0
        while True:
            iteration += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS,
                    max_tokens=_MAX_TOKENS,
                    tool_choice="auto",
                    stream=True,
                )
            except openai.APIStatusError as exc:
                body = None
                with contextlib.suppress(Exception):
                    body = exc.response.text
                print(f"[DEBUG] LLM request failed | status={exc.status_code} body={body!r}")
                print(f"[DEBUG] last 2 messages sent: {json.dumps(messages[-2:], ensure_ascii=False, indent=2)}")
                raise

            # Assemble the streamed response, echoing final-answer text live.
            printer = _StreamPrinter(echo=stream)
            tool_call_parts: dict[int, dict] = {}
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    printer.feed(delta.content)
                for tc in delta.tool_calls or []:
                    part = tool_call_parts.setdefault(
                        tc.index, {"id": tc.id, "name": "", "arguments": ""}
                    )
                    if tc.id:
                        part["id"] = tc.id
                    if tc.function and tc.function.name:
                        part["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        part["arguments"] += tc.function.arguments
            content = printer.finish()

            # calls: list of (tool_call_id | None, tool_name, args | raw-args-json)
            if tool_call_parts:
                # Native OpenAI-style tool calls, if the server ever parses them.
                parts = [part for _, part in sorted(tool_call_parts.items())]
                for i, part in enumerate(parts):
                    part["id"] = part["id"] or f"call_{i}"
                calls = [(p["id"], p["name"], p["arguments"]) for p in parts]
                messages.append(
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": [
                            {
                                "id": p["id"],
                                "type": "function",
                                "function": {"name": p["name"], "arguments": p["arguments"]},
                            }
                            for p in parts
                        ],
                    }
                )
            else:
                inline_calls = _parse_inline_tool_calls(content)
                if not inline_calls:
                    answer = _strip_markup(content)
                    if stream:
                        if printer.printed == 0 and answer:
                            # Nothing was echoed live (e.g. a <think> block came
                            # first) — print the cleaned answer in one go.
                            print(answer, end="")
                        print(flush=True)
                    break
                # Normalize into proper OpenAI tool_calls so the follow-up "tool"
                # messages always have a matching tool_call_id — some backends
                # 500 on a "tool" message that doesn't pair with an assistant
                # tool_calls entry (which raw <tool_call> XML text does not provide).
                leftover = _TOOL_CALL_RE.sub("", content).strip()
                calls = []
                tool_calls_payload = []
                for i, (name, args) in enumerate(inline_calls):
                    call_id = f"call_{i}"
                    calls.append((call_id, name, args))
                    tool_calls_payload.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": leftover or None,
                        "tool_calls": tool_calls_payload,
                    }
                )

            if stream and printer.printed:
                print(flush=True)  # separate any streamed preamble from tool logs

            for call_id, tool_name, tool_args in calls:
                try:
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)
                    result_str = dispatch_tool(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        indexer=self.indexer,
                        graph=self.graph,
                        image_indexer=self.image_indexer,
                    )
                except Exception as exc:
                    result_str = json.dumps({"error": str(exc)}, ensure_ascii=False)

                result_preview = result_str[:200].replace("\n", " ")
                print(f"[TOOL result] {tool_name}: {result_preview!r}")

                # Guard against oversized tool results that would overflow the context
                if len(result_str) > _MAX_TOOL_RESULT_CHARS:
                    print(f"[WARN] tool result truncated ({len(result_str)} → {_MAX_TOOL_RESULT_CHARS} chars)")
                    result_str = (
                        result_str[:_MAX_TOOL_RESULT_CHARS]
                        + f"\n... [truncated — result exceeded {_MAX_TOOL_RESULT_CHARS} chars]"
                    )

                tool_message: dict = {"role": "tool", "content": result_str}
                if call_id is not None:
                    tool_message["tool_call_id"] = call_id
                messages.append(tool_message)

        if conversation_id:
            save_message(conversation_id, "user", user_message)
            save_message(conversation_id, "assistant", answer)

        return answer


# ── REPL helpers ──────────────────────────────────────────────────────────────

_SESSION_FILE = os.path.join(os.path.dirname(__file__), ".conversation_id")
_IMAGE_PREFIX_RE = re.compile(r"^image:\s*(.+)$", re.IGNORECASE)


def _parse_image_input(raw: str) -> tuple[str | None, str]:
    match = _IMAGE_PREFIX_RE.match(raw.strip())
    if not match:
        return None, raw

    rest = match.group(1).strip()
    if rest.startswith('"') or rest.startswith("'"):
        try:
            parts = shlex.split(rest)
            image_path = parts[0]
            text_query = " ".join(parts[1:])
        except ValueError:
            image_path = rest
            text_query = ""
    else:
        parts = rest.split(None, 1)
        image_path = parts[0]
        text_query = parts[1] if len(parts) > 1 else ""

    return image_path, text_query or "Βρες παρόμοιες ιατρικές εικόνες και τους συνδεδεμένους ασθενείς."


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
        print("Medical Search Agent (Local LLM) — type 'quit' to exit.")
        print("[Warning] Redis not available — conversation history disabled.\n")
        conversation_id = None
    else:
        if conversation_id is None:
            conversation_id = _load_session_id()
        _save_session_id(conversation_id)
        print(f"Medical Search Agent (Local LLM) — conversation: {conversation_id}")
        print("Commands: 'quit' to exit, 'new' to start a new conversation, 'clear' to clear history.\n")

    # print(f"[LLM] {_LOCAL_LLM_MODEL} @ {_LOCAL_LLM_URL}\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input:
            continue

        image_path, user_input = _parse_image_input(user_input)
        if image_path is not None:
            if not os.path.isfile(image_path):
                print(f"[Error] Image file not found: {image_path}\n")
                continue
            user_input = f"[Image attached: {image_path}]\n{user_input}"

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
            agent.query(user_input, conversation_id=conversation_id)
        except Exception as exc:
            print(f"[Error] {exc}")
        elapsed = time.perf_counter() - t0

        print(f"  ({elapsed:.2f}s)\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # On Windows, stdin/stdout attached to a non-native console (VS Code's
    # integrated/debug terminal, a pipe, etc.) fall back to the system's
    # legacy codepage (e.g. cp1252) instead of UTF-8. Typing Greek text then
    # gets mis-decoded into lone surrogate characters that crash later when
    # the request body is UTF-8 encoded for the LLM server. Force UTF-8 here
    # so REPL input/output is decoded/encoded consistently regardless of how
    # the process was launched.
    if sys.platform == "win32":
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Medical Search Agent REPL (Local LLM)")
    parser.add_argument("--uri",             default="bolt://localhost:7687", help="Neo4j bolt URI")
    parser.add_argument("--user",            default="neo4j",                help="Neo4j username")
    parser.add_argument("--password",        default="quantapass",           help="Neo4j password")
    parser.add_argument("--conversation-id", default=None,                   help="Resume an existing conversation by ID")
    parser.add_argument("--llm-url",         default=None,                   help="Override LOCAL_LLM_URL env var")
    parser.add_argument("--llm-model",       default=None,                   help="Override LOCAL_LLM_MODEL env var")
    args = parser.parse_args()

    if args.llm_url:
        os.environ["LOCAL_LLM_URL"] = args.llm_url
    if args.llm_model:
        os.environ["LOCAL_LLM_MODEL"] = args.llm_model

    print(f"[INIT] Connecting to Neo4j at {args.uri} and building indexes …")
    indexer = build_indexer(uri=args.uri, user=args.user, password=args.password)
    graph = indexer._graph

    image_indexer = build_image_indexer(graph=graph)

    agent = MedicalSearchAgent(indexer=indexer, graph=graph, image_indexer=image_indexer)
    run_repl(agent, conversation_id=args.conversation_id)
