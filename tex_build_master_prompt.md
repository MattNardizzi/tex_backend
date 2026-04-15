# Tex Build Prompt v2

## What Tex Is

Tex is the **last-mile content gate for AI agent actions**. It reads the actual content an AI agent is about to release into the real world and returns PERMIT, ABSTAIN, or FORBID.

Tex is **not** the identity, permissions, behavioral monitoring, or tool-access layer. Every funded player (Zenity, Microsoft Agent Governance Toolkit, Cisco, Geordie) works upstream — discovering agents, managing identities, enforcing tool access, monitoring behavior. None of them answer: **this specific content, right now — should it go out?** That is Tex.

## Current Codebase State (April 2026)

The project is a **10,775-line Python codebase** (FastAPI + Pydantic v2) that boots, runs, and evaluates. The architecture is correct and the domain models are production-grade.

**Working:** Deterministic gate (7 recognizers), specialist judges (4 heuristic judges), score fusion router, policy system (default + strict snapshots, versioning, calibration), evidence chain (SHA-256 hash-chained JSONL), outcome classification, full API (6 routes), explicit abstention logic.

**Not working:** The LLM semantic provider has no adapter — every evaluation runs through a keyword-matching fallback. Retrieval returns empty context every time (stores exist but aren't wired to the orchestrator). All 11 test files are empty. `pyproject.toml`, `config.py`, and `telemetry.py` are empty. `domain/semantic_result.py` is orphaned (its dimensions don't match the pipeline's actual schema in `semantic/schema.py`).

## Architecture (Do Not Change)

**Sacred evaluation order:**
deterministic recognizers → retrieval grounding → specialist judges → structured semantic judge → router/abstention → evidence → outcome learning

**File structure:** `src/tex/` with subpackages: `domain/`, `deterministic/`, `retrieval/`, `specialists/`, `semantic/`, `engine/`, `learning/`, `evidence/`, `stores/`, `commands/`, `api/`, `policies/`, `observability/`

## What To Build Next (Priority Order)

### 1. LLM Semantic Provider Adapter
Build the `StructuredSemanticProvider` implementation that calls OpenAI or Anthropic with the existing prompts (`semantic/prompt.py`) and parses structured output into `SemanticAnalysis` (`semantic/schema.py`).

**Current state of the art (April 2026):** Both OpenAI and Anthropic now offer native structured outputs with 100% schema compliance via constrained decoding. OpenAI uses `response_format: { type: "json_schema", json_schema: { schema: ..., strict: true } }`. Anthropic uses `output_config: { format: { type: "json_schema", schema: ... } }` with `messages.parse()` returning typed Pydantic objects directly. Use `client.beta.chat.completions.parse(response_format=PydanticModel)` for OpenAI or `client.messages.parse(output_format=PydanticModel)` for Anthropic. Set temperature to 0 for evaluation consistency.

The prompt in `semantic/prompt.py` is already written and strong. The schema in `semantic/schema.py` is already defined. The adapter is ~200 lines.

### 2. Wire Retrieval Stores to Orchestrator
Connect `InMemoryPrecedentStore` and `InMemoryEntityStore` to `RetrievalOrchestrator`. The stores already have `find_similar()` and `find_matching()` methods. The orchestrator already has `PolicyClauseStore`, `PrecedentStore`, and `EntityStore` protocols. Write thin adapter classes that bridge the stores to the protocols. ~150 lines.

### 3. Fill `pyproject.toml` and `config.py`
Make the project installable. Add environment variable handling for API keys, model selection, evidence path. Use `pydantic-settings` for config.

### 4. Write Core Tests
Priority test files: `test_deterministic.py` (recognizer coverage), `test_router.py` (verdict logic at threshold boundaries), `test_pdp.py` (end-to-end pipeline), `test_outcomes.py` (classification correctness). Use `factories.py` for test fixtures.

### 5. Delete `domain/semantic_result.py`
It defines 6 dimensions (privilege_escalation, external_exfiltration, etc.) that conflict with the 5 dimensions actually used in the pipeline (policy_compliance, data_leakage, external_sharing, unauthorized_commitment, destructive_or_bypass). The real schema lives in `semantic/schema.py`. The orphan creates confusion.

## Coding Standards

- Python 3.12+, FastAPI, Pydantic v2, full type hints
- Immutable models: `ConfigDict(frozen=True, extra="forbid")`
- UTC timestamps, absolute imports from `tex.*`
- No placeholder TODOs, no fake abstractions, no magical hidden behavior
- Tests must be real and must run

## Rules

1. **Do not rewrite working code** unless there is a concrete bug or architectural violation.
2. **Do not add files** without justifying their existence.
3. **Do not drift** into identity, permissions, dashboards, vector DBs, RL pipelines, multi-agent debate, or SLM fine-tuning. Those come later.
4. **Do not build retrieval infrastructure** (embeddings, vector stores, RAG) yet. In-memory stores are correct for now.
5. **Tell me bluntly** if something is overkill, premature, or wrong.
6. **Write complete, runnable code** when asked. No partial snippets.

## Competitive Context (April 2026)

The OWASP Top 10 for Agentic Applications (2026) explicitly identifies sensitive data disclosure, tool misuse, and missing guardrails as top risks — all of which Tex addresses at the content layer. Microsoft's Agent Governance Toolkit (April 2026) provides sub-millisecond policy enforcement for agent *actions* (tool calls, permissions) but explicitly does not do content moderation. Cisco's agentic security focuses on identity, access control, and SOC tooling. None of these evaluate the *content* of outbound agent actions. That gap is Tex's positioning.

## How To Work

1. Orient to the architecture above
2. Ask which file we're writing
3. Write it completely
4. Do not drift
