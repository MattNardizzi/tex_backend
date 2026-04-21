from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tex.api.routes import build_api_router
from tex.commands.activate_policy import ActivatePolicyCommand
from tex.commands.calibrate_policy import CalibratePolicyCommand
from tex.commands.evaluate_action import EvaluateActionCommand
from tex.commands.export_bundle import ExportBundleCommand
from tex.commands.report_outcome import ReportOutcomeCommand
from tex.domain.evaluation import EvaluationRequest
from tex.domain.policy import PolicySnapshot
from tex.domain.retrieval import RetrievedEntity, RetrievedPolicyClause, RetrievedPrecedent
from tex.engine.pdp import PolicyDecisionPoint
from tex.evidence.exporter import EvidenceExporter
from tex.evidence.recorder import EvidenceRecorder
from tex.learning.calibrator import ThresholdCalibrator, build_default_calibrator
from tex.policies.defaults import build_default_policy, build_strict_policy
from tex.retrieval.orchestrator import RetrievalOrchestrator
from tex.stores.decision_store import InMemoryDecisionStore
from tex.stores.entity_store import InMemoryEntityStore
from tex.stores.outcome_store import InMemoryOutcomeStore
from tex.stores.policy_store import InMemoryPolicyStore
from tex.stores.precedent_store import InMemoryPrecedentStore


DEFAULT_EVIDENCE_PATH = Path("var/tex/evidence/evidence.jsonl")
APP_TITLE = "Tex"
APP_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class TexRuntime:
    """
    Fully wired in-process runtime for Tex.

    This is Tex's composition root. It keeps dependency wiring explicit and
    local instead of spreading startup behavior across modules and globals.
    """

    pdp: PolicyDecisionPoint
    calibrator: ThresholdCalibrator

    policy_store: InMemoryPolicyStore
    decision_store: InMemoryDecisionStore
    outcome_store: InMemoryOutcomeStore
    precedent_store: InMemoryPrecedentStore
    entity_store: InMemoryEntityStore

    evidence_recorder: EvidenceRecorder
    evidence_exporter: EvidenceExporter

    evaluate_action_command: EvaluateActionCommand
    report_outcome_command: ReportOutcomeCommand
    activate_policy_command: ActivatePolicyCommand
    calibrate_policy_command: CalibratePolicyCommand
    export_bundle_command: ExportBundleCommand


class InMemoryPolicyClauseStoreAdapter:
    """
    Thin retrieval adapter that projects policy snapshot data into grounding clauses.

    Tex does not need a separate policy-clause database yet. For local runtime,
    the active policy snapshot already contains enough structured material to
    create usable retrieval grounding:
    - blocked terms
    - sensitive entities
    - enabled recognizers
    - operator metadata

    This is deliberately lightweight and deterministic.
    """

    __slots__ = ()

    def retrieve_policy_clauses(
        self,
        *,
        policy: PolicySnapshot,
        request: EvaluationRequest,
        top_k: int,
    ) -> tuple[RetrievedPolicyClause, ...]:
        if top_k <= 0:
            return tuple()

        candidates: list[RetrievedPolicyClause] = []
        request_text = f"{request.action_type} {request.channel} {request.environment} {request.content}".casefold()

        rank = 1
        for term in policy.blocked_terms:
            relevance = 0.98 if term.casefold() in request_text else 0.72
            candidates.append(
                RetrievedPolicyClause(
                    clause_id=f"{policy.version}:blocked_term:{rank}",
                    policy_id=policy.policy_id,
                    policy_version=policy.version,
                    title="Blocked term restriction",
                    # Clause text carries the policy payload only. Downstream
                    # overlap matchers tokenize this text, so any generic
                    # English boilerplate here becomes false-positive surface
                    # area against benign request content.
                    text=term,
                    channel=request.channel,
                    action_type=request.action_type,
                    relevance_score=relevance,
                    rank=rank,
                    metadata={
                        "source": "policy_snapshot.blocked_terms",
                        "blocked_term": term,
                    },
                )
            )
            rank += 1

        for entity in policy.sensitive_entities:
            relevance = 0.95 if entity.casefold() in request_text else 0.68
            candidates.append(
                RetrievedPolicyClause(
                    clause_id=f"{policy.version}:sensitive_entity:{rank}",
                    policy_id=policy.policy_id,
                    policy_version=policy.version,
                    title="Sensitive entity handling",
                    # Clause text is just the entity name for the same reason.
                    text=entity,
                    channel=request.channel,
                    action_type=request.action_type,
                    relevance_score=relevance,
                    rank=rank,
                    metadata={
                        "source": "policy_snapshot.sensitive_entities",
                        "sensitive_entity": entity,
                    },
                )
            )
            rank += 1

        for recognizer_name in policy.enabled_recognizers:
            relevance = 0.60
            if recognizer_name.casefold().replace("_", " ") in request_text:
                relevance = 0.82
            candidates.append(
                RetrievedPolicyClause(
                    clause_id=f"{policy.version}:recognizer:{rank}",
                    policy_id=policy.policy_id,
                    policy_version=policy.version,
                    title="Enabled recognizer policy",
                    # Recognizer name only. The human-readable framing is in
                    # the title; keeping the text minimal avoids accidental
                    # overlap hits on generic English tokens.
                    text=recognizer_name.replace("_", " "),
                    channel=request.channel,
                    action_type=request.action_type,
                    relevance_score=relevance,
                    rank=rank,
                    metadata={
                        "source": "policy_snapshot.enabled_recognizers",
                        "recognizer": recognizer_name,
                    },
                )
            )
            rank += 1

        metadata_description = policy.metadata.get("description")
        if isinstance(metadata_description, str) and metadata_description.strip():
            candidates.append(
                RetrievedPolicyClause(
                    clause_id=f"{policy.version}:metadata:{rank}",
                    policy_id=policy.policy_id,
                    policy_version=policy.version,
                    title="Policy description",
                    text=metadata_description.strip(),
                    channel=request.channel,
                    action_type=request.action_type,
                    relevance_score=0.55,
                    rank=rank,
                    metadata={"source": "policy_snapshot.metadata.description"},
                )
            )

        ranked = sorted(
            candidates,
            key=lambda item: (-item.relevance_score, item.rank),
        )[:top_k]

        return tuple(
            item.model_copy(update={"rank": index})
            for index, item in enumerate(ranked, start=1)
        )


class InMemoryPrecedentStoreAdapter:
    """
    Thin adapter from the concrete in-memory precedent store to the retrieval protocol.
    """

    __slots__ = ("_store",)

    def __init__(self, store: InMemoryPrecedentStore) -> None:
        self._store = store

    def retrieve_precedents(
        self,
        *,
        request: EvaluationRequest,
        limit: int,
    ) -> tuple[RetrievedPrecedent, ...]:
        if limit <= 0:
            return tuple()

        return self._store.find_similar(
            action_type=request.action_type,
            channel=request.channel,
            environment=request.environment,
            recipient=request.recipient,
            limit=limit,
        )


class InMemoryEntityStoreAdapter:
    """
    Thin adapter from the concrete in-memory entity store to the retrieval protocol.
    """

    __slots__ = ("_store",)

    def __init__(self, store: InMemoryEntityStore) -> None:
        self._store = store

    def retrieve_entities(
        self,
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
        top_k: int,
    ) -> tuple[RetrievedEntity, ...]:
        if top_k <= 0:
            return tuple()

        return self._store.find_matching(
            text=request.content,
            limit=top_k,
        )


def build_runtime(
    *,
    evidence_path: str | Path = DEFAULT_EVIDENCE_PATH,
) -> TexRuntime:
    """
    Build Tex's local in-process runtime.

    Important fixes in this composition root:
    - retrieval is actually wired into the live PDP
    - default policies are seeded exactly once
    - default sensitive entities are seeded into the entity store
    - evidence path is normalized and directory-safe
    """
    normalized_evidence_path = Path(evidence_path)

    policy_store = InMemoryPolicyStore()
    decision_store = InMemoryDecisionStore()
    outcome_store = InMemoryOutcomeStore()
    precedent_store = InMemoryPrecedentStore()
    entity_store = InMemoryEntityStore()

    _seed_default_policies(policy_store)
    _seed_default_entities(policy_store=policy_store, entity_store=entity_store)

    recorder = EvidenceRecorder(normalized_evidence_path)
    exporter = EvidenceExporter(recorder)

    retrieval_orchestrator = RetrievalOrchestrator(
        policy_store=InMemoryPolicyClauseStoreAdapter(),
        precedent_store=InMemoryPrecedentStoreAdapter(precedent_store),
        entity_store=InMemoryEntityStoreAdapter(entity_store),
    )

    pdp = PolicyDecisionPoint(
        retrieval_orchestrator=retrieval_orchestrator,
    )
    calibrator = build_default_calibrator()

    evaluate_action_command = EvaluateActionCommand(
        pdp=pdp,
        policy_store=policy_store,
        decision_store=decision_store,
        precedent_store=precedent_store,
        evidence_recorder=recorder,
    )

    report_outcome_command = ReportOutcomeCommand(
        decision_store=decision_store,
        outcome_store=outcome_store,
        evidence_recorder=recorder,
    )

    activate_policy_command = ActivatePolicyCommand(
        policy_store=policy_store,
    )

    calibrate_policy_command = CalibratePolicyCommand(
        policy_store=policy_store,
        outcome_store=outcome_store,
        calibrator=calibrator,
    )

    export_bundle_command = ExportBundleCommand(
        exporter=exporter,
    )

    return TexRuntime(
        pdp=pdp,
        calibrator=calibrator,
        policy_store=policy_store,
        decision_store=decision_store,
        outcome_store=outcome_store,
        precedent_store=precedent_store,
        entity_store=entity_store,
        evidence_recorder=recorder,
        evidence_exporter=exporter,
        evaluate_action_command=evaluate_action_command,
        report_outcome_command=report_outcome_command,
        activate_policy_command=activate_policy_command,
        calibrate_policy_command=calibrate_policy_command,
        export_bundle_command=export_bundle_command,
    )


def create_app(
    *,
    runtime: TexRuntime | None = None,
    evidence_path: str | Path = DEFAULT_EVIDENCE_PATH,
) -> FastAPI:
    """
    Create and configure the FastAPI application for Tex.

    If no runtime is supplied, this builds the default in-process runtime.
    """
    resolved_runtime = runtime or build_runtime(evidence_path=evidence_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Idempotent safety net. Runtime is already attached below at app
        # construction time so that synchronous entry points (TestClient,
        # direct import-time probes, test fixtures that skip lifespan)
        # always observe a fully populated app.state.
        _attach_runtime_to_app(app, resolved_runtime)
        yield

    app = FastAPI(
        title=APP_TITLE,
        version=APP_VERSION,
        description=(
            "Tex is a retrieval-grounded, evidence-aware, abstention-capable "
            "content adjudication engine for AI actions."
        ),
        lifespan=lifespan,
    )

    # Attach runtime state eagerly so that app.state is populated the moment
    # create_app returns, regardless of whether the caller enters lifespan.
    _attach_runtime_to_app(app, resolved_runtime)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(build_api_router())

    @app.get("/", tags=["tex"], summary="Tex service metadata")
    def root() -> dict[str, object]:
        active_policy = resolved_runtime.policy_store.get_active()

        return {
            "service": APP_TITLE,
            "version": APP_VERSION,
            "status": "ok",
            "active_policy_version": active_policy.version if active_policy else None,
            "retrieval_enabled": True,
            "precedent_count": len(resolved_runtime.precedent_store.list_all()),
            "entity_count": len(resolved_runtime.entity_store.list_all()),
            "evidence_path": str(resolved_runtime.evidence_recorder.path),
        }

    return app


def _attach_runtime_to_app(app: FastAPI, runtime: TexRuntime) -> None:
    """
    Publish the runtime and command stack into FastAPI app state.

    The route layer depends on these exact names.
    """
    app.state.runtime = runtime

    app.state.pdp = runtime.pdp
    app.state.calibrator = runtime.calibrator

    app.state.policy_store = runtime.policy_store
    app.state.decision_store = runtime.decision_store
    app.state.outcome_store = runtime.outcome_store
    app.state.precedent_store = runtime.precedent_store
    app.state.entity_store = runtime.entity_store

    app.state.evidence_recorder = runtime.evidence_recorder
    app.state.evidence_exporter = runtime.evidence_exporter

    app.state.evaluate_action_command = runtime.evaluate_action_command
    app.state.report_outcome_command = runtime.report_outcome_command
    app.state.activate_policy_command = runtime.activate_policy_command
    app.state.calibrate_policy_command = runtime.calibrate_policy_command
    app.state.export_bundle_command = runtime.export_bundle_command


def _seed_default_policies(policy_store: InMemoryPolicyStore) -> None:
    """
    Load the baseline policy snapshots into the policy store exactly once.
    """
    default_policy = build_default_policy()
    strict_policy = build_strict_policy()

    if default_policy.version not in policy_store:
        policy_store.save(default_policy)

    if strict_policy.version not in policy_store:
        policy_store.save(strict_policy)


def _seed_default_entities(
    *,
    policy_store: InMemoryPolicyStore,
    entity_store: InMemoryEntityStore,
) -> None:
    """
    Seed the entity store from policy-defined sensitive entities.

    This keeps retrieval alive in local development without introducing a
    separate persistence layer before it is justified.
    """
    seen_names: set[str] = set()
    rank = 1

    for policy in policy_store.list_policies():
        for entity_name in policy.sensitive_entities:
            dedupe_key = entity_name.casefold()
            if dedupe_key in seen_names:
                continue
            seen_names.add(dedupe_key)

            entity_store.save(
                RetrievedEntity(
                    entity_id=f"{policy.version}:entity:{rank}",
                    entity_type="policy_sensitive_entity",
                    canonical_name=entity_name,
                    aliases=tuple(),
                    sensitivity="high",
                    description=(
                        "Seeded from policy.sensitive_entities for local retrieval grounding."
                    ),
                    relevance_score=0.90,
                    rank=rank,
                    metadata={
                        "source_policy_id": policy.policy_id,
                        "source_policy_version": policy.version,
                        "seeded_from": "policy_snapshot.sensitive_entities",
                    },
                )
            )
            rank += 1


app = create_app()