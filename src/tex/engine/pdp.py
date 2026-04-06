from __future__ import annotations

import hashlib
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from tex.deterministic.gate import (
    DeterministicGate,
    DeterministicGateResult,
    build_default_deterministic_gate,
)
from tex.domain.decision import Decision
from tex.domain.evaluation import EvaluationRequest, EvaluationResponse
from tex.domain.policy import PolicySnapshot
from tex.domain.retrieval import RetrievalContext
from tex.engine.router import RoutingResult, build_default_router
from tex.retrieval.orchestrator import (
    RetrievalOrchestrator,
    build_noop_retrieval_orchestrator,
)
from tex.semantic.analyzer import (
    SemanticAnalyzer,
    build_default_semantic_analyzer,
)
from tex.semantic.schema import SemanticAnalysis
from tex.specialists.base import SpecialistBundle
from tex.specialists.judges import SpecialistSuite, build_default_specialist_suite


@runtime_checkable
class Router(Protocol):
    """Contract for Tex's routing and fusion layer."""

    def route(
        self,
        *,
        deterministic_result: DeterministicGateResult,
        specialist_bundle: SpecialistBundle,
        semantic_analysis: SemanticAnalysis,
        policy: PolicySnapshot,
        action_type: str,
        channel: str,
        environment: str,
    ) -> RoutingResult:
        """Returns the fused routing result for one evaluation."""


class PDPResult(BaseModel):
    """
    Full internal output of a single Tex evaluation pass.

    This is intentionally richer than the public EvaluationResponse because the
    engine needs to preserve intermediate artifacts for audit, evidence
    recording, replay, debugging, and later outcome analysis.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: EvaluationRequest
    policy: PolicySnapshot

    retrieval_context: RetrievalContext
    deterministic_result: DeterministicGateResult
    specialist_bundle: SpecialistBundle
    semantic_analysis: SemanticAnalysis
    routing_result: RoutingResult

    decision: Decision
    response: EvaluationResponse


class PolicyDecisionPoint:
    """
    Tex's orchestration engine.

    Fixed evaluation order:
    deterministic recognizers
    -> retrieval grounding
    -> specialist judges
    -> semantic judge
    -> routing / abstention
    -> durable decision + public response

    This class coordinates evaluation only. It does not own transport,
    persistence, policy activation, or outcome reporting.
    """

    __slots__ = (
        "_deterministic_gate",
        "_retrieval_orchestrator",
        "_specialist_suite",
        "_semantic_analyzer",
        "_router",
    )

    def __init__(
        self,
        *,
        deterministic_gate: DeterministicGate | None = None,
        retrieval_orchestrator: RetrievalOrchestrator | None = None,
        specialist_suite: SpecialistSuite | None = None,
        semantic_analyzer: SemanticAnalyzer | None = None,
        router: Router | None = None,
    ) -> None:
        self._deterministic_gate = (
            deterministic_gate or build_default_deterministic_gate()
        )
        self._retrieval_orchestrator = (
            retrieval_orchestrator or build_noop_retrieval_orchestrator()
        )
        self._specialist_suite = specialist_suite or build_default_specialist_suite()
        self._semantic_analyzer = semantic_analyzer or build_default_semantic_analyzer()
        self._router = router or build_default_router()

    def evaluate(
        self,
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
    ) -> PDPResult:
        """
        Evaluates one action request against one immutable policy snapshot.

        Returns both the durable internal Decision and the outward-facing
        EvaluationResponse.
        """
        deterministic_result = self._deterministic_gate.evaluate(
            request=request,
            policy=policy,
        )

        retrieval_context = self._retrieval_orchestrator.retrieve(
            request=request,
            policy=policy,
        )

        specialist_bundle = self._specialist_suite.evaluate(
            request=request,
            retrieval_context=retrieval_context,
        )

        semantic_analysis = self._semantic_analyzer.analyze(
            request=request,
            retrieval_context=retrieval_context,
        )

        routing_result = self._router.route(
            deterministic_result=deterministic_result,
            specialist_bundle=specialist_bundle,
            semantic_analysis=semantic_analysis,
            policy=policy,
            action_type=request.action_type,
            channel=request.channel,
            environment=request.environment,
        )

        decision = self._build_decision(
            request=request,
            policy=policy,
            retrieval_context=retrieval_context,
            deterministic_result=deterministic_result,
            specialist_bundle=specialist_bundle,
            semantic_analysis=semantic_analysis,
            routing_result=routing_result,
        )

        response = self._build_response(
            decision=decision,
            routing_result=routing_result,
        )

        return PDPResult(
            request=request,
            policy=policy,
            retrieval_context=retrieval_context,
            deterministic_result=deterministic_result,
            specialist_bundle=specialist_bundle,
            semantic_analysis=semantic_analysis,
            routing_result=routing_result,
            decision=decision,
            response=response,
        )

    def _build_decision(
        self,
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
        retrieval_context: RetrievalContext,
        deterministic_result: DeterministicGateResult,
        specialist_bundle: SpecialistBundle,
        semantic_analysis: SemanticAnalysis,
        routing_result: RoutingResult,
    ) -> Decision:
        content_sha256 = self._sha256_hex(request.content)

        metadata = self._build_decision_metadata(
            request=request,
            policy=policy,
            retrieval_context=retrieval_context,
            deterministic_result=deterministic_result,
            specialist_bundle=specialist_bundle,
            semantic_analysis=semantic_analysis,
            routing_result=routing_result,
            content_sha256=content_sha256,
        )

        return Decision(
            request_id=request.request_id,
            verdict=routing_result.verdict,
            confidence=routing_result.confidence,
            final_score=routing_result.final_score,
            action_type=request.action_type,
            channel=request.channel,
            environment=request.environment,
            recipient=request.recipient,
            content_excerpt=self._build_content_excerpt(request.content),
            content_sha256=content_sha256,
            policy_id=policy.policy_id,
            policy_version=policy.version,
            scores=dict(routing_result.scores),
            findings=list(routing_result.findings),
            reasons=list(routing_result.reasons),
            uncertainty_flags=list(routing_result.uncertainty_flags),
            retrieval_context=self._serialize_retrieval_context(retrieval_context),
            metadata=metadata,
        )

    @staticmethod
    def _build_response(
        *,
        decision: Decision,
        routing_result: RoutingResult,
    ) -> EvaluationResponse:
        return EvaluationResponse(
            decision_id=decision.decision_id,
            verdict=decision.verdict,
            confidence=decision.confidence,
            final_score=decision.final_score,
            reasons=list(routing_result.reasons),
            findings=list(routing_result.findings),
            scores=dict(routing_result.scores),
            uncertainty_flags=list(routing_result.uncertainty_flags),
            policy_version=decision.policy_version,
            evidence_hash=decision.evidence_hash,
            evaluated_at=decision.decided_at,
        )

    def _build_decision_metadata(
        self,
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
        retrieval_context: RetrievalContext,
        deterministic_result: DeterministicGateResult,
        specialist_bundle: SpecialistBundle,
        semantic_analysis: SemanticAnalysis,
        routing_result: RoutingResult,
        content_sha256: str,
    ) -> dict[str, Any]:
        """
        Produces a compact execution summary for audit, replay, and debugging.

        The durable Decision should not store full intermediate objects. This
        summary keeps the high-signal operational facts while leaving the full
        artifacts available in PDPResult when needed in-process.
        """
        metadata = dict(request.metadata)
        metadata["pdp"] = {
            "pdp_version": "v2",
            "request_id": str(request.request_id),
            "request_fingerprint": self._request_fingerprint(
                request=request,
                policy=policy,
                content_sha256=content_sha256,
            ),
            "evaluation_order": [
                "deterministic_recognizers",
                "policy_retrieval",
                "specialist_judges",
                "semantic_judge",
                "routing",
                "decision_materialization",
            ],
            "policy": {
                "policy_id": policy.policy_id,
                "policy_version": policy.version,
                "policy_active": policy.is_active,
                "permit_threshold": policy.permit_threshold,
                "forbid_threshold": policy.forbid_threshold,
                "minimum_confidence": policy.minimum_confidence,
            },
            "request": {
                "request_id": str(request.request_id),
                "action_type": request.action_type,
                "channel": request.channel,
                "environment": request.environment,
                "recipient": request.recipient,
                "has_recipient": request.recipient is not None,
                "content_sha256": content_sha256,
                "requested_at": request.requested_at.isoformat(),
                "policy_id_hint": request.policy_id,
            },
            "deterministic": self._summarize_deterministic(deterministic_result),
            "retrieval": self._summarize_retrieval(retrieval_context),
            "specialists": self._summarize_specialists(specialist_bundle),
            "semantic": self._summarize_semantic(semantic_analysis),
            "routing": self._summarize_routing(routing_result),
        }
        return metadata

    @staticmethod
    def _summarize_deterministic(
        deterministic_result: DeterministicGateResult,
    ) -> dict[str, Any]:
        return {
            "blocked": deterministic_result.blocked,
            "suggested_verdict": PolicyDecisionPoint._stringify_optional_enum(
                deterministic_result.suggested_verdict
            ),
            "enabled_recognizers": list(deterministic_result.enabled_recognizers),
            "finding_count": len(deterministic_result.findings),
            "critical_finding_count": len(deterministic_result.critical_findings),
            "warning_finding_count": len(deterministic_result.warning_findings),
            "info_finding_count": len(deterministic_result.info_findings),
            "blocking_reasons": list(deterministic_result.blocking_reasons),
        }

    @staticmethod
    def _summarize_retrieval(retrieval_context: RetrievalContext) -> dict[str, Any]:
        return {
            "is_empty": retrieval_context.is_empty,
            "policy_clause_count": len(retrieval_context.policy_clauses),
            "precedent_count": len(retrieval_context.precedents),
            "entity_count": len(retrieval_context.entities),
            "warning_count": len(retrieval_context.retrieval_warnings),
            "matched_policy_clause_ids": list(
                retrieval_context.matched_policy_clause_ids
            ),
            "matched_entity_names": list(retrieval_context.matched_entity_names),
            "retrieved_at": retrieval_context.retrieved_at.isoformat(),
        }

    @staticmethod
    def _summarize_specialists(specialist_bundle: SpecialistBundle) -> dict[str, Any]:
        return {
            "judge_count": len(specialist_bundle.results),
            "is_empty": specialist_bundle.is_empty,
            "max_risk_score": specialist_bundle.max_risk_score,
            "min_confidence": specialist_bundle.min_confidence,
            "matched_policy_clause_ids": list(
                specialist_bundle.matched_policy_clause_ids
            ),
            "matched_entity_names": list(specialist_bundle.matched_entity_names),
            "uncertainty_flags": list(specialist_bundle.uncertainty_flags),
            "results": [
                {
                    "specialist_name": result.specialist_name,
                    "risk_score": result.risk_score,
                    "confidence": result.confidence,
                    "matched_policy_clause_ids": list(
                        result.matched_policy_clause_ids
                    ),
                    "matched_entity_names": list(result.matched_entity_names),
                    "uncertainty_flags": list(result.uncertainty_flags),
                    "evidence_count": len(result.evidence),
                }
                for result in specialist_bundle.results
            ],
        }

    @staticmethod
    def _summarize_semantic(semantic_analysis: SemanticAnalysis) -> dict[str, Any]:
        return {
            "provider_name": semantic_analysis.provider_name,
            "model_name": semantic_analysis.model_name,
            "summary": semantic_analysis.summary,
            "overall_confidence": semantic_analysis.overall_confidence,
            "evidence_sufficiency": semantic_analysis.evidence_sufficiency,
            "rationale_quality": semantic_analysis.rationale_quality,
            "recommended_verdict": semantic_analysis.recommended_verdict.verdict.value,
            "recommended_verdict_confidence": (
                semantic_analysis.recommended_verdict.confidence
            ),
            "matched_policy_clause_ids": list(
                semantic_analysis.matched_policy_clause_ids
            ),
            "uncertainty_flags": list(semantic_analysis.uncertainty_flags),
            "has_any_evidence": semantic_analysis.has_any_evidence,
            "has_low_confidence_dimension": (
                semantic_analysis.has_low_confidence_dimension
            ),
            "dimension_scores": dict(semantic_analysis.dimension_scores),
            "dimension_confidences": dict(semantic_analysis.dimension_confidences),
            "metadata": dict(semantic_analysis.metadata),
        }

    @staticmethod
    def _summarize_routing(routing_result: RoutingResult) -> dict[str, Any]:
        return {
            "verdict": routing_result.verdict.value,
            "confidence": routing_result.confidence,
            "final_score": routing_result.final_score,
            "finding_count": len(routing_result.findings),
            "reason_count": len(routing_result.reasons),
            "uncertainty_flag_count": len(routing_result.uncertainty_flags),
            "scores": dict(routing_result.scores),
            "reasons": list(routing_result.reasons),
            "uncertainty_flags": list(routing_result.uncertainty_flags),
        }

    @staticmethod
    def _request_fingerprint(
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
        content_sha256: str,
    ) -> str:
        raw = "|".join(
            (
                str(request.request_id),
                request.action_type,
                request.channel,
                request.environment,
                request.recipient or "",
                policy.policy_id,
                policy.version,
                request.requested_at.isoformat(),
                content_sha256,
            )
        )
        return PolicyDecisionPoint._sha256_hex(raw)

    @staticmethod
    def _build_content_excerpt(content: str, limit: int = 400) -> str:
        normalized = " ".join(content.strip().split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 1].rstrip() + "…"

    @staticmethod
    def _sha256_hex(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_retrieval_context(
        retrieval_context: RetrievalContext,
    ) -> dict[str, object]:
        """
        Produces a compact retrieval summary for durable storage.

        Decision intentionally keeps retrieval_context generic for now. This
        preserves the contract without hard-coupling persistence to the full
        retrieval schema too early.
        """
        return {
            "is_empty": retrieval_context.is_empty,
            "policy_clause_ids": list(retrieval_context.matched_policy_clause_ids),
            "entity_names": list(retrieval_context.matched_entity_names),
            "precedent_decision_ids": [
                str(precedent.decision_id) for precedent in retrieval_context.precedents
            ],
            "retrieval_warnings": list(retrieval_context.retrieval_warnings),
            "policy_clause_count": len(retrieval_context.policy_clauses),
            "precedent_count": len(retrieval_context.precedents),
            "entity_count": len(retrieval_context.entities),
            "retrieved_at": retrieval_context.retrieved_at.isoformat(),
            "metadata": dict(retrieval_context.metadata),
        }

    @staticmethod
    def _stringify_optional_enum(value: object) -> str | None:
        if value is None:
            return None

        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, str):
            return enum_value

        if isinstance(value, str):
            return value

        return str(value)


def build_default_pdp() -> PolicyDecisionPoint:
    """Returns Tex's default local PDP stack."""
    return PolicyDecisionPoint()


PDPResult.model_rebuild()
