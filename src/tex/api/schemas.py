from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tex.commands.activate_policy import ActivatePolicyResult
from tex.commands.calibrate_policy import CalibratePolicyResult
from tex.commands.evaluate_action import EvaluateActionResult
from tex.commands.export_bundle import ExportBundleResult
from tex.commands.report_outcome import ReportOutcomeResult
from tex.domain.evaluation import EvaluationRequest, EvaluationResponse
from tex.domain.finding import Finding
from tex.domain.outcome import OutcomeKind, OutcomeLabel, OutcomeRecord
from tex.domain.policy import PolicySnapshot
from tex.domain.severity import Severity
from tex.domain.verdict import Verdict
from tex.learning.calibrator import CalibrationRecommendation
from tex.learning.outcomes import OutcomeClassification, OutcomeSummary


def _normalize_non_blank_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _normalize_string_sequence(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a sequence of strings, not a plain string")
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise TypeError(f"{field_name} must be a list, tuple, or set")

    normalized_items: list[str] = []
    seen: set[str] = set()

    for item in value:
        normalized = _normalize_non_blank_string(item, field_name=field_name)
        assert normalized is not None
        dedupe_key = normalized.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized_items.append(normalized)

    return tuple(normalized_items)


def _normalize_dict(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dictionary")
    return dict(value)


def _normalize_timezone_aware_datetime(
    value: datetime | None,
    *,
    field_name: str,
) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


class FindingDTO(BaseModel):
    """
    Public API representation of a single finding.

    The schema stays close to the domain model so the public surface remains
    transparent and audit-friendly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    rule_name: str
    severity: Severity
    message: str
    matched_text: str | None = None
    start_index: int | None = Field(default=None, ge=0)
    end_index: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, finding: Finding) -> "FindingDTO":
        return cls.model_validate(finding.model_dump())


class PolicySnapshotDTO(BaseModel):
    """
    Public API representation of one policy snapshot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    is_active: bool

    permit_threshold: float = Field(ge=0.0, le=1.0)
    forbid_threshold: float = Field(ge=0.0, le=1.0)
    minimum_confidence: float = Field(ge=0.0, le=1.0)

    deterministic_block_severities: tuple[Severity, ...]
    enabled_recognizers: tuple[str, ...]
    blocked_terms: tuple[str, ...]
    sensitive_entities: tuple[str, ...]

    retrieval_top_k: int = Field(ge=1)
    precedent_lookback_limit: int = Field(ge=1)

    specialist_thresholds: dict[str, float] = Field(default_factory=dict)
    action_criticality: dict[str, float] = Field(default_factory=dict)
    channel_criticality: dict[str, float] = Field(default_factory=dict)
    environment_criticality: dict[str, float] = Field(default_factory=dict)
    fusion_weights: dict[str, float] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime

    @classmethod
    def from_domain(cls, policy: PolicySnapshot) -> "PolicySnapshotDTO":
        return cls.model_validate(policy.model_dump())


class OutcomeRecordDTO(BaseModel):
    """
    Public API representation of a durable outcome record.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome_id: UUID
    decision_id: UUID
    request_id: UUID

    verdict: Verdict
    outcome_kind: OutcomeKind
    was_safe: bool | None = None
    human_override: bool = False

    summary: str | None = None
    reporter: str | None = None

    label: OutcomeLabel
    recorded_at: datetime

    @classmethod
    def from_domain(cls, outcome: OutcomeRecord) -> "OutcomeRecordDTO":
        return cls.model_validate(outcome.model_dump())


class OutcomeClassificationDTO(BaseModel):
    """
    Public API representation of one classified outcome judgment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str
    verdict: Verdict
    outcome_label: OutcomeLabel
    classification: str

    is_correct: bool
    is_error: bool
    is_false_permit: bool
    is_false_forbid: bool
    is_abstain_review: bool
    is_unknown: bool

    @classmethod
    def from_domain(
        cls,
        classification: OutcomeClassification,
    ) -> "OutcomeClassificationDTO":
        return cls(
            decision_id=classification.decision_id,
            verdict=classification.verdict,
            outcome_label=classification.outcome_label,
            classification=classification.classification,
            is_correct=classification.is_correct,
            is_error=classification.is_error,
            is_false_permit=classification.is_false_permit,
            is_false_forbid=classification.is_false_forbid,
            is_abstain_review=classification.is_abstain_review,
            is_unknown=classification.is_unknown,
        )

    def to_domain(self) -> OutcomeClassification:
        return OutcomeClassification(
            decision_id=self.decision_id,
            verdict=self.verdict,
            outcome_label=self.outcome_label,
            classification=self.classification,
            is_correct=self.is_correct,
            is_error=self.is_error,
            is_false_permit=self.is_false_permit,
            is_false_forbid=self.is_false_forbid,
            is_abstain_review=self.is_abstain_review,
            is_unknown=self.is_unknown,
        )


class OutcomeSummaryDTO(BaseModel):
    """
    Public API representation of aggregate outcome statistics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total: int = Field(ge=0)
    correct_permits: int = Field(ge=0)
    false_permits: int = Field(ge=0)
    correct_forbids: int = Field(ge=0)
    false_forbids: int = Field(ge=0)
    abstain_reviews: int = Field(ge=0)
    unknown: int = Field(ge=0)

    error_count: int = Field(ge=0)
    correctness_count: int = Field(ge=0)
    error_rate: float = Field(ge=0.0, le=1.0)

    @classmethod
    def from_domain(cls, summary: OutcomeSummary) -> "OutcomeSummaryDTO":
        return cls(
            total=summary.total,
            correct_permits=summary.correct_permits,
            false_permits=summary.false_permits,
            correct_forbids=summary.correct_forbids,
            false_forbids=summary.false_forbids,
            abstain_reviews=summary.abstain_reviews,
            unknown=summary.unknown,
            error_count=summary.error_count,
            correctness_count=summary.correctness_count,
            error_rate=summary.error_rate,
        )


class CalibrationRecommendationDTO(BaseModel):
    """
    Public API representation of a calibrator recommendation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    current_permit_threshold: float = Field(ge=0.0, le=1.0)
    recommended_permit_threshold: float = Field(ge=0.0, le=1.0)

    current_forbid_threshold: float = Field(ge=0.0, le=1.0)
    recommended_forbid_threshold: float = Field(ge=0.0, le=1.0)

    current_minimum_confidence: float = Field(ge=0.0, le=1.0)
    recommended_minimum_confidence: float = Field(ge=0.0, le=1.0)

    summary: OutcomeSummaryDTO
    reasons: tuple[str, ...] = Field(default_factory=tuple)

    false_permit_rate: float = Field(ge=0.0, le=1.0)
    false_forbid_rate: float = Field(ge=0.0, le=1.0)
    abstain_review_rate: float = Field(ge=0.0, le=1.0)
    unknown_rate: float = Field(ge=0.0, le=1.0)

    sample_weight: float = Field(ge=0.0, le=1.0)
    permit_threshold_delta: float
    forbid_threshold_delta: float
    minimum_confidence_delta: float

    changed: bool

    @field_validator("reasons", mode="before")
    @classmethod
    def normalize_reasons(cls, value: Any) -> tuple[str, ...]:
        return _normalize_string_sequence(value, field_name="reasons")

    @classmethod
    def from_domain(
        cls,
        recommendation: CalibrationRecommendation,
    ) -> "CalibrationRecommendationDTO":
        return cls(
            current_permit_threshold=recommendation.current_permit_threshold,
            recommended_permit_threshold=recommendation.recommended_permit_threshold,
            current_forbid_threshold=recommendation.current_forbid_threshold,
            recommended_forbid_threshold=recommendation.recommended_forbid_threshold,
            current_minimum_confidence=recommendation.current_minimum_confidence,
            recommended_minimum_confidence=recommendation.recommended_minimum_confidence,
            summary=OutcomeSummaryDTO.from_domain(recommendation.summary),
            reasons=recommendation.reasons,
            false_permit_rate=recommendation.false_permit_rate,
            false_forbid_rate=recommendation.false_forbid_rate,
            abstain_review_rate=recommendation.abstain_review_rate,
            unknown_rate=recommendation.unknown_rate,
            sample_weight=recommendation.sample_weight,
            permit_threshold_delta=recommendation.permit_threshold_delta,
            forbid_threshold_delta=recommendation.forbid_threshold_delta,
            minimum_confidence_delta=recommendation.minimum_confidence_delta,
            changed=recommendation.changed,
        )


class EvaluateRequestDTO(BaseModel):
    """
    Public inbound request for evaluating one action through Tex.

    request_id is required at the API boundary so the caller, audit trail, and
    internal decision pipeline all refer to the same concrete evaluation event.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: UUID
    action_type: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=50_000)

    recipient: str | None = Field(default=None, max_length=500)
    channel: str = Field(min_length=1, max_length=50)
    environment: str = Field(min_length=1, max_length=50)

    metadata: dict[str, Any] = Field(default_factory=dict)
    policy_id: str | None = Field(default=None, max_length=100)
    requested_at: datetime | None = None

    @field_validator("requested_at", mode="after")
    @classmethod
    def normalize_requested_at(cls, value: datetime | None) -> datetime | None:
        return _normalize_timezone_aware_datetime(value, field_name="requested_at")

    @field_validator("metadata", mode="before")
    @classmethod
    def normalize_metadata(cls, value: Any) -> dict[str, Any]:
        return _normalize_dict(value, field_name="metadata")

    def to_domain(self) -> EvaluationRequest:
        payload: dict[str, Any] = {
            "request_id": self.request_id,
            "action_type": self.action_type,
            "content": self.content,
            "recipient": self.recipient,
            "channel": self.channel,
            "environment": self.environment,
            "metadata": dict(self.metadata),
            "policy_id": self.policy_id,
        }
        if self.requested_at is not None:
            payload["requested_at"] = self.requested_at
        return EvaluationRequest(**payload)


class DeterministicLayerDTO(BaseModel):
    """Frontend-facing deterministic gate breakdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    blocked: bool
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    findings: tuple[FindingDTO, ...] = Field(default_factory=tuple)


class RetrievalClauseDTO(BaseModel):
    """Frontend-facing policy clause summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    clause_id: str
    title: str
    text: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class RetrievalLayerDTO(BaseModel):
    """Frontend-facing retrieval grounding breakdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_empty: bool
    clauses: tuple[RetrievalClauseDTO, ...] = Field(default_factory=tuple)
    entities: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)


class SpecialistEvidenceDTO(BaseModel):
    """Frontend-facing specialist evidence item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    keyword: str = ""
    text: str = ""
    explanation: str = ""


class SpecialistResultDTO(BaseModel):
    """Frontend-facing single specialist judge result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_name: str
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    evidence: tuple[SpecialistEvidenceDTO, ...] = Field(default_factory=tuple)


class SpecialistsLayerDTO(BaseModel):
    """Frontend-facing specialist judges breakdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    specialists: tuple[SpecialistResultDTO, ...] = Field(default_factory=tuple)


class SemanticDimensionDTO(BaseModel):
    """Frontend-facing single semantic dimension."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_spans: tuple[dict[str, str], ...] = Field(default_factory=tuple)


class SemanticLayerDTO(BaseModel):
    """Frontend-facing semantic analysis breakdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimensions: dict[str, SemanticDimensionDTO] = Field(default_factory=dict)
    recommended_verdict: str = "ABSTAIN"
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class RouterLayerDTO(BaseModel):
    """Frontend-facing router / fusion breakdown."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    final_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    verdict: str
    evidence_sufficiency: float = Field(ge=0.0, le=1.0, default=1.0)
    layer_scores: dict[str, float] = Field(default_factory=dict)
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    uncertainty_flags: tuple[str, ...] = Field(default_factory=tuple)


class EvidenceLayerDTO(BaseModel):
    """Frontend-facing evidence chain summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_hash: str = ""
    chain_valid: bool = True
    record_count: int = Field(ge=0, default=0)


class EvaluateResponseDTO(BaseModel):
    """
    Public outbound response for a completed Tex evaluation.

    Includes both the top-level verdict and per-layer breakdowns consumed
    by the frontend decision panel.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: UUID
    request_id: UUID | None = None
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)

    reasons: tuple[str, ...] = Field(default_factory=tuple)
    findings: tuple[FindingDTO, ...] = Field(default_factory=tuple)
    scores: dict[str, float] = Field(default_factory=dict)
    uncertainty_flags: tuple[str, ...] = Field(default_factory=tuple)

    policy_version: str
    evidence_hash: str | None = None
    evaluated_at: datetime

    deterministic: DeterministicLayerDTO | None = None
    retrieval: RetrievalLayerDTO | None = None
    specialists: SpecialistsLayerDTO | None = None
    semantic: SemanticLayerDTO | None = None
    router: RouterLayerDTO | None = None
    evidence: EvidenceLayerDTO | None = None

    @field_validator("reasons", "uncertainty_flags", mode="before")
    @classmethod
    def normalize_string_collections(cls, value: Any, info: Any) -> tuple[str, ...]:
        return _normalize_string_sequence(value, field_name=info.field_name)

    @field_validator("scores", mode="before")
    @classmethod
    def normalize_scores(cls, value: Any) -> dict[str, float]:
        return _normalize_dict(value, field_name="scores")

    @classmethod
    def from_domain(
        cls,
        response: EvaluationResponse,
    ) -> "EvaluateResponseDTO":
        return cls(
            decision_id=response.decision_id,
            verdict=response.verdict,
            confidence=response.confidence,
            final_score=response.final_score,
            reasons=tuple(response.reasons),
            findings=tuple(FindingDTO.from_domain(item) for item in response.findings),
            scores=dict(response.scores),
            uncertainty_flags=tuple(response.uncertainty_flags),
            policy_version=response.policy_version,
            evidence_hash=response.evidence_hash,
            evaluated_at=response.evaluated_at,
        )

    @classmethod
    def from_command_result(
        cls,
        result: EvaluateActionResult,
    ) -> "EvaluateResponseDTO":
        """
        Build the full response including per-layer breakdowns from the
        internal PDPResult, so the frontend can render the pipeline view.
        """
        response = result.response
        pdp = result.pdp_result

        det_result = pdp.deterministic_result
        finding_count = len(det_result.findings)
        det_score = min(1.0, finding_count * 0.15) if finding_count > 0 else 0.0
        det_confidence = 0.95 if finding_count > 0 else 0.85

        deterministic_dto = DeterministicLayerDTO(
            blocked=det_result.blocked,
            score=det_score,
            confidence=det_confidence,
            findings=tuple(FindingDTO.from_domain(f) for f in det_result.findings),
        )

        ret_ctx = pdp.retrieval_context
        retrieval_dto = RetrievalLayerDTO(
            is_empty=ret_ctx.is_empty,
            clauses=tuple(
                RetrievalClauseDTO(
                    clause_id=clause.clause_id,
                    title=clause.title or "Matched Policy Clause",
                    text=clause.text,
                    relevance_score=clause.relevance_score,
                )
                for clause in ret_ctx.policy_clauses
            ),
            entities=tuple(ret_ctx.matched_entity_names),
            warnings=tuple(ret_ctx.retrieval_warnings),
        )

        spec_bundle = pdp.specialist_bundle
        specialists_dto = SpecialistsLayerDTO(
            specialists=tuple(
                SpecialistResultDTO(
                    specialist_name=sr.specialist_name,
                    risk_score=sr.risk_score,
                    confidence=sr.confidence,
                    summary=sr.summary,
                    evidence=tuple(
                        SpecialistEvidenceDTO(
                            keyword=ev.text[:40] if ev.text else "signal",
                            text=ev.text,
                            explanation=ev.explanation or "Matched specialist signal.",
                        )
                        for ev in sr.evidence
                    ),
                )
                for sr in spec_bundle.results
            ),
        )

        sem = pdp.semantic_analysis
        semantic_dto = SemanticLayerDTO(
            dimensions={
                dr.dimension: SemanticDimensionDTO(
                    score=dr.score,
                    confidence=dr.confidence,
                    evidence_spans=tuple(
                        {"text": span.text}
                        for span in dr.evidence_spans
                    ),
                )
                for dr in sem.dimension_results
            },
            recommended_verdict=sem.recommended_verdict.verdict.value,
            overall_confidence=sem.overall_confidence,
        )

        routing = pdp.routing_result
        router_dto = RouterLayerDTO(
            final_score=routing.final_score,
            confidence=routing.confidence,
            verdict=routing.verdict.value,
            evidence_sufficiency=1.0,
            layer_scores=dict(routing.scores),
            reasons=tuple(routing.reasons),
            uncertainty_flags=tuple(routing.uncertainty_flags),
        )

        total_evidence = (
            len(det_result.findings)
            + len(ret_ctx.policy_clauses)
            + sum(len(sr.evidence) for sr in spec_bundle.results)
            + sum(len(dr.evidence_spans) for dr in sem.dimension_results)
            + 1
        )
        evidence_dto = EvidenceLayerDTO(
            evidence_hash=response.evidence_hash or "",
            chain_valid=True,
            record_count=total_evidence,
        )

        return cls(
            decision_id=response.decision_id,
            request_id=pdp.request.request_id,
            verdict=response.verdict,
            confidence=response.confidence,
            final_score=response.final_score,
            reasons=tuple(response.reasons),
            findings=tuple(FindingDTO.from_domain(item) for item in response.findings),
            scores=dict(response.scores),
            uncertainty_flags=tuple(response.uncertainty_flags),
            policy_version=response.policy_version,
            evidence_hash=response.evidence_hash,
            evaluated_at=response.evaluated_at,
            deterministic=deterministic_dto,
            retrieval=retrieval_dto,
            specialists=specialists_dto,
            semantic=semantic_dto,
            router=router_dto,
            evidence=evidence_dto,
        )


class ReportOutcomeRequestDTO(BaseModel):
    """
    Public inbound request for recording what happened after a Tex decision.

    The API computes the calibration label automatically instead of trusting
    clients to provide one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: UUID
    request_id: UUID
    verdict: Verdict
    outcome_kind: OutcomeKind

    was_safe: bool | None = None
    human_override: bool = False

    summary: str | None = Field(default=None, max_length=2_000)
    reporter: str | None = Field(default=None, max_length=200)
    recorded_at: datetime | None = None

    @field_validator("recorded_at", mode="after")
    @classmethod
    def normalize_recorded_at(cls, value: datetime | None) -> datetime | None:
        return _normalize_timezone_aware_datetime(value, field_name="recorded_at")

    def to_domain(self) -> OutcomeRecord:
        outcome = OutcomeRecord.create(
            decision_id=self.decision_id,
            request_id=self.request_id,
            verdict=self.verdict,
            outcome_kind=self.outcome_kind,
            was_safe=self.was_safe,
            human_override=self.human_override,
            summary=self.summary,
            reporter=self.reporter,
        )

        if self.recorded_at is not None:
            outcome = outcome.model_copy(update={"recorded_at": self.recorded_at})

        return outcome


class ReportOutcomeResponseDTO(BaseModel):
    """
    Public outbound response for one recorded outcome.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: OutcomeRecordDTO
    classification: OutcomeClassificationDTO
    evidence_recorded: bool

    @classmethod
    def from_command_result(
        cls,
        result: ReportOutcomeResult,
    ) -> "ReportOutcomeResponseDTO":
        return cls(
            outcome=OutcomeRecordDTO.from_domain(result.outcome),
            classification=OutcomeClassificationDTO.from_domain(result.classification),
            evidence_recorded=result.evidence_record is not None,
        )


class ActivatePolicyRequestDTO(BaseModel):
    """
    Public inbound request for activating an existing policy version.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(min_length=1, max_length=100)

    @field_validator("version", mode="before")
    @classmethod
    def normalize_version(cls, value: Any) -> str:
        normalized = _normalize_non_blank_string(value, field_name="version")
        assert normalized is not None
        return normalized


class ActivatePolicyResponseDTO(BaseModel):
    """
    Public outbound response for policy activation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    activated_policy: PolicySnapshotDTO
    previous_active_policy: PolicySnapshotDTO | None = None

    @classmethod
    def from_command_result(
        cls,
        result: ActivatePolicyResult,
    ) -> "ActivatePolicyResponseDTO":
        return cls(
            activated_policy=PolicySnapshotDTO.from_domain(result.activated_policy),
            previous_active_policy=(
                None
                if result.previous_active_policy is None
                else PolicySnapshotDTO.from_domain(result.previous_active_policy)
            ),
        )


class CalibratePolicyRequestDTO(BaseModel):
    """
    Public inbound request for a calibration pass.

    This keeps calibration explicit: the caller supplies already-classified
    outcome judgments instead of hiding classification discovery inside the API.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_policy_version: str | None = Field(default=None, max_length=100)
    classifications: tuple[OutcomeClassificationDTO, ...] = Field(default_factory=tuple)

    new_version: str | None = Field(default=None, max_length=100)
    save: bool = False
    activate: bool = False

    metadata_updates: dict[str, object] = Field(default_factory=dict)

    @field_validator("source_policy_version", "new_version", mode="before")
    @classmethod
    def normalize_optional_versions(cls, value: Any, info: Any) -> str | None:
        return _normalize_non_blank_string(value, field_name=info.field_name)

    @field_validator("metadata_updates", mode="before")
    @classmethod
    def normalize_metadata_updates(cls, value: Any) -> dict[str, Any]:
        return _normalize_dict(value, field_name="metadata_updates")

    def to_domain_classifications(self) -> tuple[OutcomeClassification, ...]:
        return tuple(item.to_domain() for item in self.classifications)


class CalibratePolicyResponseDTO(BaseModel):
    """
    Public outbound response for a calibration pass.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_policy: PolicySnapshotDTO
    recommendation: CalibrationRecommendationDTO
    summary: OutcomeSummaryDTO
    classifications: tuple[OutcomeClassificationDTO, ...] = Field(default_factory=tuple)
    calibrated_policy: PolicySnapshotDTO | None = None

    @classmethod
    def from_command_result(
        cls,
        result: CalibratePolicyResult,
    ) -> "CalibratePolicyResponseDTO":
        return cls(
            source_policy=PolicySnapshotDTO.from_domain(result.source_policy),
            recommendation=CalibrationRecommendationDTO.from_domain(result.recommendation),
            summary=OutcomeSummaryDTO.from_domain(result.summary),
            classifications=tuple(
                OutcomeClassificationDTO.from_domain(item)
                for item in result.classifications
            ),
            calibrated_policy=(
                None
                if result.calibrated_policy is None
                else PolicySnapshotDTO.from_domain(result.calibrated_policy)
            ),
        )


class ExportBundleRequestDTO(BaseModel):
    """
    Public inbound request for exporting evidence.

    The route can decide which exporter method to call based on export_format
    and the optional filter fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1, max_length=2_000)
    export_format: str = Field(default="json", min_length=1, max_length=20)

    export_name: str = Field(
        default="tex-evidence-bundle",
        min_length=1,
        max_length=200,
    )
    verify_chain: bool = True
    indent: int = Field(default=2, ge=0, le=8)

    record_type: str | None = Field(default=None, max_length=100)
    decision_id: str | None = Field(default=None, max_length=100)
    outcome_id: str | None = Field(default=None, max_length=100)

    @field_validator(
        "path",
        "export_name",
        "record_type",
        "decision_id",
        "outcome_id",
        mode="before",
    )
    @classmethod
    def normalize_optional_string_fields(cls, value: Any, info: Any) -> str | None:
        return _normalize_non_blank_string(value, field_name=info.field_name)

    @field_validator("export_format", mode="before")
    @classmethod
    def normalize_export_format(cls, value: Any) -> str:
        normalized = _normalize_non_blank_string(value, field_name="export_format")
        assert normalized is not None
        normalized = normalized.lower()
        if normalized not in {"json", "jsonl"}:
            raise ValueError("export_format must be 'json' or 'jsonl'")
        return normalized


class ExportBundleResponseDTO(BaseModel):
    """
    Public outbound response for evidence export.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    output_path: str
    export_format: str
    bundle_included: bool

    @classmethod
    def from_command_result(
        cls,
        result: ExportBundleResult,
    ) -> "ExportBundleResponseDTO":
        return cls(
            output_path=str(Path(result.output_path)),
            export_format=result.export_format,
            bundle_included=result.bundle is not None,
        )