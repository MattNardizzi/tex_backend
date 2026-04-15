from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tex.deterministic.gate import DeterministicGateResult
from tex.domain.finding import Finding
from tex.domain.policy import PolicySnapshot
from tex.domain.verdict import Verdict
from tex.semantic.schema import SemanticAnalysis
from tex.specialists.base import SpecialistBundle


class RoutingResult(BaseModel):
    """
    Structured fusion and routing result for Tex's decision engine.

    This is the output of the routing layer before the final durable decision
    record is created by the PDP.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)

    reasons: tuple[str, ...] = Field(default_factory=tuple)
    uncertainty_flags: tuple[str, ...] = Field(default_factory=tuple)
    findings: tuple[Finding, ...] = Field(default_factory=tuple)

    scores: dict[str, float] = Field(default_factory=dict)

    @field_validator("reasons", "uncertainty_flags", mode="before")
    @classmethod
    def normalize_string_sequences(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return tuple()
        if isinstance(value, str):
            raise TypeError("sequence fields must not be plain strings")
        if not isinstance(value, (list, tuple)):
            raise TypeError("sequence fields must be lists or tuples")

        normalized: list[str] = []
        seen: set[str] = set()

        for item in value:
            if not isinstance(item, str):
                raise TypeError("sequence items must be strings")
            candidate = item.strip()
            if not candidate:
                raise ValueError("sequence items must not be blank")
            dedupe_key = candidate.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(candidate)

        return tuple(normalized)

    @model_validator(mode="after")
    def validate_scores(self) -> "RoutingResult":
        for key, value in self.scores.items():
            if not isinstance(key, str):
                raise TypeError("score keys must be strings")
            if not 0.0 <= value <= 1.0:
                raise ValueError("score values must be between 0.0 and 1.0")
        return self


class DecisionRouter:
    """
    Fuses Tex's upstream signals into a final routed verdict.

    Evaluation order is preserved conceptually:
    - deterministic output is respected first
    - specialist and semantic signals are fused
    - policy criticality is added
    - abstention is treated as first-class, not an afterthought
    """

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
        criticality_score = policy.criticality_for(
            action_type=action_type,
            channel=channel,
            environment=environment,
        )

        deterministic_score = self._deterministic_score(deterministic_result)
        specialist_score = specialist_bundle.max_risk_score
        semantic_score = semantic_analysis.max_dimension_score

        final_score = self._fuse_scores(
            deterministic_score=deterministic_score,
            specialist_score=specialist_score,
            semantic_score=semantic_score,
            criticality_score=criticality_score,
            policy=policy,
        )

        confidence = self._compute_confidence(
            deterministic_result=deterministic_result,
            specialist_bundle=specialist_bundle,
            semantic_analysis=semantic_analysis,
        )

        reasons = self._build_reasons(
            deterministic_result=deterministic_result,
            specialist_bundle=specialist_bundle,
            semantic_analysis=semantic_analysis,
            final_score=final_score,
            policy=policy,
        )

        uncertainty_flags = self._build_uncertainty_flags(
            deterministic_result=deterministic_result,
            specialist_bundle=specialist_bundle,
            semantic_analysis=semantic_analysis,
            confidence=confidence,
            policy=policy,
            final_score=final_score,
        )

        verdict = self._determine_verdict(
            deterministic_result=deterministic_result,
            semantic_analysis=semantic_analysis,
            specialist_bundle=specialist_bundle,
            final_score=final_score,
            confidence=confidence,
            policy=policy,
            uncertainty_flags=uncertainty_flags,
        )

        return RoutingResult(
            verdict=verdict,
            confidence=round(confidence, 4),
            final_score=round(final_score, 4),
            reasons=reasons,
            uncertainty_flags=uncertainty_flags,
            findings=deterministic_result.findings,
            scores={
                "deterministic": round(deterministic_score, 4),
                "specialists": round(specialist_score, 4),
                "semantic": round(semantic_score, 4),
                "criticality": round(criticality_score, 4),
            },
        )

    def _deterministic_score(self, deterministic_result: DeterministicGateResult) -> float:
        if deterministic_result.blocked:
            return 1.0

        if not deterministic_result.findings:
            return 0.0

        severity_scores = {
            "CRITICAL": 1.0,
            "WARNING": 0.55,
            "INFO": 0.20,
        }

        highest = 0.0
        for finding in deterministic_result.findings:
            highest = max(highest, severity_scores.get(finding.severity.value, 0.0))

        return min(1.0, highest)

    def _fuse_scores(
        self,
        *,
        deterministic_score: float,
        specialist_score: float,
        semantic_score: float,
        criticality_score: float,
        policy: PolicySnapshot,
    ) -> float:
        weights = policy.fusion_weights

        fused = (
            deterministic_score * weights["deterministic"]
            + specialist_score * weights["specialists"]
            + semantic_score * weights["semantic"]
            + criticality_score * weights["criticality"]
        )

        return min(1.0, max(0.0, fused))

    def _compute_confidence(
        self,
        *,
        deterministic_result: DeterministicGateResult,
        specialist_bundle: SpecialistBundle,
        semantic_analysis: SemanticAnalysis,
    ) -> float:
        deterministic_confidence = 0.95 if deterministic_result.blocked else (
            0.75 if deterministic_result.findings else 0.85
        )

        if specialist_bundle.is_empty:
            specialist_confidence = 0.0
        else:
            specialist_confidence = sum(
                result.confidence for result in specialist_bundle.results
            ) / len(specialist_bundle.results)

        semantic_confidence = semantic_analysis.overall_confidence

        base = (
            deterministic_confidence * 0.25
            + specialist_confidence * 0.20
            + semantic_confidence * 0.55
        )

        if semantic_analysis.has_low_confidence_dimension:
            base -= 0.08

        if semantic_analysis.evidence_sufficiency < 0.30:
            base -= 0.05

        return min(1.0, max(0.0, base))

    def _determine_verdict(
        self,
        *,
        deterministic_result: DeterministicGateResult,
        semantic_analysis: SemanticAnalysis,
        specialist_bundle: SpecialistBundle,
        final_score: float,
        confidence: float,
        policy: PolicySnapshot,
        uncertainty_flags: tuple[str, ...],
    ) -> Verdict:
        if deterministic_result.blocked:
            return Verdict.FORBID

        if semantic_analysis.recommended_verdict.verdict == Verdict.FORBID:
            if final_score >= max(policy.permit_threshold, 0.45):
                return Verdict.FORBID

        if final_score >= policy.forbid_threshold:
            return Verdict.FORBID

        if self._should_abstain(
            semantic_analysis=semantic_analysis,
            specialist_bundle=specialist_bundle,
            final_score=final_score,
            confidence=confidence,
            policy=policy,
            uncertainty_flags=uncertainty_flags,
        ):
            return Verdict.ABSTAIN

        if (
            final_score <= policy.permit_threshold
            and confidence >= policy.minimum_confidence
            and semantic_analysis.recommended_verdict.verdict == Verdict.PERMIT
        ):
            return Verdict.PERMIT

        return Verdict.ABSTAIN

    def _should_abstain(
        self,
        *,
        semantic_analysis: SemanticAnalysis,
        specialist_bundle: SpecialistBundle,
        final_score: float,
        confidence: float,
        policy: PolicySnapshot,
        uncertainty_flags: tuple[str, ...],
    ) -> bool:
        if semantic_analysis.recommended_verdict.verdict == Verdict.ABSTAIN:
            return True

        if confidence < policy.minimum_confidence:
            return True

        if semantic_analysis.has_low_confidence_dimension:
            return True

        if semantic_analysis.evidence_sufficiency < 0.25 and final_score >= policy.permit_threshold:
            return True

        if specialist_bundle.max_risk_score >= 0.60 and final_score < policy.forbid_threshold:
            return True

        if any(flag.casefold() == "no_retrieval_context" for flag in uncertainty_flags):
            if final_score >= policy.permit_threshold:
                return True

        if policy.permit_threshold < final_score < policy.forbid_threshold:
            return True

        return False

    def _build_reasons(
        self,
        *,
        deterministic_result: DeterministicGateResult,
        specialist_bundle: SpecialistBundle,
        semantic_analysis: SemanticAnalysis,
        final_score: float,
        policy: PolicySnapshot,
    ) -> tuple[str, ...]:
        reasons: list[str] = []

        if deterministic_result.blocked:
            reasons.extend(deterministic_result.blocking_reasons)

        if deterministic_result.findings and not deterministic_result.blocked:
            reasons.append(
                f"Deterministic layer produced {len(deterministic_result.findings)} finding(s)."
            )

        if not specialist_bundle.is_empty:
            highest_specialist = max(
                specialist_bundle.results,
                key=lambda result: result.risk_score,
            )
            reasons.append(
                f"Highest specialist risk came from {highest_specialist.specialist_name} "
                f"({highest_specialist.risk_score:.2f})."
            )

        reasons.append(
            f"Semantic layer recommended {semantic_analysis.recommended_verdict.verdict.value} "
            f"with confidence {semantic_analysis.recommended_verdict.confidence:.2f}."
        )
        reasons.append(
            f"Fused final score was {final_score:.2f} "
            f"(permit <= {policy.permit_threshold:.2f}, forbid >= {policy.forbid_threshold:.2f})."
        )

        if semantic_analysis.matched_policy_clause_ids:
            reasons.append(
                f"Matched {len(semantic_analysis.matched_policy_clause_ids)} policy clause(s) in semantic analysis."
            )

        return tuple(reasons)

    def _build_uncertainty_flags(
        self,
        *,
        deterministic_result: DeterministicGateResult,
        specialist_bundle: SpecialistBundle,
        semantic_analysis: SemanticAnalysis,
        confidence: float,
        policy: PolicySnapshot,
        final_score: float,
    ) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()

        def add(flag: str) -> None:
            key = flag.casefold()
            if key in seen:
                return
            seen.add(key)
            ordered.append(flag)

        for flag in semantic_analysis.uncertainty_flags:
            add(flag)

        for flag in semantic_analysis.recommended_verdict.uncertainty_flags:
            add(flag)

        for flag in specialist_bundle.uncertainty_flags:
            add(flag)

        if deterministic_result.findings and not deterministic_result.blocked:
            add("deterministic_findings_present")

        if confidence < policy.minimum_confidence:
            add("confidence_below_policy_minimum")

        if semantic_analysis.has_low_confidence_dimension:
            add("low_confidence_semantic_dimension")

        if semantic_analysis.evidence_sufficiency < 0.25:
            add("weak_semantic_evidence")

        if policy.permit_threshold < final_score < policy.forbid_threshold:
            add("borderline_fused_score")

        return tuple(ordered)


def build_default_router() -> DecisionRouter:
    """Convenience constructor for the default decision router."""
    return DecisionRouter()