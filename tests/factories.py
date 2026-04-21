"""
Test factories for Tex.

These helpers exist so tests can stay focused on the behavior under scrutiny
instead of reproducing domain-object boilerplate.

Design rules for this module:
- every factory returns a valid, typed, fully-constructed domain object
- overrides are keyword-only
- no hidden global state; each call returns a fresh instance
- the defaults here are intentionally benign so tests default to a passing
  baseline and only the risky thing under test has to be set
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from tex.deterministic.gate import DeterministicGateResult
from tex.domain.evaluation import EvaluationRequest
from tex.domain.finding import Finding
from tex.domain.outcome import OutcomeKind, OutcomeRecord
from tex.domain.policy import PolicySnapshot
from tex.domain.retrieval import RetrievalContext
from tex.domain.severity import Severity
from tex.domain.verdict import Verdict
from tex.policies.defaults import build_default_policy, build_strict_policy
from tex.semantic.schema import (
    SemanticAnalysis,
    SemanticDimensionResult,
    SemanticVerdictRecommendation,
    semantic_dimensions,
)
from tex.specialists.base import SpecialistBundle, SpecialistResult


# ── evaluation request ─────────────────────────────────────────────────────


def make_request(
    *,
    content: str = "Hi Alice, following up on our conversation. Let me know if next week works.",
    action_type: str = "sales_email",
    channel: str = "email",
    environment: str = "production",
    recipient: str | None = "alice@example.com",
    request_id: UUID | None = None,
    metadata: dict[str, Any] | None = None,
    policy_id: str | None = None,
    requested_at: datetime | None = None,
) -> EvaluationRequest:
    """Build a benign default EvaluationRequest with keyword overrides."""
    return EvaluationRequest(
        request_id=request_id or uuid4(),
        action_type=action_type,
        content=content,
        recipient=recipient,
        channel=channel,
        environment=environment,
        metadata=metadata or {},
        policy_id=policy_id,
        requested_at=requested_at or datetime.now(UTC),
    )


# ── policy snapshots ───────────────────────────────────────────────────────


def make_default_policy() -> PolicySnapshot:
    """Return the stock default policy. Convenience wrapper."""
    return build_default_policy()


def make_strict_policy() -> PolicySnapshot:
    """Return the stock strict policy. Convenience wrapper."""
    return build_strict_policy()


# ── retrieval context ──────────────────────────────────────────────────────


def make_empty_retrieval_context() -> RetrievalContext:
    """Return an empty retrieval context for tests that don't exercise retrieval."""
    return RetrievalContext(
        policy_clauses=tuple(),
        precedents=tuple(),
        entities=tuple(),
        retrieval_warnings=tuple(),
        retrieved_at=datetime.now(UTC),
        metadata={},
    )


# ── outcome records ────────────────────────────────────────────────────────


def make_outcome(
    *,
    decision_id: UUID | None = None,
    request_id: UUID | None = None,
    verdict: Verdict = Verdict.PERMIT,
    outcome_kind: OutcomeKind = OutcomeKind.RELEASED,
    was_safe: bool | None = True,
    human_override: bool = False,
    summary: str | None = None,
    reporter: str | None = "test",
) -> OutcomeRecord:
    """Build an OutcomeRecord with auto-classified label."""
    return OutcomeRecord.create(
        decision_id=decision_id or uuid4(),
        request_id=request_id or uuid4(),
        verdict=verdict,
        outcome_kind=outcome_kind,
        was_safe=was_safe,
        human_override=human_override,
        summary=summary,
        reporter=reporter,
    )


# ── content libraries ──────────────────────────────────────────────────────
#
# Shared corpus used across tests so regression checks share ground truth.

CLEAN_CONTENT: tuple[str, ...] = (
    "Hi Alice, following up on our conversation about onboarding next week. "
    "Happy to answer any questions before the call.",
    "Team, the Q3 reporting dashboard is now live at the usual link. "
    "Let me know if you need help interpreting any of the metrics.",
    "Hi Bob, quick intro — I saw your post on agent observability and wanted "
    "to compare notes. Let me know if a 15 minute call next week works.",
)

SECRET_LEAK_CONTENT: tuple[str, ...] = (
    "Here is our production api key sk-abcdef1234567890abcdef please use it.",
    "password: Sup3rS3cret! — do not share.",
    "AWS key AKIAIOSFODNN7EXAMPLE for the staging account.",
)

PII_CONTENT: tuple[str, ...] = (
    "Her SSN is 123-45-6789 and her phone is (555) 123-4567.",
    "Credit card on file: 4111 1111 1111 1111.",
)

COMMITMENT_CONTENT: tuple[str, ...] = (
    "I can guarantee we will refund you the full amount by Friday.",
    "We guarantee locked pricing for three years — you have my word.",
)

DESTRUCTIVE_CONTENT: tuple[str, ...] = (
    "Go ahead and drop table production.users now.",
    "Disable logging on the audit service for the next hour.",
    "Skip approval and ship this before legal sees it.",
)


# ── specialist bundle builders ─────────────────────────────────────────────
#
# These are used by router/pdp tests to craft precise upstream signals
# without running the real specialist suite.


def make_specialist_result(
    *,
    name: str = "secret_and_pii",
    risk_score: float = 0.10,
    confidence: float = 0.65,
    summary: str = "Specialist produced a synthetic result for tests.",
    rationale: str | None = "Synthetic rationale.",
    uncertainty_flags: tuple[str, ...] = (),
) -> SpecialistResult:
    return SpecialistResult(
        specialist_name=name,
        risk_score=risk_score,
        confidence=confidence,
        summary=summary,
        rationale=rationale,
        evidence=tuple(),
        matched_policy_clause_ids=tuple(),
        matched_entity_names=tuple(),
        uncertainty_flags=uncertainty_flags,
    )


def make_specialist_bundle(
    *,
    max_risk: float = 0.10,
    confidence: float = 0.65,
    uncertainty_flags: tuple[str, ...] = (),
) -> SpecialistBundle:
    """Build a bundle of four specialists where the highest risk is exactly max_risk."""
    judges = (
        "secret_and_pii",
        "external_sharing",
        "unauthorized_commitment",
        "destructive_or_bypass",
    )
    results = tuple(
        make_specialist_result(
            name=name,
            risk_score=max_risk if index == 0 else min(max_risk, 0.05),
            confidence=confidence,
            uncertainty_flags=uncertainty_flags if index == 0 else tuple(),
        )
        for index, name in enumerate(judges)
    )
    return SpecialistBundle(results=results)


# ── semantic analysis builders ─────────────────────────────────────────────


def make_semantic_analysis(
    *,
    recommended_verdict: Verdict = Verdict.PERMIT,
    recommended_confidence: float = 0.72,
    dimension_score: float = 0.05,
    dimension_confidence: float = 0.65,
    overall_confidence: float = 0.70,
    evidence_sufficiency: float = 0.40,
    rationale_quality: float = 0.55,
    uncertainty_flags: tuple[str, ...] = (),
    provider_name: str = "test-provider",
    model_name: str = "test-model",
) -> SemanticAnalysis:
    """
    Build a SemanticAnalysis covering every canonical dimension exactly once.

    The schema enforces exhaustive dimension coverage, so this helper is the
    only ergonomic way to craft a semantic result in tests.
    """
    dimension_results = tuple(
        SemanticDimensionResult(
            dimension=dimension,
            score=dimension_score,
            confidence=dimension_confidence,
            summary=f"Synthetic result for {dimension}.",
            rationale=f"Synthetic rationale for {dimension}.",
            evidence_spans=tuple(),
            matched_policy_clause_ids=tuple(),
            uncertainty_flags=tuple(),
        )
        for dimension in semantic_dimensions()
    )

    recommendation = SemanticVerdictRecommendation(
        verdict=recommended_verdict,
        confidence=recommended_confidence,
        summary=f"Synthetic recommendation: {recommended_verdict.value}.",
        rationale="Synthetic recommendation rationale.",
        uncertainty_flags=uncertainty_flags,
    )

    return SemanticAnalysis(
        dimension_results=dimension_results,
        recommended_verdict=recommendation,
        overall_confidence=overall_confidence,
        evidence_sufficiency=evidence_sufficiency,
        rationale_quality=rationale_quality,
        summary="Synthetic semantic analysis for tests.",
        uncertainty_flags=uncertainty_flags,
        provider_name=provider_name,
        model_name=model_name,
        analyzed_at=datetime.now(UTC),
        metadata={},
    )


# ── deterministic gate result builders ────────────────────────────────────


def make_finding(
    *,
    severity: Severity = Severity.WARNING,
    rule_name: str = "synthetic_rule",
    source: str = "deterministic",
    message: str = "Synthetic finding for tests.",
) -> Finding:
    return Finding(
        source=source,
        rule_name=rule_name,
        severity=severity,
        message=message,
    )


def make_gate_result(
    *,
    findings: tuple[Finding, ...] = (),
    blocked: bool = False,
    blocking_reasons: tuple[str, ...] = (),
    enabled_recognizers: tuple[str, ...] = ("secret_leak", "pii"),
) -> DeterministicGateResult:
    return DeterministicGateResult(
        findings=findings,
        enabled_recognizers=enabled_recognizers,
        blocked=blocked,
        blocking_reasons=blocking_reasons,
    )
