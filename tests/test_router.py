"""
Tests for Tex's routing / verdict-fusion layer.

The router is the single point where all upstream signals meet. It is the
file most likely to drift during calibration work, and its behavior at the
threshold boundaries is what turns correct scores into correct verdicts.

Test plan:
1. deterministic block wins over everything
2. clean inputs with full confidence reach PERMIT
3. `has_low_confidence_dimension` forces ABSTAIN even with a clean score
4. `confidence < minimum_confidence` forces ABSTAIN
5. semantic FORBID with a non-trivial score is respected
6. a high final_score crosses the forbid_threshold even without determinism
7. borderline fused scores (strictly between permit and forbid thresholds)
   must ABSTAIN
8. very high specialist risk forces ABSTAIN even when semantic is permissive
9. fusion weights sum correctly under the default policy
10. scores emitted on the result match the per-layer inputs
"""

from __future__ import annotations

from tex.domain.severity import Severity
from tex.domain.verdict import Verdict
from tex.engine.router import DecisionRouter, build_default_router

from tests.factories import (
    make_default_policy,
    make_finding,
    make_gate_result,
    make_semantic_analysis,
    make_specialist_bundle,
)


# ── helpers ───────────────────────────────────────────────────────────────


def _route(
    *,
    gate=None,
    specialists=None,
    semantic=None,
    policy=None,
    action_type: str = "sales_email",
    channel: str = "email",
    environment: str = "production",
):
    router = build_default_router()
    return router.route(
        deterministic_result=gate or make_gate_result(),
        specialist_bundle=specialists or make_specialist_bundle(max_risk=0.05, confidence=0.75),
        semantic_analysis=semantic or make_semantic_analysis(),
        policy=policy or make_default_policy(),
        action_type=action_type,
        channel=channel,
        environment=environment,
    )


# ── deterministic block short-circuits everything ─────────────────────────


def test_deterministic_block_forces_forbid_regardless_of_semantic_permit() -> None:
    gate = make_gate_result(
        findings=(make_finding(severity=Severity.CRITICAL, rule_name="secret_leak"),),
        blocked=True,
        blocking_reasons=("secret_leak produced CRITICAL finding: api key detected.",),
    )
    result = _route(
        gate=gate,
        semantic=make_semantic_analysis(
            recommended_verdict=Verdict.PERMIT,
            recommended_confidence=0.99,
        ),
    )
    assert result.verdict is Verdict.FORBID
    assert any("secret_leak" in reason for reason in result.reasons)
    # Deterministic score should be 1.0 when blocked.
    assert result.scores["deterministic"] == 1.0


# ── clean path reaches PERMIT ─────────────────────────────────────────────


def test_clean_signals_reach_permit() -> None:
    result = _route(
        specialists=make_specialist_bundle(max_risk=0.05, confidence=0.80),
        semantic=make_semantic_analysis(
            recommended_verdict=Verdict.PERMIT,
            recommended_confidence=0.80,
            overall_confidence=0.80,
            dimension_confidence=0.70,
            evidence_sufficiency=0.50,
        ),
    )
    assert result.verdict is Verdict.PERMIT
    assert result.final_score <= make_default_policy().permit_threshold


# ── low semantic dimension confidence forces abstention ───────────────────


def test_low_confidence_dimension_triggers_abstain() -> None:
    # dimension_confidence=0.40 is below the schema's 0.50 is_low_confidence
    # cutoff, so has_low_confidence_dimension will be True.
    result = _route(
        specialists=make_specialist_bundle(max_risk=0.05, confidence=0.75),
        semantic=make_semantic_analysis(
            recommended_verdict=Verdict.PERMIT,
            recommended_confidence=0.80,
            overall_confidence=0.75,
            dimension_confidence=0.40,
        ),
    )
    assert result.verdict is Verdict.ABSTAIN
    assert "low_confidence_semantic_dimension" in result.uncertainty_flags


# ── confidence below policy minimum forces abstention ─────────────────────


def test_confidence_below_policy_minimum_triggers_abstain() -> None:
    # Craft semantic and specialists so their weighted confidence lands
    # well below the default minimum (0.62).
    result = _route(
        specialists=make_specialist_bundle(max_risk=0.05, confidence=0.30),
        semantic=make_semantic_analysis(
            recommended_verdict=Verdict.PERMIT,
            recommended_confidence=0.40,
            overall_confidence=0.35,
            dimension_confidence=0.55,
        ),
    )
    assert result.verdict is Verdict.ABSTAIN
    assert "confidence_below_policy_minimum" in result.uncertainty_flags


# ── semantic FORBID is honored when final_score supports it ───────────────


def test_semantic_forbid_with_sufficient_score_yields_forbid() -> None:
    # Semantic recommends FORBID with high dimension scores, so fused score
    # clears the 0.45 gate inside _determine_verdict.
    result = _route(
        specialists=make_specialist_bundle(max_risk=0.60, confidence=0.70),
        semantic=make_semantic_analysis(
            recommended_verdict=Verdict.FORBID,
            recommended_confidence=0.80,
            overall_confidence=0.78,
            dimension_score=0.80,
            dimension_confidence=0.70,
            evidence_sufficiency=0.60,
        ),
    )
    assert result.verdict is Verdict.FORBID


# ── fused score crossing forbid threshold produces FORBID ─────────────────


def test_fused_score_above_forbid_threshold_forbids() -> None:
    # To reach fused score >= 0.72 without a deterministic block, we need a
    # CRITICAL finding present (deterministic_score=1.0 without blocking) plus
    # high specialist and semantic risk. Default forbid_threshold is 0.72.
    gate = make_gate_result(
        findings=(make_finding(severity=Severity.CRITICAL, rule_name="synthetic"),),
        blocked=False,  # finding present but policy didn't block
    )
    result = _route(
        gate=gate,
        specialists=make_specialist_bundle(max_risk=0.90, confidence=0.80),
        semantic=make_semantic_analysis(
            recommended_verdict=Verdict.PERMIT,  # semantic says PERMIT
            recommended_confidence=0.75,
            overall_confidence=0.75,
            dimension_score=0.80,
            dimension_confidence=0.70,
            evidence_sufficiency=0.60,
        ),
    )
    # Even though semantic said PERMIT, the fused score on its own crosses
    # forbid_threshold. The router must honor that threshold.
    assert result.verdict is Verdict.FORBID
    assert result.final_score >= make_default_policy().forbid_threshold


# ── borderline fused score abstains ───────────────────────────────────────


def test_borderline_fused_score_triggers_abstain() -> None:
    # Target a final_score strictly inside (permit_threshold, forbid_threshold)
    # with semantic recommending PERMIT. Should ABSTAIN via the borderline
    # rule inside _should_abstain.
    result = _route(
        specialists=make_specialist_bundle(max_risk=0.55, confidence=0.75),
        semantic=make_semantic_analysis(
            recommended_verdict=Verdict.PERMIT,
            recommended_confidence=0.70,
            overall_confidence=0.70,
            dimension_score=0.50,
            dimension_confidence=0.65,
            evidence_sufficiency=0.45,
        ),
    )
    assert result.verdict is Verdict.ABSTAIN
    # borderline flag should surface on abstention
    assert "borderline_fused_score" in result.uncertainty_flags


# ── specialist high-risk escape hatch forces abstention ───────────────────


def test_high_specialist_risk_forces_abstain_even_with_semantic_permit() -> None:
    result = _route(
        specialists=make_specialist_bundle(max_risk=0.72, confidence=0.78),
        semantic=make_semantic_analysis(
            recommended_verdict=Verdict.PERMIT,
            recommended_confidence=0.75,
            overall_confidence=0.75,
            dimension_score=0.10,
            dimension_confidence=0.70,
            evidence_sufficiency=0.50,
        ),
    )
    # When specialist risk >= 0.60 and fused score is below forbid threshold,
    # the router must abstain rather than auto-permitting.
    assert result.verdict is Verdict.ABSTAIN


# ── missing retrieval context flag promotes abstention near threshold ─────


def test_no_retrieval_context_flag_pushes_borderline_score_to_abstain() -> None:
    # The no_retrieval_context abstention branch only fires when final_score
    # is at or above permit_threshold. Craft fixture so fused score lands
    # just above permit (0.34) but well below forbid (0.72), and carry the
    # flag on both specialist and semantic side.
    result = _route(
        specialists=make_specialist_bundle(
            max_risk=0.55,
            confidence=0.70,
            uncertainty_flags=("no_retrieval_context",),
        ),
        semantic=make_semantic_analysis(
            recommended_verdict=Verdict.PERMIT,
            recommended_confidence=0.70,
            overall_confidence=0.70,
            dimension_score=0.50,
            dimension_confidence=0.65,
            evidence_sufficiency=0.40,
            uncertainty_flags=("no_retrieval_context",),
        ),
    )
    assert result.verdict is Verdict.ABSTAIN


# ── per-layer score exposure ──────────────────────────────────────────────


def test_result_scores_expose_each_layer() -> None:
    result = _route()
    for key in ("deterministic", "specialists", "semantic", "criticality"):
        assert key in result.scores
        assert 0.0 <= result.scores[key] <= 1.0


def test_final_score_respects_fusion_weights() -> None:
    # Set all non-criticality layers to 0 and semantic to 1.0. With default
    # semantic weight = 0.35, final score should be 0.35 + (criticality *
    # 0.10). We can bound this tightly.
    result = _route(
        gate=make_gate_result(findings=(), blocked=False),
        specialists=make_specialist_bundle(max_risk=0.0, confidence=0.75),
        semantic=make_semantic_analysis(
            recommended_verdict=Verdict.PERMIT,
            recommended_confidence=0.70,
            overall_confidence=0.70,
            dimension_score=1.0,
            dimension_confidence=0.70,
            evidence_sufficiency=0.50,
        ),
        action_type="sales_email",
        channel="email",
        environment="production",
    )
    # Deterministic=0, specialists=0, semantic=1.0*0.35=0.35,
    # criticality capped at 1.0 * 0.10 = at most 0.10. So final_score must
    # be at least 0.35 and at most 0.45.
    assert 0.34 < result.final_score < 0.46


# ── confidence rounding ───────────────────────────────────────────────────


def test_confidence_and_final_score_are_rounded_to_four_places() -> None:
    result = _route()
    # Since both floats are rounded at emit time, their string forms must
    # contain no more than four fractional digits.
    for value in (result.confidence, result.final_score):
        as_str = f"{value:.10f}".rstrip("0")
        fractional_digits = as_str.split(".")[-1] if "." in as_str else ""
        assert len(fractional_digits) <= 4


# ── uncertainty flag dedupe ───────────────────────────────────────────────


def test_uncertainty_flags_are_deduplicated_across_sources() -> None:
    result = _route(
        specialists=make_specialist_bundle(
            max_risk=0.60,
            confidence=0.65,
            uncertainty_flags=("borderline_risk",),
        ),
        semantic=make_semantic_analysis(
            recommended_verdict=Verdict.ABSTAIN,
            recommended_confidence=0.55,
            uncertainty_flags=("borderline_risk", "weak_evidence"),
        ),
    )
    # borderline_risk appears from two sources; router must dedupe.
    assert result.uncertainty_flags.count("borderline_risk") == 1


# ── router is pure: same inputs produce same outputs ──────────────────────


def test_router_is_deterministic_for_same_inputs() -> None:
    policy = make_default_policy()
    gate = make_gate_result()
    specialists = make_specialist_bundle(max_risk=0.20, confidence=0.70)
    semantic = make_semantic_analysis()

    router = DecisionRouter()
    first = router.route(
        deterministic_result=gate,
        specialist_bundle=specialists,
        semantic_analysis=semantic,
        policy=policy,
        action_type="sales_email",
        channel="email",
        environment="production",
    )
    second = router.route(
        deterministic_result=gate,
        specialist_bundle=specialists,
        semantic_analysis=semantic,
        policy=policy,
        action_type="sales_email",
        channel="email",
        environment="production",
    )
    assert first.verdict == second.verdict
    assert first.final_score == second.final_score
    assert first.confidence == second.confidence
    assert first.scores == second.scores
