"""
Tests for Tex's deterministic gate.

What this file guarantees:
- each recognizer fires on the content it is supposed to catch
- none of the recognizers fire on benign content
- severity-to-verdict mapping is correct
- policy-driven blocking activates only when a finding's severity crosses the
  policy's configured block threshold
"""

from __future__ import annotations

from tex.deterministic.gate import (
    DeterministicGate,
    DeterministicGateResult,
    build_default_deterministic_gate,
)
from tex.deterministic.recognizers import default_recognizers
from tex.domain.severity import Severity
from tex.domain.verdict import Verdict

from tests.factories import (
    CLEAN_CONTENT,
    COMMITMENT_CONTENT,
    DESTRUCTIVE_CONTENT,
    PII_CONTENT,
    SECRET_LEAK_CONTENT,
    make_default_policy,
    make_request,
    make_strict_policy,
)


# ── baseline: clean content passes cleanly ─────────────────────────────────


def test_clean_content_produces_no_findings() -> None:
    gate = build_default_deterministic_gate()
    policy = make_default_policy()

    for content in CLEAN_CONTENT:
        result = gate.evaluate(
            request=make_request(content=content),
            policy=policy,
        )
        assert isinstance(result, DeterministicGateResult)
        assert result.findings == (), f"unexpected findings for content: {content!r}"
        assert result.blocked is False
        assert result.suggested_verdict is None


# ── recognizer coverage: secrets ──────────────────────────────────────────


def test_secret_leak_recognizer_catches_api_keys_and_tokens() -> None:
    gate = build_default_deterministic_gate()
    policy = make_default_policy()

    for content in SECRET_LEAK_CONTENT:
        result = gate.evaluate(request=make_request(content=content), policy=policy)
        assert result.has_findings, f"secret content did not trigger gate: {content!r}"
        assert any(
            f.severity is Severity.CRITICAL
            and f.rule_name in {"secret_leak", "unauthorized_commitment"}
            for f in result.findings
        )


def test_secret_leak_critical_severity_triggers_block_under_default_policy() -> None:
    gate = build_default_deterministic_gate()
    policy = make_default_policy()

    result = gate.evaluate(
        request=make_request(content="api_key = abc123"),
        policy=policy,
    )
    # Default policy blocks CRITICAL severities, so blocked should be True
    # and the suggested verdict must be FORBID.
    assert result.blocked is True
    assert result.suggested_verdict is Verdict.FORBID
    assert result.blocking_reasons, "blocked=True must yield at least one reason"


# ── recognizer coverage: PII ──────────────────────────────────────────────


def test_pii_recognizer_catches_ssn_and_phone_and_email() -> None:
    gate = build_default_deterministic_gate()
    policy = make_default_policy()

    for content in PII_CONTENT:
        result = gate.evaluate(request=make_request(content=content), policy=policy)
        assert result.has_findings, f"PII content did not trigger gate: {content!r}"
        # PII recognizer is critical severity under default policy.
        assert any(f.rule_name == "pii" and f.severity is Severity.CRITICAL for f in result.findings)


# ── recognizer coverage: commitments ──────────────────────────────────────


def test_unauthorized_commitment_recognizer_is_warning_not_critical() -> None:
    gate = build_default_deterministic_gate()
    policy = make_default_policy()

    # Commitment language is warning-level in default policy — it should
    # surface a finding but not hard-block.
    for content in COMMITMENT_CONTENT:
        result = gate.evaluate(request=make_request(content=content), policy=policy)
        assert result.has_findings
        commitment_findings = [
            f for f in result.findings if f.rule_name == "unauthorized_commitment"
        ]
        assert commitment_findings
        assert all(f.severity is Severity.WARNING for f in commitment_findings)


# ── recognizer coverage: destructive / bypass ─────────────────────────────


def test_destructive_or_bypass_recognizer_catches_destructive_language() -> None:
    gate = build_default_deterministic_gate()
    policy = make_default_policy()

    for content in DESTRUCTIVE_CONTENT:
        result = gate.evaluate(request=make_request(content=content), policy=policy)
        assert result.has_findings, f"destructive content did not trigger gate: {content!r}"
        assert any(
            f.rule_name == "destructive_or_bypass" and f.severity is Severity.CRITICAL
            for f in result.findings
        )


# ── recognizer coverage: blocked terms (policy-driven) ────────────────────


def test_blocked_terms_recognizer_uses_policy_not_hardcoded_list() -> None:
    gate = build_default_deterministic_gate()
    policy = make_default_policy()

    # Blocked-terms recognizer reads from request.metadata["blocked_terms"].
    request = make_request(
        content="please do a custom_blocked_phrase on the report",
        metadata={"blocked_terms": ["custom_blocked_phrase"]},
    )
    result = gate.evaluate(request=request, policy=policy)
    blocked_findings = [f for f in result.findings if f.rule_name == "blocked_terms"]
    assert blocked_findings, "policy-driven blocked term was not matched"
    assert all(f.severity is Severity.CRITICAL for f in blocked_findings)


# ── policy behavior: enabled recognizers ──────────────────────────────────


def test_disabled_recognizer_is_not_executed() -> None:
    # Drop the secret_leak recognizer from the enabled list via strict policy
    # override — it should then not fire even on obvious secret content.
    gate = build_default_deterministic_gate()
    policy = make_default_policy().model_copy(
        update={"enabled_recognizers": ("pii",)}
    )
    result = gate.evaluate(
        request=make_request(content="api_key = abc123xyz"),
        policy=policy,
    )
    assert "secret_leak" not in result.enabled_recognizers
    assert all(f.rule_name != "secret_leak" for f in result.findings)


# ── policy behavior: block severities ─────────────────────────────────────


def test_warning_does_not_block_under_default_policy() -> None:
    gate = build_default_deterministic_gate()
    policy = make_default_policy()

    result = gate.evaluate(
        request=make_request(content="we guarantee this will ship"),
        policy=policy,
    )
    assert result.has_findings
    assert all(f.severity is Severity.WARNING for f in result.findings)
    assert result.blocked is False
    assert result.suggested_verdict is Verdict.ABSTAIN


def test_strict_policy_block_severities_same_as_default_for_critical() -> None:
    # Strict mode is stricter in thresholds and specialist weights but
    # critical-severity blocking is shared. This test pins that contract
    # so a future reshuffle of strict policy doesn't silently let CRITICAL
    # findings through without a block.
    gate = build_default_deterministic_gate()
    policy = make_strict_policy()

    result = gate.evaluate(
        request=make_request(content="password: hunter2abc"),
        policy=policy,
    )
    assert result.blocked is True
    assert result.suggested_verdict is Verdict.FORBID


# ── determinism: same input, same output ──────────────────────────────────


def test_same_request_yields_same_findings_across_calls() -> None:
    gate = build_default_deterministic_gate()
    policy = make_default_policy()
    request = make_request(content=SECRET_LEAK_CONTENT[0])

    first = gate.evaluate(request=request, policy=policy)
    second = gate.evaluate(request=request, policy=policy)

    assert len(first.findings) == len(second.findings)
    for a, b in zip(first.findings, second.findings):
        assert a.rule_name == b.rule_name
        assert a.severity == b.severity
        assert a.start_index == b.start_index
        assert a.end_index == b.end_index


# ── dedupe: the same pattern matched multiple ways is collapsed ───────────


def test_findings_are_deduplicated_by_rule_and_span() -> None:
    # An injection of the same secret twice should produce two distinct
    # findings at different spans — dedupe only collapses identical spans.
    gate = build_default_deterministic_gate()
    policy = make_default_policy()
    content = "api_key = abc123 and later api_key = xyz789"
    result = gate.evaluate(request=make_request(content=content), policy=policy)

    secret_findings = [f for f in result.findings if f.rule_name == "secret_leak"]
    span_pairs = {(f.start_index, f.end_index) for f in secret_findings}
    assert len(span_pairs) == len(secret_findings)


# ── recognizer construction is pure ───────────────────────────────────────


def test_default_recognizers_list_is_non_empty_and_named() -> None:
    recognizers = default_recognizers()
    assert recognizers
    names = {r.name for r in recognizers}
    assert {
        "blocked_terms",
        "sensitive_entities",
        "secret_leak",
        "pii",
        "unauthorized_commitment",
        "external_sharing",
        "destructive_or_bypass",
    } <= names


def test_gate_can_be_constructed_with_custom_recognizers() -> None:
    # The gate should accept an arbitrary recognizer tuple so that tests and
    # bespoke deployments can inject their own.
    class NoopRecognizer:
        name = "noop"

        def scan(self, request):  # type: ignore[no-untyped-def]
            return tuple()

    gate = DeterministicGate(recognizers=(NoopRecognizer(),))
    policy = make_default_policy().model_copy(
        update={"enabled_recognizers": ("noop",)}
    )
    result = gate.evaluate(
        request=make_request(content="anything goes here"),
        policy=policy,
    )
    assert result.findings == ()
    assert result.enabled_recognizers == ("noop",)
