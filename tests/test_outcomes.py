"""
Tests for Tex's outcome classification and calibration labels.

Outcome labels power the learning / calibration loop. If this classification
is wrong, the calibrator drifts in the wrong direction. The rules are small
and deterministic, so the tests are too — but they are load-bearing.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from tex.domain.outcome import OutcomeKind, OutcomeLabel, OutcomeRecord
from tex.domain.verdict import Verdict

from tests.factories import make_outcome


# ── verdict + was_safe truth table ────────────────────────────────────────


def test_permit_plus_safe_is_correct_permit() -> None:
    outcome = make_outcome(
        verdict=Verdict.PERMIT,
        outcome_kind=OutcomeKind.RELEASED,
        was_safe=True,
    )
    assert outcome.label is OutcomeLabel.CORRECT_PERMIT


def test_permit_plus_unsafe_is_false_permit() -> None:
    outcome = make_outcome(
        verdict=Verdict.PERMIT,
        outcome_kind=OutcomeKind.RELEASED,
        was_safe=False,
    )
    assert outcome.label is OutcomeLabel.FALSE_PERMIT


def test_forbid_plus_unsafe_is_correct_forbid() -> None:
    outcome = make_outcome(
        verdict=Verdict.FORBID,
        outcome_kind=OutcomeKind.BLOCKED,
        was_safe=False,
    )
    assert outcome.label is OutcomeLabel.CORRECT_FORBID


def test_forbid_plus_safe_is_false_forbid() -> None:
    outcome = make_outcome(
        verdict=Verdict.FORBID,
        outcome_kind=OutcomeKind.BLOCKED,
        was_safe=True,
    )
    assert outcome.label is OutcomeLabel.FALSE_FORBID


# ── abstain + escalate path ───────────────────────────────────────────────


def test_abstain_always_yields_abstain_review_regardless_of_was_safe() -> None:
    for was_safe in (True, False, None):
        outcome = OutcomeRecord.create(
            decision_id=uuid4(),
            request_id=uuid4(),
            verdict=Verdict.ABSTAIN,
            outcome_kind=OutcomeKind.ESCALATED,
            was_safe=was_safe,
        )
        assert outcome.label is OutcomeLabel.ABSTAIN_REVIEW


# ── unknown safety yields UNKNOWN for non-abstain verdicts ────────────────


def test_permit_with_unknown_safety_is_unknown() -> None:
    outcome = make_outcome(
        verdict=Verdict.PERMIT,
        outcome_kind=OutcomeKind.RELEASED,
        was_safe=None,
    )
    assert outcome.label is OutcomeLabel.UNKNOWN


def test_forbid_with_unknown_safety_is_unknown() -> None:
    outcome = make_outcome(
        verdict=Verdict.FORBID,
        outcome_kind=OutcomeKind.BLOCKED,
        was_safe=None,
    )
    assert outcome.label is OutcomeLabel.UNKNOWN


# ── human override structural invariant ───────────────────────────────────


def test_overridden_outcome_kind_requires_human_override_flag() -> None:
    # The model validator enforces this invariant — without human_override=True,
    # OutcomeKind.OVERRIDDEN must raise at construction time.
    with pytest.raises(ValueError, match="OVERRIDDEN"):
        OutcomeRecord.create(
            decision_id=uuid4(),
            request_id=uuid4(),
            verdict=Verdict.FORBID,
            outcome_kind=OutcomeKind.OVERRIDDEN,
            was_safe=True,
            human_override=False,
        )


def test_overridden_outcome_accepts_human_override_flag() -> None:
    outcome = OutcomeRecord.create(
        decision_id=uuid4(),
        request_id=uuid4(),
        verdict=Verdict.FORBID,
        outcome_kind=OutcomeKind.OVERRIDDEN,
        was_safe=True,
        human_override=True,
    )
    assert outcome.human_override is True
    assert outcome.outcome_kind is OutcomeKind.OVERRIDDEN


# ── outcome_kind parsing ──────────────────────────────────────────────────


def test_outcome_kind_from_str_is_case_insensitive_and_trims() -> None:
    assert OutcomeKind.from_str("released") is OutcomeKind.RELEASED
    assert OutcomeKind.from_str("  BLOCKED  ") is OutcomeKind.BLOCKED
    assert OutcomeKind.from_str("Escalated") is OutcomeKind.ESCALATED


def test_outcome_kind_from_str_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Invalid outcome kind"):
        OutcomeKind.from_str("not_a_real_kind")


# ── classify is a pure function; call it directly ────────────────────────


def test_classify_permit_safe_without_creating_record() -> None:
    label = OutcomeRecord.classify(
        verdict=Verdict.PERMIT,
        outcome_kind=OutcomeKind.RELEASED,
        was_safe=True,
    )
    assert label is OutcomeLabel.CORRECT_PERMIT


def test_classify_is_pure() -> None:
    # Calling twice yields the same label for the same inputs.
    first = OutcomeRecord.classify(
        verdict=Verdict.PERMIT,
        outcome_kind=OutcomeKind.RELEASED,
        was_safe=False,
    )
    second = OutcomeRecord.classify(
        verdict=Verdict.PERMIT,
        outcome_kind=OutcomeKind.RELEASED,
        was_safe=False,
    )
    assert first is second
    assert first is OutcomeLabel.FALSE_PERMIT


# ── summary and reporter normalization ────────────────────────────────────


def test_summary_and_reporter_are_stripped() -> None:
    outcome = OutcomeRecord.create(
        decision_id=uuid4(),
        request_id=uuid4(),
        verdict=Verdict.PERMIT,
        outcome_kind=OutcomeKind.RELEASED,
        was_safe=True,
        summary="   released cleanly   ",
        reporter="   qa-bot   ",
    )
    assert outcome.summary == "released cleanly"
    assert outcome.reporter == "qa-bot"


def test_blank_summary_and_reporter_become_none() -> None:
    outcome = OutcomeRecord.create(
        decision_id=uuid4(),
        request_id=uuid4(),
        verdict=Verdict.PERMIT,
        outcome_kind=OutcomeKind.RELEASED,
        was_safe=True,
        summary="   ",
        reporter="   ",
    )
    assert outcome.summary is None
    assert outcome.reporter is None


# ── coverage: every (verdict, was_safe) combination lands somewhere ──────


def test_all_verdict_safety_combinations_produce_a_label() -> None:
    # Cartesian product as a defensive check in case somebody adds a new
    # verdict or outcome kind and forgets to update the classifier.
    combinations = [
        (Verdict.PERMIT, True, OutcomeLabel.CORRECT_PERMIT),
        (Verdict.PERMIT, False, OutcomeLabel.FALSE_PERMIT),
        (Verdict.PERMIT, None, OutcomeLabel.UNKNOWN),
        (Verdict.FORBID, True, OutcomeLabel.FALSE_FORBID),
        (Verdict.FORBID, False, OutcomeLabel.CORRECT_FORBID),
        (Verdict.FORBID, None, OutcomeLabel.UNKNOWN),
        (Verdict.ABSTAIN, True, OutcomeLabel.ABSTAIN_REVIEW),
        (Verdict.ABSTAIN, False, OutcomeLabel.ABSTAIN_REVIEW),
        (Verdict.ABSTAIN, None, OutcomeLabel.ABSTAIN_REVIEW),
    ]
    for verdict, was_safe, expected in combinations:
        outcome_kind = {
            Verdict.PERMIT: OutcomeKind.RELEASED,
            Verdict.FORBID: OutcomeKind.BLOCKED,
            Verdict.ABSTAIN: OutcomeKind.ESCALATED,
        }[verdict]
        label = OutcomeRecord.classify(
            verdict=verdict,
            outcome_kind=outcome_kind,
            was_safe=was_safe,
        )
        assert label is expected, (
            f"verdict={verdict.value} was_safe={was_safe!r} "
            f"expected {expected.value} got {label.value}"
        )
