from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from tex.domain.decision import Decision
from tex.domain.outcome import OutcomeLabel, OutcomeRecord
from tex.domain.verdict import Verdict

OutcomeClassName = Literal[
    "correct_permit",
    "false_permit",
    "correct_forbid",
    "false_forbid",
    "abstain_review",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class OutcomeClassification:
    """
    Classification result for a single decision/outcome pair.

    This converts a raw observed outcome into a compact signal that Tex can use
    for calibration, reporting, and quality tracking.
    """

    decision_id: str
    request_id: str
    verdict: Verdict
    outcome_label: OutcomeLabel
    classification: OutcomeClassName

    is_correct: bool
    is_error: bool
    is_false_permit: bool
    is_false_forbid: bool
    is_abstain_review: bool
    is_unknown: bool


@dataclass(frozen=True, slots=True)
class OutcomeSummary:
    """
    Aggregate statistics across a batch of classified outcomes.
    """

    total: int
    correct_permits: int
    false_permits: int
    correct_forbids: int
    false_forbids: int
    abstain_reviews: int
    unknown: int

    @property
    def error_count(self) -> int:
        return self.false_permits + self.false_forbids

    @property
    def correctness_count(self) -> int:
        return self.correct_permits + self.correct_forbids

    @property
    def reviewed_count(self) -> int:
        return self.correctness_count + self.error_count + self.abstain_reviews

    @property
    def error_rate(self) -> float:
        """
        Error rate over all classified outcomes.
        """
        if self.total == 0:
            return 0.0
        return self.error_count / self.total

    @property
    def reviewed_error_rate(self) -> float:
        """
        Error rate over non-unknown reviewed outcomes.
        """
        denominator = self.reviewed_count
        if denominator == 0:
            return 0.0
        return self.error_count / denominator


def classify_outcome(
    *,
    decision: Decision,
    outcome: OutcomeRecord,
) -> OutcomeClassification:
    """
    Classifies a single observed outcome against the original decision.

    Explicit mapping:
    - PERMIT + CORRECT_PERMIT -> correct_permit
    - PERMIT + FALSE_PERMIT -> false_permit
    - FORBID + CORRECT_FORBID -> correct_forbid
    - FORBID + FALSE_FORBID -> false_forbid
    - ABSTAIN + ABSTAIN_REVIEW -> abstain_review
    - all other combinations -> unknown

    This function stays intentionally literal. Calibration logic should be built
    on top of stable, auditable outcome labels rather than clever hidden rules.
    """
    _validate_alignment(decision=decision, outcome=outcome)

    verdict = decision.verdict
    label = outcome.label

    if verdict == Verdict.PERMIT and label == OutcomeLabel.CORRECT_PERMIT:
        return _build_classification(
            decision=decision,
            outcome=outcome,
            classification="correct_permit",
            is_correct=True,
            is_false_permit=False,
            is_false_forbid=False,
            is_abstain_review=False,
            is_unknown=False,
        )

    if verdict == Verdict.PERMIT and label == OutcomeLabel.FALSE_PERMIT:
        return _build_classification(
            decision=decision,
            outcome=outcome,
            classification="false_permit",
            is_correct=False,
            is_false_permit=True,
            is_false_forbid=False,
            is_abstain_review=False,
            is_unknown=False,
        )

    if verdict == Verdict.FORBID and label == OutcomeLabel.CORRECT_FORBID:
        return _build_classification(
            decision=decision,
            outcome=outcome,
            classification="correct_forbid",
            is_correct=True,
            is_false_permit=False,
            is_false_forbid=False,
            is_abstain_review=False,
            is_unknown=False,
        )

    if verdict == Verdict.FORBID and label == OutcomeLabel.FALSE_FORBID:
        return _build_classification(
            decision=decision,
            outcome=outcome,
            classification="false_forbid",
            is_correct=False,
            is_false_permit=False,
            is_false_forbid=True,
            is_abstain_review=False,
            is_unknown=False,
        )

    if verdict == Verdict.ABSTAIN and label == OutcomeLabel.ABSTAIN_REVIEW:
        return _build_classification(
            decision=decision,
            outcome=outcome,
            classification="abstain_review",
            is_correct=False,
            is_false_permit=False,
            is_false_forbid=False,
            is_abstain_review=True,
            is_unknown=False,
        )

    return _build_classification(
        decision=decision,
        outcome=outcome,
        classification="unknown",
        is_correct=False,
        is_false_permit=False,
        is_false_forbid=False,
        is_abstain_review=False,
        is_unknown=True,
    )


def summarize_outcomes(
    classifications: Iterable[OutcomeClassification],
) -> OutcomeSummary:
    """
    Produces aggregate counts from a batch of classified outcomes.
    """
    correct_permits = 0
    false_permits = 0
    correct_forbids = 0
    false_forbids = 0
    abstain_reviews = 0
    unknown = 0
    total = 0

    for item in classifications:
        total += 1

        if item.classification == "correct_permit":
            correct_permits += 1
        elif item.classification == "false_permit":
            false_permits += 1
        elif item.classification == "correct_forbid":
            correct_forbids += 1
        elif item.classification == "false_forbid":
            false_forbids += 1
        elif item.classification == "abstain_review":
            abstain_reviews += 1
        else:
            unknown += 1

    return OutcomeSummary(
        total=total,
        correct_permits=correct_permits,
        false_permits=false_permits,
        correct_forbids=correct_forbids,
        false_forbids=false_forbids,
        abstain_reviews=abstain_reviews,
        unknown=unknown,
    )


def classify_batch(
    *,
    decisions: Iterable[Decision],
    outcomes: Iterable[OutcomeRecord],
) -> tuple[OutcomeClassification, ...]:
    """
    Classifies a batch of outcomes against their matching decisions.

    Orphan outcomes are skipped on purpose. This function is for paired analysis,
    not integrity enforcement.
    """
    decisions_by_id = {decision.decision_id: decision for decision in decisions}
    classified: list[OutcomeClassification] = []

    for outcome in outcomes:
        decision = decisions_by_id.get(outcome.decision_id)
        if decision is None:
            continue
        classified.append(classify_outcome(decision=decision, outcome=outcome))

    return tuple(classified)


def _validate_alignment(
    *,
    decision: Decision,
    outcome: OutcomeRecord,
) -> None:
    if outcome.decision_id != decision.decision_id:
        raise ValueError("outcome.decision_id must match decision.decision_id")

    if outcome.request_id != decision.request_id:
        raise ValueError("outcome.request_id must match decision.request_id")


def _build_classification(
    *,
    decision: Decision,
    outcome: OutcomeRecord,
    classification: OutcomeClassName,
    is_correct: bool,
    is_false_permit: bool,
    is_false_forbid: bool,
    is_abstain_review: bool,
    is_unknown: bool,
) -> OutcomeClassification:
    return OutcomeClassification(
        decision_id=str(decision.decision_id),
        request_id=str(decision.request_id),
        verdict=decision.verdict,
        outcome_label=outcome.label,
        classification=classification,
        is_correct=is_correct,
        is_error=is_false_permit or is_false_forbid,
        is_false_permit=is_false_permit,
        is_false_forbid=is_false_forbid,
        is_abstain_review=is_abstain_review,
        is_unknown=is_unknown,
    )