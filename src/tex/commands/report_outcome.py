from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tex.domain.decision import Decision
from tex.domain.evidence import EvidenceRecord
from tex.domain.outcome import OutcomeRecord
from tex.learning.outcomes import OutcomeClassification, classify_outcome
from tex.stores.decision_store import InMemoryDecisionStore
from tex.stores.outcome_store import InMemoryOutcomeStore


@runtime_checkable
class OutcomeEvidenceRecorder(Protocol):
    """
    Narrow protocol for recorders capable of appending outcome evidence.
    """

    def record_outcome(
        self,
        outcome: OutcomeRecord,
        *,
        metadata: dict[str, object] | None = None,
        policy_version: str | None = None,
    ) -> EvidenceRecord:
        """
        Persists an evidence envelope for an outcome.
        """
        ...


@dataclass(frozen=True, slots=True)
class ReportOutcomeResult:
    """
    Application-layer result for recording a single observed outcome.
    """

    outcome: OutcomeRecord
    decision: Decision
    classification: OutcomeClassification
    evidence_record: EvidenceRecord | None = None


class ReportOutcomeCommand:
    """
    Application service for recording observed outcomes against prior decisions.

    Responsibilities:
    - verify the referenced decision exists
    - verify the request linkage is consistent
    - persist the outcome
    - classify the decision/outcome pair
    - optionally append evidence

    This command does not perform calibration. Calibration belongs in the
    learning layer after enough labeled outcomes exist.
    """

    __slots__ = (
        "_decision_store",
        "_outcome_store",
        "_evidence_recorder",
    )

    def __init__(
        self,
        *,
        decision_store: InMemoryDecisionStore,
        outcome_store: InMemoryOutcomeStore,
        evidence_recorder: OutcomeEvidenceRecorder | None = None,
    ) -> None:
        self._decision_store = decision_store
        self._outcome_store = outcome_store
        self._evidence_recorder = evidence_recorder

    def execute(self, outcome: OutcomeRecord) -> ReportOutcomeResult:
        """
        Records an outcome for an existing decision and returns the classification.
        """
        decision = self._resolve_decision(outcome)
        self._validate_decision_alignment(outcome=outcome, decision=decision)

        self._outcome_store.save(outcome)

        classification = classify_outcome(
            decision=decision,
            outcome=outcome,
        )

        evidence_record = None
        if self._evidence_recorder is not None:
            evidence_record = self._record_outcome_evidence(
                outcome=outcome,
                decision=decision,
                classification=classification,
            )

        return ReportOutcomeResult(
            outcome=outcome,
            decision=decision,
            classification=classification,
            evidence_record=evidence_record,
        )

    def _resolve_decision(self, outcome: OutcomeRecord) -> Decision:
        try:
            return self._decision_store.require(outcome.decision_id)
        except KeyError as exc:
            raise LookupError(
                f"cannot record outcome for missing decision: {outcome.decision_id}"
            ) from exc

    @staticmethod
    def _validate_decision_alignment(
        *,
        outcome: OutcomeRecord,
        decision: Decision,
    ) -> None:
        """
        Enforces that the outcome is being attached to the correct evaluated request.
        """
        if outcome.request_id != decision.request_id:
            raise ValueError(
                "outcome request_id does not match the referenced decision request_id"
            )

    def _record_outcome_evidence(
        self,
        *,
        outcome: OutcomeRecord,
        decision: Decision,
        classification: OutcomeClassification,
    ) -> EvidenceRecord:
        recorder = self._evidence_recorder
        if recorder is None:
            raise RuntimeError("evidence recorder is not configured")

        if not isinstance(recorder, OutcomeEvidenceRecorder):
            raise TypeError(
                "evidence_recorder must implement record_outcome("
                "outcome, *, metadata=None, policy_version=None)"
            )

        return recorder.record_outcome(
            outcome,
            policy_version=decision.policy_version,
            metadata={
                "decision_verdict": decision.verdict.value,
                "decision_final_score": decision.final_score,
                "decision_confidence": decision.confidence,
                "decision_policy_version": decision.policy_version,
                "classification": classification.classification,
                "is_correct": classification.is_correct,
                "is_error": classification.is_error,
                "is_false_permit": classification.is_false_permit,
                "is_false_forbid": classification.is_false_forbid,
                "is_abstain_review": classification.is_abstain_review,
                "is_unknown": classification.is_unknown,
            },
        )