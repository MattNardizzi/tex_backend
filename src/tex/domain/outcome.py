from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tex.domain.verdict import Verdict


class OutcomeKind(StrEnum):
    """
    Human- or system-reported result observed after Tex made a decision.

    This describes what happened in the world after evaluation, not whether
    Tex was right about it.
    """

    RELEASED = "RELEASED"
    BLOCKED = "BLOCKED"
    ESCALATED = "ESCALATED"
    OVERRIDDEN = "OVERRIDDEN"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_str(cls, value: str) -> "OutcomeKind":
        normalized = value.strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Invalid outcome kind {value!r}. Expected one of: {allowed}."
            ) from exc


class OutcomeLabel(StrEnum):
    """
    Tex performance label derived by comparing a final verdict to what actually
    happened afterward.

    This is the label used by the learning loop and calibrator.
    """

    CORRECT_PERMIT = "CORRECT_PERMIT"
    FALSE_PERMIT = "FALSE_PERMIT"
    CORRECT_FORBID = "CORRECT_FORBID"
    FALSE_FORBID = "FALSE_FORBID"
    ABSTAIN_REVIEW = "ABSTAIN_REVIEW"
    UNKNOWN = "UNKNOWN"


class OutcomeRecord(BaseModel):
    """
    Durable record of what happened after a Tex decision.

    This powers:
    - learning / calibration
    - audit
    - precedent retrieval
    - operator review
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome_id: UUID = Field(default_factory=uuid4)
    decision_id: UUID
    request_id: UUID

    verdict: Verdict
    outcome_kind: OutcomeKind

    was_safe: bool | None = Field(
        default=None,
        description=(
            "Whether the action was ultimately safe / acceptable in hindsight. "
            "Can be unknown at record time."
        ),
    )
    human_override: bool = Field(
        default=False,
        description="Whether a human explicitly overrode Tex's original verdict.",
    )

    summary: str | None = Field(
        default=None,
        max_length=2_000,
        description="Optional human-readable description of what happened.",
    )
    reporter: str | None = Field(
        default=None,
        max_length=200,
        description="Human or system that reported the outcome.",
    )

    label: OutcomeLabel = Field(
        default=OutcomeLabel.UNKNOWN,
        description="Derived calibration label for the decision/outcome pair.",
    )

    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("summary", "reporter", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("Value must be a string when provided.")
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_consistency(self) -> "OutcomeRecord":
        if self.outcome_kind is OutcomeKind.OVERRIDDEN and not self.human_override:
            raise ValueError(
                "Outcome kind OVERRIDDEN requires human_override=True."
            )
        return self

    @staticmethod
    def classify(
        *,
        verdict: Verdict,
        outcome_kind: OutcomeKind,
        was_safe: bool | None,
    ) -> OutcomeLabel:
        """
        Classify whether Tex's original decision appears to have been correct.

        Rules:
        - PERMIT + unsafe outcome => FALSE_PERMIT
        - PERMIT + safe outcome => CORRECT_PERMIT
        - FORBID + safe outcome => FALSE_FORBID
        - FORBID + unsafe outcome => CORRECT_FORBID
        - ABSTAIN => ABSTAIN_REVIEW
        - anything with insufficient hindsight => UNKNOWN
        """
        if verdict is Verdict.ABSTAIN:
            return OutcomeLabel.ABSTAIN_REVIEW

        if was_safe is None:
            return OutcomeLabel.UNKNOWN

        if verdict is Verdict.PERMIT:
            return (
                OutcomeLabel.CORRECT_PERMIT
                if was_safe
                else OutcomeLabel.FALSE_PERMIT
            )

        if verdict is Verdict.FORBID:
            return (
                OutcomeLabel.FALSE_FORBID
                if was_safe
                else OutcomeLabel.CORRECT_FORBID
            )

        return OutcomeLabel.UNKNOWN

    @classmethod
    def create(
        cls,
        *,
        decision_id: UUID,
        request_id: UUID,
        verdict: Verdict,
        outcome_kind: OutcomeKind,
        was_safe: bool | None = None,
        human_override: bool = False,
        summary: str | None = None,
        reporter: str | None = None,
    ) -> "OutcomeRecord":
        """
        Convenience constructor that computes the calibration label automatically.
        """
        label = cls.classify(
            verdict=verdict,
            outcome_kind=outcome_kind,
            was_safe=was_safe,
        )
        return cls(
            decision_id=decision_id,
            request_id=request_id,
            verdict=verdict,
            outcome_kind=outcome_kind,
            was_safe=was_safe,
            human_override=human_override,
            summary=summary,
            reporter=reporter,
            label=label,
        )