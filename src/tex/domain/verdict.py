from __future__ import annotations

from enum import StrEnum


class Verdict(StrEnum):
    """
    Final release decision produced by Tex.

    Tex exists to decide whether a specific AI-generated action should be
    allowed to cross the boundary into the real world. The decision surface
    is intentionally strict and minimal:

    - PERMIT: safe enough to release automatically
    - ABSTAIN: insufficient confidence or conflicting signals; escalate
    - FORBID: unsafe to release

    This stays small on purpose. Real-world control systems need a decision
    that is clear, enforceable, and operationally stable. More verdict types
    would add ambiguity without adding real capability.
    """

    PERMIT = "PERMIT"
    ABSTAIN = "ABSTAIN"
    FORBID = "FORBID"

    @property
    def is_terminal(self) -> bool:
        """
        Whether this verdict is a final control decision.

        All Tex verdicts are terminal from the perspective of the evaluation
        pipeline: each one tells the caller exactly what to do next.
        """
        return True

    @property
    def allows_release(self) -> bool:
        """
        Whether the action may proceed automatically into the real world.
        """
        return self is Verdict.PERMIT

    @property
    def requires_human_review(self) -> bool:
        """
        Whether the action should be escalated instead of auto-released.
        """
        return self is Verdict.ABSTAIN

    @property
    def blocks_release(self) -> bool:
        """
        Whether the action must be stopped from proceeding.
        """
        return self is Verdict.FORBID

    @classmethod
    def from_str(cls, value: str) -> "Verdict":
        """
        Parse a verdict from external input.

        Accepts case-insensitive string input and normalizes surrounding
        whitespace, but only permits the three canonical verdicts.
        """
        normalized = value.strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Invalid verdict {value!r}. Expected one of: {allowed}."
            ) from exc