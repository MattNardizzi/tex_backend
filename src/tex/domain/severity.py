from __future__ import annotations

from enum import IntEnum, StrEnum


class Severity(StrEnum):
    """
    Severity level for deterministic findings and other risk signals.

    Severity is deliberately separate from the final verdict. A finding can be
    informational, warning-level, or critical, while the final release decision
    is still produced by the full Tex pipeline.

    This keeps the system honest:
    - findings describe what was detected
    - verdicts decide what happens next
    """

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        """
        Numeric ordering for comparisons and aggregation.
        """
        return _SEVERITY_RANK[self]

    @property
    def is_informational(self) -> bool:
        return self is Severity.INFO

    @property
    def is_warning(self) -> bool:
        return self is Severity.WARNING

    @property
    def is_critical(self) -> bool:
        return self is Severity.CRITICAL

    @classmethod
    def from_str(cls, value: str) -> "Severity":
        """
        Parse severity from external input.

        Accepts case-insensitive string input with surrounding whitespace.
        """
        normalized = value.strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Invalid severity {value!r}. Expected one of: {allowed}."
            ) from exc

    @classmethod
    def max(cls, severities: list["Severity"] | tuple["Severity", ...]) -> "Severity | None":
        """
        Return the highest severity present, or None for an empty collection.
        """
        if not severities:
            return None
        return max(severities, key=lambda severity: severity.rank)


class SeverityScore(IntEnum):
    """
    Stable numeric mapping for severity-aware comparisons.

    Kept separate from the string enum so the public API stays human-readable
    while internal ordering remains explicit and type-safe.
    """

    INFO = 1
    WARNING = 2
    CRITICAL = 3


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.INFO: SeverityScore.INFO,
    Severity.WARNING: SeverityScore.WARNING,
    Severity.CRITICAL: SeverityScore.CRITICAL,
}