from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tex.domain.severity import Severity


class Finding(BaseModel):
    """
    Structured signal emitted by Tex during deterministic or specialist analysis.

    A finding is not the final decision. It is a piece of evidence that a later
    stage in the pipeline can use to justify PERMIT / ABSTAIN / FORBID.

    Examples:
    - a credit card number detected in outbound content
    - a JWT-like token pattern found in an API payload
    - a blocked phrase such as "internal only" in a customer-facing email
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(
        min_length=1,
        max_length=100,
        description="Subsystem that produced the finding, such as deterministic.pii.",
    )
    rule_name: str = Field(
        min_length=1,
        max_length=150,
        description="Stable identifier for the recognizer or rule that matched.",
    )
    severity: Severity = Field(
        description="Importance level of the finding independent of final verdict.",
    )
    message: str = Field(
        min_length=1,
        max_length=500,
        description="Human-readable explanation of what was found.",
    )
    matched_text: str | None = Field(
        default=None,
        max_length=1000,
        description="Exact text span that matched, when safe to retain.",
    )
    start_index: int | None = Field(
        default=None,
        ge=0,
        description="Inclusive start character offset in the evaluated content.",
    )
    end_index: int | None = Field(
        default=None,
        ge=0,
        description="Exclusive end character offset in the evaluated content.",
    )
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Optional structured context for debugging, audit, or routing.",
    )

    @field_validator("source", "rule_name", "message")
    @classmethod
    def validate_non_blank_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Value must not be blank.")
        return normalized

    @field_validator("matched_text")
    @classmethod
    def validate_matched_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("matched_text must not be blank when provided.")
        return normalized

    @field_validator("end_index")
    @classmethod
    def validate_index_order(
        cls,
        end_index: int | None,
        info,
    ) -> int | None:
        start_index = info.data.get("start_index")
        if start_index is None and end_index is None:
            return None
        if start_index is None or end_index is None:
            raise ValueError(
                "start_index and end_index must either both be provided or both be omitted."
            )
        if end_index <= start_index:
            raise ValueError("end_index must be greater than start_index.")
        return end_index

    @property
    def has_span(self) -> bool:
        """
        Whether this finding includes a precise character span.
        """
        return self.start_index is not None and self.end_index is not None

    @property
    def is_blocking(self) -> bool:
        """
        Whether this finding is severe enough to justify immediate blocking
        in the deterministic gate.
        """
        return self.severity.is_critical