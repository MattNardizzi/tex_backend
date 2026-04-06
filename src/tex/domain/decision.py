from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tex.domain.finding import Finding
from tex.domain.verdict import Verdict


class Decision(BaseModel):
    """
    Immutable durable decision record produced by the Tex pipeline.

    EvaluationResponse is the public product surface.
    Decision is the internal record retained for:
    - evidence logging
    - replay and audit
    - outcome analysis
    - calibration
    - precedent retrieval

    This model is intentionally stricter than a transport DTO. It must be safe
    to persist, compare, and use as the canonical record of what Tex decided.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the durable decision record.",
    )
    request_id: UUID = Field(
        description="Caller-supplied stable identifier for the evaluated request.",
    )

    verdict: Verdict

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall system confidence in the final verdict.",
    )
    final_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fused risk score used to derive the verdict.",
    )

    action_type: str = Field(min_length=1, max_length=100)
    channel: str = Field(min_length=1, max_length=50)
    environment: str = Field(min_length=1, max_length=50)
    recipient: str | None = Field(default=None, max_length=500)

    content_excerpt: str = Field(
        min_length=1,
        max_length=2_000,
        description="Safe excerpt of evaluated content for debugging and audit.",
    )
    content_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="SHA-256 hex digest of the full evaluated content.",
    )

    policy_id: str | None = Field(default=None, max_length=100)
    policy_version: str = Field(min_length=1, max_length=100)

    scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-layer or per-dimension scores used in the decision.",
    )
    findings: list[Finding] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)

    retrieval_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Retrieved policy / precedent context used during evaluation.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution metadata retained for audit.",
    )

    evidence_hash: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="Hash-chain reference once recorded in the evidence system.",
    )

    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator(
        "action_type",
        "channel",
        "environment",
        "content_excerpt",
        "policy_version",
        mode="before",
    )
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("recipient", "policy_id", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("value must be a string when provided")
        normalized = value.strip()
        return normalized or None

    @field_validator("content_sha256", "evidence_hash")
    @classmethod
    def validate_sha256_hex(cls, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = value.strip().lower()
        if len(normalized) != 64:
            raise ValueError("hash must be a 64-character SHA-256 hex digest")

        allowed = set("0123456789abcdef")
        if any(char not in allowed for char in normalized):
            raise ValueError(
                "hash must contain only lowercase hexadecimal characters"
            )

        return normalized

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}

        for raw_key, score in value.items():
            if not isinstance(raw_key, str):
                raise TypeError("score keys must be strings")

            key = raw_key.strip()
            if not key:
                raise ValueError("score keys must be non-blank strings")

            if not 0.0 <= score <= 1.0:
                raise ValueError(f"score for {key!r} must be between 0.0 and 1.0")

            normalized[key] = score

        return normalized

    @field_validator("reasons", "uncertainty_flags")
    @classmethod
    def validate_text_list(cls, values: list[str]) -> list[str]:
        normalized_values: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = value.strip()
            if not normalized:
                raise ValueError("list values must not be blank")
            if normalized not in seen:
                normalized_values.append(normalized)
                seen.add(normalized)

        return normalized_values

    @field_validator("retrieval_context", "metadata")
    @classmethod
    def validate_mapping_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

    @field_validator("decided_at")
    @classmethod
    def validate_decided_at_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decided_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_verdict_consistency(self) -> Decision:
        if self.verdict is Verdict.ABSTAIN and not self.uncertainty_flags:
            raise ValueError(
                "ABSTAIN decisions must include at least one uncertainty flag"
            )

        if self.verdict is Verdict.FORBID:
            has_signal = self.final_score > 0.0 or bool(self.findings) or bool(self.reasons)
            if not has_signal:
                raise ValueError(
                    "FORBID decisions must include non-zero risk, findings, or reasons"
                )

        return self

    @property
    def is_permit(self) -> bool:
        return self.verdict.allows_release

    @property
    def is_abstain(self) -> bool:
        return self.verdict.requires_human_review

    @property
    def is_forbid(self) -> bool:
        return self.verdict.blocks_release

    @property
    def blocking_findings(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.is_blocking]
