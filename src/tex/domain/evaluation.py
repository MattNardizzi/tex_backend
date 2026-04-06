from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tex.domain.finding import Finding
from tex.domain.verdict import Verdict


class EvaluationRequest(BaseModel):
    """
    Canonical input to Tex for one content adjudication event.

    Tex does not own identity, permissions, or runtime authorization. It judges
    one concrete action request in context:
    - what action is being attempted
    - what content is about to be released
    - where it is going
    - under which policy context
    - under which explicit request identity

    request_id is first-class and must enter the system at the edge. The PDP and
    downstream decision record must preserve it unchanged.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: UUID

    action_type: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=50_000)

    recipient: str | None = Field(default=None, max_length=500)
    channel: str = Field(min_length=1, max_length=50)
    environment: str = Field(min_length=1, max_length=50)

    metadata: dict[str, Any] = Field(default_factory=dict)
    policy_id: str | None = Field(default=None, max_length=100)

    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("action_type", "channel", "environment")
    @classmethod
    def normalize_lower(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("content must not be empty")
        return normalized

    @field_validator("recipient")
    @classmethod
    def normalize_recipient(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("policy_id")
    @classmethod
    def normalize_policy_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("requested_at")
    @classmethod
    def validate_requested_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("requested_at must be timezone-aware")
        return value.astimezone(UTC)


class EvaluationResponse(BaseModel):
    """
    Public adjudication result returned by Tex.

    This is the product surface:
    - final verdict
    - calibrated confidence and fused score
    - reasons and findings
    - uncertainty signals
    - policy/audit references

    decision_id must be supplied by the engine. The response should reflect the
    durable decision that was actually created, not invent a new identifier.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: UUID
    verdict: Verdict

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Tex's confidence in the final adjudication outcome.",
    )
    final_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fused risk score across deterministic, specialist, and semantic layers.",
    )

    reasons: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    uncertainty_flags: list[str] = Field(default_factory=list)

    policy_version: str = Field(min_length=1, max_length=100)
    evidence_hash: str | None = Field(default=None, min_length=1, max_length=128)

    evaluated_at: datetime

    @field_validator("reasons", "uncertainty_flags")
    @classmethod
    def normalize_string_list(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()

        for value in values:
            item = value.strip()
            if not item:
                raise ValueError("list entries must not be blank")
            if item not in seen:
                normalized.append(item)
                seen.add(item)

        return normalized

    @field_validator("scores")
    @classmethod
    def validate_scores(cls, values: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}

        for raw_key, raw_value in values.items():
            key = raw_key.strip()
            if not key:
                raise ValueError("score keys must not be blank")
            if not 0.0 <= raw_value <= 1.0:
                raise ValueError("score values must be between 0.0 and 1.0")
            normalized[key] = raw_value

        return normalized

    @field_validator("policy_version")
    @classmethod
    def normalize_policy_version(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("policy_version must not be blank")
        return normalized

    @field_validator("evidence_hash")
    @classmethod
    def normalize_evidence_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("evidence_hash must not be blank when provided")
        return normalized

    @field_validator("evaluated_at")
    @classmethod
    def validate_evaluated_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evaluated_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_verdict_consistency(self) -> EvaluationResponse:
        if self.verdict.requires_human_review and not self.uncertainty_flags:
            raise ValueError(
                "uncertainty_flags must be present when verdict is ABSTAIN"
            )
        return self

    @property
    def is_permit(self) -> bool:
        return self.verdict.allows_release

    @property
    def is_forbid(self) -> bool:
        return self.verdict.blocks_release

    @property
    def is_abstain(self) -> bool:
        return self.verdict.requires_human_review