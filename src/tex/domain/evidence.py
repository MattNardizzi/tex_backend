from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidenceRecord(BaseModel):
    """
    Append-only audit record for a Tex decision.

    This is the atomic unit written into the evidence log and hash chain.
    It is intentionally narrow and stable:
    - identifies the decision and request
    - captures the serialized payload being chained
    - stores the cryptographic linkage to the previous record
    - records when the entry was written

    The evidence layer should be tamper-evident, not overloaded with business
    logic. Rich decision semantics belong in the Decision model; this record is
    the durable audit envelope around that data.

    Verification of record_hash against payload_json + previous_hash belongs in
    the evidence chain layer, not here. That logic should be implemented in
    tex.evidence.chain so chain verification stays centralized and consistent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: UUID = Field(default_factory=uuid4)
    decision_id: UUID
    request_id: UUID

    record_type: str = Field(
        min_length=1,
        max_length=100,
        description="Stable type identifier such as 'decision' or 'outcome'.",
    )
    payload_json: str = Field(
        min_length=2,
        description="Canonical serialized JSON payload included in the evidence chain.",
    )

    payload_sha256: str = Field(
        min_length=64,
        max_length=64,
        description="SHA-256 hex digest of payload_json.",
    )
    previous_hash: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        description="Hash of the previous evidence record in the chain, if any.",
    )
    record_hash: str = Field(
        min_length=64,
        max_length=64,
        description="SHA-256 hex digest for this chained evidence record.",
    )

    policy_version: str = Field(
        min_length=1,
        max_length=100,
        description="Policy version active when the decision was made.",
    )

    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("record_type", "payload_json", "policy_version", mode="before")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Value must be a string.")
        normalized = value.strip()
        if not normalized:
            raise ValueError("Value must not be blank.")
        return normalized

    @field_validator("payload_sha256", "record_hash", mode="before")
    @classmethod
    def validate_required_sha256_hex(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("Hash value must be a string.")
        normalized = value.strip().lower()
        if len(normalized) != 64:
            raise ValueError("Hash values must be 64-character SHA-256 hex digests.")
        allowed = set("0123456789abcdef")
        if any(char not in allowed for char in normalized):
            raise ValueError(
                "Hash values must contain only lowercase hexadecimal characters."
            )
        return normalized

    @field_validator("previous_hash", mode="before")
    @classmethod
    def validate_optional_sha256_hex(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("previous_hash must be a string when provided.")
        normalized = value.strip().lower()
        if len(normalized) != 64:
            raise ValueError("previous_hash must be a 64-character SHA-256 hex digest.")
        allowed = set("0123456789abcdef")
        if any(char not in allowed for char in normalized):
            raise ValueError(
                "previous_hash must contain only lowercase hexadecimal characters."
            )
        return normalized