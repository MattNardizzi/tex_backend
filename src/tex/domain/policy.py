from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tex.domain.severity import Severity


_ALLOWED_FUSION_KEYS: frozenset[str] = frozenset(
    {
        "deterministic",
        "specialists",
        "semantic",
        "criticality",
    }
)

_DEFAULT_FUSION_WEIGHTS: dict[str, float] = {
    "deterministic": 0.30,
    "specialists": 0.25,
    "semantic": 0.35,
    "criticality": 0.10,
}


class PolicySnapshot(BaseModel):
    """
    Immutable, versioned policy configuration for Tex.

    A policy snapshot controls the tunable behavior of Tex's decision pipeline:
    - which deterministic recognizers are enabled
    - which finding severities hard-block immediately
    - retrieval depth and precedent lookback
    - specialist judge thresholds
    - context criticality by action, channel, and environment
    - fusion weights used by the orchestration layer

    `policy_id` is the stable identity of the policy family.
    `version` is the exact immutable snapshot used for a decision.

    Every decision should be traceable back to both:
    - the stable policy identity
    - the exact policy snapshot version
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_id: str = Field(
        min_length=1,
        max_length=255,
        description="Stable identifier for the policy family, such as 'default' or 'customer-outbound'.",
    )
    version: str = Field(
        min_length=1,
        max_length=100,
        description="Exact immutable version identifier for this policy snapshot.",
    )
    is_active: bool = Field(
        default=False,
        description="Whether this policy snapshot is currently active.",
    )

    permit_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Maximum fused risk score that still permits release.",
    )
    forbid_threshold: float = Field(
        ge=0.0,
        le=1.0,
        description="Minimum fused risk score that forbids release.",
    )
    minimum_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Minimum confidence required for an automatic PERMIT.",
    )

    deterministic_block_severities: tuple[Severity, ...] = Field(
        default=(Severity.CRITICAL,),
        description="Finding severities that trigger an immediate deterministic block.",
    )
    enabled_recognizers: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Names of deterministic recognizers enabled in layer one.",
    )
    blocked_terms: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Policy-specific blocked terms for deterministic matching.",
    )
    sensitive_entities: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Customer- or domain-specific sensitive entities to detect.",
    )

    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of retrieval items to gather for grounding.",
    )
    precedent_lookback_limit: int = Field(
        default=25,
        ge=1,
        le=500,
        description="Maximum number of historical precedents to consider.",
    )

    specialist_thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Per-specialist escalation thresholds keyed by specialist name.",
    )
    action_criticality: dict[str, float] = Field(
        default_factory=dict,
        description="Per-action criticality values between 0.0 and 1.0.",
    )
    channel_criticality: dict[str, float] = Field(
        default_factory=dict,
        description="Per-channel criticality values between 0.0 and 1.0.",
    )
    environment_criticality: dict[str, float] = Field(
        default_factory=dict,
        description="Per-environment criticality values between 0.0 and 1.0.",
    )

    fusion_weights: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_FUSION_WEIGHTS),
        description="Normalized weights used to fuse multi-layer signals.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional operator annotations and policy metadata.",
    )

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when the policy snapshot was created.",
    )

    @field_validator("policy_id", mode="before")
    @classmethod
    def normalize_policy_id(cls, value: Any) -> str:
        return _normalize_required_string(
            value=value,
            field_name="policy_id",
            lowercase=True,
        )

    @field_validator("version", mode="before")
    @classmethod
    def normalize_version(cls, value: Any) -> str:
        return _normalize_required_string(
            value=value,
            field_name="version",
            lowercase=False,
        )

    @field_validator(
        "enabled_recognizers",
        "blocked_terms",
        "sensitive_entities",
        mode="before",
    )
    @classmethod
    def normalize_string_tuple(cls, value: Any) -> tuple[str, ...]:
        return _normalize_string_sequence(value=value)

    @field_validator("deterministic_block_severities", mode="before")
    @classmethod
    def normalize_block_severities(cls, value: Any) -> tuple[Severity, ...]:
        if value is None:
            return (Severity.CRITICAL,)
        if isinstance(value, Severity):
            return (value,)
        if isinstance(value, str):
            raise TypeError(
                "deterministic_block_severities must be a sequence, not a plain string"
            )
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                "deterministic_block_severities must be a list or tuple"
            )

        normalized: list[Severity] = []
        seen: set[Severity] = set()

        for item in value:
            severity = _coerce_severity(item)
            if severity in seen:
                continue
            seen.add(severity)
            normalized.append(severity)

        if not normalized:
            raise ValueError("deterministic_block_severities must not be empty")

        return tuple(normalized)

    @field_validator(
        "specialist_thresholds",
        "action_criticality",
        "channel_criticality",
        "environment_criticality",
        mode="before",
    )
    @classmethod
    def normalize_score_mapping(cls, value: Any) -> dict[str, float]:
        return _normalize_float_mapping(
            value=value,
            field_name="mapping",
            allowed_keys=None,
        )

    @field_validator("fusion_weights", mode="before")
    @classmethod
    def normalize_fusion_weights(cls, value: Any) -> dict[str, float]:
        normalized = _normalize_float_mapping(
            value=value,
            field_name="fusion_weights",
            allowed_keys=_ALLOWED_FUSION_KEYS,
        )

        missing_keys = _ALLOWED_FUSION_KEYS.difference(normalized)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(
                f"fusion_weights is missing required keys: {missing}"
            )

        return normalized

    @field_validator("created_at", mode="after")
    @classmethod
    def enforce_timezone_aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_policy_snapshot(self) -> "PolicySnapshot":
        if self.permit_threshold >= self.forbid_threshold:
            raise ValueError("permit_threshold must be lower than forbid_threshold")

        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0.0 and 1.0")

        weight_sum = sum(self.fusion_weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError("fusion_weights must sum to 1.0")

        return self

    def criticality_for(
        self,
        *,
        action_type: str,
        channel: str,
        environment: str,
    ) -> float:
        """
        Returns the average policy-defined criticality for the evaluation context.

        Missing keys default to 0.0. This keeps policy lookup deterministic and
        avoids hidden fallback behavior elsewhere in the engine.
        """
        normalized_action_type = _normalize_lookup_key(action_type)
        normalized_channel = _normalize_lookup_key(channel)
        normalized_environment = _normalize_lookup_key(environment)

        action_score = self.action_criticality.get(normalized_action_type, 0.0)
        channel_score = self.channel_criticality.get(normalized_channel, 0.0)
        environment_score = self.environment_criticality.get(
            normalized_environment,
            0.0,
        )

        return (action_score + channel_score + environment_score) / 3.0

    def specialist_threshold_for(self, specialist_name: str) -> float | None:
        """
        Returns the configured escalation threshold for a specialist judge, if any.
        """
        return self.specialist_thresholds.get(
            _normalize_lookup_key(specialist_name)
        )

    def blocks_severity(self, severity: Severity) -> bool:
        """
        Returns True when the given finding severity is configured to hard-block.
        """
        return severity in self.deterministic_block_severities

    @property
    def ordered_fusion_weights(self) -> tuple[tuple[str, float], ...]:
        """
        Returns fusion weights in stable engine order.
        """
        return tuple(
            (key, self.fusion_weights[key])
            for key in sorted(_ALLOWED_FUSION_KEYS)
        )


def _normalize_required_string(
    *,
    value: Any,
    field_name: str,
    lowercase: bool = True,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")

    return normalized.casefold() if lowercase else normalized


def _normalize_string_sequence(*, value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()

    if isinstance(value, str):
        raise TypeError("sequence fields must not be plain strings")

    if not isinstance(value, (list, tuple)):
        raise TypeError("sequence fields must be a list or tuple")

    normalized_items: list[str] = []
    seen: set[str] = set()

    for item in value:
        if not isinstance(item, str):
            raise TypeError("sequence items must be strings")

        normalized = item.strip()
        if not normalized:
            raise ValueError("sequence items must not be blank")

        dedupe_key = normalized.casefold()
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        normalized_items.append(normalized)

    return tuple(normalized_items)


def _normalize_float_mapping(
    *,
    value: Any,
    field_name: str,
    allowed_keys: frozenset[str] | None,
) -> dict[str, float]:
    if value is None:
        return {}

    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dictionary")

    normalized_items: dict[str, float] = {}

    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise TypeError(f"{field_name} keys must be strings")

        key = raw_key.strip().casefold()
        if not key:
            raise ValueError(f"{field_name} keys must not be blank")

        if allowed_keys is not None and key not in allowed_keys:
            allowed = ", ".join(sorted(allowed_keys))
            raise ValueError(
                f"{field_name} contains unsupported key {key!r}; allowed keys: {allowed}"
            )

        if not isinstance(raw_value, (int, float)):
            raise TypeError(f"{field_name} values must be numeric")

        numeric_value = float(raw_value)
        if not 0.0 <= numeric_value <= 1.0:
            raise ValueError(
                f"value for {key!r} must be between 0.0 and 1.0"
            )

        normalized_items[key] = numeric_value

    return normalized_items


def _coerce_severity(value: Any) -> Severity:
    if isinstance(value, Severity):
        return value

    if not isinstance(value, str):
        raise TypeError("severity values must be Severity members or strings")

    normalized = value.strip()
    if not normalized:
        raise ValueError("severity values must not be blank")

    try:
        return Severity(normalized.upper())
    except ValueError:
        allowed = ", ".join(member.value for member in Severity)
        raise ValueError(
            f"unsupported severity {normalized!r}; allowed values: {allowed}"
        ) from None


def _normalize_lookup_key(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("lookup value must be a string")

    normalized = value.strip()
    if not normalized:
        raise ValueError("lookup value must not be blank")

    return normalized.casefold()