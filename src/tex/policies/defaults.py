from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from tex.domain.policy import PolicySnapshot
from tex.domain.severity import Severity

DEFAULT_POLICY_ID = "default"
STRICT_POLICY_ID = "strict"

DEFAULT_POLICY_VERSION = "default-v1"
STRICT_POLICY_VERSION = "strict-v1"

_DEFAULT_ENABLED_RECOGNIZERS: tuple[str, ...] = (
    "blocked_terms",
    "sensitive_entities",
    "secret_leak",
    "pii",
    "unauthorized_commitment",
    "external_sharing",
    "destructive_or_bypass",
)

_DEFAULT_BLOCK_SEVERITIES: tuple[Severity, ...] = (Severity.CRITICAL,)

_DEFAULT_BLOCKED_TERMS: tuple[str, ...] = (
    "ignore policy",
    "skip approval",
    "skip review",
    "disable logging",
    "remove audit",
    "drop table",
    "public link to customer data",
)

_DEFAULT_SENSITIVE_ENTITIES: tuple[str, ...] = (
    "pricing sheet",
    "customer list",
    "internal roadmap",
    "production credentials",
    "security questionnaire",
)

_DEFAULT_ACTION_CRITICALITY: dict[str, float] = {
    "email_send": 0.45,
    "sales_email": 0.55,
    "slack_message": 0.25,
    "external_message": 0.55,
    "api_response": 0.30,
    "api_export": 0.70,
    "file_export": 0.72,
    "document_share": 0.62,
    "crm_update": 0.35,
    "approval_response": 0.60,
    "workflow_instruction": 0.65,
    "admin_command": 0.82,
    "sql_execution": 0.88,
}

_DEFAULT_CHANNEL_CRITICALITY: dict[str, float] = {
    "email": 0.45,
    "sales_email": 0.55,
    "slack": 0.20,
    "teams": 0.20,
    "api": 0.30,
    "webhook": 0.45,
    "export": 0.70,
    "console": 0.80,
}

_DEFAULT_ENVIRONMENT_CRITICALITY: dict[str, float] = {
    "dev": 0.10,
    "development": 0.10,
    "test": 0.15,
    "staging": 0.35,
    "prod": 0.75,
    "production": 0.75,
}

_DEFAULT_SPECIALIST_THRESHOLDS: dict[str, float] = {
    "secret_and_pii": 0.55,
    "external_sharing": 0.58,
    "unauthorized_commitment": 0.60,
    "destructive_or_bypass": 0.50,
}

_STRICT_SPECIALIST_THRESHOLDS: dict[str, float] = {
    "secret_and_pii": 0.48,
    "external_sharing": 0.52,
    "unauthorized_commitment": 0.55,
    "destructive_or_bypass": 0.45,
}

_DEFAULT_FUSION_WEIGHTS: dict[str, float] = {
    "deterministic": 0.30,
    "specialists": 0.25,
    "semantic": 0.35,
    "criticality": 0.10,
}

_STRICT_FUSION_WEIGHTS: dict[str, float] = {
    "deterministic": 0.32,
    "specialists": 0.26,
    "semantic": 0.32,
    "criticality": 0.10,
}

_DEFAULT_METADATA: dict[str, Any] = {
    "policy_name": "Tex Default Policy",
    "description": (
        "Lean default policy for local development and early product validation. "
        "It is intentionally conservative on destructive actions, disclosure risk, "
        "and unauthorized commitments."
    ),
    "owner": "tex",
    "mode": "default",
}

_STRICT_METADATA: dict[str, Any] = {
    **_DEFAULT_METADATA,
    "policy_name": "Tex Strict Policy",
    "mode": "strict",
}

_DEFAULT_PROFILE: dict[str, Any] = {
    "permit_threshold": 0.34,
    "forbid_threshold": 0.72,
    "minimum_confidence": 0.62,
    "retrieval_top_k": 5,
    "precedent_lookback_limit": 25,
    "specialist_thresholds": _DEFAULT_SPECIALIST_THRESHOLDS,
    "fusion_weights": _DEFAULT_FUSION_WEIGHTS,
}

_STRICT_PROFILE: dict[str, Any] = {
    "permit_threshold": 0.28,
    "forbid_threshold": 0.64,
    "minimum_confidence": 0.70,
    "retrieval_top_k": 7,
    "precedent_lookback_limit": 40,
    "specialist_thresholds": _STRICT_SPECIALIST_THRESHOLDS,
    "fusion_weights": _STRICT_FUSION_WEIGHTS,
}


def default_policy_snapshot(
    *,
    policy_id: str = DEFAULT_POLICY_ID,
    version: str = DEFAULT_POLICY_VERSION,
    is_active: bool = True,
    created_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> PolicySnapshot:
    """
    Returns Tex's default policy snapshot.

    This policy is intentionally conservative enough to make Tex useful
    immediately without pretending calibration is already mature.
    """
    return _build_policy_snapshot(
        policy_id=policy_id,
        version=version,
        is_active=is_active,
        created_at=created_at,
        metadata=_merge_metadata(_DEFAULT_METADATA, metadata),
        profile=_DEFAULT_PROFILE,
    )


def strict_policy_snapshot(
    *,
    policy_id: str = STRICT_POLICY_ID,
    version: str = STRICT_POLICY_VERSION,
    is_active: bool = False,
    created_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> PolicySnapshot:
    """
    Returns a stricter policy snapshot for higher-risk environments.

    This is the same policy shape with tighter thresholds and more aggressive
    specialist escalation, not a separate architecture.
    """
    return _build_policy_snapshot(
        policy_id=policy_id,
        version=version,
        is_active=is_active,
        created_at=created_at,
        metadata=_merge_metadata(_STRICT_METADATA, metadata),
        profile=_STRICT_PROFILE,
    )


def build_default_policy() -> PolicySnapshot:
    """Convenience constructor for the active default policy."""
    return default_policy_snapshot()


def build_strict_policy() -> PolicySnapshot:
    """Convenience constructor for the inactive strict policy."""
    return strict_policy_snapshot()


def _build_policy_snapshot(
    *,
    policy_id: str,
    version: str,
    is_active: bool,
    created_at: datetime | None,
    metadata: dict[str, Any],
    profile: dict[str, Any],
) -> PolicySnapshot:
    return PolicySnapshot(
        policy_id=policy_id,
        version=version,
        is_active=is_active,
        permit_threshold=float(profile["permit_threshold"]),
        forbid_threshold=float(profile["forbid_threshold"]),
        minimum_confidence=float(profile["minimum_confidence"]),
        deterministic_block_severities=_DEFAULT_BLOCK_SEVERITIES,
        enabled_recognizers=_DEFAULT_ENABLED_RECOGNIZERS,
        blocked_terms=_DEFAULT_BLOCKED_TERMS,
        sensitive_entities=_DEFAULT_SENSITIVE_ENTITIES,
        retrieval_top_k=int(profile["retrieval_top_k"]),
        precedent_lookback_limit=int(profile["precedent_lookback_limit"]),
        specialist_thresholds=dict(profile["specialist_thresholds"]),
        action_criticality=dict(_DEFAULT_ACTION_CRITICALITY),
        channel_criticality=dict(_DEFAULT_CHANNEL_CRITICALITY),
        environment_criticality=dict(_DEFAULT_ENVIRONMENT_CRITICALITY),
        fusion_weights=dict(profile["fusion_weights"]),
        metadata=dict(metadata),
        created_at=created_at or datetime.now(UTC),
    )


def _merge_metadata(
    base: dict[str, Any],
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(base)
    if overrides:
        merged.update(overrides)
    return merged