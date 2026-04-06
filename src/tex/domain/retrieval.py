from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tex.domain.verdict import Verdict


class RetrievedPolicyClause(BaseModel):
    """
    A policy clause retrieved as grounding context for a single evaluation.

    This is intentionally lightweight and immutable. Retrieval quality can
    improve later without changing the domain contract.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    clause_id: str = Field(min_length=1, max_length=255)
    policy_id: str = Field(min_length=1, max_length=255)
    policy_version: str = Field(min_length=1, max_length=100)

    title: str | None = Field(default=None, min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=10_000)

    channel: str | None = Field(default=None, min_length=1, max_length=100)
    action_type: str | None = Field(default=None, min_length=1, max_length=100)

    relevance_score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "clause_id",
        "policy_id",
        "policy_version",
        "title",
        "text",
        "channel",
        "action_type",
        mode="before",
    )
    @classmethod
    def normalize_string_fields(cls, value: Any) -> Any:
        return _normalize_optional_string(value)

    @property
    def scope_key(self) -> tuple[str | None, str | None]:
        """Stable retrieval scope key used for grouping or diagnostics."""
        return (self.channel, self.action_type)


class RetrievedPrecedent(BaseModel):
    """
    A similar prior decision retrieved to provide case-aware grounding.

    This is not the full historical decision object. It is the minimum,
    retrieval-safe summary needed during adjudication.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str = Field(min_length=1, max_length=255)
    request_id: str = Field(min_length=1, max_length=255)

    verdict: Verdict
    action_type: str | None = Field(default=None, min_length=1, max_length=100)
    channel: str | None = Field(default=None, min_length=1, max_length=100)
    environment: str | None = Field(default=None, min_length=1, max_length=100)

    content_excerpt: str | None = Field(default=None, min_length=1, max_length=2_000)
    reasons: tuple[str, ...] = Field(default_factory=tuple)
    matched_policy_clause_ids: tuple[str, ...] = Field(default_factory=tuple)
    uncertainty_flags: tuple[str, ...] = Field(default_factory=tuple)

    relevance_score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)

    decided_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "decision_id",
        "request_id",
        "action_type",
        "channel",
        "environment",
        "content_excerpt",
        mode="before",
    )
    @classmethod
    def normalize_string_fields(cls, value: Any) -> Any:
        return _normalize_optional_string(value)

    @field_validator("verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, value: Any) -> Verdict:
        if isinstance(value, Verdict):
            return value
        if not isinstance(value, str):
            raise TypeError("verdict must be a Verdict or string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("verdict must not be blank")
        try:
            return Verdict(normalized.upper())
        except ValueError as exc:
            allowed = ", ".join(member.value for member in Verdict)
            raise ValueError(f"unsupported verdict {normalized!r}; allowed values: {allowed}") from exc

    @field_validator("reasons", "matched_policy_clause_ids", "uncertainty_flags", mode="before")
    @classmethod
    def normalize_string_sequences(cls, value: Any) -> tuple[str, ...]:
        return _normalize_string_sequence(value, dedupe=True)

    @field_validator("decided_at", mode="after")
    @classmethod
    def enforce_timezone_aware_decided_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decided_at must be timezone-aware")
        return value.astimezone(UTC)

    @property
    def matched_clause_count(self) -> int:
        return len(self.matched_policy_clause_ids)


class RetrievedEntity(BaseModel):
    """
    A customer- or domain-specific sensitive entity relevant to evaluation.

    Examples:
    - customer name
    - internal product codename
    - restricted pricing term
    - regulated dataset or account identifier
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: str = Field(min_length=1, max_length=255)
    entity_type: str = Field(min_length=1, max_length=100)
    canonical_name: str = Field(min_length=1, max_length=300)

    aliases: tuple[str, ...] = Field(default_factory=tuple)
    sensitivity: str = Field(min_length=1, max_length=100)
    description: str | None = Field(default=None, min_length=1, max_length=1_000)

    relevance_score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "entity_id",
        "entity_type",
        "canonical_name",
        "sensitivity",
        "description",
        mode="before",
    )
    @classmethod
    def normalize_string_fields(cls, value: Any) -> Any:
        return _normalize_optional_string(value)

    @field_validator("aliases", mode="before")
    @classmethod
    def normalize_aliases(cls, value: Any) -> tuple[str, ...]:
        return _normalize_string_sequence(value, dedupe=True)

    @property
    def all_names(self) -> tuple[str, ...]:
        """Returns canonical name followed by aliases."""
        return (self.canonical_name, *self.aliases)


class RetrievalContext(BaseModel):
    """
    Grounding context assembled before semantic adjudication.

    The retrieval layer may return an empty context early on. That is valid.
    The point of this model is to lock the contract now so the rest of the
    system can be built against it cleanly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_clauses: tuple[RetrievedPolicyClause, ...] = Field(default_factory=tuple)
    precedents: tuple[RetrievedPrecedent, ...] = Field(default_factory=tuple)
    entities: tuple[RetrievedEntity, ...] = Field(default_factory=tuple)

    retrieval_warnings: tuple[str, ...] = Field(default_factory=tuple)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("policy_clauses", "precedents", "entities", mode="before")
    @classmethod
    def normalize_collection_fields(cls, value: Any) -> tuple[Any, ...]:
        return _normalize_tuple_collection(value)

    @field_validator("retrieval_warnings", mode="before")
    @classmethod
    def normalize_retrieval_warnings(cls, value: Any) -> tuple[str, ...]:
        return _normalize_string_sequence(value, dedupe=True)

    @field_validator("retrieved_at", mode="after")
    @classmethod
    def enforce_timezone_aware_retrieved_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_unique_ranking_and_ids(self) -> "RetrievalContext":
        _validate_unique_ranks(items=self.policy_clauses, label="policy_clauses")
        _validate_unique_ranks(items=self.precedents, label="precedents")
        _validate_unique_ranks(items=self.entities, label="entities")

        _validate_unique_ids(
            values=[item.clause_id for item in self.policy_clauses],
            label="policy clause ids",
        )
        _validate_unique_ids(
            values=[item.decision_id for item in self.precedents],
            label="precedent decision ids",
        )
        _validate_unique_ids(
            values=[item.entity_id for item in self.entities],
            label="entity ids",
        )

        return self

    @property
    def is_empty(self) -> bool:
        """Returns True when no retrieval results were available."""
        return not self.policy_clauses and not self.precedents and not self.entities

    @property
    def matched_policy_clause_ids(self) -> tuple[str, ...]:
        """Returns retrieved policy clause IDs in rank order."""
        return tuple(clause.clause_id for clause in self.policy_clauses)

    @property
    def matched_entity_names(self) -> tuple[str, ...]:
        """Returns retrieved canonical entity names in rank order."""
        return tuple(entity.canonical_name for entity in self.entities)

    @property
    def all_entity_names(self) -> tuple[str, ...]:
        """Returns canonical and alias entity names in stable deduped order."""
        seen: set[str] = set()
        ordered: list[str] = []

        for entity in self.entities:
            for name in entity.all_names:
                dedupe_key = name.casefold()
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                ordered.append(name)

        return tuple(ordered)

    @property
    def highest_policy_relevance(self) -> float:
        return max((clause.relevance_score for clause in self.policy_clauses), default=0.0)

    @property
    def highest_precedent_relevance(self) -> float:
        return max((precedent.relevance_score for precedent in self.precedents), default=0.0)

    @property
    def highest_entity_relevance(self) -> float:
        return max((entity.relevance_score for entity in self.entities), default=0.0)

    @classmethod
    def empty(
        cls,
        *,
        warning: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "RetrievalContext":
        """
        Convenience constructor for the initial no-op retrieval phase.
        """
        warnings = tuple()
        if warning is not None:
            normalized_warning = _normalize_optional_string(warning)
            if normalized_warning is not None:
                warnings = (normalized_warning,)

        return cls(
            policy_clauses=tuple(),
            precedents=tuple(),
            entities=tuple(),
            retrieval_warnings=warnings,
            metadata=dict(metadata) if metadata is not None else {},
        )


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("string fields must not be blank")
    return normalized


def _normalize_string_sequence(value: Any, *, dedupe: bool) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        raise TypeError("sequence fields must not be plain strings")
    if not isinstance(value, (list, tuple)):
        raise TypeError("sequence fields must be lists or tuples")

    normalized_items: list[str] = []
    seen: set[str] = set()

    for item in value:
        if not isinstance(item, str):
            raise TypeError("sequence items must be strings")
        normalized = item.strip()
        if not normalized:
            raise ValueError("sequence items must not be blank")

        if dedupe:
            dedupe_key = normalized.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

        normalized_items.append(normalized)

    return tuple(normalized_items)


def _normalize_tuple_collection(value: Any) -> tuple[Any, ...]:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError("retrieval collections must be lists or tuples")


def _validate_unique_ranks(*, items: tuple[Any, ...], label: str) -> None:
    seen: set[int] = set()
    for item in items:
        if item.rank in seen:
            raise ValueError(f"{label} must not contain duplicate rank values")
        seen.add(item.rank)


def _validate_unique_ids(*, values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")