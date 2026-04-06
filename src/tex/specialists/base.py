from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tex.domain.evaluation import EvaluationRequest
from tex.domain.retrieval import RetrievalContext


class SpecialistEvidence(BaseModel):
    """
    Evidence emitted by a specialist judge.

    Specialists are narrow detectors, not final decision-makers. Their evidence
    should stay compact, explicit, and easy to fuse into the main decision.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1, max_length=2_000)
    start_index: int | None = Field(default=None, ge=0)
    end_index: int | None = Field(default=None, ge=0)
    explanation: str | None = Field(default=None, min_length=1, max_length=1_000)

    @field_validator("text", "explanation", mode="before")
    @classmethod
    def normalize_string_fields(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("string fields must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_indexes(self) -> "SpecialistEvidence":
        if self.start_index is None and self.end_index is None:
            return self
        if self.start_index is None or self.end_index is None:
            raise ValueError("start_index and end_index must both be provided together")
        if self.end_index <= self.start_index:
            raise ValueError("end_index must be greater than start_index")
        return self


class SpecialistResult(BaseModel):
    """
    Structured output from a single specialist judge.

    A specialist result is intentionally advisory. It contributes a narrow,
    high-signal slice of risk to the main engine, but it does not own the
    final verdict.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    specialist_name: str = Field(min_length=1, max_length=100)
    risk_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)

    summary: str = Field(min_length=1, max_length=1_500)
    rationale: str | None = Field(default=None, min_length=1, max_length=2_000)

    evidence: tuple[SpecialistEvidence, ...] = Field(default_factory=tuple)
    matched_policy_clause_ids: tuple[str, ...] = Field(default_factory=tuple)
    matched_entity_names: tuple[str, ...] = Field(default_factory=tuple)
    uncertainty_flags: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("specialist_name", "summary", "rationale", mode="before")
    @classmethod
    def normalize_string_fields(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        normalized = value.strip()
        if not normalized:
            raise ValueError("string fields must not be blank")
        return normalized

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_evidence(cls, value: object) -> tuple[object, ...]:
        if value is None:
            return tuple()
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        raise TypeError("evidence must be a list or tuple")

    @field_validator(
        "matched_policy_clause_ids",
        "matched_entity_names",
        "uncertainty_flags",
        mode="before",
    )
    @classmethod
    def normalize_string_sequences(cls, value: object) -> tuple[str, ...]:
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
            dedupe_key = normalized.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized_items.append(normalized)

        return tuple(normalized_items)

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)

    @property
    def should_escalate(self) -> bool:
        return self.risk_score >= 0.5 or self.confidence < 0.5


class SpecialistBundle(BaseModel):
    """
    Aggregated output from all executed specialist judges.

    This gives the router/PDP a stable contract for specialist-layer fusion.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    results: tuple[SpecialistResult, ...] = Field(default_factory=tuple)

    @field_validator("results", mode="before")
    @classmethod
    def normalize_results(cls, value: object) -> tuple[object, ...]:
        if value is None:
            return tuple()
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        raise TypeError("results must be a list or tuple")

    @model_validator(mode="after")
    def validate_unique_specialist_names(self) -> "SpecialistBundle":
        names = [result.specialist_name for result in self.results]
        if len(names) != len(set(names)):
            raise ValueError("specialist names must be unique within a bundle")
        return self

    @property
    def is_empty(self) -> bool:
        return not self.results

    @property
    def max_risk_score(self) -> float:
        return max((result.risk_score for result in self.results), default=0.0)

    @property
    def min_confidence(self) -> float:
        return min((result.confidence for result in self.results), default=0.0)

    @property
    def matched_policy_clause_ids(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []

        for result in self.results:
            for clause_id in result.matched_policy_clause_ids:
                key = clause_id.casefold()
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(clause_id)

        return tuple(ordered)

    @property
    def matched_entity_names(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []

        for result in self.results:
            for entity_name in result.matched_entity_names:
                key = entity_name.casefold()
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(entity_name)

        return tuple(ordered)

    @property
    def uncertainty_flags(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []

        for result in self.results:
            for flag in result.uncertainty_flags:
                key = flag.casefold()
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(flag)

        return tuple(ordered)

    @classmethod
    def empty(cls) -> "SpecialistBundle":
        return cls(results=tuple())


@runtime_checkable
class SpecialistJudge(Protocol):
    """
    Contract for a narrow specialist judge.

    Specialists focus on one risk slice, such as:
    - PII / secrets
    - external sharing / exfiltration
    - unauthorized commitment
    - destructive or bypass language
    """

    name: str

    def evaluate(
        self,
        *,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
    ) -> SpecialistResult:
        """Returns a structured, schema-valid specialist result."""