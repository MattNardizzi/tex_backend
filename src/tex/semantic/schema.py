from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Final, Iterable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tex.domain.verdict import Verdict

_ALLOWED_DIMENSIONS: Final[tuple[str, ...]] = (
    "policy_compliance",
    "data_leakage",
    "external_sharing",
    "unauthorized_commitment",
    "destructive_or_bypass",
)

_ALLOWED_DIMENSION_SET: Final[frozenset[str]] = frozenset(_ALLOWED_DIMENSIONS)


def _normalize_non_blank_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _normalize_string_tuple(
    value: Any,
    *,
    field_name: str,
    dedupe_casefold: bool = True,
) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        raise TypeError(f"{field_name} must be a sequence of strings, not a plain string")
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise TypeError(f"{field_name} must be a list, tuple, or set")

    normalized_items: list[str] = []
    seen: set[str] = set()

    for item in value:
        normalized = _normalize_non_blank_string(item, field_name=field_name)
        assert normalized is not None
        dedupe_key = normalized.casefold() if dedupe_casefold else normalized
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        normalized_items.append(normalized)

    return tuple(normalized_items)


def _normalize_tuple_input(value: Any, *, field_name: str) -> tuple[Any, ...]:
    if value is None:
        return tuple()
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, tuple):
        return value
    raise TypeError(f"{field_name} must be a list or tuple")


def _dedupe_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()

    for value in values:
        dedupe_key = value.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        ordered.append(value)

    return tuple(ordered)


class SemanticEvidenceSpan(BaseModel):
    """
    Concrete supporting evidence for one semantic finding.

    Character offsets are optional because not every provider can return exact
    indexes reliably. When offsets are present, both must be supplied and form a
    valid half-open range: [start_index, end_index).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1, max_length=2_000)
    start_index: int | None = Field(default=None, ge=0)
    end_index: int | None = Field(default=None, ge=0)
    explanation: str | None = Field(default=None, min_length=1, max_length=1_000)

    @field_validator("text", "explanation", mode="before")
    @classmethod
    def normalize_text_fields(cls, value: Any, info: Any) -> Any:
        return _normalize_non_blank_string(value, field_name=info.field_name)

    @model_validator(mode="after")
    def validate_indexes(self) -> "SemanticEvidenceSpan":
        if self.start_index is None and self.end_index is None:
            return self
        if self.start_index is None or self.end_index is None:
            raise ValueError("start_index and end_index must both be provided together")
        if self.end_index <= self.start_index:
            raise ValueError("end_index must be greater than start_index")
        return self


class SemanticDimensionResult(BaseModel):
    """
    Semantic assessment for one canonical Tex risk dimension.

    `score` is the estimated risk severity for the dimension.
    `confidence` is the semantic layer's confidence in that estimate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension: str = Field(min_length=1, max_length=100)
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=1_500)
    rationale: str | None = Field(default=None, min_length=1, max_length=2_000)
    evidence_spans: tuple[SemanticEvidenceSpan, ...] = Field(default_factory=tuple)
    matched_policy_clause_ids: tuple[str, ...] = Field(default_factory=tuple)
    uncertainty_flags: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("dimension", "summary", "rationale", mode="before")
    @classmethod
    def normalize_string_fields(cls, value: Any, info: Any) -> Any:
        return _normalize_non_blank_string(value, field_name=info.field_name)

    @field_validator("dimension", mode="after")
    @classmethod
    def validate_dimension(cls, value: str) -> str:
        if value not in _ALLOWED_DIMENSION_SET:
            allowed = ", ".join(_ALLOWED_DIMENSIONS)
            raise ValueError(f"dimension must be one of: {allowed}")
        return value

    @field_validator("evidence_spans", mode="before")
    @classmethod
    def normalize_evidence_spans(cls, value: Any) -> tuple[Any, ...]:
        return _normalize_tuple_input(value, field_name="evidence_spans")

    @field_validator("matched_policy_clause_ids", "uncertainty_flags", mode="before")
    @classmethod
    def normalize_string_sequences(cls, value: Any, info: Any) -> tuple[str, ...]:
        return _normalize_string_tuple(value, field_name=info.field_name)

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence_spans)

    @property
    def is_high_risk(self) -> bool:
        return self.score >= 0.8

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < 0.5


class SemanticVerdictRecommendation(BaseModel):
    """
    Advisory verdict produced by the semantic layer before downstream routing.

    The router still owns the final verdict. This object only captures the
    semantic layer's structured recommendation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=1_500)
    rationale: str | None = Field(default=None, min_length=1, max_length=2_000)
    uncertainty_flags: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("summary", "rationale", mode="before")
    @classmethod
    def normalize_string_fields(cls, value: Any, info: Any) -> Any:
        return _normalize_non_blank_string(value, field_name=info.field_name)

    @field_validator("uncertainty_flags", mode="before")
    @classmethod
    def normalize_uncertainty_flags(cls, value: Any) -> tuple[str, ...]:
        return _normalize_string_tuple(value, field_name="uncertainty_flags")


class SemanticAnalysis(BaseModel):
    """
    Schema-locked semantic output contract for Tex.

    This is the boundary object between the semantic model and the rest of the
    decision pipeline. The semantic provider can be sophisticated; its emitted
    shape cannot.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension_results: tuple[SemanticDimensionResult, ...]
    recommended_verdict: SemanticVerdictRecommendation
    overall_confidence: float = Field(ge=0.0, le=1.0)
    evidence_sufficiency: float = Field(ge=0.0, le=1.0)
    rationale_quality: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=2_000)
    uncertainty_flags: tuple[str, ...] = Field(default_factory=tuple)
    provider_name: str | None = Field(default=None, min_length=1, max_length=200)
    model_name: str | None = Field(default=None, min_length=1, max_length=200)
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dimension_results", mode="before")
    @classmethod
    def normalize_dimension_results(cls, value: Any) -> tuple[Any, ...]:
        return _normalize_tuple_input(value, field_name="dimension_results")

    @field_validator("summary", "provider_name", "model_name", mode="before")
    @classmethod
    def normalize_string_fields(cls, value: Any, info: Any) -> Any:
        return _normalize_non_blank_string(value, field_name=info.field_name)

    @field_validator("uncertainty_flags", mode="before")
    @classmethod
    def normalize_uncertainty_flags(cls, value: Any) -> tuple[str, ...]:
        return _normalize_string_tuple(value, field_name="uncertainty_flags")

    @field_validator("metadata", mode="before")
    @classmethod
    def normalize_metadata(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("metadata must be a dictionary")
        return dict(value)

    @field_validator("analyzed_at", mode="after")
    @classmethod
    def enforce_timezone_aware_analyzed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("analyzed_at must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_dimension_coverage(self) -> "SemanticAnalysis":
        if len(self.dimension_results) != len(_ALLOWED_DIMENSIONS):
            raise ValueError(
                "dimension_results must contain each semantic dimension exactly once"
            )

        seen_dimensions: list[str] = [result.dimension for result in self.dimension_results]
        unique_dimensions = set(seen_dimensions)

        if len(unique_dimensions) != len(seen_dimensions):
            duplicates = sorted(
                {
                    dimension
                    for dimension in seen_dimensions
                    if seen_dimensions.count(dimension) > 1
                }
            )
            raise ValueError(
                "dimension_results must not contain duplicate dimensions: "
                + ", ".join(duplicates)
            )

        missing = [dimension for dimension in _ALLOWED_DIMENSIONS if dimension not in unique_dimensions]
        unexpected = sorted(unique_dimensions - _ALLOWED_DIMENSION_SET)

        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing dimensions: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected dimensions: {', '.join(unexpected)}")
            raise ValueError(
                "dimension_results must cover the canonical Tex dimensions exactly; "
                + "; ".join(details)
            )

        return self

    @property
    def dimension_scores(self) -> dict[str, float]:
        return {result.dimension: result.score for result in self.dimension_results}

    @property
    def dimension_confidences(self) -> dict[str, float]:
        return {result.dimension: result.confidence for result in self.dimension_results}

    @property
    def dimension_result_by_name(self) -> dict[str, SemanticDimensionResult]:
        return {result.dimension: result for result in self.dimension_results}

    @property
    def matched_policy_clause_ids(self) -> tuple[str, ...]:
        return _dedupe_preserve_order(
            clause_id
            for result in self.dimension_results
            for clause_id in result.matched_policy_clause_ids
        )

    @property
    def all_uncertainty_flags(self) -> tuple[str, ...]:
        return _dedupe_preserve_order(
            (
                list(self.uncertainty_flags)
                + [
                    flag
                    for result in self.dimension_results
                    for flag in result.uncertainty_flags
                ]
                + list(self.recommended_verdict.uncertainty_flags)
            )
        )

    @property
    def max_dimension_score(self) -> float:
        return max(result.score for result in self.dimension_results)

    @property
    def min_dimension_confidence(self) -> float:
        return min(result.confidence for result in self.dimension_results)

    @property
    def has_low_confidence_dimension(self) -> bool:
        return any(result.is_low_confidence for result in self.dimension_results)

    @property
    def has_any_evidence(self) -> bool:
        return any(result.has_evidence for result in self.dimension_results)

    @property
    def all_evidence_spans(self) -> tuple[SemanticEvidenceSpan, ...]:
        return tuple(
            span
            for result in self.dimension_results
            for span in result.evidence_spans
        )

    @property
    def high_risk_dimensions(self) -> tuple[str, ...]:
        return tuple(
            result.dimension
            for result in self.dimension_results
            if result.is_high_risk
        )

    @property
    def low_confidence_dimensions(self) -> tuple[str, ...]:
        return tuple(
            result.dimension
            for result in self.dimension_results
            if result.is_low_confidence
        )

class SemanticAnalysisParseTarget(BaseModel):
    """
    Slim parse target for OpenAI structured output.

    This excludes runtime-populated fields (metadata, provider_name,
    model_name, analyzed_at) that OpenAI's strict schema cannot handle
    because they use open-ended dict[str, Any].

    The OpenAI provider converts this into a full SemanticAnalysis
    after parsing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    dimension_results: tuple[SemanticDimensionResult, ...]
    recommended_verdict: SemanticVerdictRecommendation
    overall_confidence: float = Field(ge=0.0, le=1.0)
    evidence_sufficiency: float = Field(ge=0.0, le=1.0)
    rationale_quality: float = Field(ge=0.0, le=1.0)
    summary: str = Field(min_length=1, max_length=2_000)
    uncertainty_flags: tuple[str, ...] = Field(default_factory=tuple)

    @field_validator("dimension_results", mode="before")
    @classmethod
    def normalize_dimension_results(cls, value: Any) -> tuple[Any, ...]:
        return _normalize_tuple_input(value, field_name="dimension_results")

    @field_validator("summary", mode="before")
    @classmethod
    def normalize_string_fields(cls, value: Any, info: Any) -> Any:
        return _normalize_non_blank_string(value, field_name=info.field_name)

    @field_validator("uncertainty_flags", mode="before")
    @classmethod
    def normalize_uncertainty_flags(cls, value: Any) -> tuple[str, ...]:
        return _normalize_string_tuple(value, field_name="uncertainty_flags")

    @model_validator(mode="after")
    def validate_dimension_coverage(self) -> "SemanticAnalysisParseTarget":
        if len(self.dimension_results) != len(_ALLOWED_DIMENSIONS):
            raise ValueError(
                "dimension_results must contain each semantic dimension exactly once"
            )

        seen_dimensions: list[str] = [result.dimension for result in self.dimension_results]
        unique_dimensions = set(seen_dimensions)

        if len(unique_dimensions) != len(seen_dimensions):
            duplicates = sorted(
                {
                    dimension
                    for dimension in seen_dimensions
                    if seen_dimensions.count(dimension) > 1
                }
            )
            raise ValueError(
                "dimension_results must not contain duplicate dimensions: "
                + ", ".join(duplicates)
            )

        missing = [dimension for dimension in _ALLOWED_DIMENSIONS if dimension not in unique_dimensions]
        unexpected = sorted(unique_dimensions - _ALLOWED_DIMENSION_SET)

        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing dimensions: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected dimensions: {', '.join(unexpected)}")
            raise ValueError(
                "dimension_results must cover the canonical Tex dimensions exactly; "
                + "; ".join(details)
            )

        return self

    def to_full_analysis(
        self,
        *,
        provider_name: str | None = None,
        model_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SemanticAnalysis:
        """Convert to a full SemanticAnalysis with runtime fields."""
        return SemanticAnalysis(
            dimension_results=self.dimension_results,
            recommended_verdict=self.recommended_verdict,
            overall_confidence=self.overall_confidence,
            evidence_sufficiency=self.evidence_sufficiency,
            rationale_quality=self.rationale_quality,
            summary=self.summary,
            uncertainty_flags=self.uncertainty_flags,
            provider_name=provider_name,
            model_name=model_name,
            metadata=metadata or {},
        )

def semantic_dimensions() -> tuple[str, ...]:
    """Return the canonical semantic risk dimensions in stable order."""
    return _ALLOWED_DIMENSIONS
