from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import ValidationError

from tex.config import get_settings
from tex.domain.evaluation import EvaluationRequest
from tex.domain.retrieval import RetrievalContext
from tex.semantic.fallback import HeuristicSemanticFallback, SemanticFallbackAnalyzer
from tex.semantic.prompt import (
    build_semantic_system_prompt,
    build_semantic_user_prompt,
    semantic_prompt_bundle,
)
from tex.semantic.schema import SemanticAnalysis


class SemanticProviderError(RuntimeError):
    """Raised when a configured semantic provider cannot return a valid result."""


class SemanticExecutionMode(str, Enum):
    """
    Execution mode for one semantic analysis pass.

    - PRIMARY_PROVIDER: the configured provider returned a valid analysis
    - DEFAULT_FALLBACK: no provider was configured, so fallback was the primary path
    - FAILURE_FALLBACK: provider execution failed and fallback was used instead
    """

    PRIMARY_PROVIDER = "primary_provider"
    DEFAULT_FALLBACK = "default_fallback"
    FAILURE_FALLBACK = "failure_fallback"


@dataclass(frozen=True, slots=True)
class SemanticExecutionTrace:
    """
    Audit-friendly trace of one semantic analysis execution.

    This stays intentionally small. It captures what happened at runtime
    without bloating the core semantic schema or leaking full prompt contents.
    """

    mode: SemanticExecutionMode
    provider_error: str | None
    system_prompt_sha256: str
    user_prompt_sha256: str

    @property
    def used_fallback(self) -> bool:
        return self.mode in {
            SemanticExecutionMode.DEFAULT_FALLBACK,
            SemanticExecutionMode.FAILURE_FALLBACK,
        }


@runtime_checkable
class StructuredSemanticProvider(Protocol):
    """
    Provider contract for Tex's schema-locked semantic layer.

    Tex owns prompt construction and schema validation.
    The provider owns transport and model execution only.
    """

    def analyze(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> SemanticAnalysis | Mapping[str, Any]:
        """Return a semantic payload for the supplied prompt pair."""


@runtime_checkable
class SemanticAnalyzer(Protocol):
    """Contract for Tex's semantic analysis boundary."""

    def analyze(
        self,
        *,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
    ) -> SemanticAnalysis:
        """Return a schema-valid SemanticAnalysis for the request."""


class DefaultSemanticAnalyzer:
    """
    Real semantic analyzer boundary for Tex.

    Responsibilities:
    1. build the semantic prompt bundle
    2. invoke the configured provider when present
    3. strictly validate provider output
    4. fail safely into deterministic fallback when allowed
    5. attach runtime metadata for auditability

    Non-responsibilities:
    - retrieval
    - routing / final verdict fusion
    - evidence persistence
    - policy activation/versioning
    """

    __slots__ = (
        "_provider",
        "_fallback_analyzer",
        "_allow_fallback",
        "_provider_name",
        "_model_name",
    )

    def __init__(
        self,
        *,
        provider: StructuredSemanticProvider | None = None,
        fallback_analyzer: SemanticFallbackAnalyzer | None = None,
        allow_fallback: bool = True,
        provider_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        self._provider = provider
        self._fallback_analyzer = fallback_analyzer or HeuristicSemanticFallback()
        self._allow_fallback = allow_fallback
        self._provider_name = self._normalize_optional_label(
            value=provider_name,
            field_name="provider_name",
        )
        self._model_name = self._normalize_optional_label(
            value=model_name,
            field_name="model_name",
        )

    def analyze(
        self,
        *,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
    ) -> SemanticAnalysis:
        """
        Return a schema-valid SemanticAnalysis for the request.

        Behavior:
        - if no provider is configured, Tex uses fallback as the default path
        - if the provider fails and fallback is allowed, Tex falls back safely
        - if the provider fails and fallback is disabled, an explicit error is raised
        """
        system_prompt, user_prompt = self.build_prompts(
            request=request,
            retrieval_context=retrieval_context,
        )

        if self._provider is None:
            fallback_analysis = self._fallback_analyzer.analyze(
                request=request,
                retrieval_context=retrieval_context,
            )
            trace = self._build_trace(
                mode=SemanticExecutionMode.DEFAULT_FALLBACK,
                provider_error=None,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            return self._decorate_analysis(
                analysis=fallback_analysis,
                request=request,
                retrieval_context=retrieval_context,
                execution_trace=trace,
            )

        try:
            provider_payload = self._provider.analyze(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            validated_analysis = self._coerce_provider_result(provider_payload)
        except Exception as exc:
            if not self._allow_fallback:
                raise SemanticProviderError(
                    "semantic provider failed and fallback is disabled"
                ) from exc

            fallback_analysis = self._fallback_analyzer.analyze(
                request=request,
                retrieval_context=retrieval_context,
            )
            trace = self._build_trace(
                mode=SemanticExecutionMode.FAILURE_FALLBACK,
                provider_error=f"{type(exc).__name__}: {exc}",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            return self._decorate_analysis(
                analysis=fallback_analysis,
                request=request,
                retrieval_context=retrieval_context,
                execution_trace=trace,
            )

        trace = self._build_trace(
            mode=SemanticExecutionMode.PRIMARY_PROVIDER,
            provider_error=None,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return self._decorate_analysis(
            analysis=validated_analysis,
            request=request,
            retrieval_context=retrieval_context,
            execution_trace=trace,
        )

    def build_prompts(
        self,
        *,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
    ) -> tuple[str, str]:
        """
        Return the exact semantic prompt pair for one request.

        Useful for provider adapters, debugging, snapshots, and tests.
        """
        return semantic_prompt_bundle(
            request=request,
            retrieval_context=retrieval_context,
        )

    @staticmethod
    def _coerce_provider_result(
        provider_result: SemanticAnalysis | Mapping[str, Any],
    ) -> SemanticAnalysis:
        if isinstance(provider_result, SemanticAnalysis):
            return provider_result

        if isinstance(provider_result, Mapping):
            try:
                return SemanticAnalysis.model_validate(dict(provider_result))
            except ValidationError as exc:
                raise SemanticProviderError(
                    "semantic provider returned a payload that failed schema validation"
                ) from exc

        raise SemanticProviderError(
            "semantic provider must return SemanticAnalysis or a mapping "
            "compatible with SemanticAnalysis"
        )

    def _decorate_analysis(
        self,
        *,
        analysis: SemanticAnalysis,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
        execution_trace: SemanticExecutionTrace,
    ) -> SemanticAnalysis:
        """
        Return the analysis enriched with normalized runtime metadata.

        The semantic schema stays the source of truth. This method only adds
        auditable metadata and fills provider/model identity when absent.
        """
        merged_metadata = dict(analysis.metadata)
        merged_metadata["semantic_runtime"] = {
            "mode": execution_trace.mode.value,
            "used_fallback": execution_trace.used_fallback,
            "provider_error": execution_trace.provider_error,
            "request_action_type": request.action_type,
            "request_channel": request.channel,
            "request_environment": request.environment,
            "request_has_recipient": request.recipient is not None,
            "request_content_sha256": self._sha256_hex(request.content),
            "retrieval_empty": retrieval_context.is_empty,
            "policy_clause_count": len(retrieval_context.policy_clauses),
            "precedent_count": len(retrieval_context.precedents),
            "entity_count": len(retrieval_context.entities),
            "matched_policy_clause_count": len(analysis.matched_policy_clause_ids),
            "system_prompt_sha256": execution_trace.system_prompt_sha256,
            "user_prompt_sha256": execution_trace.user_prompt_sha256,
        }

        resolved_provider_name = (
            analysis.provider_name
            or self._provider_name
            or self._infer_provider_name(execution_trace.mode)
        )
        resolved_model_name = (
            analysis.model_name
            or self._model_name
            or self._infer_model_name(execution_trace.mode)
        )

        update_payload: dict[str, Any] = {"metadata": merged_metadata}
        if resolved_provider_name is not None:
            update_payload["provider_name"] = resolved_provider_name
        if resolved_model_name is not None:
            update_payload["model_name"] = resolved_model_name

        return analysis.model_copy(update=update_payload)

    def _build_trace(
        self,
        *,
        mode: SemanticExecutionMode,
        provider_error: str | None,
        system_prompt: str,
        user_prompt: str,
    ) -> SemanticExecutionTrace:
        return SemanticExecutionTrace(
            mode=mode,
            provider_error=provider_error,
            system_prompt_sha256=self._sha256_hex(system_prompt),
            user_prompt_sha256=self._sha256_hex(user_prompt),
        )

    def _infer_provider_name(self, mode: SemanticExecutionMode) -> str | None:
        if mode in {
            SemanticExecutionMode.DEFAULT_FALLBACK,
            SemanticExecutionMode.FAILURE_FALLBACK,
        }:
            return "heuristic_fallback"

        if self._provider is None:
            return None

        raw_name = type(self._provider).__name__.strip()
        return raw_name or None

    def _infer_model_name(self, mode: SemanticExecutionMode) -> str | None:
        if mode in {
            SemanticExecutionMode.DEFAULT_FALLBACK,
            SemanticExecutionMode.FAILURE_FALLBACK,
        }:
            return "heuristic-deterministic"

        if self._provider is None:
            return None

        for attribute_name in ("model_name", "model", "deployment_name"):
            raw_value = getattr(self._provider, attribute_name, None)
            normalized = self._normalize_optional_label(
                value=raw_value,
                field_name=attribute_name,
                allow_non_string=True,
            )
            if normalized is not None:
                return normalized

        return None

    @staticmethod
    def _normalize_optional_label(
        *,
        value: object,
        field_name: str,
        allow_non_string: bool = False,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            if allow_non_string:
                value = str(value)
            else:
                raise TypeError(f"{field_name} must be a string when supplied")

        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _sha256_hex(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_default_semantic_analyzer() -> DefaultSemanticAnalyzer:
    """
    Return Tex's default semantic analyzer.

    Wiring behavior:
    - when settings.semantic_provider == "openai", use the OpenAI structured
      semantic provider
    - otherwise fall back to provider-less semantic execution, which remains
      valid because Tex's deterministic fallback analyzer is schema-safe

    This factory uses a local import to avoid a circular import between
    tex.semantic.analyzer and tex.semantic.openai.
    """
    settings = get_settings()
    provider = _build_semantic_provider_from_settings()
    return DefaultSemanticAnalyzer(
        provider=provider,
        allow_fallback=settings.allow_semantic_fallback,
    )


def build_semantic_prompts(
    *,
    request: EvaluationRequest,
    retrieval_context: RetrievalContext,
) -> tuple[str, str]:
    """
    Convenience wrapper for callers that want the exact semantic prompt pair
    without importing prompt internals directly.
    """
    return (
        build_semantic_system_prompt(),
        build_semantic_user_prompt(
            request=request,
            retrieval_context=retrieval_context,
        ),
    )


def _build_semantic_provider_from_settings() -> StructuredSemanticProvider | None:
    settings = get_settings()

    if settings.semantic_provider is None:
        return None

    if settings.semantic_provider != "openai":
        raise ValueError(
            "unsupported semantic_provider value. Expected 'openai' or None."
        )

    from tex.semantic.openai import OpenAIStructuredSemanticProvider

    return OpenAIStructuredSemanticProvider(
        api_key=settings.openai_api_key,
        model=settings.semantic_model,
        timeout_seconds=settings.semantic_timeout_seconds,
        max_retries=settings.semantic_max_retries,
        reasoning_effort=settings.semantic_reasoning_effort,
        base_url=settings.openai_base_url,
        organization=settings.openai_org_id,
        project=settings.openai_project_id,
    )
