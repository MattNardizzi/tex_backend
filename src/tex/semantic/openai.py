from __future__ import annotations

import hashlib
import os
import time
from typing import Any
from typing import Final

from tex.semantic.analyzer import SemanticProviderError
from tex.semantic.schema import SemanticAnalysis, SemanticAnalysisParseTarget

try:
    from openai import APIConnectionError
    from openai import APITimeoutError
    from openai import BadRequestError
    from openai import OpenAI
    from openai import RateLimitError
except ImportError as exc:  # pragma: no cover - import guard for environments without SDK
    OpenAI = None  # type: ignore[assignment]
    APIConnectionError = Exception  # type: ignore[assignment]
    APITimeoutError = Exception  # type: ignore[assignment]
    BadRequestError = Exception  # type: ignore[assignment]
    RateLimitError = Exception  # type: ignore[assignment]
    _OPENAI_IMPORT_ERROR = exc
else:
    _OPENAI_IMPORT_ERROR = None


_DEFAULT_MODEL: Final[str] = "gpt-5.4"
_DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
_DEFAULT_MAX_RETRIES: Final[int] = 2
_ALLOWED_REASONING_EFFORTS: Final[frozenset[str]] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh"}
)


class OpenAIStructuredSemanticProvider:
    """
    Structured semantic provider for Tex using OpenAI's Responses API.

    Design goals:
    - strict schema-bound output into SemanticAnalysis
    - zero-verbosity, evaluation-style calls
    - explicit failure surfaces for timeout/rate-limit/refusal/parse issues
    - rich runtime metadata for audit and semantic evals
    - no prompt construction here; Tex already owns that boundary

    This class intentionally keeps a synchronous interface because Tex's
    existing StructuredSemanticProvider protocol is synchronous.
    """

    __slots__ = (
        "_api_key",
        "_base_url",
        "_organization",
        "_project",
        "_model",
        "_timeout_seconds",
        "_max_retries",
        "_reasoning_effort",
        "_client",
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        reasoning_effort: str = "low",
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        self._api_key = self._normalize_optional_string(api_key) or os.getenv(
            "OPENAI_API_KEY"
        )
        self._base_url = self._normalize_optional_string(base_url) or os.getenv(
            "OPENAI_BASE_URL"
        )
        self._organization = self._normalize_optional_string(
            organization
        ) or self._normalize_optional_string(os.getenv("OPENAI_ORG_ID"))
        self._project = self._normalize_optional_string(
            project
        ) or self._normalize_optional_string(os.getenv("OPENAI_PROJECT_ID"))
        self._model = self._normalize_required_string(model, field_name="model")
        self._timeout_seconds = self._validate_timeout_seconds(timeout_seconds)
        self._max_retries = self._validate_max_retries(max_retries)
        self._reasoning_effort = self._validate_reasoning_effort(reasoning_effort)
        self._client: OpenAI | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "openai"

    def analyze(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> SemanticAnalysis:
        """
        Executes one schema-locked semantic evaluation call.

        Returns:
            SemanticAnalysis

        Raises:
            SemanticProviderError: on transport, refusal, or schema/parse failure.
        """
        instructions = self._normalize_required_string(
            system_prompt,
            field_name="system_prompt",
        )
        request_prompt = self._normalize_required_string(
            user_prompt,
            field_name="user_prompt",
        )

        client = self._get_client()
        started_at = time.perf_counter()

        try:
            response = client.responses.parse(
                model=self._model,
                input=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": request_prompt},
                ],
                text_format=SemanticAnalysisParseTarget,
                reasoning={"effort": self._reasoning_effort},
                timeout=self._timeout_seconds,
            )
        except APITimeoutError as exc:
            raise SemanticProviderError(
                f"OpenAI semantic request timed out after {self._timeout_seconds:.1f}s"
            ) from exc
        except RateLimitError as exc:
            raise SemanticProviderError(
                "OpenAI semantic request was rate-limited"
            ) from exc
        except APIConnectionError as exc:
            raise SemanticProviderError(
                "OpenAI semantic request failed due to connection error"
            ) from exc
        except BadRequestError as exc:
            raise SemanticProviderError(
                f"OpenAI semantic request was rejected: {exc}"
            ) from exc
        except Exception as exc:
            raise SemanticProviderError(
                f"unexpected OpenAI semantic provider failure: {type(exc).__name__}: {exc}"
            ) from exc

        elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 3)

        refusal_text = self._extract_refusal_text(response)
        if refusal_text is not None:
            raise SemanticProviderError(
                f"OpenAI semantic provider refused the request: {refusal_text}"
            )

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raw_text = self._extract_output_text(response)
            if raw_text is not None:
                raise SemanticProviderError(
                    "OpenAI semantic provider returned text but no parsed SemanticAnalysisParseTarget"
                )
            raise SemanticProviderError(
                "OpenAI semantic provider returned neither parsed output nor usable text"
            )

        if not isinstance(parsed, SemanticAnalysisParseTarget):
            try:
                parsed = SemanticAnalysisParseTarget.model_validate(parsed)
            except Exception as exc:
                raise SemanticProviderError(
                    "OpenAI semantic provider returned a parsed object that "
                    "failed SemanticAnalysisParseTarget validation"
                ) from exc

        usage = getattr(response, "usage", None)

        openai_metadata = {
            "provider": "openai",
            "sdk_surface": "responses.parse",
            "response_id": getattr(response, "id", None),
            "latency_ms": elapsed_ms,
            "reasoning_effort": self._reasoning_effort,
            "timeout_seconds": self._timeout_seconds,
            "prompt_fingerprints": {
                "system_prompt_sha256": self._sha256_hex(instructions),
                "user_prompt_sha256": self._sha256_hex(request_prompt),
            },
            "usage": self._serialize_usage(usage),
        }

        return parsed.to_full_analysis(
            provider_name=self.provider_name,
            model_name=self._model,
            metadata={"openai": openai_metadata},
        )

    def _get_client(self) -> OpenAI:
        if OpenAI is None:
            raise SemanticProviderError(
                "openai package is not installed. Install it before using "
                "OpenAIStructuredSemanticProvider."
            ) from _OPENAI_IMPORT_ERROR

        if self._api_key is None:
            raise SemanticProviderError(
                "OPENAI_API_KEY is not set and no api_key was provided"
            )

        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                organization=self._organization,
                project=self._project,
                timeout=self._timeout_seconds,
                max_retries=self._max_retries,
            )
        return self._client

    @staticmethod
    def _extract_refusal_text(response: object) -> str | None:
        refusal = getattr(response, "refusal", None)
        if isinstance(refusal, str):
            normalized = refusal.strip()
            return normalized or None
        return None

    @staticmethod
    def _extract_output_text(response: object) -> str | None:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            normalized = output_text.strip()
            return normalized or None
        return None

    @staticmethod
    def _serialize_usage(usage: object) -> dict[str, Any] | None:
        if usage is None:
            return None

        if hasattr(usage, "model_dump"):
            try:
                dumped = usage.model_dump()
                if isinstance(dumped, dict):
                    return dumped
            except Exception:
                pass

        if isinstance(usage, dict):
            return usage

        result: dict[str, Any] = {}
        for field_name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
        ):
            value = getattr(usage, field_name, None)
            if value is not None:
                result[field_name] = value

        return result or None

    @staticmethod
    def _sha256_hex(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_optional_string(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @classmethod
    def _normalize_required_string(cls, value: str, *, field_name: str) -> str:
        normalized = cls._normalize_optional_string(value)
        if normalized is None:
            raise ValueError(f"{field_name} must be a non-empty string")
        return normalized

    @staticmethod
    def _validate_timeout_seconds(value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        return float(value)

    @staticmethod
    def _validate_max_retries(value: int) -> int:
        if value < 0:
            raise ValueError("max_retries must be >= 0")
        return int(value)

    @staticmethod
    def _validate_reasoning_effort(value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _ALLOWED_REASONING_EFFORTS:
            allowed = ", ".join(sorted(_ALLOWED_REASONING_EFFORTS))
            raise ValueError(
                f"reasoning_effort must be one of: {allowed}. Got: {value!r}"
            )
        return normalized