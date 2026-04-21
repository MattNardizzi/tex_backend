from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


SemanticProviderName = Literal["openai"]
SemanticReasoningEffort = Literal[
    "minimal",
    "low",
    "medium",
    "high",
    "none",
    "xhigh",
]


class Settings(BaseSettings):
    """
    Central runtime configuration for Tex.

    This keeps configuration explicit, typed, and environment-driven without
    leaking env lookups across the codebase.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_name: str = Field(default="tex")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)

    evidence_path: Path = Field(
        default=Path("data/evidence/evidence.jsonl"),
        alias="TEX_EVIDENCE_PATH",
    )

    semantic_provider: SemanticProviderName | None = Field(
        default=None,
        alias="TEX_SEMANTIC_PROVIDER",
    )
    allow_semantic_fallback: bool = Field(
        default=True,
        alias="TEX_ALLOW_SEMANTIC_FALLBACK",
    )
    semantic_model: str = Field(
        default="gpt-5-mini",
        alias="TEX_SEMANTIC_MODEL",
    )
    semantic_timeout_seconds: float = Field(
        default=30.0,
        alias="TEX_SEMANTIC_TIMEOUT_SECONDS",
    )
    semantic_max_retries: int = Field(
        default=2,
        alias="TEX_SEMANTIC_MAX_RETRIES",
    )
    semantic_reasoning_effort: SemanticReasoningEffort = Field(
        default="minimal",
        alias="TEX_SEMANTIC_REASONING_EFFORT",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_org_id: str | None = Field(default=None, alias="OPENAI_ORG_ID")
    openai_project_id: str | None = Field(default=None, alias="OPENAI_PROJECT_ID")

    @field_validator(
        "app_name",
        "app_env",
        "host",
        "semantic_model",
        "openai_api_key",
        "openai_base_url",
        "openai_org_id",
        "openai_project_id",
        mode="before",
    )
    @classmethod
    def _strip_optional_strings(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        return normalized or None

    @field_validator("semantic_provider", mode="before")
    @classmethod
    def _normalize_semantic_provider(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("TEX_SEMANTIC_PROVIDER must be a string when supplied")
        normalized = value.strip().lower()
        return normalized or None

    @field_validator("semantic_timeout_seconds")
    @classmethod
    def _validate_semantic_timeout_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("TEX_SEMANTIC_TIMEOUT_SECONDS must be greater than 0")
        return value

    @field_validator("semantic_max_retries")
    @classmethod
    def _validate_semantic_max_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("TEX_SEMANTIC_MAX_RETRIES must be >= 0")
        return value

    @field_validator("port")
    @classmethod
    def _validate_port(cls, value: int) -> int:
        if value <= 0 or value > 65535:
            raise ValueError("port must be between 1 and 65535")
        return value

    @field_validator("evidence_path", mode="before")
    @classmethod
    def _coerce_evidence_path(cls, value: object) -> object:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("TEX_EVIDENCE_PATH cannot be blank")
            return Path(normalized)
        return value

    @property
    def semantic_provider_enabled(self) -> bool:
        return self.semantic_provider is not None

    def validate_semantic_provider_configuration(self) -> None:
        """
        Fail loudly on bad semantic-provider wiring instead of silently
        degrading to heuristic fallback when the operator explicitly asked for
        a provider.
        """
        if self.semantic_provider is None:
            return

        if self.semantic_provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "TEX_SEMANTIC_PROVIDER is set to 'openai' but OPENAI_API_KEY is missing."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_semantic_provider_configuration()
    return settings