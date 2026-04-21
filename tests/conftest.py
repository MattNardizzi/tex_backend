"""
Shared pytest configuration for Tex.

Key behaviors enforced here:

1. The semantic provider is forced off for the entire test session so tests
   are deterministic and do not make real network calls. Individual tests
   that want to exercise provider-specific code paths must construct the
   provider themselves; they should not rely on ambient environment state.

2. Evidence writes are redirected to a per-session temp file so running
   the suite never pollutes the project's evidence store.

3. ``get_settings`` is cached with ``lru_cache``. We clear that cache at
   session start so the test environment overrides take effect even when a
   previous import already filled the cache.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _disable_semantic_provider_for_tests(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """
    Force deterministic test behavior:
    - no semantic provider (fallback only)
    - semantic fallback allowed
    - evidence path pointed at a throwaway temp directory
    """
    session_dir: Path = tmp_path_factory.mktemp("tex-test-evidence")
    evidence_path = session_dir / "evidence.jsonl"

    prior: dict[str, str | None] = {
        "TEX_SEMANTIC_PROVIDER": os.environ.get("TEX_SEMANTIC_PROVIDER"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "TEX_ALLOW_SEMANTIC_FALLBACK": os.environ.get("TEX_ALLOW_SEMANTIC_FALLBACK"),
        "TEX_EVIDENCE_PATH": os.environ.get("TEX_EVIDENCE_PATH"),
    }

    os.environ.pop("TEX_SEMANTIC_PROVIDER", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ["TEX_ALLOW_SEMANTIC_FALLBACK"] = "true"
    os.environ["TEX_EVIDENCE_PATH"] = str(evidence_path)

    # Clear cached settings so the new env vars take effect if anything
    # imported tex.config before this fixture ran.
    from tex.config import get_settings

    get_settings.cache_clear()

    yield

    # Restore prior environment so we leave the process state clean.
    for key, value in prior.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()


@pytest.fixture
def evidence_path(tmp_path: Path) -> Path:
    """Per-test evidence path so concurrent tests do not share state."""
    return tmp_path / "evidence.jsonl"


@pytest.fixture
def runtime(evidence_path: Path):
    """
    Build a fresh in-process Tex runtime for one test.

    Imported lazily so that ``_disable_semantic_provider_for_tests`` has
    already reset the settings cache before build_runtime runs.
    """
    from tex.main import build_runtime

    return build_runtime(evidence_path=evidence_path)
