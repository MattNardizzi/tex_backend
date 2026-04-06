from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from tex.deterministic.recognizers import Recognizer, default_recognizers
from tex.domain.evaluation import EvaluationRequest
from tex.domain.finding import Finding
from tex.domain.policy import PolicySnapshot
from tex.domain.severity import Severity
from tex.domain.verdict import Verdict


class DeterministicGateResult(BaseModel):
    """
    Result of Tex's deterministic recognition layer.

    This layer is fast, cheap, and intentionally blunt. It exists to catch
    obvious high-signal issues before retrieval, specialists, or semantic
    analysis run.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    findings: tuple[Finding, ...] = Field(default_factory=tuple)
    enabled_recognizers: tuple[str, ...] = Field(default_factory=tuple)

    blocked: bool
    blocking_reasons: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    @property
    def critical_findings(self) -> tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == Severity.CRITICAL)

    @property
    def warning_findings(self) -> tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == Severity.WARNING)

    @property
    def info_findings(self) -> tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if finding.severity == Severity.INFO)

    @property
    def suggested_verdict(self) -> Verdict | None:
        if self.blocked:
            return Verdict.FORBID
        if self.findings:
            return Verdict.ABSTAIN
        return None


class DeterministicGate:
    """
    Runs Tex's deterministic recognizers and applies policy-based hard blocking.

    The deterministic layer is not the final decision-maker. Its job is:
    - catch obvious high-signal issues cheaply
    - surface structured findings
    - hard-block only when policy explicitly says to do so
    """

    def __init__(self, recognizers: tuple[Recognizer, ...] | None = None) -> None:
        self._recognizers: tuple[Recognizer, ...] = recognizers or default_recognizers()

    def evaluate(
        self,
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
    ) -> DeterministicGateResult:
        enabled_recognizer_names = set(policy.enabled_recognizers)
        executed_names: list[str] = []
        findings: list[Finding] = []

        for recognizer in self._recognizers:
            if enabled_recognizer_names and recognizer.name not in enabled_recognizer_names:
                continue

            executed_names.append(recognizer.name)
            findings.extend(recognizer.scan(request))

        deduped_findings = self._dedupe_findings(findings)
        blocking_reasons = self._compute_blocking_reasons(
            findings=deduped_findings,
            policy=policy,
        )

        return DeterministicGateResult(
            findings=deduped_findings,
            enabled_recognizers=tuple(executed_names),
            blocked=bool(blocking_reasons),
            blocking_reasons=blocking_reasons,
        )

    @staticmethod
    def _dedupe_findings(findings: list[Finding]) -> tuple[Finding, ...]:
        seen: set[tuple[str, str, str | None, int | None, int | None]] = set()
        ordered: list[Finding] = []

        for finding in findings:
            key = (
                finding.rule_name,
                finding.message,
                finding.matched_text,
                finding.start_index,
                finding.end_index,
            )
            if key in seen:
                continue
            seen.add(key)
            ordered.append(finding)

        ordered.sort(
            key=lambda finding: (
                0 if finding.severity == Severity.CRITICAL else 1 if finding.severity == Severity.WARNING else 2,
                finding.start_index if finding.start_index is not None else 10**9,
                finding.rule_name,
            )
        )
        return tuple(ordered)

    @staticmethod
    def _compute_blocking_reasons(
        *,
        findings: tuple[Finding, ...],
        policy: PolicySnapshot,
    ) -> tuple[str, ...]:
        reasons: list[str] = []

        for finding in findings:
            if policy.blocks_severity(finding.severity):
                reasons.append(
                    f"{finding.rule_name} produced {finding.severity.value} finding: {finding.message}"
                )

        return tuple(reasons)


def build_default_deterministic_gate() -> DeterministicGate:
    """Convenience constructor for the default deterministic gate."""
    return DeterministicGate(recognizers=default_recognizers())