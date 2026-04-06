from __future__ import annotations

import re
from typing import Protocol

from tex.domain.evaluation import EvaluationRequest
from tex.domain.finding import Finding
from tex.domain.severity import Severity


class Recognizer(Protocol):
    """Contract for fast deterministic recognizers."""

    name: str

    def scan(self, request: EvaluationRequest) -> tuple[Finding, ...]:
        """Returns zero or more deterministic findings for the request."""


class BaseRegexRecognizer:
    """
    Base class for simple regex-driven deterministic recognizers.

    These recognizers are intentionally fast, explicit, and cheap. They are not
    trying to be clever. Their job is to catch obvious signals before the more
    expensive layers run.
    """

    name: str = "base_regex"
    severity: Severity = Severity.WARNING
    message: str = "Deterministic recognizer matched suspicious content."
    patterns: tuple[re.Pattern[str], ...] = tuple()

    def scan(self, request: EvaluationRequest) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        content = request.content

        for pattern in self.patterns:
            for match in pattern.finditer(content):
                matched_text = match.group(0).strip()
                if not matched_text:
                    continue

                findings.append(
                    Finding(
                        source="deterministic",
                        rule_name=self.name,
                        severity=self.severity,
                        message=self.message,
                        matched_text=matched_text,
                        start_index=match.start(),
                        end_index=match.end(),
                        metadata={
                            "pattern": pattern.pattern,
                            "channel": request.channel,
                            "action_type": request.action_type,
                            "environment": request.environment,
                        },
                    )
                )

        return tuple(findings)


class BlockedTermsRecognizer:
    """
    Matches blocked terms defined directly in policy.

    This recognizer is policy-driven rather than hardcoded. It exists because
    some customer-specific restrictions are too simple and too important to
    wait for the semantic layer.
    """

    name = "blocked_terms"

    def scan(self, request: EvaluationRequest) -> tuple[Finding, ...]:
        blocked_terms = tuple(
            term.strip()
            for term in request.metadata.get("blocked_terms", ())
            if isinstance(term, str) and term.strip()
        )
        if not blocked_terms:
            return tuple()

        findings: list[Finding] = []
        content = request.content
        lowered_content = content.casefold()

        for term in blocked_terms:
            lowered_term = term.casefold()
            start = lowered_content.find(lowered_term)
            if start == -1:
                continue

            end = start + len(term)
            findings.append(
                Finding(
                    source="deterministic",
                    rule_name=self.name,
                    severity=Severity.CRITICAL,
                    message="Content contains a policy-blocked term.",
                    matched_text=content[start:end],
                    start_index=start,
                    end_index=end,
                    metadata={
                        "blocked_term": term,
                        "channel": request.channel,
                        "action_type": request.action_type,
                        "environment": request.environment,
                    },
                )
            )

        return tuple(findings)


class SensitiveEntitiesRecognizer:
    """
    Matches customer- or domain-specific sensitive entities.

    This is still deterministic and intentionally shallow. It catches obvious
    string matches for known sensitive entities before retrieval/semantic
    grounding gets more sophisticated.
    """

    name = "sensitive_entities"

    def scan(self, request: EvaluationRequest) -> tuple[Finding, ...]:
        sensitive_entities = tuple(
            entity.strip()
            for entity in request.metadata.get("sensitive_entities", ())
            if isinstance(entity, str) and entity.strip()
        )
        if not sensitive_entities:
            return tuple()

        findings: list[Finding] = []
        content = request.content
        lowered_content = content.casefold()

        for entity in sensitive_entities:
            lowered_entity = entity.casefold()
            start = lowered_content.find(lowered_entity)
            if start == -1:
                continue

            end = start + len(entity)
            findings.append(
                Finding(
                    source="deterministic",
                    rule_name=self.name,
                    severity=Severity.WARNING,
                    message="Content references a configured sensitive entity.",
                    matched_text=content[start:end],
                    start_index=start,
                    end_index=end,
                    metadata={
                        "entity": entity,
                        "channel": request.channel,
                        "action_type": request.action_type,
                        "environment": request.environment,
                    },
                )
            )

        return tuple(findings)


class SecretLeakRecognizer(BaseRegexRecognizer):
    """Catches obvious credential and secret leakage patterns."""

    name = "secret_leak"
    severity = Severity.CRITICAL
    message = "Content appears to contain a secret, credential, or token."
    patterns = (
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
        re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
        re.compile(r"\bapi[_\-\s]?key\b", re.IGNORECASE),
        re.compile(r"\bprivate[_\-\s]?key\b", re.IGNORECASE),
        re.compile(r"\baccess[_\-\s]?token\b", re.IGNORECASE),
        re.compile(r"\bclient[_\-\s]?secret\b", re.IGNORECASE),
        re.compile(r"\bpassword\s*[:=]\s*\S+", re.IGNORECASE),
    )


class PiiRecognizer(BaseRegexRecognizer):
    """Catches obvious personally identifiable information patterns."""

    name = "pii"
    severity = Severity.CRITICAL
    message = "Content appears to contain sensitive personal data."
    patterns = (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        re.compile(r"\b(?:\d{4}[ -]?){3}\d{1,4}\b"),
    )


class UnauthorizedCommitmentRecognizer(BaseRegexRecognizer):
    """Catches obvious commitment or approval language."""

    name = "unauthorized_commitment"
    severity = Severity.WARNING
    message = "Content appears to make a commitment, approval, or guarantee."
    patterns = (
        re.compile(r"\bwe guarantee\b", re.IGNORECASE),
        re.compile(r"\bwe commit\b", re.IGNORECASE),
        re.compile(r"\bfinal offer\b", re.IGNORECASE),
        re.compile(r"\blocked price\b", re.IGNORECASE),
        re.compile(r"\bcontract signed\b", re.IGNORECASE),
        re.compile(r"\byou have my word\b", re.IGNORECASE),
        re.compile(r"\bwe will refund\b", re.IGNORECASE),
    )


class ExternalSharingRecognizer(BaseRegexRecognizer):
    """Catches obvious external-sharing or exfiltration language."""

    name = "external_sharing"
    severity = Severity.WARNING
    message = "Content appears to describe risky external sharing or export."
    patterns = (
        re.compile(r"\bshare externally\b", re.IGNORECASE),
        re.compile(r"\bsend externally\b", re.IGNORECASE),
        re.compile(r"\bpublic link\b", re.IGNORECASE),
        re.compile(r"\bforward to customer\b", re.IGNORECASE),
        re.compile(r"\bupload to external\b", re.IGNORECASE),
        re.compile(r"\bexport all\b", re.IGNORECASE),
        re.compile(r"\bpost publicly\b", re.IGNORECASE),
    )


class DestructiveOrBypassRecognizer(BaseRegexRecognizer):
    """Catches destructive, evasive, or workflow-bypass language."""

    name = "destructive_or_bypass"
    severity = Severity.CRITICAL
    message = "Content appears to describe destructive action or control bypass."
    patterns = (
        re.compile(r"\bdelete all\b", re.IGNORECASE),
        re.compile(r"\bdrop table\b", re.IGNORECASE),
        re.compile(r"\bdisable logging\b", re.IGNORECASE),
        re.compile(r"\bskip review\b", re.IGNORECASE),
        re.compile(r"\bskip approval\b", re.IGNORECASE),
        re.compile(r"\boverride control\b", re.IGNORECASE),
        re.compile(r"\bignore policy\b", re.IGNORECASE),
        re.compile(r"\bremove audit\b", re.IGNORECASE),
        re.compile(r"\bexfiltrate\b", re.IGNORECASE),
        re.compile(r"\bwipe all\b", re.IGNORECASE),
    )


def default_recognizers() -> tuple[Recognizer, ...]:
    """
    Returns Tex's default deterministic recognizer set.

    Order matters. The cheap, highest-signal recognizers should run first.
    """
    return (
        BlockedTermsRecognizer(),
        SensitiveEntitiesRecognizer(),
        SecretLeakRecognizer(),
        PiiRecognizer(),
        UnauthorizedCommitmentRecognizer(),
        ExternalSharingRecognizer(),
        DestructiveOrBypassRecognizer(),
    )