from __future__ import annotations

from tex.domain.evaluation import EvaluationRequest
from tex.domain.retrieval import RetrievalContext
from tex.specialists.base import (
    SpecialistBundle,
    SpecialistEvidence,
    SpecialistJudge,
    SpecialistResult,
)


class SecretAndPiiSpecialist:
    """
    Narrow specialist for obvious secret, credential, and PII disclosure risk.

    This is intentionally lexical and retrieval-aware, not model-based. Its job
    is to add a sharper second opinion on disclosure risk before the semantic
    layer.
    """

    name = "secret_and_pii"

    _KEYWORDS: tuple[str, ...] = (
        "ssn",
        "social security",
        "password",
        "secret",
        "api key",
        "private key",
        "access token",
        "refresh token",
        "credential",
        "customer list",
        "pricing sheet",
        "confidential",
        "internal only",
        "dob",
        "date of birth",
        "bank account",
        "routing number",
    )

    def evaluate(
        self,
        *,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
    ) -> SpecialistResult:
        evidence = _match_keywords(
            content=request.content,
            keywords=self._KEYWORDS,
            explanation_prefix="Matched disclosure-risk keyword",
        )
        entity_hits = _match_entities(
            content=request.content,
            retrieval_context=retrieval_context,
        )

        combined_evidence = tuple((*evidence, *entity_hits))
        matched_entities = _matched_entity_names(entity_hits)

        risk_score = min(1.0, 0.08 + (0.18 * len(evidence)) + (0.10 * len(entity_hits)))
        confidence = 0.38 if not combined_evidence else min(0.86, 0.48 + (0.08 * len(combined_evidence)))

        uncertainty_flags = ["specialist_heuristic"]
        if retrieval_context.is_empty:
            uncertainty_flags.append("no_retrieval_context")

        if not combined_evidence:
            summary = "No obvious secret or PII disclosure signal detected by the specialist."
        else:
            summary = "Specialist detected possible sensitive-data disclosure signals."

        return SpecialistResult(
            specialist_name=self.name,
            risk_score=round(risk_score, 4),
            confidence=round(confidence, 4),
            summary=summary,
            rationale=(
                "This specialist focuses narrowly on lexical disclosure signals and "
                "retrieved sensitive entities. It is designed to be cheap, early, and conservative."
            ),
            evidence=combined_evidence,
            matched_policy_clause_ids=tuple(),
            matched_entity_names=matched_entities,
            uncertainty_flags=tuple(uncertainty_flags),
        )


class ExternalSharingSpecialist:
    """
    Narrow specialist for risky sharing, forwarding, export, or public exposure.
    """

    name = "external_sharing"

    _KEYWORDS: tuple[str, ...] = (
        "send externally",
        "share externally",
        "forward to customer",
        "public link",
        "anyone with the link",
        "export all",
        "bulk export",
        "download all",
        "upload to",
        "post publicly",
        "external recipient",
        "email attachment",
    )

    def evaluate(
        self,
        *,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
    ) -> SpecialistResult:
        evidence = _match_keywords(
            content=request.content,
            keywords=self._KEYWORDS,
            explanation_prefix="Matched external-sharing keyword",
        )

        policy_clause_ids = _clause_ids_with_overlap(
            content=request.content,
            retrieval_context=retrieval_context,
        )

        risk_score = min(1.0, 0.06 + (0.22 * len(evidence)) + (0.05 * len(policy_clause_ids)))
        confidence = 0.36 if not evidence else min(0.84, 0.50 + (0.09 * len(evidence)))

        uncertainty_flags = ["specialist_heuristic"]
        if retrieval_context.is_empty:
            uncertainty_flags.append("no_retrieval_context")

        if not evidence:
            summary = "No obvious risky external-sharing language detected by the specialist."
        else:
            summary = "Specialist detected possible risky sharing, forwarding, or export language."

        return SpecialistResult(
            specialist_name=self.name,
            risk_score=round(risk_score, 4),
            confidence=round(confidence, 4),
            summary=summary,
            rationale=(
                "This specialist focuses on outward movement of content or data, especially "
                "phrases associated with export, forwarding, public links, or external release."
            ),
            evidence=evidence,
            matched_policy_clause_ids=policy_clause_ids,
            matched_entity_names=tuple(),
            uncertainty_flags=tuple(uncertainty_flags),
        )


class UnauthorizedCommitmentSpecialist:
    """
    Narrow specialist for commitments, approvals, guarantees, pricing promises,
    or other potentially binding outbound language.
    """

    name = "unauthorized_commitment"

    _KEYWORDS: tuple[str, ...] = (
        "we guarantee",
        "we commit",
        "approved",
        "i approved",
        "final offer",
        "locked price",
        "guaranteed pricing",
        "contract signed",
        "you have my word",
        "we will refund",
        "we will deliver by",
        "confirmed for production",
        "approved exception",
    )

    def evaluate(
        self,
        *,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
    ) -> SpecialistResult:
        evidence = _match_keywords(
            content=request.content,
            keywords=self._KEYWORDS,
            explanation_prefix="Matched commitment-risk keyword",
        )

        policy_clause_ids = _clause_ids_with_overlap(
            content=request.content,
            retrieval_context=retrieval_context,
        )

        channel_bonus = 0.08 if request.channel.casefold() in {"email", "external_message", "sales_email"} else 0.0

        risk_score = min(1.0, 0.05 + (0.20 * len(evidence)) + (0.05 * len(policy_clause_ids)) + channel_bonus)
        confidence = 0.34 if not evidence else min(0.82, 0.47 + (0.08 * len(evidence)))

        uncertainty_flags = ["specialist_heuristic"]
        if retrieval_context.is_empty:
            uncertainty_flags.append("no_retrieval_context")

        if not evidence:
            summary = "No obvious unauthorized-commitment language detected by the specialist."
        else:
            summary = "Specialist detected language that may create unauthorized commitments or approvals."

        return SpecialistResult(
            specialist_name=self.name,
            risk_score=round(risk_score, 4),
            confidence=round(confidence, 4),
            summary=summary,
            rationale=(
                "This specialist targets language that could bind the organization operationally, "
                "commercially, or contractually without sufficient authorization."
            ),
            evidence=evidence,
            matched_policy_clause_ids=policy_clause_ids,
            matched_entity_names=tuple(),
            uncertainty_flags=tuple(uncertainty_flags),
        )


class DestructiveOrBypassSpecialist:
    """
    Narrow specialist for destructive action language, control bypass, or
    workflow circumvention.
    """

    name = "destructive_or_bypass"

    _KEYWORDS: tuple[str, ...] = (
        "delete",
        "wipe",
        "purge",
        "drop table",
        "disable logging",
        "turn off monitoring",
        "bypass approval",
        "skip review",
        "skip approval",
        "ignore policy",
        "remove audit",
        "override control",
        "exfiltrate",
        "disable guardrail",
    )

    def evaluate(
        self,
        *,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
    ) -> SpecialistResult:
        evidence = _match_keywords(
            content=request.content,
            keywords=self._KEYWORDS,
            explanation_prefix="Matched destructive-or-bypass keyword",
        )
        policy_clause_ids = _clause_ids_with_overlap(
            content=request.content,
            retrieval_context=retrieval_context,
        )

        environment_bonus = 0.10 if request.environment.casefold() in {"production", "prod"} else 0.0

        risk_score = min(1.0, 0.08 + (0.24 * len(evidence)) + (0.04 * len(policy_clause_ids)) + environment_bonus)
        confidence = 0.36 if not evidence else min(0.88, 0.52 + (0.08 * len(evidence)))

        uncertainty_flags = ["specialist_heuristic"]
        if retrieval_context.is_empty:
            uncertainty_flags.append("no_retrieval_context")

        if not evidence:
            summary = "No obvious destructive or control-bypass language detected by the specialist."
        else:
            summary = "Specialist detected language associated with destructive actions or control bypass."

        return SpecialistResult(
            specialist_name=self.name,
            risk_score=round(risk_score, 4),
            confidence=round(confidence, 4),
            summary=summary,
            rationale=(
                "This specialist focuses on deletion, disabling controls, bypassing workflow, "
                "or other language that points to unsafe operational shortcuts."
            ),
            evidence=evidence,
            matched_policy_clause_ids=policy_clause_ids,
            matched_entity_names=tuple(),
            uncertainty_flags=tuple(uncertainty_flags),
        )


class SpecialistSuite:
    """
    Executes Tex's configured specialist judges and returns a stable bundle.

    This is intentionally lean. Policy can later decide which specialists are
    enabled or weighted differently, but the execution contract stays simple.
    """

    def __init__(self, judges: tuple[SpecialistJudge, ...] | None = None) -> None:
        self._judges: tuple[SpecialistJudge, ...] = judges or default_specialist_judges()

    def evaluate(
        self,
        *,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
    ) -> SpecialistBundle:
        results = tuple(
            judge.evaluate(
                request=request,
                retrieval_context=retrieval_context,
            )
            for judge in self._judges
        )
        return SpecialistBundle(results=results)


def default_specialist_judges() -> tuple[SpecialistJudge, ...]:
    """
    Returns Tex's default specialist judge set.

    Keep this small. Specialists only deserve to exist when they add a distinct,
    high-signal slice of judgment.
    """
    return (
        SecretAndPiiSpecialist(),
        ExternalSharingSpecialist(),
        UnauthorizedCommitmentSpecialist(),
        DestructiveOrBypassSpecialist(),
    )


def build_default_specialist_suite() -> SpecialistSuite:
    """Convenience constructor for Tex's default specialist suite."""
    return SpecialistSuite(judges=default_specialist_judges())


def _match_keywords(
    *,
    content: str,
    keywords: tuple[str, ...],
    explanation_prefix: str,
) -> tuple[SpecialistEvidence, ...]:
    lowered_content = content.casefold()
    evidence: list[SpecialistEvidence] = []
    seen: set[tuple[int, int, str]] = set()

    for keyword in keywords:
        lowered_keyword = keyword.casefold()
        start_index = 0

        while True:
            found_at = lowered_content.find(lowered_keyword, start_index)
            if found_at == -1:
                break

            end_index = found_at + len(lowered_keyword)
            matched_text = content[found_at:end_index]
            dedupe_key = (found_at, end_index, lowered_keyword)

            if dedupe_key not in seen:
                seen.add(dedupe_key)
                evidence.append(
                    SpecialistEvidence(
                        text=matched_text,
                        start_index=found_at,
                        end_index=end_index,
                        explanation=f"{explanation_prefix}: {keyword}",
                    )
                )

            start_index = end_index

    evidence.sort(key=lambda item: (item.start_index or 10**9, item.text.casefold()))
    return tuple(evidence)


def _match_entities(
    *,
    content: str,
    retrieval_context: RetrievalContext,
) -> tuple[SpecialistEvidence, ...]:
    lowered_content = content.casefold()
    evidence: list[SpecialistEvidence] = []
    seen: set[tuple[int, int, str]] = set()

    for entity in retrieval_context.entities:
        for candidate in entity.all_names:
            lowered_candidate = candidate.casefold()
            start_index = 0

            while True:
                found_at = lowered_content.find(lowered_candidate, start_index)
                if found_at == -1:
                    break

                end_index = found_at + len(lowered_candidate)
                dedupe_key = (found_at, end_index, entity.entity_id)

                if dedupe_key not in seen:
                    seen.add(dedupe_key)
                    evidence.append(
                        SpecialistEvidence(
                            text=content[found_at:end_index],
                            start_index=found_at,
                            end_index=end_index,
                            explanation=(
                                f"Matched retrieved sensitive entity: {entity.canonical_name} "
                                f"({entity.sensitivity})"
                            ),
                        )
                    )

                start_index = end_index

    evidence.sort(key=lambda item: (item.start_index or 10**9, item.text.casefold()))
    return tuple(evidence)


def _matched_entity_names(
    evidence: tuple[SpecialistEvidence, ...],
) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()

    for item in evidence:
        explanation = item.explanation or ""
        marker = "Matched retrieved sensitive entity: "
        if not explanation.startswith(marker):
            continue

        remainder = explanation[len(marker):]
        entity_name = remainder.split(" (", 1)[0].strip()
        if not entity_name:
            continue

        dedupe_key = entity_name.casefold()
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        names.append(entity_name)

    return tuple(names)


def _clause_ids_with_overlap(
    *,
    content: str,
    retrieval_context: RetrievalContext,
) -> tuple[str, ...]:
    lowered_content = content.casefold()
    matched_clause_ids: list[str] = []

    for clause in retrieval_context.policy_clauses:
        tokens = _policy_clause_tokens(clause.text)
        if any(token in lowered_content for token in tokens):
            matched_clause_ids.append(clause.clause_id)

    deduped: list[str] = []
    seen: set[str] = set()

    for clause_id in matched_clause_ids:
        dedupe_key = clause_id.casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(clause_id)

    return tuple(deduped)


def _policy_clause_tokens(clause_text: str) -> tuple[str, ...]:
    raw_tokens = (
        clause_text.replace(",", " ")
        .replace(".", " ")
        .replace(";", " ")
        .replace(":", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )

    tokens: list[str] = []
    seen: set[str] = set()

    for token in raw_tokens:
        normalized = token.strip().casefold()
        if len(normalized) < 6:
            continue
        if normalized in _CLAUSE_TOKEN_STOPWORDS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(normalized)

    return tuple(tokens)


# Generic English and policy-boilerplate tokens that should never count as
# semantic overlap between a policy clause and request content. Without this
# filter, a prose clause like "requires extra care and review" would match
# any benign content that happens to contain "review" or "require".
_CLAUSE_TOKEN_STOPWORDS: frozenset[str] = frozenset(
    {
        # generic English verbs and nouns
        "should",
        "shall",
        "would",
        "could",
        "might",
        "before",
        "after",
        "during",
        "within",
        "without",
        "between",
        "because",
        "through",
        "following",
        "however",
        "therefore",
        "another",
        "several",
        "include",
        "includes",
        "including",
        "involve",
        "involves",
        "involving",
        "require",
        "requires",
        "required",
        "requiring",
        "ensure",
        "ensures",
        "ensured",
        "ensuring",
        "provide",
        "provides",
        "provided",
        "providing",
        "consider",
        "considers",
        "considered",
        "reference",
        "references",
        "referenced",
        "referencing",
        "regarding",
        "respect",
        "respects",
        "respected",
        "respecting",
        # policy / governance boilerplate
        "policy",
        "policies",
        "content",
        "contents",
        "context",
        "contexts",
        "action",
        "actions",
        "review",
        "reviews",
        "reviewed",
        "reviewing",
        "approval",
        "approvals",
        "approved",
        "approve",
        "release",
        "released",
        "releases",
        "releasing",
        "request",
        "requests",
        "requested",
        "requesting",
        "response",
        "responses",
        "recipient",
        "recipients",
        "channel",
        "channels",
        "environment",
        "environments",
        "system",
        "systems",
        "internal",
        "external",
        "explicit",
        "explicitly",
        "additional",
        "appropriate",
        "relevant",
        "related",
        "specific",
        "generic",
        "general",
        "default",
        "strict",
        "configured",
        "configuration",
        "section",
        "sections",
        "clause",
        "clauses",
        "restriction",
        "restrictions",
        "description",
        "descriptions",
        "documentation",
        "enabled",
        "disabled",
        "detect",
        "detects",
        "detected",
        "detection",
        "handling",
        "handled",
        "handles",
    }
)