from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol

from tex.domain.evaluation import EvaluationRequest
from tex.domain.retrieval import RetrievalContext
from tex.domain.verdict import Verdict
from tex.semantic.schema import (
    SemanticAnalysis,
    SemanticDimensionResult,
    SemanticEvidenceSpan,
    SemanticVerdictRecommendation,
    semantic_dimensions,
)


class SemanticFallbackAnalyzer(Protocol):
    """Protocol for deterministic fallback semantic analyzers."""

    def analyze(
        self,
        *,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
    ) -> SemanticAnalysis:
        """Returns a schema-valid semantic analysis without using an LLM."""


class HeuristicSemanticFallback:
    """
    Lean deterministic fallback for Tex's semantic layer.

    This exists for three reasons:
    1. the LLM layer may be unavailable
    2. structured parsing may fail
    3. local development still needs a valid semantic result

    Design principles:
    - clean content with no keyword hits MUST be able to reach PERMIT
    - risky content with keyword hits MUST still reach FORBID or ABSTAIN
    - confidence scores must stay above the is_low_confidence threshold
      (0.50) for zero-hit dimensions so the router does not auto-ABSTAIN
    - the fallback is not trying to be smart; it is trying to produce
      calibrated scores that let the router do its job correctly
    """

    _DATA_LEAKAGE_TERMS: tuple[str, ...] = (
        "ssn",
        "social security",
        "password",
        "secret",
        "api key",
        "private key",
        "token",
        "credential",
        "customer list",
        "pricing sheet",
        "confidential",
        "internal only",
    )

    _EXTERNAL_SHARING_TERMS: tuple[str, ...] = (
        "send externally",
        "share externally",
        "public link",
        "export",
        "download all",
        "upload",
        "forward to customer",
        "email attachment",
        "post publicly",
    )

    _UNAUTHORIZED_COMMITMENT_TERMS: tuple[str, ...] = (
        "we guarantee",
        "we commit",
        "approved",
        "i approved",
        "final offer",
        "locked price",
        "contract signed",
        "you have my word",
        "will deliver by",
        "we will refund",
    )

    _DESTRUCTIVE_OR_BYPASS_TERMS: tuple[str, ...] = (
        "delete",
        "drop table",
        "disable logging",
        "bypass approval",
        "ignore policy",
        "skip review",
        "override control",
        "turn off monitoring",
        "remove audit",
        "exfiltrate",
        "wipe",
    )

    _POLICY_RISK_TERMS: tuple[str, ...] = (
        "override",
        "exception",
        "bypass",
        "urgent send",
        "skip approval",
        "production data",
        "customer data",
        "send now",
    )

    def analyze(
        self,
        *,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
    ) -> SemanticAnalysis:
        content = request.content
        lowered_content = content.casefold()

        policy_result = self._build_policy_compliance_result(
            content=content,
            lowered_content=lowered_content,
            request=request,
            retrieval_context=retrieval_context,
        )
        data_leakage_result = self._build_keyword_dimension_result(
            dimension="data_leakage",
            content=content,
            lowered_content=lowered_content,
            keywords=self._DATA_LEAKAGE_TERMS,
            summary_low="No obvious sensitive-data disclosure signal detected by fallback heuristics.",
            summary_high="Fallback heuristics detected possible sensitive-data disclosure language.",
            retrieval_context=retrieval_context,
        )
        external_sharing_result = self._build_keyword_dimension_result(
            dimension="external_sharing",
            content=content,
            lowered_content=lowered_content,
            keywords=self._EXTERNAL_SHARING_TERMS,
            summary_low="No obvious risky external-sharing signal detected by fallback heuristics.",
            summary_high="Fallback heuristics detected possible external-sharing or export language.",
            retrieval_context=retrieval_context,
        )
        commitment_result = self._build_keyword_dimension_result(
            dimension="unauthorized_commitment",
            content=content,
            lowered_content=lowered_content,
            keywords=self._UNAUTHORIZED_COMMITMENT_TERMS,
            summary_low="No obvious unauthorized-commitment signal detected by fallback heuristics.",
            summary_high="Fallback heuristics detected possible commitment or promise language.",
            retrieval_context=retrieval_context,
        )
        destructive_result = self._build_keyword_dimension_result(
            dimension="destructive_or_bypass",
            content=content,
            lowered_content=lowered_content,
            keywords=self._DESTRUCTIVE_OR_BYPASS_TERMS,
            summary_low="No obvious destructive or workflow-bypass signal detected by fallback heuristics.",
            summary_high="Fallback heuristics detected possible destructive or control-bypass language.",
            retrieval_context=retrieval_context,
        )

        dimension_results = (
            policy_result,
            data_leakage_result,
            external_sharing_result,
            commitment_result,
            destructive_result,
        )

        recommended_verdict = self._recommend_verdict(
            dimension_results=dimension_results,
            retrieval_context=retrieval_context,
            request=request,
        )

        overall_confidence = self._compute_overall_confidence(
            dimension_results=dimension_results,
            retrieval_context=retrieval_context,
        )
        evidence_sufficiency = self._compute_evidence_sufficiency(dimension_results)
        rationale_quality = self._compute_rationale_quality(
            dimension_results=dimension_results,
            retrieval_context=retrieval_context,
        )
        uncertainty_flags = self._collect_global_uncertainty_flags(
            dimension_results=dimension_results,
            retrieval_context=retrieval_context,
        )

        return SemanticAnalysis(
            dimension_results=dimension_results,
            recommended_verdict=recommended_verdict,
            overall_confidence=overall_confidence,
            evidence_sufficiency=evidence_sufficiency,
            rationale_quality=rationale_quality,
            summary=self._build_summary(
                dimension_results=dimension_results,
                recommended_verdict=recommended_verdict,
                retrieval_context=retrieval_context,
            ),
            uncertainty_flags=uncertainty_flags,
            provider_name="fallback",
            model_name="heuristic-semantic-fallback-v2",
            analyzed_at=datetime.now(UTC),
            metadata={
                "fallback_used": True,
                "retrieval_empty": retrieval_context.is_empty,
                "dimension_order": list(semantic_dimensions()),
            },
        )

    # ── dimension builders ─────────────────────────────────────────

    def _build_policy_compliance_result(
        self,
        *,
        content: str,
        lowered_content: str,
        request: EvaluationRequest,
        retrieval_context: RetrievalContext,
    ) -> SemanticDimensionResult:
        keyword_hits = self._match_keywords(
            content=content,
            lowered_content=lowered_content,
            keywords=self._POLICY_RISK_TERMS,
        )

        clause_hits = []
        matched_clause_ids: list[str] = []

        for clause in retrieval_context.policy_clauses:
            clause_text = clause.text.casefold()
            if any(term in lowered_content for term in self._tokenize_policy_clause(clause_text)):
                clause_hits.append(
                    SemanticEvidenceSpan(
                        text=clause.text[:300],
                        explanation=f"Retrieved policy clause {clause.clause_id} may be relevant to this action.",
                    )
                )
                matched_clause_ids.append(clause.clause_id)

        evidence_spans = tuple((*keyword_hits, *clause_hits))
        hit_count = len(evidence_spans)

        if hit_count == 0 and not clause_hits:
            # ── clean content: low score, solid confidence ──
            # Score stays well under permit_threshold (0.34).
            # Confidence stays above is_low_confidence (0.50) so the
            # router does not auto-ABSTAIN on this dimension.
            if retrieval_context.is_empty:
                score = 0.06
                confidence = 0.56
                summary = "No policy-risk signals detected. Retrieval context was unavailable but no lexical risk was found."
                uncertainty_flags = ("fallback_analysis",)
            else:
                score = 0.04
                confidence = min(
                    0.80,
                    0.60 + (0.04 * min(len(retrieval_context.policy_clauses), 4)),
                )
                summary = "No clear policy conflict was detected by fallback heuristics against retrieved context."
                uncertainty_flags = ("fallback_analysis",)
        else:
            # ── risky content: escalating scores ──
            if retrieval_context.is_empty:
                score = 0.35 if keyword_hits else 0.20
                confidence = 0.52 if keyword_hits else 0.50
                summary = "Policy grounding is limited because no retrieval context was available."
                uncertainty_flags = (
                    "no_retrieval_context",
                    "policy_grounding_weak",
                )
            else:
                score = min(1.0, 0.18 + (0.14 * len(keyword_hits)) + (0.10 * len(clause_hits)))
                confidence = min(
                    0.80,
                    0.55 + (0.06 * min(len(retrieval_context.policy_clauses), 4)),
                )
                summary = (
                    "Fallback heuristics found potential policy-relevant signals and used "
                    "retrieved policy clauses for light grounding."
                )
                uncertainty_flags = ("fallback_policy_analysis",)

        return SemanticDimensionResult(
            dimension="policy_compliance",
            score=score,
            confidence=confidence,
            summary=summary,
            rationale=(
                "This fallback result is heuristic only. It uses simple lexical matching plus "
                "light policy-clause grounding and should be treated as conservative support for routing, "
                "not a substitute for a stronger semantic model."
            ),
            evidence_spans=evidence_spans,
            matched_policy_clause_ids=tuple(dict.fromkeys(matched_clause_ids)),
            uncertainty_flags=uncertainty_flags,
        )

    def _build_keyword_dimension_result(
        self,
        *,
        dimension: str,
        content: str,
        lowered_content: str,
        keywords: tuple[str, ...],
        summary_low: str,
        summary_high: str,
        retrieval_context: RetrievalContext,
    ) -> SemanticDimensionResult:
        hits = self._match_keywords(
            content=content,
            lowered_content=lowered_content,
            keywords=keywords,
        )

        hit_count = len(hits)
        uncertainty_flags: list[str] = ["fallback_analysis"]

        if retrieval_context.is_empty:
            uncertainty_flags.append("no_retrieval_context")

        if hit_count == 0:
            # ── clean dimension: low score, confidence above 0.50 ──
            # This is the critical fix. The old fallback set zero-hit
            # confidence at 0.34 then capped it at 0.42. That produced
            # is_low_confidence=True on every clean dimension, which
            # caused the router to auto-ABSTAIN via _should_abstain().
            #
            # New values: score is negligible (0.04), confidence is 0.58.
            # This keeps the dimension above the 0.50 low-confidence
            # cutoff while still being honest that this is fallback-only.
            score = 0.04
            confidence = 0.58
        else:
            # ── risky dimension: escalating score and confidence ──
            score = min(1.0, 0.08 + (0.22 * hit_count))
            confidence = min(0.82, 0.52 + (0.10 * hit_count))

        return SemanticDimensionResult(
            dimension=dimension,
            score=score,
            confidence=confidence,
            summary=summary_high if hit_count else summary_low,
            rationale=(
                "This result comes from deterministic fallback keyword matching. "
                "It provides conservative continuity when the primary semantic layer is unavailable."
            ),
            evidence_spans=hits,
            matched_policy_clause_ids=tuple(),
            uncertainty_flags=tuple(uncertainty_flags),
        )

    # ── verdict recommendation ─────────────────────────────────────

    def _recommend_verdict(
        self,
        *,
        dimension_results: tuple[SemanticDimensionResult, ...],
        retrieval_context: RetrievalContext,
        request: EvaluationRequest,
    ) -> SemanticVerdictRecommendation:
        max_score = max(result.score for result in dimension_results)
        min_confidence = min(result.confidence for result in dimension_results)
        total_evidence_spans = sum(
            len(result.evidence_spans) for result in dimension_results
        )
        critical_dimensions = [
            result.dimension for result in dimension_results if result.score >= 0.78
        ]
        medium_risk_dimensions = [
            result.dimension for result in dimension_results if result.score >= 0.45
        ]

        uncertainty_flags: list[str] = ["fallback_recommendation"]

        high_impact_channel = request.channel.casefold() in {
            "email",
            "api",
            "export",
            "webhook",
            "external_message",
        }

        # ── FORBID: high-risk dimensions crossed critical threshold ──
        if critical_dimensions:
            return SemanticVerdictRecommendation(
                verdict=Verdict.FORBID,
                confidence=min(0.82, 0.56 + (0.08 * len(critical_dimensions))),
                summary="Fallback heuristics recommend FORBID because one or more dimensions crossed a high-risk threshold.",
                rationale=(
                    "The fallback layer detected high-risk language strongly associated with unsafe release. "
                    "A stricter semantic pass or human review can still inspect this later, but defaulting to release would be weak."
                ),
                uncertainty_flags=tuple((*uncertainty_flags, "high_risk_detected")),
            )

        # ── ABSTAIN: medium-risk signals on a high-impact channel ──
        if medium_risk_dimensions:
            flags = list(uncertainty_flags)
            flags.append("borderline_risk")

            if retrieval_context.is_empty:
                flags.append("missing_grounding")

            if high_impact_channel or min_confidence < 0.50:
                return SemanticVerdictRecommendation(
                    verdict=Verdict.ABSTAIN,
                    confidence=0.55,
                    summary="Fallback heuristics recommend ABSTAIN because the action is borderline or context-sensitive.",
                    rationale=(
                        "The fallback layer saw enough risk signal that automatic release would be weak, "
                        "but not enough grounded certainty to hard-forbid confidently."
                    ),
                    uncertainty_flags=tuple(flags),
                )

        # ── ABSTAIN: evidence present but below medium-risk threshold ──
        # Only escalate when we actually found *something* concerning.
        # The old version escalated here whenever retrieval was empty OR
        # min_confidence < 0.40, which trapped clean content in ABSTAIN.
        # Threshold is 0.30 (not 0.20) because light clause matches on
        # benign content can push policy_compliance to ~0.28 without any
        # real risk signal — those should not block PERMIT.
        if total_evidence_spans > 0 and max_score >= 0.30:
            flags = list(uncertainty_flags)
            if retrieval_context.is_empty:
                flags.append("missing_grounding")
            flags.append("low_risk_signals_present")
            return SemanticVerdictRecommendation(
                verdict=Verdict.ABSTAIN,
                confidence=0.52,
                summary="Fallback heuristics recommend ABSTAIN because some risk signals were detected but they did not reach a high-risk threshold.",
                rationale=(
                    "The fallback layer found low-level lexical signals. This does not warrant "
                    "hard-blocking but the presence of any signal means automatic release is premature."
                ),
                uncertainty_flags=tuple(flags),
            )

        # ── PERMIT: no meaningful risk signals detected ──
        # This path is now reachable for clean content because:
        # - zero keyword hits → max_score ~0.04-0.06, well below 0.20
        # - zero evidence spans → total_evidence_spans == 0
        # - confidence ≥ 0.50 everywhere → no low-confidence triggers
        return SemanticVerdictRecommendation(
            verdict=Verdict.PERMIT,
            confidence=0.68,
            summary="Fallback heuristics recommend PERMIT because no meaningful risk signals were detected.",
            rationale=(
                "The fallback layer found no substantial lexical evidence of policy conflict, leakage, "
                "external sharing abuse, unauthorized commitments, or destructive bypass behavior."
            ),
            uncertainty_flags=tuple(uncertainty_flags),
        )

    # ── aggregate metrics ──────────────────────────────────────────

    def _compute_overall_confidence(
        self,
        *,
        dimension_results: tuple[SemanticDimensionResult, ...],
        retrieval_context: RetrievalContext,
    ) -> float:
        average_confidence = sum(result.confidence for result in dimension_results) / len(dimension_results)

        if retrieval_context.is_empty:
            # Smaller penalty than before (-0.04 vs -0.10) so clean
            # content without retrieval can still pass the router's
            # minimum_confidence threshold after weighted fusion.
            return max(0.0, round(average_confidence - 0.04, 4))

        grounded_bonus = min(0.08, 0.02 * len(retrieval_context.policy_clauses))
        return min(1.0, round(average_confidence + grounded_bonus, 4))

    def _compute_evidence_sufficiency(
        self,
        dimension_results: tuple[SemanticDimensionResult, ...],
    ) -> float:
        span_count = sum(len(result.evidence_spans) for result in dimension_results)
        if span_count == 0:
            # Raised from 0.18 to 0.32 so the router does not penalize
            # clean content for having no evidence (there is nothing to
            # find, so evidence sufficiency should not be critically low).
            return 0.32
        return min(1.0, round(0.20 + (0.10 * span_count), 4))

    def _compute_rationale_quality(
        self,
        *,
        dimension_results: tuple[SemanticDimensionResult, ...],
        retrieval_context: RetrievalContext,
    ) -> float:
        base = 0.42
        if any(result.evidence_spans for result in dimension_results):
            base += 0.10
        if not retrieval_context.is_empty:
            base += 0.10
        return min(1.0, round(base, 4))

    # ── uncertainty flags ──────────────────────────────────────────

    def _collect_global_uncertainty_flags(
        self,
        *,
        dimension_results: tuple[SemanticDimensionResult, ...],
        retrieval_context: RetrievalContext,
    ) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()

        if retrieval_context.is_empty:
            seen.add("no_retrieval_context")
            ordered.append("no_retrieval_context")

        seen.add("fallback_used")
        ordered.append("fallback_used")

        for result in dimension_results:
            for flag in result.uncertainty_flags:
                key = flag.casefold()
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(flag)

        return tuple(ordered)

    # ── summary ────────────────────────────────────────────────────

    def _build_summary(
        self,
        *,
        dimension_results: tuple[SemanticDimensionResult, ...],
        recommended_verdict: SemanticVerdictRecommendation,
        retrieval_context: RetrievalContext,
    ) -> str:
        highest = max(dimension_results, key=lambda result: result.score)
        grounding_state = (
            "retrieval-grounded"
            if not retrieval_context.is_empty
            else "ungrounded"
        )
        return (
            f"Fallback semantic analysis completed with a recommended verdict of "
            f"{recommended_verdict.verdict.value}. Highest-risk dimension: "
            f"{highest.dimension} ({highest.score:.2f}). Analysis mode: {grounding_state}."
        )

    # ── keyword matching ───────────────────────────────────────────

    def _match_keywords(
        self,
        *,
        content: str,
        lowered_content: str,
        keywords: tuple[str, ...],
    ) -> tuple[SemanticEvidenceSpan, ...]:
        spans: list[SemanticEvidenceSpan] = []

        for keyword in keywords:
            lowered_keyword = keyword.casefold()
            start_index = lowered_content.find(lowered_keyword)
            if start_index == -1:
                continue
            end_index = start_index + len(lowered_keyword)
            spans.append(
                SemanticEvidenceSpan(
                    text=content[start_index:end_index],
                    start_index=start_index,
                    end_index=end_index,
                    explanation=f"Matched fallback heuristic keyword: {keyword}",
                )
            )

        return tuple(spans)

    def _tokenize_policy_clause(self, clause_text: str) -> tuple[str, ...]:
        """
        Very light clause token extraction for fallback grounding.

        This is intentionally crude. It only tries to pull longer lexical units
        that may overlap with risky action content. Generic English and
        policy boilerplate is filtered via a shared stopword set so common
        prose words like 'following', 'requires', or 'review' cannot cause
        false-positive overlaps against benign request content.
        """
        from tex.specialists.judges import _CLAUSE_TOKEN_STOPWORDS

        raw_tokens = clause_text.replace(",", " ").replace(".", " ").split()
        tokens = []
        for token in raw_tokens:
            normalized = token.strip().casefold()
            if len(normalized) < 8:
                continue
            if normalized in _CLAUSE_TOKEN_STOPWORDS:
                continue
            tokens.append(normalized)
        return tuple(dict.fromkeys(tokens))