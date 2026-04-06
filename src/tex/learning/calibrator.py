from __future__ import annotations

from dataclasses import dataclass

from tex.domain.policy import PolicySnapshot
from tex.learning.outcomes import OutcomeSummary


@dataclass(frozen=True, slots=True)
class CalibrationRecommendation:
    """
    Conservative policy-threshold recommendation derived from labeled outcomes.

    Tex should learn by tightening or loosening a versioned policy snapshot,
    not by mutating core architecture or inventing unstable runtime behavior.
    """

    current_permit_threshold: float
    recommended_permit_threshold: float

    current_forbid_threshold: float
    recommended_forbid_threshold: float

    current_minimum_confidence: float
    recommended_minimum_confidence: float

    summary: OutcomeSummary
    reasons: tuple[str, ...]

    false_permit_rate: float
    false_forbid_rate: float
    abstain_review_rate: float
    unknown_rate: float

    sample_weight: float
    permit_threshold_delta: float
    forbid_threshold_delta: float
    minimum_confidence_delta: float

    @property
    def changed(self) -> bool:
        return (
            self.current_permit_threshold != self.recommended_permit_threshold
            or self.current_forbid_threshold != self.recommended_forbid_threshold
            or self.current_minimum_confidence != self.recommended_minimum_confidence
        )


class ThresholdCalibrator:
    """
    Produces conservative, sample-aware threshold recommendations.

    Calibration priorities:
    1. reduce false permits first
    2. reduce false forbids second
    3. preserve ABSTAIN as a safety valve
    4. avoid large swings from noisy or undersized batches

    Threshold semantics in Tex:
    - lower permit_threshold => PERMIT becomes harder
    - lower forbid_threshold => FORBID becomes easier
    - higher minimum_confidence => automatic PERMIT becomes harder
    """

    __slots__ = (
        "_minimum_sample_size",
        "_full_trust_sample_size",
        "_max_single_adjustment",
        "_max_confidence_adjustment",
        "_target_false_permit_rate",
        "_target_false_forbid_rate",
        "_high_abstain_review_rate",
        "_high_unknown_rate",
        "_minimum_abstain_band",
        "_round_digits",
    )

    def __init__(
        self,
        *,
        minimum_sample_size: int = 12,
        full_trust_sample_size: int = 50,
        max_single_adjustment: float = 0.05,
        max_confidence_adjustment: float = 0.03,
        target_false_permit_rate: float = 0.04,
        target_false_forbid_rate: float = 0.08,
        high_abstain_review_rate: float = 0.30,
        high_unknown_rate: float = 0.12,
        minimum_abstain_band: float = 0.10,
        round_digits: int = 4,
    ) -> None:
        if minimum_sample_size <= 0:
            raise ValueError("minimum_sample_size must be greater than 0")
        if full_trust_sample_size < minimum_sample_size:
            raise ValueError(
                "full_trust_sample_size must be >= minimum_sample_size"
            )
        if not 0.0 < max_single_adjustment <= 0.20:
            raise ValueError(
                "max_single_adjustment must be greater than 0.0 and <= 0.20"
            )
        if not 0.0 < max_confidence_adjustment <= 0.10:
            raise ValueError(
                "max_confidence_adjustment must be greater than 0.0 and <= 0.10"
            )
        if not 0.0 <= target_false_permit_rate < 1.0:
            raise ValueError("target_false_permit_rate must be in [0.0, 1.0)")
        if not 0.0 <= target_false_forbid_rate < 1.0:
            raise ValueError("target_false_forbid_rate must be in [0.0, 1.0)")
        if target_false_permit_rate > target_false_forbid_rate:
            raise ValueError(
                "target_false_permit_rate must be <= target_false_forbid_rate"
            )
        if not 0.0 <= high_abstain_review_rate < 1.0:
            raise ValueError("high_abstain_review_rate must be in [0.0, 1.0)")
        if not 0.0 <= high_unknown_rate < 1.0:
            raise ValueError("high_unknown_rate must be in [0.0, 1.0)")
        if not 0.05 <= minimum_abstain_band <= 0.30:
            raise ValueError("minimum_abstain_band must be between 0.05 and 0.30")
        if round_digits < 0:
            raise ValueError("round_digits must be >= 0")

        self._minimum_sample_size = minimum_sample_size
        self._full_trust_sample_size = full_trust_sample_size
        self._max_single_adjustment = max_single_adjustment
        self._max_confidence_adjustment = max_confidence_adjustment
        self._target_false_permit_rate = target_false_permit_rate
        self._target_false_forbid_rate = target_false_forbid_rate
        self._high_abstain_review_rate = high_abstain_review_rate
        self._high_unknown_rate = high_unknown_rate
        self._minimum_abstain_band = minimum_abstain_band
        self._round_digits = round_digits

    def recommend(
        self,
        *,
        policy: PolicySnapshot,
        summary: OutcomeSummary,
    ) -> CalibrationRecommendation:
        """
        Returns conservative threshold recommendations from observed outcomes.

        Logic:
        - false permits above target tighten the policy
        - false forbids above target loosen the policy, but only when they
          materially outweigh false permits
        - high abstain/unknown volume slightly raises confidence requirements
          rather than forcing stronger permits
        - low sample volume shrinks or suppresses adjustments
        """
        reasons: list[str] = []

        current_permit = policy.permit_threshold
        current_forbid = policy.forbid_threshold
        current_minimum_confidence = policy.minimum_confidence

        if summary.total < self._minimum_sample_size:
            reasons.append(
                f"Sample too small for calibration ({summary.total} < "
                f"{self._minimum_sample_size})."
            )
            return CalibrationRecommendation(
                current_permit_threshold=current_permit,
                recommended_permit_threshold=current_permit,
                current_forbid_threshold=current_forbid,
                recommended_forbid_threshold=current_forbid,
                current_minimum_confidence=current_minimum_confidence,
                recommended_minimum_confidence=current_minimum_confidence,
                summary=summary,
                reasons=tuple(reasons),
                false_permit_rate=self._safe_rate(summary.false_permits, summary.total),
                false_forbid_rate=self._safe_rate(summary.false_forbids, summary.total),
                abstain_review_rate=self._safe_rate(
                    summary.abstain_reviews,
                    summary.total,
                ),
                unknown_rate=self._safe_rate(summary.unknown, summary.total),
                sample_weight=0.0,
                permit_threshold_delta=0.0,
                forbid_threshold_delta=0.0,
                minimum_confidence_delta=0.0,
            )

        false_permit_rate = self._safe_rate(summary.false_permits, summary.total)
        false_forbid_rate = self._safe_rate(summary.false_forbids, summary.total)
        abstain_review_rate = self._safe_rate(summary.abstain_reviews, summary.total)
        unknown_rate = self._safe_rate(summary.unknown, summary.total)

        sample_weight = self._sample_weight(summary.total)

        recommended_permit = current_permit
        recommended_forbid = current_forbid
        recommended_minimum_confidence = current_minimum_confidence

        # False permits are the primary risk. Tighten first.
        false_permit_excess = max(
            0.0,
            false_permit_rate - self._target_false_permit_rate,
        )
        if false_permit_excess > 0.0:
            severity = self._relative_excess(
                observed=false_permit_rate,
                target=self._target_false_permit_rate,
                max_rate=0.35,
            )
            delta = self._bounded_delta(
                severity=severity,
                sample_weight=sample_weight,
                ceiling=self._max_single_adjustment,
            )

            recommended_permit -= delta
            recommended_forbid -= delta * 0.90
            recommended_minimum_confidence += min(
                self._max_confidence_adjustment,
                delta * 0.65,
            )

            reasons.append(
                "False permits exceed the target rate; tightening permit and "
                "forbid thresholds and raising minimum confidence."
            )

        # False forbids matter, but should not override elevated false permits.
        false_forbid_excess = max(
            0.0,
            false_forbid_rate - self._target_false_forbid_rate,
        )
        false_forbids_dominate = false_forbid_rate > false_permit_rate + 0.03
        if false_forbid_excess > 0.0 and false_forbids_dominate:
            severity = self._relative_excess(
                observed=false_forbid_rate,
                target=self._target_false_forbid_rate,
                max_rate=0.40,
            )
            delta = self._bounded_delta(
                severity=severity,
                sample_weight=sample_weight,
                ceiling=self._max_single_adjustment,
            )

            recommended_permit += delta * 0.85
            recommended_forbid += delta
            recommended_minimum_confidence -= min(
                self._max_confidence_adjustment,
                delta * 0.45,
            )

            reasons.append(
                "False forbids materially exceed false permits; easing thresholds "
                "slightly to reduce over-blocking."
            )

        # High abstain volume is not automatically bad, but persistent volume can
        # mean low evidence quality or overly permissive auto-release behavior.
        if abstain_review_rate >= self._high_abstain_review_rate:
            confidence_delta = self._bounded_delta(
                severity=self._relative_excess(
                    observed=abstain_review_rate,
                    target=self._high_abstain_review_rate,
                    max_rate=0.60,
                ),
                sample_weight=sample_weight,
                ceiling=self._max_confidence_adjustment,
            )
            recommended_minimum_confidence += max(
                confidence_delta * 0.70,
                min(self._max_confidence_adjustment, 0.008),
            )
            reasons.append(
                "Abstain-review volume is high; raising minimum confidence slightly "
                "to keep automatic permits conservative."
            )

        # High unknown volume is a signal that evidence quality or labeling quality
        # is weak. Do not force harder threshold shifts; just raise the quality bar.
        if unknown_rate >= self._high_unknown_rate:
            confidence_delta = self._bounded_delta(
                severity=self._relative_excess(
                    observed=unknown_rate,
                    target=self._high_unknown_rate,
                    max_rate=0.35,
                ),
                sample_weight=sample_weight,
                ceiling=self._max_confidence_adjustment,
            )
            recommended_minimum_confidence += max(
                confidence_delta,
                min(self._max_confidence_adjustment, 0.01),
            )
            reasons.append(
                "Unknown outcome volume is elevated; increasing minimum confidence "
                "to reduce unsupported automatic permits."
            )

        recommended_permit, recommended_forbid, recommended_minimum_confidence = (
            self._normalize_thresholds(
                permit_threshold=recommended_permit,
                forbid_threshold=recommended_forbid,
                minimum_confidence=recommended_minimum_confidence,
            )
        )

        if not reasons:
            reasons.append(
                "Observed outcome profile is within tolerance; no threshold change "
                "is justified."
            )

        return CalibrationRecommendation(
            current_permit_threshold=current_permit,
            recommended_permit_threshold=recommended_permit,
            current_forbid_threshold=current_forbid,
            recommended_forbid_threshold=recommended_forbid,
            current_minimum_confidence=current_minimum_confidence,
            recommended_minimum_confidence=recommended_minimum_confidence,
            summary=summary,
            reasons=tuple(reasons),
            false_permit_rate=round(false_permit_rate, self._round_digits),
            false_forbid_rate=round(false_forbid_rate, self._round_digits),
            abstain_review_rate=round(abstain_review_rate, self._round_digits),
            unknown_rate=round(unknown_rate, self._round_digits),
            sample_weight=round(sample_weight, self._round_digits),
            permit_threshold_delta=round(
                recommended_permit - current_permit,
                self._round_digits,
            ),
            forbid_threshold_delta=round(
                recommended_forbid - current_forbid,
                self._round_digits,
            ),
            minimum_confidence_delta=round(
                recommended_minimum_confidence - current_minimum_confidence,
                self._round_digits,
            ),
        )

    def apply_recommendation(
        self,
        *,
        policy: PolicySnapshot,
        recommendation: CalibrationRecommendation,
        new_version: str,
        metadata_updates: dict[str, object] | None = None,
        activate: bool | None = None,
    ) -> PolicySnapshot:
        """
        Returns a new immutable PolicySnapshot with calibrated thresholds applied.
        """
        normalized_version = new_version.strip()
        if not normalized_version:
            raise ValueError("new_version must not be blank")

        merged_metadata = dict(policy.metadata)
        merged_metadata.update(
            {
                "calibration_parent_version": policy.version,
                "calibration": {
                    "source_policy_version": policy.version,
                    "sample_total": recommendation.summary.total,
                    "sample_weight": recommendation.sample_weight,
                    "false_permit_rate": recommendation.false_permit_rate,
                    "false_forbid_rate": recommendation.false_forbid_rate,
                    "abstain_review_rate": recommendation.abstain_review_rate,
                    "unknown_rate": recommendation.unknown_rate,
                    "current_thresholds": {
                        "permit_threshold": recommendation.current_permit_threshold,
                        "forbid_threshold": recommendation.current_forbid_threshold,
                        "minimum_confidence": recommendation.current_minimum_confidence,
                    },
                    "recommended_thresholds": {
                        "permit_threshold": recommendation.recommended_permit_threshold,
                        "forbid_threshold": recommendation.recommended_forbid_threshold,
                        "minimum_confidence": (
                            recommendation.recommended_minimum_confidence
                        ),
                    },
                    "deltas": {
                        "permit_threshold_delta": recommendation.permit_threshold_delta,
                        "forbid_threshold_delta": recommendation.forbid_threshold_delta,
                        "minimum_confidence_delta": (
                            recommendation.minimum_confidence_delta
                        ),
                    },
                    "reasons": list(recommendation.reasons),
                    "changed": recommendation.changed,
                },
            }
        )

        if metadata_updates:
            merged_metadata.update(metadata_updates)

        update_payload: dict[str, object] = {
            "version": normalized_version,
            "permit_threshold": recommendation.recommended_permit_threshold,
            "forbid_threshold": recommendation.recommended_forbid_threshold,
            "minimum_confidence": recommendation.recommended_minimum_confidence,
            "metadata": merged_metadata,
        }

        if activate is not None:
            update_payload["is_active"] = activate

        return policy.model_copy(update=update_payload)

    def _sample_weight(self, total: int) -> float:
        """
        Returns a trust weight in [0.0, 1.0] based on sample volume.
        """
        if total < self._minimum_sample_size:
            return 0.0
        if total >= self._full_trust_sample_size:
            return 1.0

        span = self._full_trust_sample_size - self._minimum_sample_size
        if span <= 0:
            return 1.0

        return (total - self._minimum_sample_size) / span

    def _bounded_delta(
        self,
        *,
        severity: float,
        sample_weight: float,
        ceiling: float,
    ) -> float:
        """
        Converts normalized severity into a bounded adjustment.

        The square-root style curve makes early adjustments meaningful without
        allowing noisy high rates to cause extreme threshold swings.
        """
        normalized_severity = self._clamp(severity, lower=0.0, upper=1.0)
        normalized_weight = self._clamp(sample_weight, lower=0.0, upper=1.0)

        raw_delta = (normalized_severity**0.5) * normalized_weight * ceiling
        return round(raw_delta, self._round_digits)

    @staticmethod
    def _relative_excess(
        *,
        observed: float,
        target: float,
        max_rate: float,
    ) -> float:
        if observed <= target:
            return 0.0

        effective_span = max(max_rate - target, 1e-9)
        return min(1.0, (observed - target) / effective_span)

    def _normalize_thresholds(
        self,
        *,
        permit_threshold: float,
        forbid_threshold: float,
        minimum_confidence: float,
    ) -> tuple[float, float, float]:
        """
        Clamps thresholds into sane ranges and preserves a meaningful abstain band.
        """
        normalized_permit = self._clamp(permit_threshold, lower=0.05, upper=0.60)
        normalized_forbid = self._clamp(forbid_threshold, lower=0.40, upper=0.95)
        normalized_minimum_confidence = self._clamp(
            minimum_confidence,
            lower=0.40,
            upper=0.95,
        )

        minimum_gap = self._minimum_abstain_band

        if normalized_forbid <= normalized_permit + minimum_gap:
            normalized_forbid = min(0.95, normalized_permit + minimum_gap)

        if normalized_permit >= normalized_forbid - minimum_gap:
            normalized_permit = max(0.05, normalized_forbid - minimum_gap)

        return (
            round(normalized_permit, self._round_digits),
            round(normalized_forbid, self._round_digits),
            round(normalized_minimum_confidence, self._round_digits),
        )

    @staticmethod
    def _safe_rate(count: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return count / total

    @staticmethod
    def _clamp(value: float, *, lower: float, upper: float) -> float:
        return min(upper, max(lower, value))


def build_default_calibrator() -> ThresholdCalibrator:
    """Returns Tex's default conservative threshold calibrator."""
    return ThresholdCalibrator()