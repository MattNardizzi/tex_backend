from __future__ import annotations

from dataclasses import dataclass

from tex.domain.policy import PolicySnapshot
from tex.learning.calibrator import CalibrationRecommendation, ThresholdCalibrator
from tex.learning.outcomes import OutcomeClassification, OutcomeSummary, summarize_outcomes
from tex.stores.outcome_store import InMemoryOutcomeStore
from tex.stores.policy_store import InMemoryPolicyStore


@dataclass(frozen=True, slots=True)
class CalibratePolicyResult:
    """
    Application-layer result for a calibration pass.
    """

    source_policy: PolicySnapshot
    recommendation: CalibrationRecommendation
    summary: OutcomeSummary
    classifications: tuple[OutcomeClassification, ...]
    calibrated_policy: PolicySnapshot | None = None


class CalibratePolicyCommand:
    """
    Application service for deriving a calibrated policy snapshot from observed outcomes.

    Responsibilities:
    - resolve the source policy
    - summarize provided outcome classifications
    - ask the calibrator for a recommendation
    - optionally create and persist a new policy snapshot version

    This command intentionally does not discover outcomes by itself from the
    whole system. It consumes already-classified outcomes so the logic stays
    explicit and testable.
    """

    __slots__ = (
        "_policy_store",
        "_outcome_store",
        "_calibrator",
    )

    def __init__(
        self,
        *,
        policy_store: InMemoryPolicyStore,
        outcome_store: InMemoryOutcomeStore,
        calibrator: ThresholdCalibrator,
    ) -> None:
        self._policy_store = policy_store
        self._outcome_store = outcome_store
        self._calibrator = calibrator

    def execute(
        self,
        *,
        source_policy_version: str | None = None,
        classifications: tuple[OutcomeClassification, ...] | list[OutcomeClassification],
        new_version: str | None = None,
        save: bool = False,
        activate: bool = False,
        metadata_updates: dict[str, object] | None = None,
    ) -> CalibratePolicyResult:
        """
        Runs a calibration pass from a set of classified outcomes.

        Rules:
        - if source_policy_version is omitted, the active policy is used
        - if save is False, no new policy is created
        - if save is True, new_version is required
        - if activate is True, save must also be True
        """
        source_policy = self._resolve_source_policy(source_policy_version)

        normalized_classifications = tuple(classifications)
        summary = summarize_outcomes(normalized_classifications)

        recommendation = self._calibrator.recommend(
            policy=source_policy,
            summary=summary,
        )

        calibrated_policy: PolicySnapshot | None = None

        if activate and not save:
            raise ValueError("activate=True requires save=True")

        if save:
            if not new_version or not new_version.strip():
                raise ValueError("new_version is required when save=True")

            calibrated_policy = self._calibrator.apply_recommendation(
                policy=source_policy,
                recommendation=recommendation,
                new_version=new_version.strip(),
                metadata_updates=metadata_updates,
                activate=activate,
            )
            self._policy_store.save(calibrated_policy)

            if activate:
                self._policy_store.activate(calibrated_policy.version)

        return CalibratePolicyResult(
            source_policy=source_policy,
            recommendation=recommendation,
            summary=summary,
            classifications=normalized_classifications,
            calibrated_policy=calibrated_policy,
        )

    def execute_for_policy_outcomes(
        self,
        *,
        source_policy_version: str | None = None,
        decision_classifications: tuple[OutcomeClassification, ...]
        | list[OutcomeClassification],
        new_version: str | None = None,
        save: bool = False,
        activate: bool = False,
        metadata_updates: dict[str, object] | None = None,
    ) -> CalibratePolicyResult:
        """
        Convenience wrapper for the common case where the caller already filtered
        classifications for the target policy.
        """
        return self.execute(
            source_policy_version=source_policy_version,
            classifications=decision_classifications,
            new_version=new_version,
            save=save,
            activate=activate,
            metadata_updates=metadata_updates,
        )

    def _resolve_source_policy(self, version: str | None) -> PolicySnapshot:
        if version is not None:
            normalized = version.strip()
            if not normalized:
                raise ValueError("source_policy_version must not be blank")
            try:
                return self._policy_store.require(normalized)
            except KeyError as exc:
                raise LookupError(
                    f"source policy version not found: {normalized}"
                ) from exc

        try:
            return self._policy_store.require_active()
        except LookupError as exc:
            raise LookupError("no active policy is available for calibration") from exc