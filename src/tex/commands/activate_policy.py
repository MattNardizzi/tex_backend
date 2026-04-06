from __future__ import annotations

from dataclasses import dataclass

from tex.domain.policy import PolicySnapshot
from tex.stores.policy_store import InMemoryPolicyStore


@dataclass(frozen=True, slots=True)
class ActivatePolicyResult:
    """
    Application-layer result for activating a policy snapshot.
    """

    activated_policy: PolicySnapshot
    previous_active_policy: PolicySnapshot | None


class ActivatePolicyCommand:
    """
    Application service for activating a specific policy version.

    Responsibilities:
    - resolve the target policy version
    - capture the currently active policy, if any
    - activate the requested version through the policy store

    This stays intentionally narrow and does not perform calibration or
    evidence recording by itself.
    """

    __slots__ = ("_policy_store",)

    def __init__(self, *, policy_store: InMemoryPolicyStore) -> None:
        self._policy_store = policy_store

    def execute(self, version: str) -> ActivatePolicyResult:
        """
        Activates the requested policy version and returns the before/after state.
        """
        if not isinstance(version, str):
            raise TypeError("version must be a string")

        normalized_version = version.strip()
        if not normalized_version:
            raise ValueError("version must not be blank")

        previous_active = self._policy_store.get_active()

        try:
            activated_policy = self._policy_store.activate(normalized_version)
        except KeyError as exc:
            raise LookupError(
                f"cannot activate missing policy version: {normalized_version}"
            ) from exc

        return ActivatePolicyResult(
            activated_policy=activated_policy,
            previous_active_policy=previous_active,
        )