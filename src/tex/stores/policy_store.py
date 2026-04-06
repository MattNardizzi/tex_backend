from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

from tex.domain.policy import PolicySnapshot


class InMemoryPolicyStore:
    """
    Simple in-memory policy store for local development and testing.

    This store supports:
    - saving versioned policy snapshots
    - resolving by exact version
    - resolving the latest snapshot for a stable policy_id
    - resolving the active policy
    - activating a specific version while deactivating all others

    It is intentionally small and explicit. No hidden persistence, no magic.
    """

    __slots__ = (
        "_lock",
        "_by_version",
        "_ordered_versions",
        "_versions_by_policy_id",
    )

    def __init__(self, initial_policies: Iterable[PolicySnapshot] | None = None) -> None:
        self._lock = RLock()
        self._by_version: dict[str, PolicySnapshot] = {}
        self._ordered_versions: list[str] = []
        self._versions_by_policy_id: dict[str, list[str]] = {}

        if initial_policies is not None:
            for policy in initial_policies:
                self.save(policy)

    def save(self, policy: PolicySnapshot) -> None:
        """
        Saves or replaces a policy snapshot by exact version.

        Ordering is preserved by last save, so the most recently saved policy
        version appears last in global listings and policy-family listings.
        """
        with self._lock:
            existing = self._by_version.get(policy.version)
            if existing is not None:
                self._remove_version_references(
                    version=existing.version,
                    policy_id=existing.policy_id,
                )

            self._by_version[policy.version] = policy
            self._ordered_versions.append(policy.version)
            self._versions_by_policy_id.setdefault(policy.policy_id, []).append(policy.version)

    def get(self, version: str) -> PolicySnapshot | None:
        """Returns a policy snapshot by exact version, or None if missing."""
        with self._lock:
            return self._by_version.get(version)

    def require(self, version: str) -> PolicySnapshot:
        """Returns a policy snapshot by exact version or raises KeyError."""
        policy = self.get(version)
        if policy is None:
            raise KeyError(f"policy version not found: {version}")
        return policy

    def get_by_policy_id(self, policy_id: str) -> PolicySnapshot | None:
        """
        Returns the most recently saved snapshot for a stable policy_id.

        This does not require the snapshot to be active.
        """
        with self._lock:
            versions = self._versions_by_policy_id.get(policy_id)
            if not versions:
                return None
            latest_version = versions[-1]
            return self._by_version[latest_version]

    def require_by_policy_id(self, policy_id: str) -> PolicySnapshot:
        """Returns the latest snapshot for a policy_id or raises KeyError."""
        policy = self.get_by_policy_id(policy_id)
        if policy is None:
            raise KeyError(f"policy_id not found: {policy_id}")
        return policy

    def list_versions(self, policy_id: str | None = None) -> tuple[str, ...]:
        """
        Returns stored versions in save order.

        When policy_id is provided, only versions belonging to that policy family
        are returned.
        """
        with self._lock:
            if policy_id is None:
                return tuple(self._ordered_versions)
            return tuple(self._versions_by_policy_id.get(policy_id, []))

    def list_policies(self, policy_id: str | None = None) -> tuple[PolicySnapshot, ...]:
        """
        Returns stored policies in save order.

        When policy_id is provided, only policies belonging to that policy family
        are returned.
        """
        with self._lock:
            if policy_id is None:
                versions = self._ordered_versions
            else:
                versions = self._versions_by_policy_id.get(policy_id, [])

            return tuple(self._by_version[version] for version in versions)

    def get_active(self) -> PolicySnapshot | None:
        """
        Returns the active policy snapshot.

        If multiple active policies somehow exist, the most recently saved one wins.
        That should not be the normal state, but this makes resolution explicit.
        """
        with self._lock:
            for version in reversed(self._ordered_versions):
                policy = self._by_version[version]
                if policy.is_active:
                    return policy
            return None

    def require_active(self) -> PolicySnapshot:
        """Returns the active policy snapshot or raises LookupError."""
        policy = self.get_active()
        if policy is None:
            raise LookupError("no active policy is available")
        return policy

    def activate(self, version: str) -> PolicySnapshot:
        """
        Activates the requested policy version and deactivates all others.

        Because PolicySnapshot is immutable, activation produces replacement
        snapshots rather than mutating the existing objects in place.
        """
        with self._lock:
            target = self._by_version.get(version)
            if target is None:
                raise KeyError(f"policy version not found: {version}")

            for existing_version in list(self._ordered_versions):
                existing = self._by_version[existing_version]
                should_be_active = existing_version == version

                if existing.is_active == should_be_active:
                    continue

                self._by_version[existing_version] = existing.model_copy(
                    update={"is_active": should_be_active}
                )

            return self._by_version[version]

    def delete(self, version: str) -> None:
        """
        Deletes a stored policy version.

        This is useful for local development/testing only. In a real durable
        system, policy deletion might be disallowed or replaced by archival.
        """
        with self._lock:
            policy = self._by_version.get(version)
            if policy is None:
                raise KeyError(f"policy version not found: {version}")

            del self._by_version[version]
            self._remove_version_references(
                version=version,
                policy_id=policy.policy_id,
            )

    def clear(self) -> None:
        """Removes all stored policies."""
        with self._lock:
            self._by_version.clear()
            self._ordered_versions.clear()
            self._versions_by_policy_id.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_version)

    def __contains__(self, version: object) -> bool:
        if not isinstance(version, str):
            return False
        with self._lock:
            return version in self._by_version

    def _remove_version_references(self, *, version: str, policy_id: str) -> None:
        self._ordered_versions = [
            stored_version
            for stored_version in self._ordered_versions
            if stored_version != version
        ]

        family_versions = self._versions_by_policy_id.get(policy_id, [])
        updated_family_versions = [
            stored_version
            for stored_version in family_versions
            if stored_version != version
        ]

        if updated_family_versions:
            self._versions_by_policy_id[policy_id] = updated_family_versions
        else:
            self._versions_by_policy_id.pop(policy_id, None)