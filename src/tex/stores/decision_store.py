from __future__ import annotations

from collections.abc import Iterable
from threading import RLock
from uuid import UUID

from tex.domain.decision import Decision
from tex.domain.verdict import Verdict


class InMemoryDecisionStore:
    """
    Simple in-memory decision store for local development and testing.

    This store is intentionally explicit and small. It supports:
    - save/update by decision_id
    - lookup by decision_id
    - lookup by request_id
    - ordered listing
    - simple filtering by verdict, policy version, channel, environment, action type
    """

    __slots__ = (
        "_lock",
        "_by_id",
        "_ordered_ids",
        "_by_request_id",
    )

    def __init__(self, initial_decisions: Iterable[Decision] | None = None) -> None:
        self._lock = RLock()
        self._by_id: dict[UUID, Decision] = {}
        self._ordered_ids: list[UUID] = []
        self._by_request_id: dict[UUID, UUID] = {}

        if initial_decisions is not None:
            for decision in initial_decisions:
                self.save(decision)

    def save(self, decision: Decision) -> None:
        """
        Saves or replaces a decision.

        Re-saving the same decision_id updates the stored record and moves that
        decision to the end of the ordered list.
        """
        with self._lock:
            existing = self._by_id.get(decision.decision_id)
            if existing is not None:
                self._ordered_ids = [
                    stored_id
                    for stored_id in self._ordered_ids
                    if stored_id != decision.decision_id
                ]
                if existing.request_id in self._by_request_id:
                    del self._by_request_id[existing.request_id]

            self._by_id[decision.decision_id] = decision
            self._ordered_ids.append(decision.decision_id)
            self._by_request_id[decision.request_id] = decision.decision_id

    def get(self, decision_id: UUID) -> Decision | None:
        """Returns a decision by decision_id, or None if missing."""
        with self._lock:
            return self._by_id.get(decision_id)

    def require(self, decision_id: UUID) -> Decision:
        """Returns a decision by decision_id or raises KeyError."""
        decision = self.get(decision_id)
        if decision is None:
            raise KeyError(f"decision not found: {decision_id}")
        return decision

    def get_by_request_id(self, request_id: UUID) -> Decision | None:
        """Returns the decision associated with a request_id, if present."""
        with self._lock:
            decision_id = self._by_request_id.get(request_id)
            if decision_id is None:
                return None
            return self._by_id.get(decision_id)

    def require_by_request_id(self, request_id: UUID) -> Decision:
        """Returns the decision associated with a request_id or raises KeyError."""
        decision = self.get_by_request_id(request_id)
        if decision is None:
            raise KeyError(f"decision for request_id not found: {request_id}")
        return decision

    def list_all(self) -> tuple[Decision, ...]:
        """Returns all stored decisions in save order."""
        with self._lock:
            return tuple(self._by_id[decision_id] for decision_id in self._ordered_ids)

    def list_recent(self, limit: int = 50) -> tuple[Decision, ...]:
        """Returns the most recently saved decisions, newest first."""
        if limit <= 0:
            return tuple()

        with self._lock:
            selected_ids = list(reversed(self._ordered_ids[-limit:]))
            return tuple(self._by_id[decision_id] for decision_id in selected_ids)

    def find(
        self,
        *,
        verdict: Verdict | None = None,
        policy_version: str | None = None,
        channel: str | None = None,
        environment: str | None = None,
        action_type: str | None = None,
        limit: int | None = None,
    ) -> tuple[Decision, ...]:
        """
        Returns decisions matching the supplied filters.

        Results are returned newest first because this is the more operationally
        useful default for inspection and precedent lookup.
        """
        with self._lock:
            matched: list[Decision] = []

            for decision_id in reversed(self._ordered_ids):
                decision = self._by_id[decision_id]

                if verdict is not None and decision.verdict != verdict:
                    continue

                if policy_version is not None and decision.policy_version != policy_version:
                    continue

                if channel is not None and decision.channel != channel:
                    continue

                if environment is not None and decision.environment != environment:
                    continue

                if action_type is not None and decision.action_type != action_type:
                    continue

                matched.append(decision)

                if limit is not None and len(matched) >= limit:
                    break

            return tuple(matched)

    def delete(self, decision_id: UUID) -> None:
        """Deletes a stored decision by decision_id."""
        with self._lock:
            decision = self._by_id.get(decision_id)
            if decision is None:
                raise KeyError(f"decision not found: {decision_id}")

            del self._by_id[decision_id]
            self._ordered_ids = [
                stored_id
                for stored_id in self._ordered_ids
                if stored_id != decision_id
            ]

            if self._by_request_id.get(decision.request_id) == decision_id:
                del self._by_request_id[decision.request_id]

    def clear(self) -> None:
        """Removes all stored decisions."""
        with self._lock:
            self._by_id.clear()
            self._ordered_ids.clear()
            self._by_request_id.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_id)

    def __contains__(self, decision_id: object) -> bool:
        if not isinstance(decision_id, UUID):
            return False
        with self._lock:
            return decision_id in self._by_id