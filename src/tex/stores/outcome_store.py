from __future__ import annotations

from collections.abc import Iterable
from threading import RLock
from uuid import UUID

from tex.domain.outcome import OutcomeKind, OutcomeLabel, OutcomeRecord
from tex.domain.verdict import Verdict


class InMemoryOutcomeStore:
    """
    In-memory outcome store for local development, testing, and calibration work.

    Design goals:
    - strict alignment with the current OutcomeRecord domain contract
    - explicit indexes for the fields that actually exist
    - deterministic iteration order
    - small, boring behavior with no hidden side effects

    This store intentionally does not pretend permit_id exists, because the
    current OutcomeRecord model does not carry it.
    """

    __slots__ = (
        "_lock",
        "_by_id",
        "_ordered_ids",
        "_decision_index",
        "_request_index",
        "_kind_index",
        "_label_index",
    )

    def __init__(self, initial_outcomes: Iterable[OutcomeRecord] | None = None) -> None:
        self._lock = RLock()
        self._by_id: dict[UUID, OutcomeRecord] = {}
        self._ordered_ids: list[UUID] = []
        self._decision_index: dict[UUID, list[UUID]] = {}
        self._request_index: dict[UUID, list[UUID]] = {}
        self._kind_index: dict[OutcomeKind, list[UUID]] = {}
        self._label_index: dict[OutcomeLabel, list[UUID]] = {}

        if initial_outcomes is not None:
            for outcome in initial_outcomes:
                self.save(outcome)

    def save(self, outcome: OutcomeRecord) -> None:
        """
        Saves or replaces an outcome record.

        Re-saving the same outcome_id updates the stored record and moves it to
        the end of insertion order.
        """
        with self._lock:
            existing = self._by_id.get(outcome.outcome_id)
            if existing is not None:
                self._remove_from_indexes(existing)
                self._ordered_ids = [
                    stored_id
                    for stored_id in self._ordered_ids
                    if stored_id != outcome.outcome_id
                ]

            self._by_id[outcome.outcome_id] = outcome
            self._ordered_ids.append(outcome.outcome_id)
            self._add_to_indexes(outcome)

    def get(self, outcome_id: UUID) -> OutcomeRecord | None:
        """Returns an outcome by outcome_id, or None if missing."""
        with self._lock:
            return self._by_id.get(outcome_id)

    def require(self, outcome_id: UUID) -> OutcomeRecord:
        """Returns an outcome by outcome_id or raises KeyError."""
        outcome = self.get(outcome_id)
        if outcome is None:
            raise KeyError(f"outcome not found: {outcome_id}")
        return outcome

    def list_all(self) -> tuple[OutcomeRecord, ...]:
        """Returns all stored outcomes in insertion order."""
        with self._lock:
            return tuple(self._by_id[outcome_id] for outcome_id in self._ordered_ids)

    def list_recent(self, limit: int = 50) -> tuple[OutcomeRecord, ...]:
        """Returns the most recently saved outcomes, newest first."""
        if limit <= 0:
            return tuple()

        with self._lock:
            selected_ids = list(reversed(self._ordered_ids[-limit:]))
            return tuple(self._by_id[outcome_id] for outcome_id in selected_ids)

    def list_for_decision(self, decision_id: UUID) -> tuple[OutcomeRecord, ...]:
        """Returns all outcomes associated with a decision_id, oldest first."""
        with self._lock:
            outcome_ids = tuple(self._decision_index.get(decision_id, ()))
            return tuple(self._by_id[outcome_id] for outcome_id in outcome_ids)

    def list_for_request(self, request_id: UUID) -> tuple[OutcomeRecord, ...]:
        """Returns all outcomes associated with a request_id, oldest first."""
        with self._lock:
            outcome_ids = tuple(self._request_index.get(request_id, ()))
            return tuple(self._by_id[outcome_id] for outcome_id in outcome_ids)

    def list_for_kind(self, outcome_kind: OutcomeKind) -> tuple[OutcomeRecord, ...]:
        """Returns all outcomes for a specific OutcomeKind, oldest first."""
        with self._lock:
            outcome_ids = tuple(self._kind_index.get(outcome_kind, ()))
            return tuple(self._by_id[outcome_id] for outcome_id in outcome_ids)

    def list_for_label(self, label: OutcomeLabel) -> tuple[OutcomeRecord, ...]:
        """Returns all outcomes for a specific OutcomeLabel, oldest first."""
        with self._lock:
            outcome_ids = tuple(self._label_index.get(label, ()))
            return tuple(self._by_id[outcome_id] for outcome_id in outcome_ids)

    def find(
        self,
        *,
        decision_id: UUID | None = None,
        request_id: UUID | None = None,
        outcome_kind: OutcomeKind | None = None,
        label: OutcomeLabel | None = None,
        verdict: Verdict | None = None,
        was_safe: bool | None = None,
        human_override: bool | None = None,
        reporter: str | None = None,
        limit: int | None = None,
    ) -> tuple[OutcomeRecord, ...]:
        """
        Returns outcomes matching the supplied filters.

        Results are newest first because that is the most useful default for
        operational inspection.
        """
        normalized_reporter = reporter.strip() if reporter is not None else None
        if normalized_reporter == "":
            raise ValueError("reporter filter must not be blank")

        with self._lock:
            matched: list[OutcomeRecord] = []

            for outcome_id in reversed(self._ordered_ids):
                outcome = self._by_id[outcome_id]

                if decision_id is not None and outcome.decision_id != decision_id:
                    continue

                if request_id is not None and outcome.request_id != request_id:
                    continue

                if outcome_kind is not None and outcome.outcome_kind != outcome_kind:
                    continue

                if label is not None and outcome.label != label:
                    continue

                if verdict is not None and outcome.verdict != verdict:
                    continue

                if was_safe is not None and outcome.was_safe != was_safe:
                    continue

                if human_override is not None and outcome.human_override != human_override:
                    continue

                if normalized_reporter is not None and outcome.reporter != normalized_reporter:
                    continue

                matched.append(outcome)

                if limit is not None and len(matched) >= limit:
                    break

            return tuple(matched)

    def delete(self, outcome_id: UUID) -> None:
        """Deletes a stored outcome by outcome_id."""
        with self._lock:
            outcome = self._by_id.get(outcome_id)
            if outcome is None:
                raise KeyError(f"outcome not found: {outcome_id}")

            self._remove_from_indexes(outcome)
            del self._by_id[outcome_id]
            self._ordered_ids = [
                stored_id
                for stored_id in self._ordered_ids
                if stored_id != outcome_id
            ]

    def clear(self) -> None:
        """Removes all stored outcomes and resets indexes."""
        with self._lock:
            self._by_id.clear()
            self._ordered_ids.clear()
            self._decision_index.clear()
            self._request_index.clear()
            self._kind_index.clear()
            self._label_index.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_id)

    def __contains__(self, outcome_id: object) -> bool:
        if not isinstance(outcome_id, UUID):
            return False
        with self._lock:
            return outcome_id in self._by_id

    def _add_to_indexes(self, outcome: OutcomeRecord) -> None:
        self._decision_index.setdefault(outcome.decision_id, []).append(outcome.outcome_id)
        self._request_index.setdefault(outcome.request_id, []).append(outcome.outcome_id)
        self._kind_index.setdefault(outcome.outcome_kind, []).append(outcome.outcome_id)
        self._label_index.setdefault(outcome.label, []).append(outcome.outcome_id)

    def _remove_from_indexes(self, outcome: OutcomeRecord) -> None:
        self._remove_id_from_bucket(
            index=self._decision_index,
            key=outcome.decision_id,
            outcome_id=outcome.outcome_id,
        )
        self._remove_id_from_bucket(
            index=self._request_index,
            key=outcome.request_id,
            outcome_id=outcome.outcome_id,
        )
        self._remove_id_from_bucket(
            index=self._kind_index,
            key=outcome.outcome_kind,
            outcome_id=outcome.outcome_id,
        )
        self._remove_id_from_bucket(
            index=self._label_index,
            key=outcome.label,
            outcome_id=outcome.outcome_id,
        )

    @staticmethod
    def _remove_id_from_bucket(
        *,
        index: dict[object, list[UUID]],
        key: object,
        outcome_id: UUID,
    ) -> None:
        bucket = index.get(key)
        if bucket is None:
            return

        updated_bucket = [stored_id for stored_id in bucket if stored_id != outcome_id]
        if updated_bucket:
            index[key] = updated_bucket
        else:
            del index[key]