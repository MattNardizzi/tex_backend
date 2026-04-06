from __future__ import annotations

from collections.abc import Iterable
from threading import RLock
from uuid import UUID

from tex.domain.decision import Decision
from tex.domain.retrieval import RetrievedPrecedent
from tex.domain.verdict import Verdict


class InMemoryPrecedentStore:
    """
    In-memory precedent store for local development, tests, and early retrieval.

    The durable source of truth remains the internal ``Decision`` model. This
    store projects those decisions into ``RetrievedPrecedent`` records that are
    safe for the retrieval layer to consume.

    Current behavior is intentionally simple:
    - newest decisions are preferred
    - matching is metadata-based, not semantic
    - ranking is deterministic and stable
    - duplicate decision IDs replace the old record
    """

    __slots__ = ("_lock", "_by_decision_id", "_ordered_ids")

    def __init__(self, initial_decisions: Iterable[Decision] | None = None) -> None:
        self._lock = RLock()
        self._by_decision_id: dict[UUID, Decision] = {}
        self._ordered_ids: list[UUID] = []

        if initial_decisions is not None:
            for decision in initial_decisions:
                self.save(decision)

    def save(self, decision: Decision) -> None:
        """
        Save or replace a decision as a precedent candidate.

        Re-saving an existing ``decision_id`` updates the stored record and
        moves it to the newest position.
        """
        with self._lock:
            if decision.decision_id in self._by_decision_id:
                self._ordered_ids = [
                    stored_id
                    for stored_id in self._ordered_ids
                    if stored_id != decision.decision_id
                ]

            self._by_decision_id[decision.decision_id] = decision
            self._ordered_ids.append(decision.decision_id)

    def save_many(self, decisions: Iterable[Decision]) -> None:
        """Save multiple decisions in iteration order."""
        for decision in decisions:
            self.save(decision)

    def get(self, decision_id: UUID) -> RetrievedPrecedent | None:
        """Return one stored precedent projection by ``decision_id``."""
        with self._lock:
            decision = self._by_decision_id.get(decision_id)
            if decision is None:
                return None
            return self._to_precedent(decision, relevance_score=1.0, rank=1)

    def require(self, decision_id: UUID) -> RetrievedPrecedent:
        """Return one stored precedent projection or raise ``KeyError``."""
        precedent = self.get(decision_id)
        if precedent is None:
            raise KeyError(f"precedent not found for decision_id: {decision_id}")
        return precedent

    def get_decision(self, decision_id: UUID) -> Decision | None:
        """Return the raw stored ``Decision`` object, or ``None`` if missing."""
        with self._lock:
            return self._by_decision_id.get(decision_id)

    def list_all(self) -> tuple[RetrievedPrecedent, ...]:
        """
        Return all stored precedents in save order.

        Ranks are assigned from oldest to newest because this method is an
        inspection view, not the similarity retrieval path.
        """
        with self._lock:
            precedents: list[RetrievedPrecedent] = []
            for index, decision_id in enumerate(self._ordered_ids, start=1):
                precedents.append(
                    self._to_precedent(
                        self._by_decision_id[decision_id],
                        relevance_score=1.0,
                        rank=index,
                    )
                )
            return tuple(precedents)

    def find_similar(
        self,
        *,
        action_type: str | None = None,
        channel: str | None = None,
        environment: str | None = None,
        recipient: str | None = None,
        verdict: Verdict | None = None,
        policy_version: str | None = None,
        exclude_decision_id: UUID | None = None,
        limit: int = 10,
    ) -> tuple[RetrievedPrecedent, ...]:
        """
        Return similar prior decisions using lightweight metadata filtering.

        Similarity is intentionally simple for now:
        - exact metadata filters
        - newest matching decisions first
        - deterministic rank assignment
        - a small heuristic relevance score based on filter overlap
        """
        if limit <= 0:
            return tuple()

        with self._lock:
            matched: list[RetrievedPrecedent] = []

            for decision_id in reversed(self._ordered_ids):
                if exclude_decision_id is not None and decision_id == exclude_decision_id:
                    continue

                decision = self._by_decision_id[decision_id]

                if action_type is not None and decision.action_type != action_type:
                    continue
                if channel is not None and decision.channel != channel:
                    continue
                if environment is not None and decision.environment != environment:
                    continue
                if recipient is not None and decision.recipient != recipient:
                    continue
                if verdict is not None and decision.verdict != verdict:
                    continue
                if policy_version is not None and decision.policy_version != policy_version:
                    continue

                matched.append(
                    self._to_precedent(
                        decision,
                        relevance_score=self._compute_relevance_score(
                            decision=decision,
                            action_type=action_type,
                            channel=channel,
                            environment=environment,
                            recipient=recipient,
                            verdict=verdict,
                            policy_version=policy_version,
                        ),
                        rank=len(matched) + 1,
                    )
                )

                if len(matched) >= limit:
                    break

            return tuple(matched)

    def retrieve_precedents(
        self,
        *,
        request,
        limit: int,
    ) -> tuple[RetrievedPrecedent, ...]:
        """
        Retrieval-orchestrator adapter method.

        This bridges the in-memory store directly into the
        ``tex.retrieval.orchestrator.PrecedentStore`` protocol without forcing
        a separate adapter class.
        """
        return self.find_similar(
            action_type=request.action_type,
            channel=request.channel,
            environment=request.environment,
            recipient=request.recipient,
            exclude_decision_id=None,
            limit=limit,
        )

    def delete(self, decision_id: UUID) -> None:
        """Delete a stored precedent by ``decision_id``."""
        with self._lock:
            if decision_id not in self._by_decision_id:
                raise KeyError(f"precedent not found for decision_id: {decision_id}")

            del self._by_decision_id[decision_id]
            self._ordered_ids = [
                stored_id
                for stored_id in self._ordered_ids
                if stored_id != decision_id
            ]

    def clear(self) -> None:
        """Remove all stored precedents."""
        with self._lock:
            self._by_decision_id.clear()
            self._ordered_ids.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_decision_id)

    def __contains__(self, decision_id: object) -> bool:
        if not isinstance(decision_id, UUID):
            return False
        with self._lock:
            return decision_id in self._by_decision_id

    @staticmethod
    def _compute_relevance_score(
        *,
        decision: Decision,
        action_type: str | None,
        channel: str | None,
        environment: str | None,
        recipient: str | None,
        verdict: Verdict | None,
        policy_version: str | None,
    ) -> float:
        """
        Compute a lightweight deterministic relevance score.

        This is not semantic ranking. It is just enough signal to let the rest
        of Tex consume a consistent retrieval contract before a real ranking
        layer exists.
        """
        possible_matches = 0
        actual_matches = 0

        def check(expected: object, actual: object) -> None:
            nonlocal possible_matches, actual_matches
            if expected is None:
                return
            possible_matches += 1
            if expected == actual:
                actual_matches += 1

        check(action_type, decision.action_type)
        check(channel, decision.channel)
        check(environment, decision.environment)
        check(recipient, decision.recipient)
        check(verdict, decision.verdict)
        check(policy_version, decision.policy_version)

        if possible_matches == 0:
            return 0.5

        base_score = actual_matches / possible_matches
        # Keep the score away from zero so valid retrieved precedents still
        # carry usable weight in downstream prompts and diagnostics.
        return max(0.1, min(1.0, base_score))

    @staticmethod
    def _to_precedent(
        decision: Decision,
        *,
        relevance_score: float,
        rank: int,
    ) -> RetrievedPrecedent:
        """
        Project a durable ``Decision`` into Tex's retrieval-safe precedent shape.
        """
        matched_policy_clause_ids = _extract_matched_policy_clause_ids(decision)

        return RetrievedPrecedent(
            decision_id=str(decision.decision_id),
            request_id=str(decision.request_id),
            verdict=decision.verdict,
            action_type=decision.action_type,
            channel=decision.channel,
            environment=decision.environment,
            content_excerpt=decision.content_excerpt,
            reasons=tuple(decision.reasons),
            matched_policy_clause_ids=matched_policy_clause_ids,
            uncertainty_flags=tuple(decision.uncertainty_flags),
            relevance_score=relevance_score,
            rank=rank,
            decided_at=decision.decided_at,
            metadata={
                "policy_version": decision.policy_version,
                "policy_id": decision.policy_id,
                "recipient": decision.recipient,
                "confidence": decision.confidence,
                "final_score": decision.final_score,
                "scores": dict(decision.scores),
                "finding_count": len(decision.findings),
                "content_sha256": decision.content_sha256,
            },
        )


def _extract_matched_policy_clause_ids(decision: Decision) -> tuple[str, ...]:
    """
    Best-effort extraction of retrieved or semantic clause references from a decision.

    The decision model does not currently guarantee a single canonical field for
    matched clause IDs, so this helper looks in the most likely locations and
    returns a stable deduped tuple.
    """
    ordered: list[str] = []
    seen: set[str] = set()

    def add(candidate: object) -> None:
        if not isinstance(candidate, str):
            return
        normalized = candidate.strip()
        if not normalized:
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        ordered.append(normalized)

    retrieval_context = decision.retrieval_context
    if isinstance(retrieval_context, dict):
        for key in ("matched_policy_clause_ids", "policy_clause_ids"):
            values = retrieval_context.get(key)
            if isinstance(values, (list, tuple)):
                for value in values:
                    add(value)

    metadata = decision.metadata
    if isinstance(metadata, dict):
        values = metadata.get("matched_policy_clause_ids")
        if isinstance(values, (list, tuple)):
            for value in values:
                add(value)

    return tuple(ordered)
