from __future__ import annotations

from typing import Protocol

from tex.domain.evaluation import EvaluationRequest
from tex.domain.policy import PolicySnapshot
from tex.domain.retrieval import (
    RetrievalContext,
    RetrievedEntity,
    RetrievedPolicyClause,
    RetrievedPrecedent,
)


class PolicyClauseStore(Protocol):
    """Store contract for retrieving policy clauses relevant to an evaluation."""

    def retrieve_policy_clauses(
        self,
        *,
        policy: PolicySnapshot,
        request: EvaluationRequest,
        top_k: int,
    ) -> tuple[RetrievedPolicyClause, ...]:
        """Returns policy clauses ranked by relevance."""


class PrecedentStore(Protocol):
    """Store contract for retrieving similar prior decisions."""

    def retrieve_precedents(
        self,
        *,
        request: EvaluationRequest,
        limit: int,
    ) -> tuple[RetrievedPrecedent, ...]:
        """Returns prior decision summaries ranked by relevance."""


class EntityStore(Protocol):
    """Store contract for retrieving relevant sensitive entities."""

    def retrieve_entities(
        self,
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
        top_k: int,
    ) -> tuple[RetrievedEntity, ...]:
        """Returns relevant entities ranked by relevance."""


class RetrievalOrchestrator:
    """
    Lean retrieval-grounding orchestrator for Tex.

    This layer deliberately stays simple for now:
    - fetch relevant policy clauses
    - fetch similar precedents
    - fetch relevant sensitive entities
    - normalize failure into warnings instead of exploding the whole request

    It is a contract-first orchestrator, not a full RAG system.
    """

    def __init__(
        self,
        *,
        policy_store: PolicyClauseStore | None = None,
        precedent_store: PrecedentStore | None = None,
        entity_store: EntityStore | None = None,
    ) -> None:
        self._policy_store = policy_store
        self._precedent_store = precedent_store
        self._entity_store = entity_store

    def retrieve(
        self,
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
    ) -> RetrievalContext:
        warnings: list[str] = []

        policy_clauses = self._retrieve_policy_clauses(
            request=request,
            policy=policy,
            warnings=warnings,
        )
        precedents = self._retrieve_precedents(
            request=request,
            policy=policy,
            warnings=warnings,
        )
        entities = self._retrieve_entities(
            request=request,
            policy=policy,
            warnings=warnings,
        )

        return RetrievalContext(
            policy_clauses=policy_clauses,
            precedents=precedents,
            entities=entities,
            retrieval_warnings=tuple(warnings),
            metadata={
                "policy_version": policy.version,
                "retrieval_top_k": policy.retrieval_top_k,
                "precedent_lookback_limit": policy.precedent_lookback_limit,
            },
        )

    def _retrieve_policy_clauses(
        self,
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
        warnings: list[str],
    ) -> tuple[RetrievedPolicyClause, ...]:
        if self._policy_store is None:
            warnings.append("policy_clause_store_unavailable")
            return tuple()

        try:
            return self._policy_store.retrieve_policy_clauses(
                policy=policy,
                request=request,
                top_k=policy.retrieval_top_k,
            )
        except Exception as exc:
            warnings.append(f"policy_clause_retrieval_failed:{type(exc).__name__}")
            return tuple()

    def _retrieve_precedents(
        self,
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
        warnings: list[str],
    ) -> tuple[RetrievedPrecedent, ...]:
        if self._precedent_store is None:
            warnings.append("precedent_store_unavailable")
            return tuple()

        try:
            return self._precedent_store.retrieve_precedents(
                request=request,
                limit=policy.precedent_lookback_limit,
            )
        except Exception as exc:
            warnings.append(f"precedent_retrieval_failed:{type(exc).__name__}")
            return tuple()

    def _retrieve_entities(
        self,
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
        warnings: list[str],
    ) -> tuple[RetrievedEntity, ...]:
        if self._entity_store is None:
            warnings.append("entity_store_unavailable")
            return tuple()

        try:
            return self._entity_store.retrieve_entities(
                request=request,
                policy=policy,
                top_k=policy.retrieval_top_k,
            )
        except Exception as exc:
            warnings.append(f"entity_retrieval_failed:{type(exc).__name__}")
            return tuple()


class NoOpPolicyClauseStore:
    """Default empty policy-clause store for early development."""

    def retrieve_policy_clauses(
        self,
        *,
        policy: PolicySnapshot,
        request: EvaluationRequest,
        top_k: int,
    ) -> tuple[RetrievedPolicyClause, ...]:
        return tuple()


class NoOpPrecedentStore:
    """Default empty precedent store for early development."""

    def retrieve_precedents(
        self,
        *,
        request: EvaluationRequest,
        limit: int,
    ) -> tuple[RetrievedPrecedent, ...]:
        return tuple()


class NoOpEntityStore:
    """Default empty entity store for early development."""

    def retrieve_entities(
        self,
        *,
        request: EvaluationRequest,
        policy: PolicySnapshot,
        top_k: int,
    ) -> tuple[RetrievedEntity, ...]:
        return tuple()


def build_noop_retrieval_orchestrator() -> RetrievalOrchestrator:
    """
    Returns a retrieval orchestrator wired to empty stores.

    This is the correct starter configuration while the retrieval contracts
    exist but real retrieval is still intentionally minimal.
    """
    return RetrievalOrchestrator(
        policy_store=NoOpPolicyClauseStore(),
        precedent_store=NoOpPrecedentStore(),
        entity_store=NoOpEntityStore(),
    )