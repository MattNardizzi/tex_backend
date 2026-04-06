from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

from tex.domain.retrieval import RetrievedEntity


class InMemoryEntityStore:
    """
    Simple in-memory entity store for local development and testing.

    This store backs Tex's retrieval layer for customer/domain-sensitive
    entities without prematurely introducing a database or vector index.

    It supports:
    - save/update by entity name
    - lookup by exact name
    - lightweight matching by query text
    - filtering by entity type
    - newest-first retrieval for operational convenience
    """

    __slots__ = (
        "_lock",
        "_by_key",
        "_ordered_keys",
    )

    def __init__(self, initial_entities: Iterable[RetrievedEntity] | None = None) -> None:
        self._lock = RLock()
        self._by_key: dict[str, RetrievedEntity] = {}
        self._ordered_keys: list[str] = []

        if initial_entities is not None:
            for entity in initial_entities:
                self.save(entity)

    def save(self, entity: RetrievedEntity) -> None:
        """
        Saves or replaces an entity.

        Entity identity is normalized by name so retrieval stays predictable even
        if callers vary capitalization.
        """
        key = self._normalize_key(entity.canonical_name)

        with self._lock:
            if key in self._by_key:
                self._ordered_keys = [
                    stored_key
                    for stored_key in self._ordered_keys
                    if stored_key != key
                ]

            self._by_key[key] = entity
            self._ordered_keys.append(key)

    def get(self, name: str) -> RetrievedEntity | None:
        """Returns an entity by exact normalized name, or None if missing."""
        key = self._normalize_key(name)
        with self._lock:
            return self._by_key.get(key)

    def require(self, name: str) -> RetrievedEntity:
        """Returns an entity by name or raises KeyError."""
        entity = self.get(name)
        if entity is None:
            raise KeyError(f"entity not found: {name}")
        return entity

    def list_all(self) -> tuple[RetrievedEntity, ...]:
        """Returns all stored entities in save order."""
        with self._lock:
            return tuple(self._by_key[key] for key in self._ordered_keys)

    def list_recent(self, limit: int = 50) -> tuple[RetrievedEntity, ...]:
        """Returns the most recently saved entities, newest first."""
        if limit <= 0:
            return tuple()

        with self._lock:
            selected_keys = list(reversed(self._ordered_keys[-limit:]))
            return tuple(self._by_key[key] for key in selected_keys)

    def find_matching(
        self,
        *,
        text: str,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> tuple[RetrievedEntity, ...]:
        """
        Returns entities whose names or aliases appear in the supplied text.

        This is intentionally simple lexical matching for now. It gives Tex a
        usable sensitive-entity retrieval surface without premature complexity.
        """
        if limit <= 0:
            return tuple()

        haystack = self._normalize_search_text(text)
        if not haystack:
            return tuple()

        with self._lock:
            matched: list[RetrievedEntity] = []

            for key in reversed(self._ordered_keys):
                entity = self._by_key[key]

                if entity_type is not None and entity.entity_type != entity_type:
                    continue

                if self._entity_matches_text(entity, haystack):
                    matched.append(entity)

                if len(matched) >= limit:
                    break

            return tuple(matched)

    def filter_by_type(
        self,
        entity_type: str,
        *,
        limit: int | None = None,
    ) -> tuple[RetrievedEntity, ...]:
        """Returns entities of a given type, newest first."""
        normalized_type = entity_type.strip()
        if not normalized_type:
            raise ValueError("entity_type must not be blank")

        with self._lock:
            matched: list[RetrievedEntity] = []

            for key in reversed(self._ordered_keys):
                entity = self._by_key[key]
                if entity.entity_type != normalized_type:
                    continue

                matched.append(entity)

                if limit is not None and len(matched) >= limit:
                    break

            return tuple(matched)

    def delete(self, name: str) -> None:
        """Deletes a stored entity by normalized name."""
        key = self._normalize_key(name)

        with self._lock:
            if key not in self._by_key:
                raise KeyError(f"entity not found: {name}")

            del self._by_key[key]
            self._ordered_keys = [
                stored_key
                for stored_key in self._ordered_keys
                if stored_key != key
            ]

    def clear(self) -> None:
        """Removes all stored entities."""
        with self._lock:
            self._by_key.clear()
            self._ordered_keys.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._by_key)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False

        key = self._normalize_key(name)
        with self._lock:
            return key in self._by_key

    @staticmethod
    def _normalize_key(value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("entity name must not be blank")
        return candidate.casefold()

    @staticmethod
    def _normalize_search_text(value: str) -> str:
        return " ".join(value.strip().casefold().split())

    @staticmethod
    def _entity_matches_text(entity: RetrievedEntity, haystack: str) -> bool:
        candidates = [entity.canonical_name]

        aliases = getattr(entity, "aliases", ())
        if aliases:
            candidates.extend(aliases)

        for candidate in candidates:
            normalized_candidate = " ".join(candidate.strip().casefold().split())
            if normalized_candidate and normalized_candidate in haystack:
                return True

        return False