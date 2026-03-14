"""Long-term memory interfaces for user preference recall."""

from __future__ import annotations


class MemoryStore:
    """Base class for memory stores (Redis, Mem0, Postgres, etc.)."""

    def get_facts(self, session_id: str) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def upsert_fact(self, session_id: str, fact: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryPreferenceStore(MemoryStore):
    """In-memory store useful for local development."""

    def __init__(self) -> None:
        self._facts: dict[str, list[str]] = {}

    def get_facts(self, session_id: str) -> list[str]:
        return self._facts.get(session_id, [])

    def upsert_fact(self, session_id: str, fact: str) -> None:
        self._facts.setdefault(session_id, [])
        if fact not in self._facts[session_id]:
            self._facts[session_id].append(fact)
