"""Simple vector store interface and in-memory fallback implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SearchResult:
    policy_id: str
    title: str
    chunk: str
    score: float


class VectorStore:
    """Protocol-like base for retrieval backends."""

    def search(self, query: str, k: int = 4) -> list[SearchResult]:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """Keyword-overlap fallback for local demos and tests."""

    def __init__(self, docs: list[dict[str, str]]) -> None:
        self._docs = docs

    def search(self, query: str, k: int = 4) -> list[SearchResult]:
        query_tokens = set(query.lower().split())
        scored: list[SearchResult] = []
        for doc in self._docs:
            chunk = doc.get("chunk", "")
            overlap = query_tokens.intersection(chunk.lower().split())
            score = float(len(overlap))
            if score > 0:
                scored.append(
                    SearchResult(
                        policy_id=doc.get("policy_id", "unknown"),
                        title=doc.get("title", "Policy"),
                        chunk=chunk,
                        score=score,
                    )
                )
        return sorted(scored, key=lambda r: r.score, reverse=True)[:k]
