"""State definitions for the insurance advisor graph."""

from __future__ import annotations

from typing import Any, Literal, NotRequired
from typing_extensions import TypedDict


Intent = Literal["question", "profile_update", "quote_request", "unknown"]


class UserProfile(TypedDict, total=False):
    """Attributes collected about the customer over the session."""

    age: int
    life_stage: str
    budget_monthly: int
    health_status: str
    has_preexisting_conditions: bool
    business_owner: bool
    dependents: int
    risk_tolerance: Literal["low", "medium", "high"]


class RetrievedDoc(TypedDict):
    """A policy snippet returned from the vector store."""

    policy_id: str
    title: str
    chunk: str
    score: float


class AgentState(TypedDict, total=False):
    """Full graph state persisted by LangGraph checkpointers."""

    # Conversation context
    session_id: str
    messages: list[dict[str, str]]
    latest_user_message: str

    # Classified intent + missing fields for routing
    intent: Intent
    required_fields_missing: list[str]

    # Personalization context
    user_profile: UserProfile
    memory_facts: list[str]

    # RAG context
    retrieved_docs: list[RetrievedDoc]
    retrieval_query: str

    # Output
    recommendation: str
    follow_up_question: str
    debug: NotRequired[dict[str, Any]]


def empty_state(session_id: str) -> AgentState:
    """Create an initialized state object for a new session."""
    return {
        "session_id": session_id,
        "messages": [],
        "user_profile": {},
        "memory_facts": [],
        "retrieved_docs": [],
        "required_fields_missing": [],
        "intent": "unknown",
        "latest_user_message": "",
        "retrieval_query": "",
        "recommendation": "",
        "follow_up_question": "",
    }
