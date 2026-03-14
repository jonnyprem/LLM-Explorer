"""Graph nodes for intent detection, retrieval, memory sync, and recommendation."""

from __future__ import annotations

import json
from typing import Callable

from src.database.vector_store import VectorStore
from src.graph.state import AgentState
from src.memory.store import MemoryStore
from src.prompts.insurance_product_match import render_suggestion_prompt


def identify_intent(state: AgentState) -> AgentState:
    """Classify intent and detect missing profile fields required for quoting."""
    text = state.get("latest_user_message", "").lower()

    if any(word in text for word in ["quote", "premium", "price"]):
        intent = "quote_request"
    elif any(word in text for word in ["i am", "my age", "budget", "condition"]):
        intent = "profile_update"
    elif text.strip().endswith("?"):
        intent = "question"
    else:
        intent = "unknown"

    profile = state.get("user_profile", {})
    missing = [
        field
        for field in ["age", "budget_monthly", "health_status", "life_stage"]
        if field not in profile
    ]

    state["intent"] = intent
    state["required_fields_missing"] = missing
    return state


def retrieve_policy_context(state: AgentState, vector_store: VectorStore) -> AgentState:
    """Retrieve policy snippets relevant to the latest request."""
    query = state.get("retrieval_query") or state.get("latest_user_message", "")
    results = vector_store.search(query=query, k=4)
    state["retrieved_docs"] = [
        {
            "policy_id": r.policy_id,
            "title": r.title,
            "chunk": r.chunk,
            "score": r.score,
        }
        for r in results
    ]
    return state


def memory_sync(state: AgentState, memory_store: MemoryStore) -> AgentState:
    """Load historical user preferences from long-term memory."""
    session_id = state.get("session_id", "")
    state["memory_facts"] = memory_store.get_facts(session_id)
    return state


def generate_suggestion(state: AgentState, llm: Callable[[str], str]) -> AgentState:
    """Create recommendation grounded in profile, memory, and retrieved docs."""

    prompt = render_suggestion_prompt(
        user_message=state.get("latest_user_message", ""),
        user_profile=state.get("user_profile", {}),
        memory_facts=state.get("memory_facts", []),
        docs=state.get("retrieved_docs", []),
    )
    response = llm(prompt)

    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = {
            "recommendation": response,
            "rationale": "Response was not JSON; raw output used.",
            "trade_off": "Unknown",
            "follow_up_question": "Could you share your age, monthly budget, and health status?",
        }

    state["recommendation"] = (
        f"{parsed.get('recommendation', '')}\n"
        f"Rationale: {parsed.get('rationale', '')}\n"
        f"Trade-off: {parsed.get('trade_off', '')}"
    ).strip()
    state["follow_up_question"] = parsed.get("follow_up_question", "")
    return state


def clarification(state: AgentState) -> AgentState:
    """Ask user for missing attributes before pricing-oriented advice."""
    missing = state.get("required_fields_missing", [])
    if not missing:
        state["follow_up_question"] = "Could you share a bit more about your goals for coverage?"
        return state

    state["follow_up_question"] = (
        "Before I can provide a quote-ready recommendation, please share: "
        + ", ".join(missing)
        + "."
    )
    return state
