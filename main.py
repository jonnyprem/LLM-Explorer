"""Entry point that compiles and runs the insurance advisory LangGraph."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.database.vector_store import InMemoryVectorStore
from src.graph.edges import route_after_intent, route_after_memory
from src.graph.nodes import (
    clarification,
    generate_suggestion,
    identify_intent,
    memory_sync,
    retrieve_policy_context,
)
from src.graph.state import AgentState, empty_state
from src.memory.store import InMemoryPreferenceStore


def fake_llm(prompt: str) -> str:
    """Local stand-in for an LLM provider."""
    return (
        '{"recommendation":"Family Care Silver Plan",'
        '"rationale":"Matches moderate budget and dependent coverage goals.",'
        '"trade_off":"Higher co-pay than Gold tier.",'
        '"follow_up_question":"Do you want maternity add-on coverage included?"}'
    )


def build_graph(vector_store: InMemoryVectorStore, memory_store: InMemoryPreferenceStore):
    """Construct a compiled graph with dependency-injected nodes."""
    graph = StateGraph(AgentState)

    graph.add_node("intent", identify_intent)
    graph.add_node("clarification", clarification)
    graph.add_node("memory_sync", lambda state: memory_sync(state, memory_store))
    graph.add_node("retrieve", lambda state: retrieve_policy_context(state, vector_store))
    graph.add_node("suggest", lambda state: generate_suggestion(state, fake_llm))

    graph.add_edge(START, "intent")
    graph.add_conditional_edges("intent", route_after_intent)
    graph.add_conditional_edges("memory_sync", route_after_memory)
    graph.add_edge("clarification", END)
    graph.add_edge("retrieve", "suggest")
    graph.add_edge("suggest", END)

    return graph.compile()


def run_example() -> dict[str, Any]:
    """Execute a single example conversation turn."""
    vector_store = InMemoryVectorStore(
        docs=[
            {
                "policy_id": "plan-101",
                "title": "Family Care Silver",
                "chunk": "Affordable family plan with pediatric visits and preventive care.",
            },
            {
                "policy_id": "plan-202",
                "title": "Business Shield",
                "chunk": "Coverage for business owners including liability and employee benefits.",
            },
        ]
    )

    memory_store = InMemoryPreferenceStore()
    memory_store.upsert_fact("demo-session", "User rejected high-deductible plans.")

    app = build_graph(vector_store, memory_store)

    state = empty_state("demo-session")
    state["latest_user_message"] = "I need a quote for my family."
    state["user_profile"] = {"life_stage": "new_parent", "budget_monthly": 300}

    return app.invoke(state)


if __name__ == "__main__":
    output = run_example()
    print(output.get("recommendation") or output.get("follow_up_question"))
