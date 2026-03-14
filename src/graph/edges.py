"""Routing logic for conditional edges in the insurance advisor graph."""

from __future__ import annotations

from src.graph.state import AgentState


def route_after_intent(state: AgentState) -> str:
    """Route to clarification when quote requests are missing profile fields."""
    if state.get("intent") == "quote_request" and state.get("required_fields_missing"):
        return "clarification"
    return "memory_sync"


def route_after_memory(state: AgentState) -> str:
    """Always retrieve policy context after memory load."""
    return "retrieve"
