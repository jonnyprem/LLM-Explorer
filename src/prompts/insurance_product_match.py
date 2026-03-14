"""Prompt templates used by the insurance advisory workflow."""

SYSTEM_PROMPT = """You are an insurance advisory assistant.
Your job is to recommend plans using only the retrieved policy context and known user profile.

Rules:
1. Do not invent policy benefits, exclusions, or pricing.
2. If key profile fields are missing (age, budget, health status, life stage), ask a short follow-up.
3. Use memory facts to avoid suggesting options the user already rejected.
4. Return a concise recommendation with rationale and one trade-off.
"""


def render_suggestion_prompt(*, user_message: str, user_profile: dict, memory_facts: list[str], docs: list[dict]) -> str:
    """Render a single prompt for recommendation generation."""

    context = "\n\n".join(
        f"[{idx + 1}] {doc.get('title', 'Policy')} (score={doc.get('score', 'n/a')}):\n{doc.get('chunk', '')}"
        for idx, doc in enumerate(docs)
    )

    return f"""
{SYSTEM_PROMPT}

User message:
{user_message}

User profile:
{user_profile}

Memory facts:
{memory_facts}

Retrieved policy context:
{context if context else 'No policy context retrieved.'}

Output JSON with keys:
- recommendation
- rationale
- trade_off
- follow_up_question
""".strip()
