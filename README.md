# End-to-End Insurance Advisory Agent

A reference project for building a **stateful insurance recommendation assistant** using:

- **LangGraph** for workflow orchestration and state transitions
- **RAG (Retrieval-Augmented Generation)** for grounded policy lookup
- **Memory** for long-term user preferences across sessions

The goal is to solve the “static advisor gap”: users should not get generic answers, and they should not have to repeat important constraints (budget, health context, rejected plan types, etc.).

---

## What this project does

This agent processes each user turn through a graph-based workflow:

1. **Intent detection**
   - Classifies whether the user is asking a question, updating profile data, or requesting a quote.
2. **Routing / clarification**
   - If the user asks for quote-like guidance but required fields are missing (age, budget, health, life stage), the agent asks follow-up questions.
3. **Memory sync**
   - Loads known preferences from memory (for example: “user rejected high-deductible plans”).
4. **Policy retrieval (RAG)**
   - Pulls relevant policy snippets from a vector store.
5. **Suggestion generation**
   - Produces a recommendation based on user profile + memory + retrieved policy context.

---

## Repository structure

```text
insurance-advisor/
├── data/                         # Raw insurance documents (PDFs/brochures)
├── vector_db/                    # Persisted vector index (optional, future production use)
├── src/
│   ├── graph/
│   │   ├── state.py              # AgentState schema and defaults
│   │   ├── nodes.py              # Node logic (intent, retrieval, memory, suggestion)
│   │   └── edges.py              # Conditional routing
│   ├── database/
│   │   └── vector_store.py       # Vector store interface + in-memory fallback
│   ├── memory/
│   │   └── store.py              # Memory interface + in-memory fallback
│   └── prompts/
│       └── insurance_product_match.py   # Recommendation prompt template
├── requirements.txt              # Python dependencies
└── main.py                       # Demo entrypoint (builds and invokes graph)
```

---

## Run everything inside a virtual environment

> Recommended: run **all project commands** from a virtual environment.

### 1) Create virtual environment

```bash
python -m venv .venv
```

### 2) Activate virtual environment

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies (inside the venv)

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4) Run the demo (inside the venv)

```bash
python main.py
```

### 5) Deactivate when done

```bash
deactivate
```

You should see either:

- a generated recommendation (if enough profile data exists), or
- a clarification question (if required profile fields are missing).

---

## Quick usage notes

- The demo currently uses:
  - a **fake LLM** (`fake_llm` in `main.py`) for deterministic local output
  - an **in-memory vector store** and **in-memory memory store**
- This makes it easy to iterate on graph logic before integrating production services.

---

## Current limitations

This is a starter scaffold and intentionally simple:

- Intent classification is keyword-based.
- Retrieval scoring is token-overlap, not embedding similarity.
- Memory is process-local (not persistent across restarts).
- LLM output is stubbed via `fake_llm`.

---

## Next steps for production

1. Replace `fake_llm` with a real model provider.
2. Replace `InMemoryVectorStore` with Chroma/Pinecone/pgvector retrieval.
3. Replace `InMemoryPreferenceStore` with Redis/Postgres/Mem0-backed memory.
4. Add document ingestion pipeline for policy PDFs.
5. Add evaluation tests for recommendation quality and safety.
6. Add authentication + per-user session management.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'langgraph'`

Usually this means dependencies were installed outside the virtual environment.

Fix:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### I only get clarification questions

This is expected when required quote fields are missing. Provide:

- age
- monthly budget
- health status
- life stage

Then rerun with updated state input in `main.py` or your own caller.
