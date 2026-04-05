# Multi-Agent Research Pipeline

A production-style multi-agent LLM system built with Python and Streamlit, demonstrating orchestration, observability, memory, evaluation, and debugging patterns for production AI infrastructure.

## Architecture

Four collaborating agents coordinated by a central orchestrator:

1. **Research Agent** — produces structured research across 5 sections (Background, Key Concepts, Market Dynamics, Risks, Conclusion)
2. **Summarisation Agent** — distills research into 5 key bullet points, one per section
3. **Validation Agent** — fact-checks summary against original research, returns PASS/FAIL
4. **Report Writer Agent** — generates a formatted executive report in markdown for stakeholder consumption

Agents communicate via a shared context object — no direct inter-agent calls. The orchestrator owns all state, handles retries (max 2 per agent), and enforces a configurable per-run budget limit.

## Features

### LLM Routing Layer
A meta-classifier scores each topic 1-10 for complexity and routes to the appropriate model:
- Score 5 or below goes to Claude Haiku (fast, cheap)
- Score above 5 goes to Claude Sonnet (more capable, higher cost)

### Vector DB Memory
ChromaDB stores past research outputs as embeddings. On each new run, semantically similar past research is retrieved and injected as context into the Research Agent so the system builds on prior knowledge rather than starting from scratch.

### Observability Dashboard
Every agent call is logged to SQLite with per-agent latency, input/output token counts, cost in USD, validation pass/fail status, and model used. Surfaced as a real-time dashboard with metrics, bar charts, and a full audit log table.

### Eval Framework
A fixed eval set of 3 topics with ground-truth keyword criteria. Each pipeline run is scored on keyword coverage, length score, and overall weighted score. Results are tracked per prompt version so regressions from prompt changes are immediately visible.

### Debug Console
Full structured logging of every agent call including complete prompt input, raw model output, error messages for failed calls, retry attempt number, and a per-task inspector where you can select any past task ID and see the full execution trace.

## Tech Stack

- Python 3.11
- Streamlit — UI framework
- Anthropic Claude Haiku and Sonnet — LLM agents via AICredits gateway
- ChromaDB — local vector database for semantic memory
- SQLite — observability, eval, and debug logs
- OpenAI SDK — used as drop-in client for AICredits API gateway

## Setup

Install dependencies:

pip install anthropic openai streamlit python-dotenv chromadb

Create a .env file in the project root:

ANTHROPIC_API_KEY=your-api-key-here

Run the app:

streamlit run dashboard.py

## Project Structure

agents.py — 4 LLM agents and LLM complexity router
orchestrator.py — central state machine, budget control, eval scoring
observability.py — SQLite logging for all agent calls
memory.py — ChromaDB vector memory interface
memory_worker.py — isolated subprocess worker for ChromaDB
evaluator.py — eval framework, scoring logic, prompt version tracking
debugger.py — structured debug logging per agent call
dashboard.py — Streamlit UI with Pipeline, Observability, Evals, and Debug tabs
.gitignore — excludes .env, databases, __pycache__, chroma_db

## Key Design Decisions

Central orchestrator pattern — the orchestrator holds all state and mediates between agents. Agents never call each other directly. This enables independent scaling, full state replay, and a single point of observability and control.

Subprocess isolation for ChromaDB — ChromaDB's persistent client conflicts with Streamlit's threading model. Memory operations run in an isolated subprocess via memory_worker.py to prevent runtime crashes.

Budget control at orchestrator level — cost is tracked cumulatively across all agent calls per run. If the budget is exceeded after any stage, the pipeline halts immediately rather than continuing to burn credits.

Prompt versioning — PROMPT_VERSION in orchestrator.py is bumped manually when agent prompts change. The eval framework groups results by version so quality regressions are immediately detectable.