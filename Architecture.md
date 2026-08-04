# Agent Orchestration System — Architecture

## Platform
CLI-first. No UI in MVP; a Streamlit or React frontend is a possible later addition, not part of this build.

## Stack
- Orchestration: Python (LangGraph)
- AI: OpenAI + Anthropic
- Memory: PostgreSQL (task history, agent configs, escalation logs) + ChromaDB (conversation memory)
- Queue: Redis + Celery
- Auth: OAuth and/or magic link (Assumed: needed even though the tool is CLI-first, since Aksh confirmed user auth is required — likely for a future hosted/multi-user mode. Flag if this should instead be deferred out of MVP.)
- Hosting/deploy: Docker, self-hosted
- Third-party APIs/services: None beyond OpenAI + Anthropic

## Folder structure
TBD — Claude Code proposes a standard Python/LangGraph project structure on Phase 1 start, inside a top-level `Agent Orchestration System/` folder.

## Data flow
1. User submits a goal via CLI.
2. Supervisor agent (LangGraph) decomposes the goal into subtasks.
3. Supervisor dynamically generates and dispatches subagents per subtask, queued via Redis/Celery for async execution.
4. Subagents use tools to execute their subtask; results and intermediate state are written to PostgreSQL (task history) and ChromaDB (conversation memory).
5. On subagent failure: self-repair loop attempts diagnosis and retry.
6. If self-repair fails repeatedly: escalate to human, logged in PostgreSQL (escalation logs).
7. Supervisor aggregates subagent results and returns the final outcome to the user.
