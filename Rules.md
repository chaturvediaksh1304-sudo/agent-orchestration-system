# Agent Orchestration System — Rules

## Libraries
- Prefer: Aksh's default stack (Python, type hints) plus LangGraph conventions for orchestration. No explicit avoid-list — Claude Code should use standard, well-supported choices for each piece (PostgreSQL driver, ChromaDB client, Celery/Redis integration) unless a conflict arises.
- Avoid: Nothing explicitly excluded — flag any nonstandard or niche library choice before adopting it.

## Error handling
Subagents attempt self-diagnosis and self-repair on failure before anything else. Only after repeated self-repair attempts fail does the system escalate to a human (logged in PostgreSQL escalation logs). This applies system-wide — not just at the subagent level, but any component failure should default to "try to fix it, then escalate" rather than failing loud and stopping outright.

## Testing
TDD. Write the test (or done-criteria check) for a unit of work before implementing it, for every phase.

## Requires explicit approval before doing
- Any browser use
- Any `git push` — do not push to GitHub until explicitly told to

## Execution discipline (Karpathy guidelines — always included)
1. **Think before coding** — state assumptions explicitly; if multiple interpretations exist, present them, don't silently pick; stop and ask if genuinely unclear.
2. **Simplicity first** — minimum code that solves the task; no speculative abstraction, no unrequested configurability, no error handling for impossible cases.
3. **Surgical changes** — touch only what the task requires; don't refactor or "improve" adjacent code; match existing style; remove only orphans your own change created.
4. **Goal-driven execution** — every task starts with a stated, checkable success criterion; work in verify-then-proceed loops (implement → check against criterion → fix → re-check) rather than declaring done by feel.

## Looping mandate
Work phase-by-phase per Phases.md. Do not attempt multiple phases in a single pass. Each phase's done-criteria must be verifiably met before starting the next.
