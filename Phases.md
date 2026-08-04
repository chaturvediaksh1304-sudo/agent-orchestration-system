# Agent Orchestration System — Phases

Each phase is a loop: implement → verify against done-criteria → fix → re-check → proceed. Never skip ahead. Do not attempt multiple phases in a single pass.

## Phase 1: Core orchestration loop
- Goal: Supervisor agent (LangGraph) that decomposes a goal into subtasks, dynamically generates subagents, and dispatches them with basic tool use.
- Done-criteria:
  - [ ] Supervisor takes a goal as input and produces a subtask decomposition
  - [ ] Subagents are dynamically generated (not hardcoded) and can call at least one tool
  - [ ] A full end-to-end run (goal in, aggregated result out) completes for a simple multi-step task
- Depends on: none

## Phase 2: Persistent memory
- Goal: Task history and agent configs persist in PostgreSQL; conversation memory persists in ChromaDB.
- Done-criteria:
  - [ ] Task history and agent configs are written to and readable from PostgreSQL
  - [ ] Conversation memory is written to and retrievable from ChromaDB
  - [ ] A second run can reference memory from a prior run
- Depends on: Phase 1

## Phase 3: Task queue
- Goal: Subagent dispatch runs async via Redis + Celery instead of synchronously in-process.
- Done-criteria:
  - [ ] Subagent tasks are enqueued to Celery via Redis
  - [ ] Multiple subagents can execute concurrently
  - [ ] Supervisor correctly waits for and aggregates async results
- Depends on: Phase 1

## Phase 4: Self-repair
- Goal: Failing subagents auto-diagnose and attempt retry/fix before any escalation.
- Done-criteria:
  - [ ] A deliberately failing subagent triggers a diagnosis step
  - [ ] The system attempts at least one automated retry/fix based on the diagnosis
  - [ ] Repeated failure after self-repair attempts is correctly detected
- Depends on: Phase 1, Phase 2

## Phase 5: Human-in-the-loop escalation
- Goal: When self-repair fails repeatedly, the system escalates to a human and logs it.
- Done-criteria:
  - [ ] Escalation triggers correctly after self-repair exhausts its attempts
  - [ ] Escalation is logged in PostgreSQL (escalation logs)
  - [ ] A human can respond to the escalation and the run resumes or terminates cleanly
- Depends on: Phase 4

## Phase 6: Auth + CLI polish
- Goal: OAuth/magic link auth wired in; system packaged and runnable via Docker.
- Done-criteria:
  - [ ] OAuth and/or magic link auth flow works
  - [ ] Full system runs via `docker compose up` (or equivalent) self-hosted
  - [ ] CLI end-to-end run works from a clean environment
- Depends on: Phase 1-5
