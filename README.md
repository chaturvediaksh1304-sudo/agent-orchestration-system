# Agent Orchestration System

A multi-agent orchestration system: a supervisor decomposes a goal, designs a
subagent for each subtask at runtime, dispatches them with tool use, and
aggregates their results.

**Status: complete (Phases 1–6).** A supervisor decomposes a goal and designs a
subagent per subtask at runtime; subagents dispatch concurrently to Celery
workers over Redis; runs and agent configs persist to PostgreSQL with
conversation memory in ChromaDB; failing subagents diagnose themselves and
retry; whatever repair can't rescue is escalated to a human. Runs via
`docker compose up`.

## Run it with Docker (recommended)

```bash
export ANTHROPIC_API_KEY=...        # or OPENAI_API_KEY
docker compose up -d                # postgres, redis, celery worker
docker compose --profile cli build  # the cli service sits behind a profile
docker compose run --rm cli "your goal here"
```

`docker compose build` alone skips the `cli` service because of its profile —
use `--profile cli` when rebuilding after a change.

## Or run it locally

Requires Python 3.9+, plus PostgreSQL and (for `--queue`) Redis.

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"
docker compose up -d postgres redis        # or bring your own
export DATABASE_URL=postgresql://orchestration:orchestration@localhost:5432/orchestration
export ANTHROPIC_API_KEY=...               # or OPENAI_API_KEY
```

## Authentication

Optional, and off unless you configure it. With `OAUTH_CLIENT_ID` set to a
GitHub OAuth app's client id, the CLI requires a login and stamps each run and
escalation with the user identity:

```bash
export OAUTH_CLIENT_ID=...
orchestrate login      # shows a code to enter in your browser
orchestrate whoami
orchestrate logout
```

It's the device flow (RFC 8628), so there is no client secret and no local web
server. With no `OAUTH_CLIENT_ID` the CLI runs unauthenticated — the tool stays
usable standalone, which is the point of being CLI-first.

## Run

```bash
.venv/bin/python -m agent_orchestration.cli "Write three facts about the Apollo program to facts.txt, then read it back and summarise it"
```

Options: `--provider anthropic|openai`, `--model <name>`, `--workspace <dir>`
(default `./workspace`). Subtask progress goes to stderr, the final answer to
stdout, so you can pipe the result on its own.

Task history and escalations go to PostgreSQL (`--database-url`, or
`$DATABASE_URL`); conversation memory and checkpoints live under
`.orchestration/` (`--memory`, `--checkpoints`). `--no-memory` disables
persistence entirely. On first use with memory enabled, ChromaDB downloads an
~80MB local embedding model.

### Concurrent dispatch

By default subagents run in-process, one at a time — no broker needed. Add
`--queue` to dispatch them concurrently to Celery workers. That needs Redis:

```bash
docker compose up -d redis
```

and at least one worker (`CELERY_BROKER_URL` overrides the default
`redis://localhost:6379/0`):

```bash
.venv/bin/python -m celery -A agent_orchestration.queue worker --concurrency 4
```

Workers are separate processes, so nothing live crosses the boundary: each one
rebuilds its model, tools, and agent from the `AgentSpec`, which is already pure
data. Results are collected in submission order regardless of completion order.

## Tests

```bash
.venv/bin/python -m pytest
```

Unit tests drive the graph through a scripted stub model — no API calls, no
cost, deterministic. Two suites are excluded by default.

Integration tests, which need PostgreSQL and Redis running (no API key
required) — each gets a throwaway schema:

```bash
docker compose up -d postgres redis
.venv/bin/python -m pytest -m integration
```

The live end-to-end test, which needs a real API key:

```bash
.venv/bin/python -m pytest -m live
```

## How it works

```
             ┌── recall prior runs ──┐
goal → decompose → dispatch ──────────────→ aggregate → result
                       │                        ↑   └── persist run ──┘
                       └─→ escalate ────────────┘
                          (repair exhausted)    └─→ abort → END
```

- **`supervisor.py`** — the LangGraph graph. `decompose` asks the model for
  subtasks *and* an `AgentSpec` for each in one structured-output call, with any
  relevant prior-run memory folded into the prompt; `dispatch` runs them in
  order; `aggregate` writes the final answer and persists the run.
- **`store.py`** — task history, agent configs, and escalation logs in
  PostgreSQL. Every statement lives here, so nothing else knows the schema.
- **`auth.py`** — the OAuth device flow and local token cache.
- **`memory.py`** — conversation memory in ChromaDB, retrieved by similarity to
  the current goal.
- **`queue.py`** — the Celery app and the subagent task. JSON-only serialisation,
  so an unserialisable payload fails loudly rather than silently pickling.
- **`repair.py`** — the only place a subtask is actually run. On failure it asks
  the model to diagnose the cause and rewrite the agent's prompt and tool set,
  then retries. Both dispatch paths go through here, so `--queue` doesn't change
  failure behaviour.
- **`escalation.py`** — how a human is asked, and what they can answer. Both the
  interactive and durable routes return the same `EscalationDecision`.
- **`subagent.py`** — the only place a subagent is constructed. It knows nothing
  about any specific role: role, system prompt, and tool subset all arrive in
  the `AgentSpec` at runtime.
- **`tools.py`** — `read_file`, `write_file`, `list_files`, all confined to the
  workspace directory by a single path guard.
- **`state.py`** — the schemas. `AgentSpec` is the pivot: it is the only thing
  distinguishing one subagent from another.

### Self-repair

A failing subagent is diagnosed and retried up to `--max-attempts` (default 3)
before being given up on. Failure is reported, never raised: one unrecoverable
subtask doesn't destroy its siblings' results. Whatever repair can't rescue is
escalated.

### Escalation

When self-repair is exhausted the run escalates to a human, and every escalation
is logged with its full error and diagnosis history. Two modes:

**Interactive** (default at a terminal) — you're prompted to `retry` with
guidance, `skip` the failed subtasks, or `abort`. When stdin isn't a terminal the
run aborts instead of hanging, so automation never blocks on an unanswered
prompt.

**Durable** (`--durable`) — the run halts, persists its state, and exits so a
human can respond later from a different process:

```bash
orchestrate "some goal" --durable          # exits 3, prints an escalation id
orchestrate --list-escalations
orchestrate --respond 1 --action retry --guidance "use read_file instead"
```

Exit codes: `0` completed, `1` usage error, `2` aborted, `3` awaiting a human.

Both modes produce the same `EscalationDecision`, so the retry/skip/abort logic
is shared rather than duplicated. A guided retry that fails again finishes the
run with the failure recorded — it does not bounce back to the operator forever.

## License

MIT — see [LICENSE](LICENSE).
