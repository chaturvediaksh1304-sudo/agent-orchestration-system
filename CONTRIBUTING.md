# Contributing

This project exists to be read and forked, so the conventions below are less
about style and more about keeping the properties the design depends on.

## Getting set up

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"
docker compose up -d postgres redis
export DATABASE_URL=postgresql://orchestration:orchestration@localhost:5432/orchestration
```

## Running the tests

Three suites, deliberately separated by what they need:

```bash
pytest                  # 112 tests. Offline, deterministic, free. Must always pass.
pytest -m integration   # 46 tests. Needs PostgreSQL + Redis. No API key.
pytest -m live          #  1 test.  Needs a real ANTHROPIC_API_KEY or OPENAI_API_KEY.
```

The offline suite drives the graph through `StubChatModel` (see `tests/conftest.py`),
which replays scripted responses. Nothing in the default run costs money, touches
the network, or downloads a model — please keep it that way. If a change makes an
offline test need a service, move it behind the `integration` marker instead of
adding a service dependency to the default run.

Integration tests each get a throwaway Postgres schema and skip cleanly when no
database is reachable, so they never fail for want of infrastructure.

## Conventions worth preserving

A few things are load-bearing, and a well-meant refactor can quietly break them:

- **`subagent.py` is the only place a subagent is constructed.** Role, system
  prompt and tool subset all arrive in the `AgentSpec` at runtime. Introducing a
  registry of predefined roles would defeat the point — subagents are generated,
  not selected.
- **`store.py` holds every SQL statement.** Nothing else imports `psycopg` or
  knows the schema. That confinement is what made the SQLite-to-Postgres swap a
  one-file change.
- **`repair.py` is the only place a subtask actually runs.** Both the in-process
  and Celery paths go through it, so failure behaviour can't diverge between them.
- **Failure is returned, not raised**, across the Celery boundary. Exceptions
  don't survive a JSON result serializer with their detail intact, and a raise
  would let one doomed subtask destroy its siblings' results.
- **Anything placed before `interrupt()` in the escalate node runs twice.**
  LangGraph re-executes a node from the top on resume. This already caused one
  duplicate-escalation bug; the escalation log is idempotent per thread as a result.
- **`StubChatModel.responses` is typed `List[Any]` on purpose.** Pydantic's smart
  union tries to coerce a scripted exception into an `AIMessage` and fails inside
  its validator.

## Making a change

Write the test first — the whole project was built that way, and every
done-criterion in `Phases.md` has tests behind it. Then:

```bash
pytest && pytest -m integration
```

CI runs both suites plus a Docker build on every pull request.

## Project documents

`PRD.md`, `Architecture.md`, `Rules.md`, and `Phases.md` are the original
specification, kept unedited. `Memory.md` is the running record: what was built,
which done-criteria are verified, decisions and why, and the gaps that are
deliberately still open. Read it before assuming something is an oversight —
several are documented tradeoffs.
