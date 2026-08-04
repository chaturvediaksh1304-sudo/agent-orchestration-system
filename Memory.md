# Agent Orchestration System — Memory

## Last updated
2026-08-04 — end of Phase 6. **All six phases complete.**

## Done
- **Phase 1 (core orchestration loop) — 2 of 3 done-criteria verified.**
  - C1 *supervisor produces a subtask decomposition* — **verified** (`tests/test_supervisor.py`).
  - C2 *subagents dynamically generated, not hardcoded, calling ≥1 tool* — **verified**
    (`tests/test_subagent.py`). Structural check: one `create_react_agent` call site
    (`subagent.py`), no role-named classes, no subagent role prompts in source.
  - C3 *full end-to-end run against a real model* — **still NOT verified**, see "In progress".
- **Phase 2 (persistent memory) — all 3 done-criteria verified.**
  - C1 *task history + agent configs written to and readable from PostgreSQL* — **verified
    against real PostgreSQL** (`tests/test_store.py`). Was provisional on SQLite from Phase 2
    until the Phase 6 swap closed it.
  - C2 *conversation memory written to and retrievable from ChromaDB* — **verified**
    (`tests/test_memory.py`, 7 tests).
  - C3 *a second run can reference memory from a prior run* — **verified**
    (`tests/test_persistence_wiring.py`, 6 tests) **and confirmed across a real process
    boundary**: two separate `python` invocations against the same on-disk stores — the
    first saw no prior memory, the second recalled the first's exact output.
- **Phase 3 (task queue) — all 3 done-criteria verified.**
  - C1 *subagent tasks enqueued to Celery via Redis* — **verified** (`tests/test_queue.py`).
  - C2 *multiple subagents execute concurrently* — **verified against real Redis and real
    worker processes** (`tests/test_queue_integration.py`, `-m integration`). Measured:
    4 × 2s tasks finished in **2.02s wall clock (3.96× speedup)** across **4 distinct pids**,
    with all execution windows overlapping. Three independent checks — wall-clock, distinct
    pids, and overlapping start/finish windows — so a lucky fast sequential run can't pass.
  - C3 *supervisor waits for and aggregates async results* — **verified**
    (`tests/test_async_dispatch.py`), including that results aggregate in **submission order,
    not completion order**.
- **Phase 4 (self-repair) — all 3 done-criteria verified.**
  - C1 *a deliberately failing subagent triggers a diagnosis step* — **verified**
    (`tests/test_repair.py`).
  - C2 *at least one automated retry/fix based on the diagnosis* — **verified**, including
    that the retry actually *uses* the diagnosed prompt and tool set rather than replaying
    the original spec (a retry that ignored the diagnosis would just repeat the failure).
  - C3 *repeated failure after self-repair is correctly detected* — **verified** at both the
    repair level and the supervisor level (`state["unrepaired"]`), and **through real Celery
    workers** (`-m integration`) with the failure history intact across the JSON boundary.
- **Phase 5 (human-in-the-loop escalation) — all 3 done-criteria verified.**
  - C1 *escalation triggers after self-repair exhausts its attempts* — **verified**
    (`tests/test_escalation_wiring.py`), including that it waits for repair to finish first
    rather than escalating early, and does not fire when everything succeeds.
  - C2 *escalation is logged in PostgreSQL* — **verified against real PostgreSQL**
    (`tests/test_escalation_store.py`). Full error + diagnosis history is logged. Was
    provisional on SQLite until the Phase 6 swap closed it.
  - C3 *a human can respond and the run resumes or terminates cleanly* — **verified both
    ways**: in-process handler (`test_escalation_wiring.py`) and durable interrupt/resume
    (`test_durable_escalation.py`), plus a **genuine four-process cycle** — run halts (exit 3),
    a fresh process lists the escalation, a third responds `--action retry --guidance ...`,
    the run resumes and completes (exit 0), a fourth confirms one resolved escalation and
    the persisted run.
- **Phase 6 (auth + CLI polish + Docker) — all 3 done-criteria met, one partially.**
  - C1 *OAuth and/or magic link auth flow works* — **implemented and tested against a stub
    provider** (`tests/test_auth.py`, 24 tests): device flow per RFC 8628, honouring
    authorization_pending / slow_down / expired_token / access_denied, token cached at 0600.
    **The live handshake is unverified** — it needs a GitHub OAuth app client id that only
    Aksh can register. Not faked.
  - C2 *full system runs via `docker compose up`* — **verified**. postgres + redis + celery
    worker come up healthy, schema is created in a clean database, the worker answers
    `app.control.ping`, and the CLI runs inside the container.
  - C3 *CLI end-to-end run works from a clean environment* — **verified except the live model
    call**. A full orchestration ran inside a fresh container: decompose → dispatch → tools
    writing to the workspace volume → aggregate → persisted to compose Postgres with
    `user_id` attached. A real goal stops at `ANTHROPIC_API_KEY is not set`, the same
    long-standing blocker.
- **SQLite → PostgreSQL swap done (Phase 6).** Phase 2 C1 and Phase 5 C2 are no longer
  provisional; they are now genuinely met against the database Architecture.md specifies.
- 88 tests in the default (offline) suite + 46 integration + 1 live.

## In progress
Nothing is mid-implementation. Two things need Aksh's credentials to finish verifying:

- **Phase 1 C3** — `tests/test_live_e2e.py` skips for want of a key. `ANTHROPIC_BASE_URL` is
  plain `https://api.anthropic.com`, **not** an authenticating proxy, so a placeholder does
  not work (tried in Phase 3; don't retry it).
  ```bash
  export ANTHROPIC_API_KEY=...
  .venv/bin/python -m pytest -m live
  ```
- **Phase 6 C1 live handshake** — needs a GitHub OAuth app.
  ```bash
  export OAUTH_CLIENT_ID=...
  orchestrate login
  ```

## Key decisions made (and why)
- **Both providers, switchable at runtime** (P1) — `llm.py`, two branches, `--provider` flag.
- **Pin to system Python 3.9.6** (P1) — no new interpreter. See Deviations for the cost.
- **Stub-model unit tests + one `@pytest.mark.live` e2e** (P1) — TDD needs determinism.
  `StubChatModel` **must** override `bind_tools`: `BaseChatModel.with_structured_output`
  raises NotImplementedError otherwise and routes the schema through `bind_tools` +
  `PydanticToolsParser`. Load-bearing for every supervisor test.
- **`AgentSpec` fully LLM-authored** (P1) — role, system prompt *and* tool subset written at
  runtime. Strictest reading of "not hardcoded"; C2's tests defend it. Adding a role registry
  in a later phase would break that claim — revisit deliberately if ever tempted.
- **Tilde in a model-authored path stays literal** (P1) — `tools.py::_resolve` deliberately
  omits `expanduser()`, so `~/x` is a directory named `~` inside the workspace, not the real
  home dir. `expanduser()` on the workspace root is kept. A test pins this.
- **SQLite via stdlib `sqlite3`, all SQL confined to `store.py`** (P2) — nothing outside that
  module imports sqlite3 or knows the schema, so the Postgres swap is one file. No ORM added.
- **`chromadb==1.5.9`** (P2) — checked 0.4.24 / 0.5.23 / 1.0.0 / 1.5.9 on py3.9: the newest is
  also the **lightest** (80 packages, 8 heavy markers vs 85–90 / 13). There is no lighter pin.
- **Stub embedding function in tests, Chroma's local ONNX model at runtime** (P2) — keeps the
  suite offline, free, and free of the ~80MB download. `StubEmbeddingFunction` uses **md5, not
  `hash()`** — Python salts string hashing per process, which would corrupt persisted
  embeddings across restarts. Chroma 1.x also requires `name`/`get_config`/`build_from_config`
  on a custom embedding function, not just `__call__`.
- **`store`/`memory` are optional on `build_supervisor`** (P2) — Phase 1 behaviour survives, and
  `--no-memory` gives a genuinely persistence-free run (tested: creates nothing on disk).
- **Celery payloads are JSON-only, never pickle** (P3) — `queue.py` sets
  `accept_content=["json"]`. Celery's pickle default would silently accept unserialisable
  payloads; JSON makes the worker boundary fail loudly instead. A test pins this.
- **Workers rebuild everything from the `AgentSpec`** (P3) — no model client or tool object
  crosses the process boundary, only plain data plus workspace/provider/model strings. This
  needed **no redesign**: Phase 1's decision to make `AgentSpec` pure data already made
  subagents serialisable. Direct payoff.
- **Sync in-process dispatch stays the default; `--queue` opts in** (P3) — the system still
  runs with no broker at all. A test asserts the queue is never touched unless asked for.
- **Concurrency proven with a sleeping probe task, not a real subagent** (P3) — measures the
  queue itself, so C2 needed **no API key**. The probe lives in `tests/worker_probe.py`, not
  the package, so no test-only task ships in production code. Workers load it via
  `-A worker_probe` with `PYTHONPATH=tests`.
- **Integration tests wait on `app.control.ping` before timing** (P3) — starting the clock
  before the worker is ready would make a concurrency assertion meaningless.
- **Repair lives in ONE place, `repair.py::execute_with_repair`** (P4) — both `supervisor.py`
  (sync) and `queue.py::_execute` (worker-side) call it, so `--queue` cannot change failure
  behaviour. Implementing it per-path would have been two things to keep in sync.
- **Failure is returned as a `SubtaskOutcome`, never raised** (P4) — two reasons. Exceptions
  don't survive Celery's JSON result serializer with their detail intact, and a raise would
  let one doomed subtask destroy its siblings' results. A test pins the sibling case.
- **Diagnosis is skipped on the final attempt** (P4) — no retry follows it, so diagnosing
  would burn a model call for nothing.
- **Diagnosis failure is caught** (P4) — the repair machinery must not become a new failure
  mode of its own; a failed diagnosis ends the loop cleanly with the reason recorded.
- **`StubChatModel.responses` is `List[Any]`, not a Union** (P4) — pydantic's smart union
  tried to coerce a scripted `RuntimeError` into an `AIMessage` and blew up inside AIMessage's
  own validator. `_generate` dispatches on `isinstance` instead. Don't "tighten" this back.
- **Resuming an interrupt RE-RUNS the node body from the top** (P5) — this bit, and cost a
  real bug: `save_escalation` ran a second time on resume, creating a duplicate and stranding
  the original as `pending` forever. Fixed with `store.find_pending_escalation(thread_id)`, so
  the escalate node reuses the open escalation. **Any side effect placed before `interrupt()`
  will run twice** — remember this before adding anything to that node.
- **Both escalation routes return the same `EscalationDecision`** (P5) — the in-process handler
  and the durable interrupt converge on one type, so the retry/skip/abort logic exists once.
  This was the explicit mitigation for the "two paths to keep in sync" risk Aksh accepted when
  choosing to build both.
- **Unattended runs abort rather than prompt** (P5) — the default handler is `abort_handler`,
  and the CLI switches to it whenever stdin is not a tty. Automation must never block on a
  prompt nobody will answer. Interactive mode also aborts on EOF instead of spinning.
- **A guided retry that fails again does NOT re-escalate** (P5) — it proceeds to aggregate with
  the failure recorded. Re-escalating would loop forever. A test asserts exactly one escalation.
- **`--durable` is rejected with `--no-memory`** (P5) — found by actually running the CLI: with
  no store the escalation is never logged, so the halted run prints `--respond None` and can
  never be resumed. Now a clear error.
- **Escalations deliberately do not reference `runs(id)`** (P5) — escalation happens before
  aggregate, and an aborted run never saves a run row at all.
- **Postgres store, all SQL still in one file** (P6) — the swap touched `store.py` only,
  which is what the Phase 2 "confine the SQL" decision was for. Notable differences from the
  SQLite version: `%s` placeholders not `?`, `SERIAL` not `AUTOINCREMENT`, `RETURNING id`
  instead of `lastrowid`, and **JSONB comes back already decoded** — no `json.loads` on read.
  The SQLite `PRAGMA`-based column migration is gone; `CREATE TABLE IF NOT EXISTS` covers it.
- **Store tests are now integration tests** (P6) — the honest cost of the swap. They need a
  live Postgres and each gets a throwaway schema (`store` fixture in conftest), dropped after.
  They skip cleanly, not fail, when no database is reachable.
- **Provider is validated before the database is opened** (P6) — otherwise a bad `--provider`
  surfaced as a connection error instead of its own message.
- **The `cli` compose service sits behind a `profiles: ["cli"]` guard** (P6) — so
  `docker compose up` doesn't start a container that just prints help and exits. **Gotcha:
  plain `docker compose build` therefore SKIPS it**, which silently served a stale image and
  cost real debugging time. Rebuild with `docker compose --profile cli build`.
- **Auth gates only when configured** (P6) — with no `OAUTH_CLIENT_ID` the CLI runs
  unauthenticated. A CLI-first tool that is unusable without an OAuth provider would be worse
  than no auth at all. Device flow (RFC 8628) was chosen because it needs no client secret
  and no local web server.
- **`store.py` migrates old databases on open** (P4) — `CREATE TABLE IF NOT EXISTS` does not
  add columns, so a pre-Phase-4 database would silently read every failed subtask back as
  `ok=True`. Opening now checks `PRAGMA table_info` and `ALTER TABLE ADD COLUMN` for missing
  ones. A test builds a pre-Phase-4 schema and asserts it survives.

## Deviations from original plan
- **SQLite instead of PostgreSQL** (P2, Aksh's call) — **RESOLVED in Phase 6.** The swap
  touched `store.py` only, exactly as the "confine the SQL" decision intended.
- **Python 3.9 pins the project to the LangGraph 0.6 maintenance branch.** Verified with
  `uv pip compile`: 3.9 → `langgraph 0.6.11` / `langchain-core 0.3.86`; 3.10 → `langgraph 1.2.10`
  / `langchain-core 1.5.3`. The 0.6 → 1.x API differs, so a later upgrade is a real migration.
- **Phase 1 C3 left unverified while Phase 2 was built** (Aksh's call) — normally barred by the
  Rules.md looping mandate.
- **`urllib3` NotOpenSSLWarning — RISK NOW RETIRED (P3).** A live run with a placeholder key
  reached Anthropic and returned a clean HTTP 401, proving httpx negotiated TLS fine on
  LibreSSL 2.8.3. The warning is cosmetic. Also settled: `ANTHROPIC_BASE_URL` is plain
  `https://api.anthropic.com`, **not** an authenticating proxy — so Phase 1 C3 needs a genuine
  key, and the placeholder shortcut is a dead end. Don't retry it.
- No `python-dotenv`, no `typer`, no ORM — stdlib `argparse`, `sqlite3`, and env vars cover it.
- Repo is **not** a git repository. No `git init` run (not requested); no `git push` per Rules.md.

## Environment notes
- Redis runs as a Docker container: `docker run -d --name orchestration-redis -p 6379:6379
  redis:7-alpine`. **Docker Desktop must be running** — it was down at the start of Phase 3.
  Integration tests skip cleanly (not fail) when no broker is reachable.
- Start a worker with:
  `.venv/bin/python -m celery -A agent_orchestration.queue worker --concurrency 4`

## Known gaps (flagged, not silently resolved)
- **The live model call has never run.** `pytest -m live` still skips: no `ANTHROPIC_API_KEY`
  / `OPENAI_API_KEY`. This is the single outstanding item in the whole project. Everything
  around it is verified — TLS reaches Anthropic (a placeholder key returned a clean 401),
  model construction works, and the full graph runs against stubs in and out of Docker.
- **The live OAuth handshake has never run.** Needs a GitHub OAuth app client id. The flow is
  fully tested against a stub provider; only the real round trip is unproven.
- **Behaviour change in P5:** an unrepaired subtask now escalates instead of aggregating
  partial results, and unattended that means the run aborts. Phase 4's
  "a failing subtask does not destroy its siblings" test was updated to assert the sibling's
  work survives (the real intent) rather than a final answer appearing. Choosing `skip` at the
  escalation restores the old aggregate-partial-results behaviour.
- **Self-repair covers subagent failure only, not "system-wide" as Rules.md requires.**
  Rules.md says *"any component failure should default to try to fix it, then escalate."*
  Phase 4's done-criteria are all subagent-scoped, so that is what was built. Still
  unprotected: `supervisor.py::decompose` (a malformed structured output raises straight
  out) and `supervisor.py::aggregate`. **Aksh's call** whether to widen this in Phase 5 or
  leave it — it was not silently expanded.

## Next
1. **Close Phase 1 C3** — the only outstanding criterion in the whole project. Export a real
   key, run `pytest -m live`. Everything else on that path is confirmed working (TLS, model
   construction, graph); it is purely the key.
2. Register a GitHub OAuth app, `export OAUTH_CLIENT_ID=...`, and run `orchestrate login`
   to close the last piece of Phase 6 C1.
3. Await Aksh's final review. **All six phases are implemented**; Phases.md has nothing left.
4. Not started, and deliberately out of scope per PRD.md: any UI (Streamlit/React), which
   Architecture.md lists as "a possible later addition, not part of this build".
