"""Phase 2 done-criterion 1: task history and agent configs persist and read back.

Since the Phase 6 swap these run against real PostgreSQL, as Architecture.md
specifies, so they need a live database and carry the integration marker.
Each test gets a throwaway schema (see the `store` fixture in conftest).
"""

import pytest

from agent_orchestration.state import AgentSpec, RepairAttempt, Subtask, SubtaskResult
from agent_orchestration.store import Store

pytestmark = pytest.mark.integration


def _result(role, description="do a thing", tools=None, output="done"):
    return SubtaskResult(
        subtask=Subtask(
            description=description,
            spec=AgentSpec(
                role=role, system_prompt=f"You are the {role}.", tools=tools or ["read_file"]
            ),
        ),
        output=output,
    )


def test_saved_run_reads_back(store):
    run_id = store.save_run("build a thing", [_result("writer")], "the final answer")

    run = store.get_run(run_id)

    assert run.goal == "build a thing"
    assert run.final == "the final answer"


def test_agent_configs_persist_in_full(store):
    """The AgentSpec is the 'agent config' the criterion refers to."""
    run_id = store.save_run(
        "goal",
        [_result("researcher", tools=["read_file", "list_files"])],
        "final",
    )

    spec = store.get_run(run_id).results[0].subtask.spec

    assert spec.role == "researcher"
    assert spec.system_prompt == "You are the researcher."
    assert spec.tools == ["read_file", "list_files"]


def test_multiple_subtasks_keep_their_order(store):
    run_id = store.save_run(
        "goal",
        [_result("first", output="a"), _result("second", output="b"), _result("third", output="c")],
        "final",
    )

    results = store.get_run(run_id).results

    assert [r.subtask.spec.role for r in results] == ["first", "second", "third"]
    assert [r.output for r in results] == ["a", "b", "c"]


def test_runs_are_listed_newest_first(store):
    store.save_run("first goal", [_result("a")], "f1")
    store.save_run("second goal", [_result("b")], "f2")

    assert [r.goal for r in store.load_runs()] == ["second goal", "first goal"]


def test_load_runs_respects_limit(store):
    for i in range(5):
        store.save_run(f"goal {i}", [_result("a")], "f")

    assert len(store.load_runs(limit=2)) == 2


def test_empty_store_lists_nothing(store):
    assert store.load_runs() == []


def test_get_unknown_run_returns_none(store):
    assert store.get_run(999) is None


def test_history_survives_a_new_connection(store):
    """Persistence across process restarts is the whole point of the criterion."""
    run_id = store.save_run("durable goal", [_result("writer")], "final")

    reopened = Store(store.dsn).get_run(run_id)

    assert reopened.goal == "durable goal"
    assert reopened.results[0].subtask.spec.role == "writer"


def test_run_with_no_subtasks_is_still_recorded(store):
    run_id = store.save_run("trivial goal", [], "final")

    assert store.get_run(run_id).results == []


def test_repair_history_persists(store):
    """A failed subtask must not read back as successful."""
    failed = SubtaskResult(
        subtask=Subtask(
            description="doomed",
            spec=AgentSpec(role="writer", system_prompt="p", tools=[]),
        ),
        output="",
        ok=False,
        attempts=3,
        failures=[
            RepairAttempt(error="RuntimeError: fail 1", diagnosis="cause 1"),
            RepairAttempt(error="RuntimeError: fail 2", diagnosis=None),
        ],
    )

    reloaded = store.get_run(store.save_run("goal", [failed], "final")).results[0]

    assert reloaded.ok is False
    assert reloaded.attempts == 3
    assert [f.diagnosis for f in reloaded.failures] == ["cause 1", None]


def test_successful_subtasks_read_back_as_successful(store):
    reloaded = store.get_run(store.save_run("goal", [_result("writer")], "f")).results[0]

    assert reloaded.ok is True
    assert reloaded.failures == []


def test_opening_an_existing_database_is_idempotent(store):
    """Schema creation runs on every open, so it must not fail the second time."""
    run_id = store.save_run("goal", [_result("writer")], "final")

    assert Store(store.dsn).get_run(run_id).goal == "goal"


def test_the_run_records_who_ran_it(store):
    """Identity plumbing for hosted multi-user (Phase 6)."""
    run_id = store.save_run("goal", [_result("writer")], "final", user_id="octocat")

    assert store.get_run(run_id).user_id == "octocat"
