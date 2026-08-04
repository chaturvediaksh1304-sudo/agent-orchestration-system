"""Phase 5 criterion 2: escalations are logged and readable.

Since the Phase 6 swap these run against real PostgreSQL, as Architecture.md
specifies, so they need a live database and carry the integration marker.
"""

import pytest

from agent_orchestration.state import (
    AgentSpec,
    EscalationDecision,
    RepairAttempt,
    Subtask,
    SubtaskResult,
)
from agent_orchestration.store import Store

pytestmark = pytest.mark.integration


def _unrepaired(role="writer", description="doomed subtask"):
    return SubtaskResult(
        subtask=Subtask(
            description=description,
            spec=AgentSpec(role=role, system_prompt="p", tools=["write_file"]),
        ),
        output="",
        ok=False,
        attempts=3,
        failures=[
            RepairAttempt(error="RuntimeError: fail 1", diagnosis="cause 1"),
            RepairAttempt(error="RuntimeError: fail 2", diagnosis=None),
        ],
    )


def test_escalation_is_logged_as_pending(store):
    escalation_id = store.save_escalation("the goal", [_unrepaired()], thread_id="t1")

    escalation = store.get_escalation(escalation_id)

    assert escalation.goal == "the goal"
    assert escalation.status == "pending"
    assert escalation.thread_id == "t1"


def test_the_full_failure_history_is_logged(store):
    """An escalation a human can't diagnose from is not worth logging."""
    escalation_id = store.save_escalation("the goal", [_unrepaired()], thread_id="t1")

    subtask = store.get_escalation(escalation_id).unrepaired[0]

    assert subtask.attempts == 3
    assert [f.diagnosis for f in subtask.failures] == ["cause 1", None]
    assert subtask.subtask.spec.role == "writer"


def test_several_failed_subtasks_are_logged_together(store):
    escalation_id = store.save_escalation(
        "the goal", [_unrepaired("writer"), _unrepaired("reader")], thread_id="t1"
    )

    assert len(store.get_escalation(escalation_id).unrepaired) == 2


def test_resolving_records_the_decision(store):
    escalation_id = store.save_escalation("the goal", [_unrepaired()], thread_id="t1")

    store.resolve_escalation(
        escalation_id, EscalationDecision(action="retry", guidance="use read_file instead")
    )

    escalation = store.get_escalation(escalation_id)
    assert escalation.status == "resolved"
    assert escalation.action == "retry"
    assert escalation.guidance == "use read_file instead"
    assert escalation.resolved_at is not None


def test_aborting_is_recorded_distinctly_from_resolving(store):
    escalation_id = store.save_escalation("the goal", [_unrepaired()], thread_id="t1")

    store.resolve_escalation(escalation_id, EscalationDecision(action="abort"))

    assert store.get_escalation(escalation_id).status == "aborted"


def test_pending_escalations_can_be_listed(store):
    first = store.save_escalation("goal 1", [_unrepaired()], thread_id="t1")
    store.save_escalation("goal 2", [_unrepaired()], thread_id="t2")
    store.resolve_escalation(first, EscalationDecision(action="skip"))

    pending = store.load_escalations(status="pending")

    assert [e.goal for e in pending] == ["goal 2"]


def test_all_escalations_can_be_listed_newest_first(store):
    store.save_escalation("goal 1", [_unrepaired()], thread_id="t1")
    store.save_escalation("goal 2", [_unrepaired()], thread_id="t2")

    assert [e.goal for e in store.load_escalations()] == ["goal 2", "goal 1"]


def test_unknown_escalation_returns_none(store):
    assert store.get_escalation(999) is None


def test_escalation_survives_a_new_connection(store):
    """A human responds in a later process, so this must outlive the run."""
    escalation_id = store.save_escalation("durable goal", [_unrepaired()], thread_id="t9")

    reopened = Store(store.dsn).get_escalation(escalation_id)

    assert reopened.goal == "durable goal"
    assert reopened.thread_id == "t9"
    assert reopened.unrepaired[0].attempts == 3


def test_find_pending_escalation_is_scoped_to_its_thread(store):
    """The resume path looks up by thread; it must not pick up someone else's."""
    store.save_escalation("goal 1", [_unrepaired()], thread_id="thread-a")
    store.save_escalation("goal 2", [_unrepaired()], thread_id="thread-b")

    assert store.find_pending_escalation("thread-a").goal == "goal 1"
    assert store.find_pending_escalation("thread-zzz") is None


def test_a_resolved_escalation_is_no_longer_pending_for_its_thread(store):
    escalation_id = store.save_escalation("goal", [_unrepaired()], thread_id="t1")
    store.resolve_escalation(escalation_id, EscalationDecision(action="skip"))

    assert store.find_pending_escalation("t1") is None


def test_escalation_does_not_require_a_saved_run(store):
    """Escalation happens before aggregate, so no run row exists yet."""
    store.save_escalation("the goal", [_unrepaired()], thread_id="t1")

    assert store.load_runs() == []
