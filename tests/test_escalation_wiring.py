"""Phase 5 criteria 1 and 3, in-process.

  1. escalation triggers after self-repair exhausts its attempts
  3. a human can respond and the run resumes or terminates cleanly
"""

import pytest
from conftest import decomposition_message, diagnosis_message
from langchain_core.messages import AIMessage

from agent_orchestration.state import EscalationDecision
from agent_orchestration.store import Store
from agent_orchestration.supervisor import build_supervisor
from agent_orchestration.tools import build_tools

ONE_SUBTASK = decomposition_message(("do the thing", "worker", ["write_file"]))


def _handler(action, guidance=""):
    return lambda goal, unrepaired: EscalationDecision(action=action, guidance=guidance)


def _doomed(stub_model, extra=()):
    """A run whose single subtask fails on its only attempt."""
    return stub_model([ONE_SUBTASK, RuntimeError("subtask is doomed")] + list(extra))


def test_escalation_does_not_trigger_when_everything_succeeds(stub_model, tmp_path):
    seen = []
    llm = stub_model([ONE_SUBTASK, AIMessage(content="fine"), AIMessage(content="final")])

    result = build_supervisor(
        llm,
        build_tools(tmp_path),
        handler=lambda goal, unrepaired: seen.append(goal) or EscalationDecision(action="skip"),
    ).invoke({"goal": "g"})

    assert seen == []
    assert result["final"] == "final"


def test_escalation_triggers_once_repair_is_exhausted(stub_model, tmp_path):
    """Criterion 1."""
    seen = {}

    def handler(goal, unrepaired):
        seen["goal"] = goal
        seen["unrepaired"] = unrepaired
        return EscalationDecision(action="abort")

    build_supervisor(
        _doomed(stub_model), build_tools(tmp_path), max_attempts=1, handler=handler
    ).invoke({"goal": "the goal"})

    assert seen["goal"] == "the goal"
    assert len(seen["unrepaired"]) == 1
    assert seen["unrepaired"][0].attempts == 1


def test_escalation_waits_for_repair_to_finish_first(stub_model, tmp_path):
    """Escalating before self-repair has tried is a Rules.md violation."""
    seen = {}
    llm = stub_model(
        [
            ONE_SUBTASK,
            RuntimeError("fail 1"),
            diagnosis_message("cause 1"),
            RuntimeError("fail 2"),
        ]
    )

    build_supervisor(
        llm,
        build_tools(tmp_path),
        max_attempts=2,
        handler=lambda goal, unrepaired: seen.update(n=unrepaired[0].attempts)
        or EscalationDecision(action="abort"),
    ).invoke({"goal": "g"})

    assert seen["n"] == 2


def test_abort_terminates_without_aggregating(stub_model, tmp_path):
    """Criterion 3, termination half: stop cleanly, don't invent an answer."""
    result = build_supervisor(
        _doomed(stub_model), build_tools(tmp_path), max_attempts=1, handler=_handler("abort")
    ).invoke({"goal": "g"})

    assert result["status"] == "aborted"
    assert result.get("final") is None


def test_retry_with_guidance_resumes_the_run(stub_model, tmp_path):
    """Criterion 3, resume half."""
    llm = _doomed(
        stub_model, [AIMessage(content="worked with guidance"), AIMessage(content="the final")]
    )

    result = build_supervisor(
        llm,
        build_tools(tmp_path),
        max_attempts=1,
        handler=_handler("retry", "try using read_file"),
    ).invoke({"goal": "g"})

    assert result["status"] == "completed"
    assert result["final"] == "the final"
    assert result["results"][0].ok
    assert result["results"][0].output == "worked with guidance"


def test_the_humans_guidance_reaches_the_retry(stub_model, tmp_path):
    llm = _doomed(
        stub_model, [AIMessage(content="ok"), AIMessage(content="final")]
    )

    build_supervisor(
        llm,
        build_tools(tmp_path),
        max_attempts=1,
        handler=_handler("retry", "MAGIC GUIDANCE STRING"),
    ).invoke({"goal": "g"})

    assert any("MAGIC GUIDANCE STRING" in str(m) for m in llm.seen_messages)


def test_attempts_accumulate_across_the_escalation(stub_model, tmp_path):
    """History should read as one continuous struggle, not a fresh start."""
    llm = _doomed(
        stub_model, [AIMessage(content="ok"), AIMessage(content="final")]
    )

    result = build_supervisor(
        llm, build_tools(tmp_path), max_attempts=1, handler=_handler("retry", "g")
    ).invoke({"goal": "g"})

    assert result["results"][0].attempts == 2
    assert len(result["results"][0].failures) == 1


@pytest.mark.integration
def test_a_retry_that_also_fails_does_not_re_escalate(stub_model, tmp_path, store):
    """The human's answer must not create an infinite escalation loop.

    A guided retry that fails again finishes the run with the failure on record,
    rather than bouncing back to the operator forever.
    """
    calls = []
    llm = stub_model(
        [
            ONE_SUBTASK,
            RuntimeError("first failure"),
            RuntimeError("retry failed too"),
            AIMessage(content="final, noting the failure"),
        ]
    )

    def handler(goal, unrepaired):
        calls.append(goal)
        return EscalationDecision(action="retry", guidance="g")

    result = build_supervisor(
        llm, build_tools(tmp_path), store=store, max_attempts=1, handler=handler
    ).invoke({"goal": "g"})

    assert len(calls) == 1
    assert len(store.load_escalations()) == 1
    assert not result["results"][0].ok
    assert result["final"] == "final, noting the failure"


@pytest.mark.integration
def test_escalation_is_logged_and_resolved_in_the_store(stub_model, tmp_path, store):
    """Criterion 2, through the graph rather than the store directly."""

    build_supervisor(
        _doomed(stub_model),
        build_tools(tmp_path),
        store=store,
        max_attempts=1,
        handler=_handler("abort"),
    ).invoke({"goal": "the escalated goal"})

    escalations = store.load_escalations()
    assert len(escalations) == 1
    assert escalations[0].goal == "the escalated goal"
    assert escalations[0].status == "aborted"
    assert escalations[0].unrepaired[0].failures[0].error.endswith("subtask is doomed")


@pytest.mark.integration
def test_a_resolved_escalation_records_the_guidance(stub_model, tmp_path, store):
    llm = _doomed(stub_model, [AIMessage(content="ok"), AIMessage(content="final")])

    build_supervisor(
        llm,
        build_tools(tmp_path),
        store=store,
        max_attempts=1,
        handler=_handler("retry", "the operator's advice"),
    ).invoke({"goal": "g"})

    escalation = store.load_escalations()[0]
    assert escalation.status == "resolved"
    assert escalation.action == "retry"
    assert escalation.guidance == "the operator's advice"


def test_unattended_runs_default_to_aborting(stub_model, tmp_path):
    """No handler configured must not mean hanging on a prompt."""
    result = build_supervisor(
        _doomed(stub_model), build_tools(tmp_path), max_attempts=1
    ).invoke({"goal": "g"})

    assert result["status"] == "aborted"
