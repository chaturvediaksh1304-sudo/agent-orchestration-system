"""Self-repair must behave identically with and without the queue.

Both dispatch paths funnel through execute_with_repair, so these tests guard the
claim that turning --queue on doesn't change failure handling.
"""

from conftest import decomposition_message, diagnosis_message
from langchain_core.messages import AIMessage

from agent_orchestration.supervisor import build_supervisor
from agent_orchestration.tools import build_tools


def _one_subtask(stub_model_factory, extra):
    """A decomposition of exactly one subtask, then whatever `extra` scripts."""
    return stub_model_factory(
        [decomposition_message(("do the thing", "worker", ["write_file"]))] + extra
    )


def test_sync_dispatch_repairs_a_failing_subagent(stub_model, tmp_path):
    llm = _one_subtask(
        stub_model,
        [
            RuntimeError("subagent blew up"),
            diagnosis_message("it needed a clearer prompt"),
            AIMessage(content="recovered output"),
            AIMessage(content="the final answer"),
        ],
    )

    result = build_supervisor(llm, build_tools(tmp_path)).invoke({"goal": "g"})

    assert result["results"][0].ok
    assert result["results"][0].output == "recovered output"
    assert result["results"][0].attempts == 2
    assert result["unrepaired"] == []


def test_sync_dispatch_detects_unrepairable_failure(stub_model, tmp_path):
    """Criterion 3 at the supervisor level."""
    llm = _one_subtask(
        stub_model,
        [
            RuntimeError("fail 1"),
            diagnosis_message("cause 1"),
            RuntimeError("fail 2"),
            diagnosis_message("cause 2"),
            RuntimeError("fail 3"),
            AIMessage(content="the final answer"),
        ],
    )

    result = build_supervisor(llm, build_tools(tmp_path), max_attempts=3).invoke({"goal": "g"})

    assert len(result["unrepaired"]) == 1
    assert result["unrepaired"][0].attempts == 3
    assert not result["results"][0].ok


def test_a_failing_subtask_does_not_destroy_its_siblings(stub_model, two_subtasks, tmp_path):
    """One unrecoverable subtask must not take the whole run down with it.

    Since Phase 5 this escalates rather than aggregating, and the unattended
    default aborts — but the sibling's work is still intact, which is the point.
    """
    llm = stub_model(
        [
            two_subtasks,
            RuntimeError("first subtask is doomed"),
            AIMessage(content="second subtask fine"),
        ]
    )

    result = build_supervisor(llm, build_tools(tmp_path), max_attempts=1).invoke({"goal": "g"})

    assert not result["results"][0].ok
    assert result["results"][1].ok
    assert result["results"][1].output == "second subtask fine"


def test_skipping_the_failure_still_aggregates_the_siblings(stub_model, two_subtasks, tmp_path):
    """A human choosing 'skip' gets an answer built from the partial results."""
    from agent_orchestration.state import EscalationDecision

    llm = stub_model(
        [
            two_subtasks,
            RuntimeError("first subtask is doomed"),
            AIMessage(content="second subtask fine"),
            AIMessage(content="the final answer"),
        ]
    )

    result = build_supervisor(
        llm,
        build_tools(tmp_path),
        max_attempts=1,
        handler=lambda goal, unrepaired: EscalationDecision(action="skip"),
    ).invoke({"goal": "g"})

    assert result["final"] == "the final answer"
    assert result["status"] == "completed"


def test_max_attempts_reaches_the_queue_workers(stub_model, two_subtasks, tmp_path, monkeypatch):
    """Workers repair on their own side, so they must be told the budget."""
    seen = {}

    def fake_dispatch(subtasks, workspace, provider, model, timeout=300, max_attempts=3):
        seen["max_attempts"] = max_attempts
        return [
            {"ok": True, "output": "x", "attempts": 1, "failures": []} for _ in subtasks
        ]

    monkeypatch.setattr("agent_orchestration.supervisor.dispatch_subtasks", fake_dispatch)
    llm = stub_model([two_subtasks, AIMessage(content="final")])

    build_supervisor(llm, build_tools(tmp_path), use_queue=True, max_attempts=5).invoke(
        {"goal": "g"}
    )

    assert seen["max_attempts"] == 5


def test_queue_path_surfaces_unrepaired_subtasks(stub_model, two_subtasks, tmp_path, monkeypatch):
    """An ok=False outcome from a worker must reach state['unrepaired']."""

    def fake_dispatch(subtasks, workspace, provider, model, timeout=300, max_attempts=3):
        return [
            {
                "ok": False,
                "output": "",
                "attempts": 3,
                "failures": [{"error": "RuntimeError: worker gave up", "diagnosis": None}],
            }
            for _ in subtasks
        ]

    monkeypatch.setattr("agent_orchestration.supervisor.dispatch_subtasks", fake_dispatch)
    llm = stub_model([two_subtasks, AIMessage(content="final")])

    result = build_supervisor(llm, build_tools(tmp_path), use_queue=True).invoke({"goal": "g"})

    assert len(result["unrepaired"]) == 2
    assert "worker gave up" in result["unrepaired"][0].failures[0].error
