"""Phase 5 criterion 3, durable variant: the run halts and a later run resumes it.

These use an in-process checkpointer for speed; the genuine cross-process proof
is test_escalation_integration.py, which spawns separate interpreters.
"""

import sqlite3

import pytest
from conftest import decomposition_message
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent_orchestration.state import EscalationDecision
from agent_orchestration.store import Store
from agent_orchestration.supervisor import build_supervisor
from agent_orchestration.tools import build_tools

ONE_SUBTASK = decomposition_message(("do the thing", "worker", ["write_file"]))


@pytest.fixture
def saver(tmp_path):
    conn = sqlite3.connect(tmp_path / "checkpoints.db", check_same_thread=False)
    yield SqliteSaver(conn)
    conn.close()


def _durable(llm, tmp_path, saver, store=None, thread="t1"):
    return build_supervisor(
        llm,
        build_tools(tmp_path),
        store=store,
        max_attempts=1,
        use_interrupt=True,
        checkpointer=saver,
        thread_id=thread,
    )


def _doomed(stub_model, extra=()):
    return stub_model([ONE_SUBTASK, RuntimeError("subtask is doomed")] + list(extra))


CONFIG = {"configurable": {"thread_id": "t1"}}


def test_the_run_halts_instead_of_finishing(stub_model, tmp_path, saver):
    result = _durable(_doomed(stub_model), tmp_path, saver).invoke({"goal": "g"}, config=CONFIG)

    assert "__interrupt__" in result
    assert result.get("final") is None


@pytest.mark.integration
def test_the_halt_payload_tells_a_human_what_happened(stub_model, tmp_path, saver, store):

    result = _durable(_doomed(stub_model), tmp_path, saver, store).invoke(
        {"goal": "the halted goal"}, config=CONFIG
    )

    payload = result["__interrupt__"][0].value
    assert payload["goal"] == "the halted goal"
    assert payload["escalation_id"] == store.load_escalations()[0].id
    assert "subtask is doomed" in payload["detail"]


@pytest.mark.integration
def test_the_escalation_is_logged_before_the_halt(stub_model, tmp_path, saver, store):
    """A halt nobody recorded is a lost run."""

    _durable(_doomed(stub_model), tmp_path, saver, store).invoke({"goal": "g"}, config=CONFIG)

    assert store.load_escalations(status="pending")[0].thread_id == "t1"


def test_resuming_with_retry_completes_the_run(stub_model, tmp_path, saver):
    llm = _doomed(stub_model, [AIMessage(content="worked"), AIMessage(content="the final")])
    graph = _durable(llm, tmp_path, saver)
    graph.invoke({"goal": "g"}, config=CONFIG)

    resumed = graph.invoke(
        Command(resume=EscalationDecision(action="retry", guidance="do better").model_dump()),
        config=CONFIG,
    )

    assert resumed["status"] == "completed"
    assert resumed["final"] == "the final"
    assert resumed["results"][0].output == "worked"


def test_resuming_with_abort_terminates_cleanly(stub_model, tmp_path, saver):
    graph = _durable(_doomed(stub_model), tmp_path, saver)
    graph.invoke({"goal": "g"}, config=CONFIG)

    resumed = graph.invoke(
        Command(resume=EscalationDecision(action="abort").model_dump()), config=CONFIG
    )

    assert resumed["status"] == "aborted"
    assert resumed.get("final") is None


def test_resuming_with_skip_aggregates_the_partial_results(stub_model, tmp_path, saver):
    llm = _doomed(stub_model, [AIMessage(content="the final")])
    graph = _durable(llm, tmp_path, saver)
    graph.invoke({"goal": "g"}, config=CONFIG)

    resumed = graph.invoke(
        Command(resume=EscalationDecision(action="skip").model_dump()), config=CONFIG
    )

    assert resumed["final"] == "the final"


@pytest.mark.integration
def test_resuming_resolves_the_logged_escalation(stub_model, tmp_path, saver, store):
    llm = _doomed(stub_model, [AIMessage(content="ok"), AIMessage(content="final")])
    graph = _durable(llm, tmp_path, saver, store)
    graph.invoke({"goal": "g"}, config=CONFIG)

    graph.invoke(
        Command(resume=EscalationDecision(action="retry", guidance="advice").model_dump()),
        config=CONFIG,
    )

    escalation = store.load_escalations()[0]
    assert escalation.status == "resolved"
    assert escalation.guidance == "advice"


@pytest.mark.integration
def test_resuming_does_not_duplicate_the_escalation(stub_model, tmp_path, saver, store):
    """LangGraph re-runs the node body on resume, so the log must be idempotent.

    Otherwise every resume writes a second escalation and leaves the original
    stuck as 'pending' forever.
    """
    llm = _doomed(stub_model, [AIMessage(content="ok"), AIMessage(content="final")])
    graph = _durable(llm, tmp_path, saver, store)
    graph.invoke({"goal": "g"}, config=CONFIG)

    graph.invoke(
        Command(resume=EscalationDecision(action="retry", guidance="x").model_dump()),
        config=CONFIG,
    )

    escalations = store.load_escalations()
    assert len(escalations) == 1
    assert escalations[0].status == "resolved"
    assert store.load_escalations(status="pending") == []


def test_the_decision_object_is_accepted_as_well_as_a_dict(stub_model, tmp_path, saver):
    """Both escalation paths yield an EscalationDecision; resume must take either."""
    llm = _doomed(stub_model, [AIMessage(content="ok"), AIMessage(content="final")])
    graph = _durable(llm, tmp_path, saver)
    graph.invoke({"goal": "g"}, config=CONFIG)

    resumed = graph.invoke(
        Command(resume=EscalationDecision(action="retry", guidance="x")), config=CONFIG
    )

    assert resumed["status"] == "completed"


def test_a_run_that_never_escalates_needs_no_resume(stub_model, tmp_path, saver):
    llm = stub_model([ONE_SUBTASK, AIMessage(content="fine"), AIMessage(content="final")])

    result = _durable(llm, tmp_path, saver).invoke({"goal": "g"}, config=CONFIG)

    assert "__interrupt__" not in result
    assert result["final"] == "final"
