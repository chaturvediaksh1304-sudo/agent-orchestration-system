"""Phase 2 done-criterion 3: a second run can reference memory from a prior run.

Also covers the wiring itself — that a completed run lands in both stores.
"""

import pytest
from langchain_core.messages import AIMessage

from agent_orchestration.memory import ConversationMemory
from agent_orchestration.store import Store
from agent_orchestration.supervisor import build_supervisor
from agent_orchestration.tools import build_tools


pytestmark = pytest.mark.integration


@pytest.fixture
def persistence(tmp_path, stub_embeddings, store):
    return (
        store,
        ConversationMemory(tmp_path / "chroma", embedding_function=stub_embeddings),
    )


def _run(llm, tmp_path, store, memory, goal):
    return build_supervisor(
        llm, build_tools(tmp_path), store=store, memory=memory
    ).invoke({"goal": goal})


def _simple_llm(stub_model, two_subtasks, final="the final answer"):
    return stub_model(
        [
            two_subtasks,
            AIMessage(content="first done"),
            AIMessage(content="second done"),
            AIMessage(content=final),
        ]
    )


def test_completed_run_is_written_to_task_history(
    stub_model, two_subtasks, tmp_path, persistence
):
    store, memory = persistence

    _run(_simple_llm(stub_model, two_subtasks), tmp_path, store, memory, "persist me")

    runs = store.load_runs()
    assert len(runs) == 1
    assert runs[0].goal == "persist me"
    assert runs[0].final == "the final answer"
    assert [r.subtask.spec.role for r in runs[0].results] == ["writer", "summariser"]


def test_completed_run_is_written_to_conversation_memory(
    stub_model, two_subtasks, tmp_path, persistence
):
    store, memory = persistence

    _run(_simple_llm(stub_model, two_subtasks), tmp_path, store, memory, "remember me")

    assert "remember me" in memory.recall("remember me")[0]


def test_second_run_sees_the_first_runs_memory(
    stub_model, two_subtasks, tmp_path, persistence
):
    """The criterion itself: prior-run memory reaches the second run's supervisor."""
    store, memory = persistence
    _run(
        _simple_llm(stub_model, two_subtasks, final="Apollo 11 landed in 1969."),
        tmp_path,
        store,
        memory,
        "research the apollo program",
    )

    second_llm = _simple_llm(stub_model, two_subtasks)
    _run(second_llm, tmp_path, store, memory, "research the apollo program")

    decompose_prompt = str(second_llm.seen_messages[0])
    assert "Apollo 11 landed in 1969." in decompose_prompt


def test_first_run_has_no_prior_memory_to_reference(
    stub_model, two_subtasks, tmp_path, persistence
):
    store, memory = persistence
    llm = _simple_llm(stub_model, two_subtasks)

    _run(llm, tmp_path, store, memory, "a brand new goal")

    assert "Relevant memory from earlier runs" not in str(llm.seen_messages[0])


def test_supervisor_runs_without_persistence_configured(
    stub_model, two_subtasks, tmp_path
):
    """Phase 1 behaviour must survive: store and memory stay optional."""
    result = build_supervisor(
        _simple_llm(stub_model, two_subtasks), build_tools(tmp_path)
    ).invoke({"goal": "no persistence here"})

    assert result["final"] == "the final answer"


def test_memory_survives_a_fresh_process(
    stub_model, two_subtasks, tmp_path, stub_embeddings, store
):
    """Both stores reopened from scratch, as a separate CLI invocation would."""
    chroma = tmp_path / "chroma"
    _run(
        _simple_llm(stub_model, two_subtasks, final="durable knowledge"),
        tmp_path,
        Store(store.dsn),
        ConversationMemory(chroma, embedding_function=stub_embeddings),
        "a goal worth remembering",
    )

    second_llm = _simple_llm(stub_model, two_subtasks)
    _run(
        second_llm,
        tmp_path,
        Store(store.dsn),
        ConversationMemory(chroma, embedding_function=stub_embeddings),
        "a goal worth remembering",
    )

    assert "durable knowledge" in str(second_llm.seen_messages[0])
    assert len(Store(store.dsn).load_runs()) == 2
