"""Phase 3 criterion 3: the supervisor waits for and aggregates async results.

The queue layer is stubbed here so these stay deterministic; the real broker is
exercised in test_queue_integration.py.
"""

import pytest
from langchain_core.messages import AIMessage

from agent_orchestration.supervisor import build_supervisor
from agent_orchestration.tools import build_tools


@pytest.fixture
def captured_dispatch(monkeypatch):
    """Replaces the queue with a recorder that returns results out of order."""
    calls = {}

    def fake_dispatch(subtasks, workspace, provider, model, timeout=300, max_attempts=3):
        calls["subtasks"] = list(subtasks)
        calls["provider"] = provider
        calls["max_attempts"] = max_attempts
        return [
            {
                "ok": True,
                "output": f"result for {description}",
                "attempts": 1,
                "failures": [],
            }
            for _, description in subtasks
        ]

    monkeypatch.setattr(
        "agent_orchestration.supervisor.dispatch_subtasks", fake_dispatch
    )
    return calls


def _llm(stub_model, two_subtasks):
    return stub_model([two_subtasks, AIMessage(content="the final answer")])


def test_async_dispatch_enqueues_every_subtask(
    stub_model, two_subtasks, tmp_path, captured_dispatch
):
    build_supervisor(
        _llm(stub_model, two_subtasks), build_tools(tmp_path), use_queue=True
    ).invoke({"goal": "g"})

    assert len(captured_dispatch["subtasks"]) == 2
    assert [spec.role for spec, _ in captured_dispatch["subtasks"]] == [
        "writer",
        "summariser",
    ]


def test_async_results_aggregate_in_subtask_order(
    stub_model, two_subtasks, tmp_path, captured_dispatch
):
    result = build_supervisor(
        _llm(stub_model, two_subtasks), build_tools(tmp_path), use_queue=True
    ).invoke({"goal": "g"})

    assert [r.output for r in result["results"]] == [
        "result for Write the facts to a file",
        "result for Read the file back and summarise it",
    ]


def test_async_run_still_reaches_a_final_answer(
    stub_model, two_subtasks, tmp_path, captured_dispatch
):
    result = build_supervisor(
        _llm(stub_model, two_subtasks), build_tools(tmp_path), use_queue=True
    ).invoke({"goal": "g"})

    assert result["final"] == "the final answer"


@pytest.mark.integration
def test_async_results_are_persisted_like_sync_ones(
    stub_model, two_subtasks, tmp_path, captured_dispatch, stub_embeddings, store
):
    from agent_orchestration.memory import ConversationMemory

    build_supervisor(
        _llm(stub_model, two_subtasks),
        build_tools(tmp_path),
        store=store,
        memory=ConversationMemory(tmp_path / "chroma", embedding_function=stub_embeddings),
        use_queue=True,
    ).invoke({"goal": "async goal"})

    runs = store.load_runs()
    assert runs[0].goal == "async goal"
    assert [r.subtask.spec.role for r in runs[0].results] == ["writer", "summariser"]


def test_provider_is_passed_through_to_the_workers(
    stub_model, two_subtasks, tmp_path, captured_dispatch
):
    """Workers rebuild their own model, so they need the provider choice."""
    build_supervisor(
        _llm(stub_model, two_subtasks),
        build_tools(tmp_path),
        use_queue=True,
        provider="openai",
    ).invoke({"goal": "g"})

    assert captured_dispatch["provider"] == "openai"


def test_sync_dispatch_remains_the_default(stub_model, two_subtasks, tmp_path, monkeypatch):
    """Phases 1 and 2 behaviour must not change unless the queue is asked for."""

    def explode(*args, **kwargs):
        raise AssertionError("the queue must not be used unless use_queue=True")

    monkeypatch.setattr("agent_orchestration.supervisor.dispatch_subtasks", explode)
    llm = stub_model(
        [two_subtasks, AIMessage(content="a"), AIMessage(content="b"), AIMessage(content="done")]
    )

    result = build_supervisor(llm, build_tools(tmp_path)).invoke({"goal": "g"})

    assert result["final"] == "done"
