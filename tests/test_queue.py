"""Phase 3 criterion 1: subagent tasks are enqueued to Celery.

Eager mode runs tasks synchronously in-process, which pins the payload contract
and the task body deterministically. Genuine concurrency needs a real broker and
is covered by test_queue_integration.py.
"""

import pytest

from agent_orchestration.queue import app, run_subagent_task
from agent_orchestration.state import AgentSpec


@pytest.fixture
def eager():
    app.conf.task_always_eager = True
    yield
    app.conf.task_always_eager = False


SPEC = AgentSpec(role="writer", system_prompt="You write.", tools=["write_file"])


def test_payload_is_json_serialisable(tmp_path):
    """Workers are separate processes, so every argument must survive JSON."""
    import json

    payload = {
        "spec": SPEC.model_dump(),
        "description": "write a file",
        "workspace": str(tmp_path),
        "provider": "anthropic",
        "model": None,
    }

    assert json.loads(json.dumps(payload))["spec"]["role"] == "writer"


def test_task_is_registered_with_the_app():
    assert run_subagent_task.name in app.tasks


def test_app_uses_json_serialisation_only():
    """Celery's pickle default would silently accept unserialisable payloads."""
    assert app.conf.task_serializer == "json"
    assert app.conf.accept_content == ["json"]


def test_enqueueing_returns_results_in_submission_order(eager, tmp_path, monkeypatch):
    """Criterion 3's ordering guarantee, checked at the queue layer."""
    from agent_orchestration import queue as queue_module

    monkeypatch.setattr(
        queue_module,
        "_execute",
        lambda spec, description, workspace, provider, model, max_attempts=3: {
            "ok": True,
            "output": f"did: {description}",
            "attempts": 1,
            "failures": [],
        },
    )

    results = queue_module.dispatch_subtasks(
        [(SPEC, "first"), (SPEC, "second"), (SPEC, "third")],
        workspace=tmp_path,
        provider="anthropic",
        model=None,
    )

    assert [r["output"] for r in results] == ["did: first", "did: second", "did: third"]


def test_dispatch_of_nothing_returns_nothing(eager, tmp_path):
    from agent_orchestration import queue as queue_module

    assert queue_module.dispatch_subtasks([], workspace=tmp_path, provider="anthropic", model=None) == []
