"""Worker bootstrap for the concurrency integration test.

Lives in tests/ rather than the package so no test-only task ships in
production code. Celery workers import this module via `-A worker_probe`
with PYTHONPATH=tests.

The probe sleeps instead of calling a model: it measures whether the queue
genuinely runs tasks in parallel, which needs no API key.
"""

import os
import time

from agent_orchestration.queue import app  # noqa: F401  (the worker needs this app)


@app.task(name="tests.sleep_probe")
def sleep_probe(seconds: float) -> dict:
    """Sleep, then report which process ran it and when."""
    start = time.time()
    time.sleep(seconds)
    return {"pid": os.getpid(), "started": start, "finished": time.time()}


@app.task(name="tests.repair_probe")
def repair_probe(fail_times: int, max_attempts: int) -> dict:
    """Run the real self-repair loop worker-side against a scripted failure.

    Uses StubChatModel rather than a real model, so Phase 4's queue path can be
    proven end to end without an API key. The repair logic itself is the real
    `execute_with_repair`, not a reimplementation.
    """
    import tempfile

    from langchain_core.messages import AIMessage

    from agent_orchestration.repair import execute_with_repair
    from agent_orchestration.state import AgentSpec
    from agent_orchestration.tools import build_tools
    from conftest import StubChatModel, diagnosis_message

    script = []
    for i in range(fail_times):
        script.append(RuntimeError(f"scripted failure {i + 1}"))
        script.append(diagnosis_message(f"diagnosed cause {i + 1}"))
    script.append(AIMessage(content="recovered in the worker"))

    outcome = execute_with_repair(
        AgentSpec(role="worker", system_prompt="You work.", tools=["write_file"]),
        "do the thing",
        StubChatModel(responses=script),
        build_tools(tempfile.mkdtemp()),
        max_attempts=max_attempts,
    )
    return {"pid": os.getpid(), **outcome.model_dump()}
