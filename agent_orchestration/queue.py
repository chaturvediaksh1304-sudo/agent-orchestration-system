"""Async subagent dispatch via Celery and Redis.

Workers are separate processes, so nothing live can cross the boundary — no
model client, no tool objects. Only the AgentSpec (already pure data by design)
and the few strings needed to rebuild everything worker-side.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from celery import Celery

from agent_orchestration.state import AgentSpec

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")

app = Celery("agent_orchestration", broker=BROKER_URL, backend=BROKER_URL)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    # Celery defaults to pickle for results; JSON-only keeps the payload contract
    # honest and refuses anything that isn't plain data.
    accept_content=["json"],
)


def _execute(
    spec: Dict,
    description: str,
    workspace: str,
    provider: str,
    model: Optional[str],
    max_attempts: int = 3,
) -> Dict:
    """Rebuild the agent worker-side and run one subtask, with self-repair.

    Imported lazily: a worker process needs these, but a supervisor that only
    enqueues should not pay for loading model clients.

    Returns a plain dict because Celery results are JSON-serialised.
    """
    from agent_orchestration.llm import build_llm
    from agent_orchestration.repair import execute_with_repair
    from agent_orchestration.tools import build_tools

    outcome = execute_with_repair(
        AgentSpec(**spec),
        description,
        build_llm(provider, model),
        build_tools(Path(workspace)),
        max_attempts=max_attempts,
    )
    return outcome.model_dump()


@app.task(name="agent_orchestration.run_subagent")
def run_subagent_task(
    spec: Dict,
    description: str,
    workspace: str,
    provider: str,
    model: Optional[str],
    max_attempts: int = 3,
) -> Dict:
    return _execute(spec, description, workspace, provider, model, max_attempts)


def dispatch_subtasks(
    subtasks: Sequence[Tuple[AgentSpec, str]],
    workspace: Path,
    provider: str,
    model: Optional[str],
    timeout: int = 300,
    max_attempts: int = 3,
) -> List[Dict]:
    """Enqueue every subtask, then collect outcomes in submission order.

    Tasks run concurrently across workers; the returned list is ordered by
    submission, not completion, so aggregation stays deterministic.

    Each entry is a SubtaskOutcome dict — a subtask that exhausted self-repair
    comes back with ok=False rather than raising, so one unrecoverable subtask
    does not destroy the results of its siblings.
    """
    if not subtasks:
        return []

    async_results = [
        run_subagent_task.delay(
            spec.model_dump(), description, str(workspace), provider, model, max_attempts
        )
        for spec, description in subtasks
    ]
    return [result.get(timeout=timeout) for result in async_results]
