"""One task, one arm, in a process of its own.

    python -m bench.run_task <spec.json> <out.json>

The spec holds the prompt and where to work. It never contains the ground
truth; scoring happens back in the driver.

baseline  one ReAct agent with every tool. No supervisor, no repair loop,
          no queue, no persistent memory. Built with the system's own
          subagent factory, so the agent itself is identical.
full      the system as shipped: supervisor, runtime-authored subagents,
          Celery dispatch, diagnose-and-retry, PostgreSQL + ChromaDB.
"""

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict

from bench.config import EMBED_MODEL, MAX_ATTEMPTS, OLLAMA_URL

BASELINE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools when they help you answer."
)


def _clip(text, limit=400):
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + "…"


def run_baseline(spec: Dict, record: Dict) -> None:
    from agent_orchestration.llm import build_llm
    from agent_orchestration.state import AgentSpec
    from agent_orchestration.subagent import build_subagent
    from agent_orchestration.tools import build_tools

    registry = build_tools(Path(spec["workspace"]))
    agent = build_subagent(
        AgentSpec(role="assistant", system_prompt=BASELINE_SYSTEM_PROMPT, tools=list(registry)),
        build_llm("openai", spec["model"]),
        registry,
    )
    result = agent.invoke({"messages": [("user", spec["prompt"])]})
    record["final_text"] = result["messages"][-1].content
    record["subagents_spawned"] = 1
    record["retries_triggered"] = 0


def run_full(spec: Dict, record: Dict, flush) -> None:
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

    from agent_orchestration.llm import build_llm
    from agent_orchestration.memory import ConversationMemory
    from agent_orchestration.store import Store
    from agent_orchestration.supervisor import build_supervisor
    from agent_orchestration.tools import build_tools

    workspace = Path(spec["workspace"])
    graph = build_supervisor(
        build_llm("openai", spec["model"]),
        build_tools(workspace),
        store=Store(spec["dsn"]),
        memory=ConversationMemory(
            spec["chroma_dir"],
            embedding_function=OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=EMBED_MODEL),
        ),
        use_queue=True,
        workspace=workspace,
        provider="openai",
        model=spec["model"],
        max_attempts=MAX_ATTEMPTS,
    )

    # stream() runs the same graph as invoke(); it just lets progress be written
    # after each node, so a run killed by the timeout still shows how far it got.
    state: Dict = {}
    for update in graph.stream({"goal": spec["prompt"]}, stream_mode="updates"):
        for delta in update.values():
            state.update(delta or {})
        _summarise(state, record)
        flush()

    record["final_text"] = state.get("final")
    record["status"] = state.get("status")


def _summarise(state: Dict, record: Dict) -> None:
    subtasks = state.get("subtasks") or []
    results = state.get("results") or []
    record["subagents_spawned"] = len(subtasks)
    record["retries_triggered"] = sum(max(0, r.attempts - 1) for r in results)
    record["status"] = state.get("status")
    by_description = {r.subtask.description: r for r in results}
    record["subtasks"] = []
    for subtask in subtasks:
        result = by_description.get(subtask.description)
        record["subtasks"].append(
            {
                "role": subtask.spec.role,
                "tools": subtask.spec.tools,
                "description": _clip(subtask.description, 300),
                "ok": result.ok if result else None,
                "attempts": result.attempts if result else None,
                "failures": [
                    {"error": _clip(f.error), "diagnosis": _clip(f.diagnosis)}
                    for f in (result.failures if result else [])
                ],
            }
        )


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    record: Dict = {"finished": False, "final_text": None, "status": None,
                    "error_type": None, "error_message": None}

    def flush() -> None:
        temp = out.with_suffix(".tmp")
        temp.write_text(json.dumps(record))
        os.replace(temp, out)

    try:
        if spec["arm"] == "baseline":
            run_baseline(spec, record)
        else:
            run_full(spec, record, flush)
        record["finished"] = True
    except Exception as exc:
        record["error_type"] = f"{type(exc).__module__}.{type(exc).__name__}"
        record["error_message"] = _clip(exc, 1500)
        traceback.print_exc()
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
