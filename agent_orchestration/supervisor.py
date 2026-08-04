"""The supervisor graph: decompose -> dispatch -> aggregate.

The supervisor never hardcodes what a subagent is. It writes an AgentSpec per
subtask at runtime and hands it to the factory in subagent.py.
"""

from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from agent_orchestration.escalation import EscalationHandler, abort_handler, describe
from agent_orchestration.memory import ConversationMemory
from agent_orchestration.queue import dispatch_subtasks
from agent_orchestration.repair import SubtaskOutcome, execute_with_repair
from agent_orchestration.state import (
    AgentSpec,
    Decomposition,
    EscalationDecision,
    OrchestrationState,
    Subtask,
    SubtaskResult,
)
from agent_orchestration.store import Store
from agent_orchestration.subagent import build_subagent

DECOMPOSE_PROMPT = """You are a supervisor agent. Break the user's goal into the \
smallest set of ordered subtasks that accomplishes it.

For each subtask, design the agent that should carry it out: give it a role, \
write its system prompt, and grant it only the tools it actually needs.

Available tools:
{tool_descriptions}
{memory}
Goal: {goal}"""

MEMORY_PREAMBLE = """
Relevant memory from earlier runs — use it if it helps, ignore it if not:
{recalled}
"""

AGGREGATE_PROMPT = """You are a supervisor agent. Your subagents have completed \
their subtasks. Write the final answer to the user's original goal, drawing on \
their results. Answer the goal directly; do not describe the process.

Goal: {goal}

Subtask results:
{results}"""


def decompose_goal(
    goal: str,
    llm: BaseChatModel,
    registry: Optional[Dict[str, BaseTool]] = None,
    recalled: Optional[List[str]] = None,
) -> List[Subtask]:
    """Ask the model to split ``goal`` into subtasks and design an agent for each."""
    tool_descriptions = (
        "\n".join(f"- {name}: {t.description}" for name, t in registry.items())
        if registry
        else "(none)"
    )
    memory = MEMORY_PREAMBLE.format(recalled="\n\n".join(recalled)) if recalled else ""
    decomposition = llm.with_structured_output(Decomposition).invoke(
        DECOMPOSE_PROMPT.format(
            tool_descriptions=tool_descriptions, memory=memory, goal=goal
        )
    )
    if not decomposition.subtasks:
        raise ValueError(f"The supervisor returned no subtasks for goal: {goal!r}")
    return decomposition.subtasks


def build_supervisor(
    llm: BaseChatModel,
    registry: Dict[str, BaseTool],
    store: Optional[Store] = None,
    memory: Optional[ConversationMemory] = None,
    use_queue: bool = False,
    workspace: Optional[Path] = None,
    provider: str = "anthropic",
    model: Optional[str] = None,
    max_attempts: int = 3,
    handler: Optional[EscalationHandler] = None,
    use_interrupt: bool = False,
    checkpointer=None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> CompiledStateGraph:
    """Compile the orchestration graph.

    ``store`` and ``memory`` are optional so the loop still runs without any
    persistence configured, as it did in Phase 1. ``use_queue`` sends subtasks to
    Celery workers instead of running them in-process; it needs ``workspace``,
    ``provider`` and ``model`` because each worker rebuilds its own agent.

    When self-repair leaves subtasks unrecovered the run escalates. With
    ``use_interrupt`` it halts and persists via ``checkpointer``, to be resumed
    by a later process; otherwise ``handler`` is asked in-process. Both routes
    produce an EscalationDecision, so everything downstream is shared.
    """
    decide = handler or abort_handler

    def decompose(state: OrchestrationState) -> OrchestrationState:
        recalled = memory.recall(state["goal"]) if memory else None
        return {"subtasks": decompose_goal(state["goal"], llm, registry, recalled)}

    def dispatch(state: OrchestrationState) -> OrchestrationState:
        subtasks = state["subtasks"]
        if use_queue:
            outcomes = [
                SubtaskOutcome(**raw)
                for raw in dispatch_subtasks(
                    [(s.spec, s.description) for s in subtasks],
                    workspace=workspace or Path("workspace"),
                    provider=provider,
                    model=model,
                    max_attempts=max_attempts,
                )
            ]
        else:
            # ponytail: in-process fallback runs subtasks one at a time. Kept so the
            # system works with no broker; use_queue=True is the concurrent path.
            outcomes = [
                execute_with_repair(
                    s.spec, s.description, llm, registry, max_attempts=max_attempts
                )
                for s in subtasks
            ]

        results = [
            SubtaskResult(
                subtask=subtask,
                output=outcome.output,
                ok=outcome.ok,
                attempts=outcome.attempts,
                failures=outcome.failures,
            )
            for subtask, outcome in zip(subtasks, outcomes)
        ]
        # Phase 5 escalates these to a human; Phase 4 only has to detect them.
        return {"results": results, "unrepaired": [r for r in results if not r.ok]}

    def escalate(state: OrchestrationState) -> OrchestrationState:
        """Hand the unrecovered subtasks to a human and act on their answer."""
        unrepaired = state["unrepaired"]
        escalation_id = None
        if store:
            # Resuming re-runs this node from the top, so reuse the escalation
            # already open for this thread instead of logging a duplicate and
            # stranding the original as 'pending' forever.
            existing = store.find_pending_escalation(thread_id) if thread_id else None
            escalation_id = (
                existing.id
                if existing
                else store.save_escalation(
                    state["goal"], unrepaired, thread_id=thread_id, user_id=user_id
                )
            )

        if use_interrupt:
            # Halts here. The run is resumed later with Command(resume=...),
            # and execution re-enters this node with `raw` set to that value.
            raw = interrupt(
                {
                    "escalation_id": escalation_id,
                    "goal": state["goal"],
                    "detail": describe(state["goal"], unrepaired),
                }
            )
            decision = (
                raw if isinstance(raw, EscalationDecision) else EscalationDecision(**raw)
            )
        else:
            decision = decide(state["goal"], unrepaired)

        if store and escalation_id is not None:
            store.resolve_escalation(escalation_id, decision)

        if decision.action == "abort":
            return {"escalation_id": escalation_id, "decision": decision, "status": "aborted"}

        if decision.action == "retry":
            repaired = _retry_with_guidance(unrepaired, decision.guidance)
            by_description = {r.subtask.description: r for r in repaired}
            results = [by_description.get(r.subtask.description, r) for r in state["results"]]
            return {
                "escalation_id": escalation_id,
                "decision": decision,
                "results": results,
                "unrepaired": [r for r in results if not r.ok],
                "status": "running",
            }

        # skip: keep the partial results and carry on to aggregate.
        return {"escalation_id": escalation_id, "decision": decision, "status": "running"}

    def _retry_with_guidance(
        unrepaired: List[SubtaskResult], guidance: str
    ) -> List[SubtaskResult]:
        """Re-run failed subtasks with the human's guidance folded into the prompt."""
        retried = []
        for result in unrepaired:
            spec = result.subtask.spec
            guided = AgentSpec(
                role=spec.role,
                system_prompt=f"{spec.system_prompt}\n\nGuidance from a human "
                f"operator after earlier attempts failed: {guidance}",
                tools=spec.tools,
            )
            outcome = execute_with_repair(
                guided, result.subtask.description, llm, registry, max_attempts=max_attempts
            )
            retried.append(
                SubtaskResult(
                    subtask=result.subtask,
                    output=outcome.output,
                    ok=outcome.ok,
                    # Attempts accumulate across the escalation, so the history
                    # reads as one continuous struggle rather than a fresh start.
                    attempts=result.attempts + outcome.attempts,
                    failures=result.failures + outcome.failures,
                )
            )
        return retried

    def aggregate(state: OrchestrationState) -> OrchestrationState:
        rendered = "\n\n".join(
            f"[{r.subtask.spec.role}] {r.subtask.description}\n{r.output}"
            for r in state["results"]
        )
        final = llm.invoke(
            AGGREGATE_PROMPT.format(goal=state["goal"], results=rendered)
        )
        if store:
            run_id = store.save_run(
                state["goal"], state["results"], final.content, user_id=user_id
            )
            if memory:
                memory.remember(run_id, state["goal"], final.content)
        return {"final": final.content, "status": "completed"}

    graph = StateGraph(OrchestrationState)
    graph.add_node("decompose", decompose)
    graph.add_node("dispatch", dispatch)
    graph.add_node("escalate", escalate)
    graph.add_node("aggregate", aggregate)
    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "dispatch")
    graph.add_conditional_edges(
        "dispatch",
        lambda state: "escalate" if state.get("unrepaired") else "aggregate",
        {"escalate": "escalate", "aggregate": "aggregate"},
    )
    graph.add_conditional_edges(
        "escalate",
        # An aborted run stops without aggregating: there is no answer to give.
        lambda state: END if state.get("status") == "aborted" else "aggregate",
        {END: END, "aggregate": "aggregate"},
    )
    graph.add_edge("aggregate", END)
    return graph.compile(checkpointer=checkpointer)
