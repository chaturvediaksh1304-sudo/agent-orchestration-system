"""Self-repair: a failing subagent diagnoses itself and retries before giving up.

Rules.md puts self-diagnosis ahead of everything else on failure, with escalation
only once repair is exhausted. This module is the single place a subtask is run,
so both the in-process and the Celery paths repair identically.

Failure is reported as a returned SubtaskOutcome, never a raised exception:
outcomes cross the Celery boundary, and exceptions do not survive a JSON result
serializer with their detail intact.
"""

from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent_orchestration.state import AgentSpec, RepairAttempt
from agent_orchestration.subagent import build_subagent

DIAGNOSE_PROMPT = """A subagent failed while working on a subtask. Diagnose why, \
then redesign it so the retry has a better chance.

Subtask: {description}
Its role: {role}
Its system prompt: {system_prompt}
Tools it was given: {tools}
Tools that exist: {available}

It failed with:
{error}

Give the cause, then a revised system prompt and revised tool list for the retry."""


class Diagnosis(BaseModel):
    """The model's structured read on why a subagent failed."""

    cause: str = Field(description="Why the subagent failed, in one or two sentences.")
    revised_system_prompt: str = Field(
        description="A system prompt for the retry that addresses the cause."
    )
    revised_tools: List[str] = Field(
        description="Tool names the retry should get. Only names that exist."
    )


class SubtaskOutcome(BaseModel):
    ok: bool
    output: str = ""
    attempts: int = 0
    failures: List[RepairAttempt] = Field(default_factory=list)


def _diagnose(
    spec: AgentSpec,
    description: str,
    error: str,
    llm: BaseChatModel,
    registry: Dict[str, BaseTool],
) -> Diagnosis:
    return llm.with_structured_output(Diagnosis).invoke(
        DIAGNOSE_PROMPT.format(
            description=description,
            role=spec.role,
            system_prompt=spec.system_prompt,
            tools=", ".join(spec.tools) or "(none)",
            available=", ".join(registry) or "(none)",
            error=error,
        )
    )


def execute_with_repair(
    spec: AgentSpec,
    description: str,
    llm: BaseChatModel,
    registry: Dict[str, BaseTool],
    max_attempts: int = 3,
) -> SubtaskOutcome:
    """Run one subtask, diagnosing and retrying on failure up to ``max_attempts``."""
    current = spec
    failures: List[RepairAttempt] = []

    for attempt in range(1, max_attempts + 1):
        try:
            agent = build_subagent(current, llm, registry)
            output = agent.invoke({"messages": [("user", description)]})
            return SubtaskOutcome(
                ok=True,
                output=output["messages"][-1].content,
                attempts=attempt,
                failures=failures,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt == max_attempts:
                # No retry follows, so diagnosing would just burn a model call.
                failures.append(RepairAttempt(error=error))
                break
            try:
                diagnosis = _diagnose(current, description, error, llm, registry)
            except Exception as diagnosis_exc:
                # Repair must not become a new failure mode of its own.
                failures.append(
                    RepairAttempt(
                        error=f"{error} (diagnosis also failed: "
                        f"{type(diagnosis_exc).__name__}: {diagnosis_exc})"
                    )
                )
                break
            failures.append(RepairAttempt(error=error, diagnosis=diagnosis.cause))
            current = AgentSpec(
                role=current.role,
                system_prompt=diagnosis.revised_system_prompt,
                tools=diagnosis.revised_tools,
            )

    return SubtaskOutcome(ok=False, attempts=len(failures), failures=failures)
