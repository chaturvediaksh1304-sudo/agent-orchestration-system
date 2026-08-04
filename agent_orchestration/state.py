"""Schemas for the orchestration graph.

AgentSpec is the pivot of the design: the supervisor authors one per subtask at
runtime, and it is the only thing that distinguishes one subagent from another.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class AgentSpec(BaseModel):
    """A subagent, described entirely as data by the supervisor at runtime."""

    role: str = Field(description="Short label for what this agent is, e.g. 'researcher'.")
    system_prompt: str = Field(
        description="The full system prompt for this agent, written for its specific subtask."
    )
    tools: List[str] = Field(
        default_factory=list,
        description="Names of the tools this agent needs. Omit tools it has no use for.",
    )


class Subtask(BaseModel):
    """One unit of the decomposed goal, with the agent built to handle it."""

    description: str = Field(description="What this subtask must accomplish.")
    spec: AgentSpec = Field(description="The agent to carry out this subtask.")


class Decomposition(BaseModel):
    """The supervisor's structured-output schema for the decompose step."""

    subtasks: List[Subtask] = Field(
        description="Ordered subtasks that together accomplish the goal."
    )


class RepairAttempt(BaseModel):
    """One failed attempt at a subtask, and what was concluded from it."""

    error: str
    diagnosis: Optional[str] = Field(
        default=None,
        description="None when no repair followed — attempts ran out, or diagnosis itself failed.",
    )


class SubtaskResult(BaseModel):
    subtask: Subtask
    output: str
    ok: bool = True
    attempts: int = 1
    failures: List[RepairAttempt] = Field(default_factory=list)


class EscalationDecision(BaseModel):
    """A human's answer to an escalation.

    Both escalation paths — the in-process handler and the durable interrupt —
    produce this same type, which is what stops them drifting apart.
    """

    action: Literal["retry", "skip", "abort"] = Field(
        description="retry the failed subtasks with guidance, skip them, or abandon the run."
    )
    guidance: str = Field(
        default="", description="Instructions for the retry. Ignored by skip and abort."
    )


class OrchestrationState(TypedDict, total=False):
    """State threaded through the supervisor graph."""

    goal: str
    subtasks: List[Subtask]
    results: List[SubtaskResult]
    # Subtasks self-repair could not rescue; these are what get escalated.
    unrepaired: List[SubtaskResult]
    escalation_id: Optional[int]
    decision: Optional[EscalationDecision]
    status: str
    final: Optional[str]
