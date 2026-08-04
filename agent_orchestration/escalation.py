"""Handing an unrecoverable run to a human.

Two paths reach a decision — an in-process handler and a durable interrupt the
run resumes from later — but both return the same EscalationDecision, so the
supervisor acts on one code path regardless of which was used.
"""

import sys
from typing import Callable, List, TextIO

from agent_orchestration.state import EscalationDecision, SubtaskResult

EscalationHandler = Callable[[str, List[SubtaskResult]], EscalationDecision]

ACTIONS = ("retry", "skip", "abort")


def describe(goal: str, unrepaired: List[SubtaskResult]) -> str:
    """Render an escalation so a human can act on it without reading the code."""
    lines = [
        f"Self-repair could not complete {len(unrepaired)} subtask(s) for goal: {goal}",
        "",
    ]
    for result in unrepaired:
        lines.append(
            f"  [{result.subtask.spec.role}] {result.subtask.description} "
            f"({result.attempts} attempt(s))"
        )
        for failure in result.failures:
            lines.append(f"      error:     {failure.error}")
            if failure.diagnosis:
                lines.append(f"      diagnosis: {failure.diagnosis}")
    return "\n".join(lines)


def abort_handler(goal: str, unrepaired: List[SubtaskResult]) -> EscalationDecision:
    """Default when nobody is watching.

    Automation must never block on a prompt that will not be answered, so an
    unattended run stops rather than hanging. The escalation is still logged.
    """
    return EscalationDecision(action="abort")


def interactive_handler(
    goal: str,
    unrepaired: List[SubtaskResult],
    stdin: TextIO = None,
    stdout: TextIO = None,
) -> EscalationDecision:
    """Ask the operator on the terminal."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stderr

    print(describe(goal, unrepaired), file=stdout)
    print(f"\nHow should this proceed? {'/'.join(ACTIONS)}", file=stdout)

    while True:
        stdout.flush()
        action = (stdin.readline() or "abort").strip().lower()
        if action in ACTIONS:
            break
        # EOF: the stream is closed and re-prompting would spin forever.
        if action == "":
            return EscalationDecision(action="abort")
        print(f"Choose one of: {', '.join(ACTIONS)}", file=stdout)

    guidance = ""
    if action == "retry":
        print("Guidance for the retry:", file=stdout)
        stdout.flush()
        guidance = (stdin.readline() or "").strip()
    return EscalationDecision(action=action, guidance=guidance)
